import random
import sys
import time

import numpy as np
from gym_dogfight import Env
from gym_dogfight.spaces import Box, Discrete
import harfang as hg
from random import uniform
from math import radians


# 轨迹优化器类
class TrajectoryOptimizer:
    def __init__(self, tolerance=2.0):
        self.tolerance = tolerance
        self.trajectory_data = []
    
    def clear(self):
        self.trajectory_data = []
    
    def add_point(self, position, velocity, timestamp):
        self.trajectory_data.append({
            'position': position,
            'velocity': velocity,
            'timestamp': timestamp
        })
    
    def get_trajectory_data(self):
        return self.trajectory_data.copy()

sys.path.append('./src/')
sys.path.append('./src/environments/dogfightEnv/')
sys.path.append('./src/environments/dogfightEnv/dogfight_sandbox_hg2/network_client_example/')
sys.path.append('gym.envs.dogfightEnv.dogfight_sandbox_hg2')
try:
    from .dogfight_sandbox_hg2.network_client_example import \
        dogfight_client as df
    print("Gym in oneVSone")
    time.sleep(1)

except:
    from dogfight_sandbox_hg2.network_client_example import \
        dogfight_client as df
    print("DBRL")
    time.sleep(1)


class oneVSoneEnv_ap(Env):
    def __init__(self, host='192.168.5.8', port='57805', rendering=True, use_autopilot=False)-> None:
        self.host = host
        self.port = port
        self.nof = 0
        self.rendering = rendering
        self.step_game = 0  # 给本局设定结束条件，初定500步
        self.missle_count = 0  # 记录导弹的数量，发射的越多罚分越多
        self.enemy_set_mark = False
        self.target_lock_count = 0
        self.name = 'oneVSone'
        self.facing = 0
        self.aming = 0
        self.trajectory_length = 128
        self.use_autopilot = use_autopilot  # 是否使用自动驾驶功能
        self.current_skill = None  # 当前使用的技能
        self.skill_duration = 0  # 技能持续时间计数器
        
        # 初始化轨迹优化器
        self.trajectory_optimizer = TrajectoryOptimizer(tolerance=2.0)
        self.init_point = ['up', 'down', 'left_up', 'left_down', 'right_up', 'right_down', 'left', 'right']
        random.shuffle(self.init_point)
        try:
            df.get_planes_list()
        except:
            print('Run for the first time')
            df.connect(host, int(port))
            time.sleep(2)
        planes = df.get_planes_list()
        missiles = df.get_missiles_list()
        self.missleID = missiles[1]
        self.planeID = planes[0]
        self.enemyID = planes[1]
        df.disable_log()

        # 设定本方战机
        self.planeID = planes[0]

        # 为所有战机初始化 从这里开始显示各架飞机
        for i in planes:
            df.reset_machine(i)
        df.get_targets_list(self.planeID)
        # 两架飞机先飞着
        df.set_plane_thrust(self.planeID, 1)
        df.set_plane_thrust(self.enemyID, 1)
        # 设置成用户模式
        df.set_client_update_mode(True)
        if self.rendering:
            df.set_renderless_mode(False)
        else:
            df.set_renderless_mode(True)
        # 收起起落架
        df.retract_gear(planes[0])
        missles = df.get_machine_missiles_list(self.planeID)
        df.set_plane_linear_speed(self.planeID, 300)
        df.set_plane_linear_speed(self.enemyID, 300)
        self.action_space = Box(
            low=np.array([
                0,  # Roll 俯仰角
                0,  # Pitch 翻滚角
                0,  # thrust 油门
                0,  # fire 发射导弹
                0,  # skill activation 技能激活
            ]),
            high=np.array([
                1,
                1,
                1,
                1,
                1,  # skill activation 技能激活
            ]),
        )

        self.observation_space = Box(
            low=np.array([  # simple normalized
                -3,  # x / 1000
                -3,  # y / 1000
                -3,    # z / 1000
                -1,  # roll_attitude * 4
                -1,  # pitch_attitude * 4
                0,  # heading
                0,  # thrust level 油门
                0,  # linear speed 空速/1000
                -1,  # vertical speed 垂直速度/300
                0,  # horizontal speed 水平速度/500
                0,  # 是否锁定

                -3,  # x / 100 enemy
                -3,  # y / 100 enemy
                0,    # z / 50  enemy
                0,  # 是否锁定

                0,  # missile count 导弹发射计数/3
                0,  # current skill type 当前技能类型 (0=climb, 1=dive, 2=turn, 3=trace)
                0,  # skill duration remaining 技能剩余持续时间/100
            ]),
            high=np.array([
                3,  # x / 1000
                3,  # y / 1000
                3,  # z / 1000
                1,  # roll_attitude * 4
                1,  # pitch_attitude * 4
                2,  # heading
                1,  # thrust level 油门
                100,  # linear speed 空速/1000
                100,  # vertical speed 垂直速度/300
                100,  # horizontal speed 水平速度/500
                1,  # 是否锁定

                3,  # x / 100 enemy
                3,  # y / 100 enemy
                3,  # z / 50  enemy
                1,  # 是否锁定
                1,  # missile count 导弹发射计数/3
                3,  # current skill type 当前技能类型 (0=climb, 1=dive, 2=turn, 3=trace)
                1,  # skill duration remaining 技能剩余持续时间/100
            ])
        )

    def render_or_not(self, result):
        if result:
            df.set_renderless_mode(False)
        else:
            df.set_renderless_mode(True)

    def steps(self, action):
        # if self.step_game % 200 == 0:
        #     df.set_plane_pitch(self.enemyID, np.random.random()-0.5)
        #     df.set_plane_roll(self.enemyID, np.random.random()-0.5)
        while True:
            flag = df.get_finish_flag()
            if flag:
                break
        self.step_game += 1
        
        # 管理技能持续时间
        if self.current_skill and self.skill_duration > 0:
            self.skill_duration -= 1
            if self.skill_duration <= 0:
                self.current_skill = None
                print(f"技能已结束")
        
        new_plane_state = df.get_plane_state(self.planeID)
        new_plan_loc = new_plane_state['position']
        new_enemy_loc = df.get_plane_state(self.enemyID)['position']
        new_enemy_state = df.get_plane_state(self.enemyID)

        terminate_value = self.terminate(new_plane_state['health_level'], new_enemy_state['health_level'])
        terminate = True if terminate_value else False
        reward = self.reward(new_plane_state['linear_speed'], terminate_value, new_plane_state['health_level'], new_enemy_state['health_level'], new_plane_state['target_locked'], new_enemy_state['target_locked'])
        reward = np.array(reward)
        Euler_plane = new_plane_state['Euler_angles']
        plane_roll = Euler_plane[2] / np.pi
        plane_pitch = Euler_plane[0] / np.pi
        plane_yaw = Euler_plane[1] / np.pi
        Euler_enemy = new_enemy_state['Euler_angles']
        self.facing = self.facing_angle(plane_roll, new_plane_state['heading'], new_plan_loc[0], new_plan_loc[2], new_enemy_loc[0],                                        new_enemy_loc[2])
        
        # 记录轨迹点
        self._record_trajectory_point(new_plane_state)
        new_ob = [  # normalized
            (new_plan_loc[0] - 1000) / 1000,
            (new_plan_loc[2] - 4000) / 2000,
            (new_plan_loc[1] - 2000) / 1000,
            plane_roll,
            plane_pitch,
            plane_yaw,
            new_plane_state['thrust_level'],
            new_plane_state['linear_speed'] / 1000,
            new_plane_state['vertical_speed'] / 300,
            new_plane_state['horizontal_speed'] / 500,
            int(new_plane_state['target_locked']),

            (new_enemy_loc[0] - 1000) / 1000,
            (new_enemy_loc[2] - 4000) / 2000,
            (new_enemy_loc[1] - 2000) / 1000,
            int(new_enemy_state['target_locked']),

            self.missle_count / 3,
            
            # 添加技能状态信息
            self._get_skill_type_encoding(),  # 当前技能类型编码
            min(1.0, self.skill_duration / 100.0) if self.skill_duration > 0 else 0.0,  # 技能剩余持续时间（归一化）
        ]
        ob = np.array(new_ob)
        self.sendAction(action)
        return ob, reward, terminate, {}

    def reward(self, speed, terminate, health, enemy_health, lock, enemy_lock):
        if terminate == 1:
            # reward = 100
            reward = 0
        elif terminate == 2:
            reward = 0
        else:
            reward = 0
        if enemy_lock:
            reward -= 10
        if lock:
            reward += 0.02
        return reward

    def terminate(self, health, enemy_health):

        if self.step_game >= self.trajectory_length:
            return 2
        if health <= .8:
            return -1
        elif enemy_health <= 0.9:
            print('mission success')
            return 1
        else:
            return 0

    def getDistance(self):
        return ((df.get_plane_state(self.planeID)['position'][0] - df.get_plane_state(self.enemyID)['position'][0]) ** 2 +\
        (df.get_plane_state(self.planeID)['position'][1] - df.get_plane_state(self.enemyID)['position'][1]) ** 2 +\
        (df.get_plane_state(self.planeID)['position'][2] - df.get_plane_state(self.enemyID)['position'][2]) ** 2) ** .5

    def reset(self):
        if self.init_point:
            seed = self.init_point.pop()
        else:
            self.init_point = ['up', 'down', 'left_up', 'left_down', 'right_up', 'right_down', 'left', 'right']
            random.shuffle(self.init_point)
            seed = self.init_point.pop()
        planes = df.get_planes_list()
        self.step_game = 0  #给本局设定结束条件，初定500步
        self.missle_count = 0 #记录导弹的数量，发射的越多罚分越多
        # 保存上一轮的轨迹数据
        if len(self.trajectory_optimizer.trajectory_data) > 0:
            self._save_trajectory_data()
        
        # 重置技能状态
        self.current_skill = None
        self.skill_duration = 0
        
        # 清空轨迹优化器
        self.trajectory_optimizer.clear()
        for i in planes:
            df.reset_machine(i)
        df.set_plane_thrust(self.planeID, 0.5)
        df.set_plane_thrust(self.enemyID, 0.5)
        df.set_client_update_mode(True)
        if self.rendering:
            df.set_renderless_mode(False)
        else:
            df.set_renderless_mode(True)
        # 收起起落架
        df.retract_gear(planes[0])
        new_plane_state = df.get_plane_state(self.planeID)
        new_plan_loc = new_plane_state['position']
        new_enemy_loc = df.get_plane_state(self.enemyID)['position']
        new_enemy_state = df.get_plane_state(self.enemyID)

        df.get_targets_list(self.planeID)
        Euler_plane = new_plane_state['Euler_angles']
        plane_roll = Euler_plane[2] / np.pi
        plane_pitch = Euler_plane[0] / np.pi
        plane_yaw = Euler_plane[1] / np.pi

        self.facing = self.facing_angle(plane_roll, new_plane_state['heading'], new_plan_loc[0], new_plan_loc[2], new_enemy_loc[0], new_enemy_loc[2])
        new_ob = [  # normalized
            (new_plan_loc[0] - 1000) / 1000,
            (new_plan_loc[2] - 4000) / 2000,
            (new_plan_loc[1] - 2000) / 1000,
            plane_roll,
            plane_pitch,
            plane_yaw,
            new_plane_state['thrust_level'],
            new_plane_state['linear_speed'] / 1000,
            new_plane_state['vertical_speed'] / 300,
            new_plane_state['horizontal_speed'] / 500,
            int(new_plane_state['target_locked']),
            (new_enemy_loc[0] - 1000) / 1000,
            (new_enemy_loc[2] - 4000) / 2000,
            (new_enemy_loc[1] - 2000) / 1000,
            int(new_enemy_state['target_locked']),
            self.missle_count / 3,
            
            # 添加技能状态信息
            self._get_skill_type_encoding(),  # 当前技能类型编码
            min(1.0, self.skill_duration / 100.0) if self.skill_duration > 0 else 0.0,  # 技能剩余持续时间（归一化）
        ]

        self.last_obs = new_ob
        self.missle_count = 0
        
        # 记录初始轨迹点
        self._record_trajectory_point(new_plane_state)
        circle_range = 0

        center_enemy = hg.Vec3(3000, 4000, 5000)
        range = hg.Vec3(2000, 2000, 2000)
        y_orientations_range = hg.Vec2(-circle_range, circle_range)
        df.reset_machine_matrix(self.enemyID,
                                uniform(center_enemy.x - range.x / 2, center_enemy.x + range.x / 2),
                                uniform(center_enemy.y - range.y / 2, center_enemy.y + range.y / 2),
                                uniform(center_enemy.z - range.z / 2, center_enemy.z + range.z / 2),
                                0, radians(uniform(y_orientations_range.x, y_orientations_range.y)), 0)
        range_plane = hg.Vec3(0, 0, 0)
        center = hg.Vec3(1000, 4000, 1000)
        y_orientations_range = hg.Vec2(-circle_range, circle_range)
        df.reset_machine_matrix(self.planeID,
                                uniform(center.x - range_plane.x / 2, center.x + range_plane.x / 2),
                                uniform(center.y - range_plane.y / 2, center.y + range_plane.y / 2),
                                uniform(center.z - range_plane.z / 2, center.z + range_plane.z / 2),
                                0, radians(uniform(y_orientations_range.x, y_orientations_range.y)), 0)

        df.set_plane_linear_speed(self.planeID, 400)
        df.set_plane_linear_speed(self.enemyID, 400)
        df.update_scene()
        while True:
            flag = df.get_finish_flag()
            if flag:
                break
        return new_ob

    def sendAction(self, action):
        df.activate_autopilot(self.planeID)
        
        # 检查是否需要激活技能（基于动作的第5个维度，如果存在的话）
        if len(action[0]) > 4:
            skill_value = float(action[0][4])
            if skill_value > 0.8:  # 激活爬升技能
                if self.activate_skill('climb', duration=30, climb_rate=50):
                    print(f"激活爬升技能，持续时间: 30步")
            elif skill_value > 0.6:  # 激活俯冲技能
                if self.activate_skill('dive', duration=30, dive_rate=50):
                    print(f"激活俯冲技能，持续时间: 30步")
            elif skill_value > 0.4:  # 激活转弯技能
                if self.activate_skill('turn', duration=20, turn_rate=45):
                    print(f"激活转弯技能，持续时间: 20步")
            elif skill_value > 0.2:  # 激活追踪技能
                if self.activate_skill('track', duration=60):
                    print(f"激活追踪技能，持续时间: 60步")
        
        for i in range(4):
            # 常规动作处理
            heading_offset = float(action[0][2] * 360)
            df.set_plane_autopilot_heading(self.planeID, heading_offset)

            # 获取当前高度
            current_altitude = df.get_plane_state(self.planeID)['position'][1]

            # 将动作值映射为高度变化量，范围为-1000到1000米
            altitude_change = float(action[0][1] * 2000 - 1000)

            # 计算目标高度，并确保最低高度为1000米
            target_altitude = max(1000, current_altitude + altitude_change)

            # 设置目标高度
            df.set_plane_autopilot_altitude(self.planeID, target_altitude)

            thrust_value = float(action[0][0])
            if thrust_value > 0.5:
                thrust_value = 1.0
            else:
                thrust_value = 0.7
            df.set_plane_thrust(self.planeID, thrust_value)

            missile_mark = 0
            fire_value = float(action[0][3])
            if fire_value > 0.8 and self.missle_count < 3:
                self.missle_count += 1
                if missile_mark == 0:
                    df.fire_missile(self.planeID, self.missle_count)
                    missile_mark = 1
            
            # 更新场景，迭代10次
            for _ in range(10):
                df.update_scene()
                while True:
                    flag = df.get_finish_flag()
                    if flag:
                        break

    def facing_angle(self, roll, heading, x, y, x_e, y_e):
        x1 = x_e - x
        y1 = y_e - y
        v1 = np.array([x1, y1])
        east = np.array([0, 1])
        cos_heading = np.dot(east, v1)/(np.linalg.norm(east) * np.linalg.norm(v1))
        angle_e = np.arccos(cos_heading) * 180 / np.pi
        if x_e < x:
            result = -(heading - (360 - angle_e))
        else:
            result = 360 - heading + angle_e
        if result > 180:
            result = -(360 - result)
        if roll*180 > 90 or roll*180 < -90:
            result =- result
        return result

    # 技能系统相关方法
    def activate_skill(self, skill_name, **kwargs):
        if skill_name == "climb":
            return self.skill_climb(intensity=kwargs.get("intensity", 1.0), duration=kwargs.get("duration", 30))
        elif skill_name == "dive":
            return self.skill_dive(intensity=kwargs.get("intensity", 1.0), duration=kwargs.get("duration", 30))
        elif skill_name == "turn":
            return self.skill_turn(direction=kwargs.get("direction", "right"), angle=kwargs.get("angle", 90), duration=kwargs.get("duration", 20))
        elif skill_name == "track":
            return self.skill_track(duration=kwargs.get("duration", 60))
        else:
            print(f"未知技能: {skill_name}")
            return False
    
    def skill_climb(self, intensity=1.0, duration=30):
        self.current_skill = "climb"
        self.skill_duration = duration
        self.skill_params = {
            "intensity": max(0.0, min(1.0, intensity)),
            "target_altitude": df.get_plane_state(self.planeID)['position'][1] + 1000 * intensity
        }
        return True
    
    def skill_dive(self, intensity=1.0, duration=30):
        self.current_skill = "dive"
        self.skill_duration = duration
        self.skill_params = {
            "intensity": max(0.0, min(1.0, intensity)),
            "target_altitude": max(500, df.get_plane_state(self.planeID)['position'][1] - 1000 * intensity)
        }
        return True
    
    def skill_turn(self, direction="right", angle=90, duration=20):
        self.current_skill = "turn"
        self.skill_duration = duration
        current_heading = df.get_plane_state(self.planeID)['heading']
        
        # 计算目标航向
        if direction == "right":
            target_heading = (current_heading + angle) % 360
        else:  # left
            target_heading = (current_heading - angle) % 360
            
        self.skill_params = {
            "direction": direction,
            "target_heading": target_heading,
            "angle": angle
        }
        return True
    
    def skill_track(self, duration=60):
        self.current_skill = "track"
        self.skill_duration = duration
        self.skill_params = {}
        return True

    
    def _record_trajectory_point(self, plane_state):
        """记录轨迹点"""
        position = plane_state['position']
        velocity = [plane_state['linear_speed'], plane_state['vertical_speed'], plane_state['horizontal_speed']]
        timestamp = self.step_game
        
        self.trajectory_optimizer.add_point(position, velocity, timestamp)
    
    def _get_skill_type_encoding(self):
        """获取当前技能类型的数值编码"""
        if self.current_skill is None:
            return 0.0
        elif self.current_skill == "climb":
            return 0.0
        elif self.current_skill == "dive":
            return 1.0
        elif self.current_skill == "turn":
            return 2.0
        elif self.current_skill == "trace":
            return 3.0
        else:
            return 0.0
    
    def _save_trajectory_data(self):
        """保存轨迹数据到文件"""
        import json
        trajectory_data = self.trajectory_optimizer.get_trajectory_data()
        filename = f"trajectory_episode_{self.step_game}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(trajectory_data, f, indent=2)
            print(f"轨迹数据已保存到 {filename}")
        except Exception as e:
            print(f"保存轨迹数据失败: {e}")
    
    def get_trajectory_stats(self):
        """获取轨迹统计信息"""
        if not self.trajectory_optimizer.trajectory_data:
            return None
        
        positions = [point['position'] for point in self.trajectory_optimizer.trajectory_data]
        velocities = [point['velocity'] for point in self.trajectory_optimizer.trajectory_data]
        
        return {
            'total_points': len(positions),
            'avg_altitude': np.mean([pos[1] for pos in positions]),
            'max_altitude': np.max([pos[1] for pos in positions]),
            'min_altitude': np.min([pos[1] for pos in positions]),
            'avg_speed': np.mean([vel[0] for vel in velocities]),
            'max_speed': np.max([vel[0] for vel in velocities])
        }
    
    def close(self):
        # 保存最后的轨迹数据
        if len(self.trajectory_optimizer.trajectory_data) > 0:
            self._save_trajectory_data()
        df.disconnect()
