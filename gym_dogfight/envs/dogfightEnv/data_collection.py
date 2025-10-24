import random
import sys
import time

import numpy as np
from gym_dogfight import Env
from gym_dogfight.spaces import Box, Discrete
import harfang as hg
from random import uniform
from math import radians
import math
from .trajectory_optimizer import TrajectoryOptimizer

sys.path.append('./src/')
sys.path.append('./src/environments/dogfightEnv/')
sys.path.append('./src/environments/dogfightEnv/dogfight_sandbox_hg2/network_client_example/')
sys.path.append('gym.envs.dogfightEnv.dogfight_sandbox_hg2')
from gym_dogfight.envs.dogfightEnv.dogfight_sandbox_hg2.network_client_example import dogfight_client as df
print("Gym in oneVSone")
time.sleep(1)
CONSTANTS_RADIUS_OF_EARTH = 6378137.     # meters (m)


def XYtoGPS(x, y, ref_lat=24.8976763, ref_lon=160.123456):
    x_rad = float(x) / CONSTANTS_RADIUS_OF_EARTH
    y_rad = float(y) / CONSTANTS_RADIUS_OF_EARTH
    c = math.sqrt(x_rad * x_rad + y_rad * y_rad)

    ref_lat_rad = math.radians(ref_lat)
    ref_lon_rad = math.radians(ref_lon)

    ref_sin_lat = math.sin(ref_lat_rad)
    ref_cos_lat = math.cos(ref_lat_rad)

    if abs(c) > 0:
        sin_c = math.sin(c)
        cos_c = math.cos(c)

        lat_rad = math.asin(cos_c * ref_sin_lat + (x_rad * sin_c * ref_cos_lat) / c)
        lon_rad = (ref_lon_rad + math.atan2(y_rad * sin_c, c * ref_cos_lat * cos_c - x_rad * ref_sin_lat * sin_c))

        lat = math.degrees(lat_rad)
        lon = math.degrees(lon_rad)

    else:
        lat = math.degrees(ref_lat)
        lon = math.degrees(ref_lon)

    return lat, lon


class data_collection(Env):
    def __init__(self, host='192.168.5.8', port='57805', rendering=True)-> None:
        self.host = host
        self.port = port
        self.nof = 0
        self.rendering = rendering
        self.step_game = 0  # 给本局设定结束条件，初定500步
        self.missle_count = 0  # 记录导弹的数量，发射的越多罚分越多
        self.enemy_set_mark = False
        self.target_lock_count = 0
        self.name = 'action_collect'
        self.facing = 0
        self.aming = 0
        self.trajectory_length = 128  # 修改为128步
        self.inner_steps = 10  # 每步执行10次环境步骤
        self.total_epochs = 0
        # 初始化轨迹优化器，设置更小的容差以保持更多细节
        self.trajectory_optimizer = TrajectoryOptimizer(tolerance=1.0)
        
        # 技能系统相关属性
        self.current_skill = None  # 当前激活的技能
        self.skill_duration = 0    # 技能剩余持续时间
        try:
            df.get_planes_list()
        except:
            print('Run for the first time')
            df.connect(host, int(port))
            time.sleep(2)
        planes = df.get_planes_list()
        missiles = df.get_missiles_list()
        self.ally_missile = missiles[:3]
        self.planeID = planes[0]
        self.enemyID = planes[1]
        self.explosion_am = []
        self.explosion_am_index = []
        self.explosion_em = []
        self.explosion_em_index = []
        df.disable_log()

        # 设定本方战机
        self.planeID = planes[0]
        self.enemy = planes[1]

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
            ]),
            high=np.array([
                1,
                1,
                1,
                1,
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
                0,  # skill_type_encoding (技能类型编码: 0=climb, 1=dive, 2=turn, 3=trace)
                0,  # skill_remaining_duration (技能剩余持续时间，归一化)
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
                3,  # skill_type_encoding (技能类型编码: 0=climb, 1=dive, 2=turn, 3=trace)
                1,  # skill_remaining_duration (技能剩余持续时间，归一化)
            ])
        )

    def render_or_not(self, result):
        if result:
            df.set_renderless_mode(False)
        else:
            df.set_renderless_mode(True)

    def steps(self, action_base):
        # 打开文件准备写入
        file = open('C:\\Users\\kevin\\Desktop\\odt\\epoch_{}.txt'.format(self.total_epochs), 'a+')
        file.write('#{}'.format(self.step_game / 100) + '\n')
        
        # 初始化累积奖励和状态
        total_reward = 0
        
        # 执行inner_steps次环境步骤
        for step in range(self.inner_steps):
            df.update_scene()
            # 等待场景更新完成
            while True:
                flag = df.get_finish_flag()
                if flag:
                    break

        self.step_game += 1
        new_plane_state = df.get_plane_state(self.planeID)
        new_plan_loc = new_plane_state['position']
        new_enemy_loc = df.get_plane_state(self.enemyID)['position']
        new_enemy_state = df.get_plane_state(self.enemyID)

        # 获取飞机的欧拉角信息，用于写入文件
        plane_euler = new_plane_state['Euler_angles']
        plane_roll = plane_euler[2] / np.pi * 180
        plane_pitch = plane_euler[0] / np.pi * 180
        plane_yaw = ((360 - plane_euler[1] / np.pi * 180) + 90) % 360

        enemy_euler = new_enemy_state['Euler_angles']
        enemy_roll = enemy_euler[2] / np.pi * 180
        enemy_pitch = enemy_euler[0] / np.pi * 180
        enemy_yaw = ((360 - enemy_euler[1] / np.pi * 180) + 90) % 360

        # 转换坐标为GPS格式
        plane_lat, plane_lon = XYtoGPS(new_plan_loc[0], new_plan_loc[2])
        enemy_lat, enemy_lon = XYtoGPS(new_enemy_loc[0], new_enemy_loc[2])

        # 记录轨迹点到优化器
        self._record_trajectory_point(new_plane_state, new_enemy_state)
        
        # 写入玩家飞机信息
        if new_plane_state['health_level']:
            file.write("1,T={}|{}|{}|{}|{}|{},Type=Air+FixedWing,Coalition=Allies,Color=Blue,Name=F-16,Mach={:.3f},"
                        "ShortName=F-16  0  3.00,RadarMode=1,RadarRange=2000,RadarHorizontalBeamwidth=10,"
                        "RadarVerticalBeamwidth=10\n".format(
                plane_lon, plane_lat, new_plan_loc[1], plane_roll, -plane_pitch, plane_yaw,
                new_plane_state['linear_speed'] / 343.0))  # Mach计算
        else:
            file.write("1,T=||,Visible=0\n")

        # 写入敌方飞机信息
        if new_enemy_state['health_level']:
            file.write("2,T={}|{}|{}|{}|{}|{},Type=Air+FixedWing,Coalition=Enemies,Color=Red,Name=F-16,Mach={:.3f},"
                        "ShortName=F-16  0  3.00,RadarMode=1,RadarRange=2000,RadarHorizontalBeamwidth=10,"
                        "RadarVerticalBeamwidth=10\n".format(
                enemy_lon, enemy_lat, new_enemy_loc[1], enemy_roll, -enemy_pitch, enemy_yaw,
                new_enemy_state['linear_speed'] / 343.0))
        else:
            file.write("2,T=||,Visible=0\n")

        for im, missile_am in enumerate(self.ally_missile):
            if missile_am == '':
                continue
            else:
                if df.get_missile_state(missile_am)['position'][0]:
                    state_am = df.get_missile_state(missile_am)
                    euler_angle_am = state_am['Euler_angles']
                    loc_am = state_am['position']
                    lat_am, lon_am = XYtoGPS(loc_am[0], loc_am[2])
                    roll_am = euler_angle_am[2] / np.pi * 180
                    pitch_am = euler_angle_am[0] / np.pi * 180
                    yaw_am = ((360 - euler_angle_am[1] / np.pi * 180) + 90) % 360
                    if state_am['wreck']:  # state_am['crashed'] or state_am['destroyed'] or
                        file.write("{},T=||,Visible=0".format(im + 100) + '\n')
                        self.explosion_am.append((im + 2) * 1000)
                        self.explosion_am_index.append(10)
                        file.write(
                            "{},T={}|{}|{}|||,Color=Blue,Type=Misc+Explosion,Radius=250".format((im + 2) * 1000, lon_am, lat_am, loc_am[1]) + "\n")
                        self.ally_missile[self.ally_missile.index(missile_am)] = ''
                    else:
                        file.write("{},T={}|{}|{}|{}|{}|{},Type=Medium+Weapon+Missile,Coalition=Allies,Color=Red,Name=Mica,Mach=1,".format(im + 100, lon_am, lat_am, loc_am[1], roll_am, -pitch_am, yaw_am) + "\n")

        terminate_value = self.terminate(new_plane_state['health_level'], new_enemy_state['health_level'])
        terminate = True if terminate_value else False
        reward = self.reward(new_plane_state['linear_speed'], terminate_value, new_plane_state['health_level'], new_enemy_state['health_level'], new_plane_state['target_locked'], new_enemy_state['target_locked'])

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

        action = self.sendAction()

        # 累积奖励
        total_reward += reward
        # 保存最后一步的状态和动作
        final_ob = new_ob
        final_terminate = terminate
        final_action = action

        
        # 关闭文件
        file.close()
                
        # 返回最后一步的状态和累积奖励
        return final_ob, total_reward / self.inner_steps, final_terminate, final_action

    def reward(self, speed, terminate, health, enemy_health, lock, enemy_lock):
        # Base reward starts at zero
        reward = 0

        # Terminal rewards (these are kept discrete by nature of termination)
        if terminate == 1:  # Victory
            reward += 100
        elif terminate == -1:  # Defeat
            reward += -50
        elif terminate == 2:  # Time limit
            reward += 0

        # Target lock rewards - made continuous with decay factors
        if enemy_lock:
            # Penalty for being locked by enemy, with diminishing effect over time
            lock_penalty = -10 * (1 - 0.001 * self.step_game)  # Gradually reduces penalty
            reward += max(-10, lock_penalty)  # Cap minimum penalty

        if lock:
            # Continuous reward for locking enemy, increases with lock duration
            self.target_lock_count = self.target_lock_count + 1 if lock else 0
            lock_reward = 0.02 * (1 + 0.01 * self.target_lock_count)  # Reward increases with lock duration
            reward += min(0.1, lock_reward)  # Cap maximum reward

        # Missile usage - continuous reward based on conservation
        missile_conservation = (3 - self.missle_count) / 3  # Normalized between 0 and 1
        reward += 0.005 * missile_conservation  # Small continuous reward for conserving missiles

        # Facing angle - smoothly scaled reward based on angle to target
        # Uses a cosine function to create a smooth transition
        if abs(self.facing) <= 180:
            # Normalized facing angle (0 to 1, where 1 is perfect alignment)
            normalized_facing = (180 - abs(self.facing)) / 180
            # Continuous reward that peaks at perfect alignment and smoothly decreases
            facing_reward = 0.05 * np.cos((1 - normalized_facing) * np.pi / 2)
            reward += facing_reward

        # Aiming reward - smooth function based on aiming accuracy
        if hasattr(self, 'aming'):
            # Convert to normalized value (0 to 1, where 1 is perfect aim)
            normalized_aim = max(0, (90 - self.aming) / 90)
            # Exponential reward that increases as aim improves
            aim_reward = 0.02 * np.exp(normalized_aim * 2 - 2)  # Exponential scaling
            reward += aim_reward

        # Speed reward - smooth function for maintaining optimal speed
        # Optimal speed around 400-450, penalty for too slow or too fast
        speed_factor = max(0, min(1, speed / 400))  # Normalized between 0 and 1 (optimal at 1)
        speed_reward = 0.01 * (-(speed_factor - 0.9) ** 2 + 1)  # Quadratic function peaks at 90% of max speed
        reward += speed_reward

        # Distance reward - continuous reward for maintaining optimal combat distance
        # Optimal distance around 2000-3000 units
        optimal_distance = 2500

        # Health-based continuous reward
        health_reward = 0.01 * health  # Small continuous reward for maintaining health
        enemy_damage_reward = 0.02 * (1 - enemy_health)  # Reward for damaging enemy
        reward += health_reward + enemy_damage_reward

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

    def reset(self):
        self.total_epochs += 1
        # 创建新的文件并写入文件头
        with open('C:\\Users\\kevin\\Desktop\\odt\\epoch_{}.txt'.format(self.total_epochs), 'w') as file:
            file.write('FileType=text/acmi/tacview\n')
            file.write('FileVersion=2.1\n')
            file.write('0,ReferenceTime=2022-10-01T00:00:00Z\n')
            file.write('0,Title = test simple aircraft\n')
            file.write('1000000,T=160.123456|24.8976763|0, Type=Ground+Static+Building, Name=Competition, EngagementRange=30000\n')

        planes = df.get_planes_list()
        self.step_game = 0  # 给本局设定结束条件，初定500步
        self.missle_count = 0  # 记录导弹的数量，发射的越多罚分越多
        for i in planes:
            df.reset_machine(i)

        df.set_plane_thrust(self.planeID, 0.5)
        df.set_plane_thrust(self.enemyID, 0.5)
        df.set_client_update_mode(True)

        if self.step_game % 200 == 0:
            df.set_plane_pitch(self.enemyID, np.random.random() - 0.5)
            df.set_plane_roll(self.enemyID, np.random.random() - 0.5)

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

    def close(self):
         # 保存优化后的轨迹数据
         if hasattr(self, 'trajectory_optimizer'):
             self._save_optimized_trajectory()
         df.disconnect()
    
    def _record_trajectory_point(self, plane_state, enemy_state):
         """记录轨迹点到优化器"""
         # 为玩家飞机添加轨迹点
         self.trajectory_optimizer.add_trajectory_point(
             timestamp=self.step_game,
             position=tuple(plane_state['position']),
             orientation=tuple(plane_state['Euler_angles']),
             velocity=plane_state.get('linear_speed', 0),
             additional_data={
                 'health_level': plane_state['health_level'],
                 'vertical_speed': plane_state.get('vertical_speed', 0),
                 'horizontal_speed': plane_state.get('horizontal_speed', 0),
                 'aircraft_type': 'player',
                 'enemy_position': tuple(enemy_state['position']),
                 'enemy_orientation': tuple(enemy_state['Euler_angles']),
                 'enemy_velocity': enemy_state.get('linear_speed', 0),
                 'enemy_health': enemy_state['health_level']
             }
         )
    
    def _save_optimized_trajectory(self):
         """保存优化后的轨迹数据"""
         if len(self.trajectory_optimizer.trajectory_data) == 0:
             return
         
         # 执行Visvalingam-Whyatt简化
         simplified_trajectory = self.trajectory_optimizer.visvalingam_whyatt_simplify()
         
         # 执行Delta编码
         delta_encoded = self.trajectory_optimizer.delta_encode(simplified_trajectory)
         
         # 保存到文件
         trajectory_file = f'./data/optimized_trajectory_episode_{self.episode_count}.json'
         self.trajectory_optimizer.save_to_file(trajectory_file, 'delta')
         
         # 获取压缩统计信息
         stats = self.trajectory_optimizer.get_compression_stats()
         print(f"轨迹优化完成 - 原始点数: {stats['original_points']}, 简化后: {stats['simplified_points']}, 压缩比: {stats['compression_ratio']:.2%}")

    import numpy as np

    def sendAction(self):
        df.activate_autopilot(self.planeID)
        action = np.array([0, 0, 0, 0], dtype=np.float32)  # thrust altitude heading missile
        aircraft = self.planeID

        state = df.get_plane_state(aircraft)
        enemy_state = df.get_plane_state(self.enemy)
        td = df.get_target_idx(aircraft)

        aircraft_pos = state['position']
        
        # 更新技能持续时间
        if self.skill_duration > 0:
            self.skill_duration -= 1
            if self.skill_duration <= 0:
                self.current_skill = None
        
        # 基于技能的动作决策
        if self.current_skill is None:
            # 没有激活技能时，根据情况选择技能
            action = self._select_and_execute_skill(state, enemy_state, td)
        else:
            # 执行当前激活的技能
            action = self._execute_current_skill(state, enemy_state, td)
        
        return action
    
    def _select_and_execute_skill(self, state, enemy_state, td):
        """根据当前情况选择并执行技能"""
        action = np.array([0, 0, 0, 0], dtype=np.float32)
        aircraft = self.planeID
        
        if td['target_idx'] == 0:  # 没有目标
            # 激活搜索技能（转弯技能）
            self.current_skill = 'turn'
            self.skill_duration = 30  # 技能持续30步
            return self._execute_turn_skill(state, action)
        else:
            # 有目标时，根据距离和相对位置选择技能
            target_pos = enemy_state['position']
            if target_pos is not None:
                aircraft_pos = state['position']
                delta_alt = target_pos[1] - aircraft_pos[1]
                target_distance = np.sqrt(
                    (target_pos[0] - aircraft_pos[0]) ** 2 + 
                    (target_pos[2] - aircraft_pos[2]) ** 2 + 
                    delta_alt ** 2
                )
                
                # 根据高度差和距离选择技能
                if abs(delta_alt) > 500:  # 高度差较大
                    if delta_alt > 0:  # 敌机在上方
                        self.current_skill = 'climb'
                        self.skill_duration = 20
                        return self._execute_climb_skill(state, enemy_state, action)
                    else:  # 敌机在下方
                        self.current_skill = 'dive'
                        self.skill_duration = 20
                        return self._execute_dive_skill(state, enemy_state, action)
                elif target_distance > 3000:  # 距离较远
                    self.current_skill = 'track'
                    self.skill_duration = 40
                    return self._execute_track_skill(state, enemy_state, action)
                else:  # 距离适中，进行机动
                    self.current_skill = 'turn'
                    self.skill_duration = 25
                    return self._execute_turn_skill(state, action)
            else:
                # 没有目标位置信息，使用搜索
                self.current_skill = 'turn'
                self.skill_duration = 30
                return self._execute_turn_skill(state, action)
    
    def _execute_current_skill(self, state, enemy_state, td):
        """执行当前激活的技能"""
        action = np.array([0, 0, 0, 0], dtype=np.float32)
        
        if self.current_skill == 'climb':
            return self._execute_climb_skill(state, enemy_state, action)
        elif self.current_skill == 'dive':
            return self._execute_dive_skill(state, enemy_state, action)
        elif self.current_skill == 'turn':
            return self._execute_turn_skill(state, action)
        elif self.current_skill == 'track':
            return self._execute_track_skill(state, enemy_state, action)
        else:
            # 默认行为
            return self._execute_track_skill(state, enemy_state, action)
    
    def _execute_climb_skill(self, state, enemy_state, action):
        """执行爬升技能"""
        aircraft = self.planeID
        
        # 爬升到更高高度
        current_altitude = state['position'][1]
        target_altitude = current_altitude + 1000  # 爬升1000米
        
        df.set_plane_autopilot_altitude(aircraft, target_altitude)
        action[1] = min(target_altitude / 10000, 1.0)
        
        # 如果有目标，朝目标方向飞行
        if enemy_state['position'] is not None:
            target_pos = enemy_state['position']
            aircraft_pos = state['position']
            delta_x = target_pos[0] - aircraft_pos[0]
            delta_y = target_pos[2] - aircraft_pos[2]
            
            heading = np.degrees(np.arctan2(delta_x, delta_y))
            if heading < 0:
                heading += 360
            
            df.set_plane_autopilot_heading(aircraft, heading)
            action[2] = heading / 360.0
        
        # 中等推力
        df.set_plane_thrust(aircraft, 0.8)
        action[0] = 0.8
        
        return action
    
    def _execute_dive_skill(self, state, enemy_state, action):
        """执行俯冲技能"""
        aircraft = self.planeID
        
        # 俯冲到更低高度
        current_altitude = state['position'][1]
        target_altitude = max(current_altitude - 800, 500)  # 俯冲800米，但不低于500米
        
        df.set_plane_autopilot_altitude(aircraft, target_altitude)
        action[1] = target_altitude / 10000
        
        # 如果有目标，朝目标方向飞行
        if enemy_state['position'] is not None:
            target_pos = enemy_state['position']
            aircraft_pos = state['position']
            delta_x = target_pos[0] - aircraft_pos[0]
            delta_y = target_pos[2] - aircraft_pos[2]
            
            heading = np.degrees(np.arctan2(delta_x, delta_y))
            if heading < 0:
                heading += 360
            
            df.set_plane_autopilot_heading(aircraft, heading)
            action[2] = heading / 360.0
        
        # 高推力
        df.set_plane_thrust(aircraft, 1.0)
        action[0] = 1.0
        
        return action
    
    def _execute_turn_skill(self, state, action):
        """执行转弯技能"""
        aircraft = self.planeID
        
        # 执行转弯机动
        current_heading = state['heading']
        turn_angle = 45  # 转弯45度
        new_heading = (current_heading + turn_angle) % 360
        
        df.set_plane_autopilot_heading(aircraft, new_heading)
        action[2] = new_heading / 360.0
        
        # 保持当前高度
        current_altitude = state['position'][1]
        df.set_plane_autopilot_altitude(aircraft, current_altitude)
        action[1] = current_altitude / 10000
        
        # 中等推力
        df.set_plane_thrust(aircraft, 0.7)
        action[0] = 0.7
        
        return action
    
    def _execute_track_skill(self, state, enemy_state, action):
        """执行追踪技能"""
        aircraft = self.planeID
        
        if enemy_state['position'] is not None:
            target_pos = enemy_state['position']
            aircraft_pos = state['position']
            
            # 计算到目标的向量
            delta_x = target_pos[0] - aircraft_pos[0]
            delta_y = target_pos[2] - aircraft_pos[2]
            delta_alt = target_pos[1] - aircraft_pos[1]
            
            # 计算距离
            target_distance = np.sqrt(delta_x ** 2 + delta_y ** 2 + delta_alt ** 2)
            
            # 计算航向
            heading = np.degrees(np.arctan2(delta_x, delta_y))
            if heading < 0:
                heading += 360
            
            df.set_plane_autopilot_heading(aircraft, heading)
            action[2] = heading / 360.0
            
            # 匹配目标高度
            target_altitude = target_pos[1]
            df.set_plane_autopilot_altitude(aircraft, target_altitude)
            action[1] = target_altitude / 10000
            
            # 根据距离调整推力
            if target_distance > 2000:
                df.set_plane_thrust(aircraft, 1.0)
                action[0] = 1.0
            else:
                df.set_plane_thrust(aircraft, 0.7)
                action[0] = 0.7
            
            # 武器管理
            if target_distance <= 2000:
                aircraft_heading_rad = np.radians(state['heading'])
                target_heading_rad = np.radians(heading)
                
                angle_diff = np.degrees(np.arctan2(
                    np.sin(target_heading_rad - aircraft_heading_rad), 
                    np.cos(target_heading_rad - aircraft_heading_rad)
                ))
                target_angle = abs(angle_diff)
                
                if target_angle < 10 and state.get('target_locked', False):
                    df.fire_missile(aircraft, 0)
                    action[3] = 1.0
        
        return action
    
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