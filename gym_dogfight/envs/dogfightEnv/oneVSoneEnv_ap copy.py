import random
import sys
import time

import numpy as np
from gym_dogfight import Env
from gym_dogfight.spaces import Box, Discrete
import harfang as hg
from random import uniform
from math import radians

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

    def sendAction(self, action):
        df.activate_autopilot(self.planeID)
        for i in range(4):

            heading_offset = float(action[0][2] * 360)
            df.set_plane_autopilot_heading(self.planeID, heading_offset)
            
            altitude_change = float(action[0][1] * 10000)
            df.set_plane_autopilot_altitude(self.planeID, altitude_change)
            
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

    def close(self):
        df.disconnect()
