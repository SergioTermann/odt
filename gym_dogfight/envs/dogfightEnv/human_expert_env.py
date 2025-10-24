import sys
import time
import numpy as np
from gym_dogfight import Env
from gym_dogfight.spaces import Box
import harfang as hg
from random import uniform
from datetime import datetime
from gym_dogfight.envs.dogfightEnv.util import RolloutBuffer

sys.path.append('./src/')
sys.path.append('./src/environments/dogfightEnv/')
sys.path.append('./src/environments/dogfightEnv/dogfight_sandbox_hg2/network_client_example/')
sys.path.append('gym.envs.dogfightEnv.dogfight_sandbox_hg2')

from gym_dogfight.envs.dogfightEnv.dogfight_sandbox_hg2.network_client_example import dogfight_client as df
print("Gym in oneVSone______df")
time.sleep(1)


class human_expert_env(Env):
    def __init__(self, host='192.168.5.11', port='50786', rendering=True, dir='', step_total=100000) -> None:
        self.host = host
        self.port = port
        self.rendering = rendering
        self.step_game = 0  #给本局设定结束条件，初定500步
        self.missle_count = 0 #记录导弹的数量，发射的越多罚分越多

        self.last_obs = None
        self.buffer_size = step_total
        self.total_step = 0
        self.finish_collecting = False
        self.reset_start_point = []
        self.time_count = 0.01
        self.dir = dir
        try:
            df.get_planes_list()
        except:
            print('Run for the first time')
            df.connect(host, int(port))
            time.sleep(2)
        self.planes = df.get_planes_list()
        self.planeID = self.planes[0]
        self.enemyID = self.planes[1]
        df.disable_log()
        self.planeID = self.planes[0]
        for i in self.planes:
           df.reset_machine(i)
        df.set_plane_thrust(self.planeID, 1)
        df.set_plane_thrust(self.enemyID, 1)
        df.set_plane_linear_speed(self.planeID, 85)
        df.set_plane_linear_speed(self.enemyID, 85)
        df.set_client_update_mode(True)
        df.get_targets_list(self.planeID)
        if self.rendering:
            df.set_renderless_mode(False)
        else:
            df.set_renderless_mode(True)
        #收起起落架
        df.retract_gear(self.planes[0])
        self.action_space = Box(
            low=np.array([
                -1,  # Roll 俯仰角
                -1,  # Pitch 翻滚角
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
                # -360,  # roll_attitude * 4
                # -360,  # pitch_attitude * 4
                # 0,  # heading
                # 0,  # thrust level 油门
                # 0,  # linear speed 空速/1000
                # -1,  # vertical speed 垂直速度/300
                # 0,  # horizontal speed 水平速度/500
                0,  # 是否锁定

                0,  # missile count 导弹发射计数/3
                0,    # 距离
                # -180,    # 罗盘角度
                # 0,  # 视线角度
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
                # 360,  # roll_attitude * 4
                # 360,  # pitch_attitude * 4
                # 360,  # heading
                # 1,  # thrust level 油门
                # 100,  # linear speed 空速/1000
                # 100,  # vertical speed 垂直速度/300
                # 100,  # horizontal speed 水平速度/500
                1,  # 是否锁定

                1,  # missile count 导弹发射计数/3
                2,  # 距离
                # 180,  # 罗盘角度
                # 180,  # 视线角度
            ])
        )
        self.buffer = RolloutBuffer()

    def render(self, id=0):
        df.set_renderless_mode(False)

    def steps(self):
        file = open('trained_epoch_0.txt', 'a')
        file.write('#{}'.format(self.time_count))
        self.time_count += 0.01
        self.step_game += 1
        self.total_step += 1
        # if self.step_game % 200 == 0:
        #     df.set_plane_pitch(self.enemyID, 0.1 * (np.random.random()-0.5))
        #     df.set_plane_roll(self.enemyID, 0.1 * (np.random.random()-0.5))
        action_get = df.get_gamepad_action(self.planeID, False)
        action = [action_get["ROLL"], action_get["PITCH"], action_get["THRUST"], action_get["FIRE"]]
        action = np.array(action)
        for i in range(4):
            if i == 3:
                df.set_gamepad_action(self.planeID, action_get["ROLL"], action_get["PITCH"], action_get["THRUST"], False)
            else:
                df.set_gamepad_action(self.planeID, action_get["ROLL"], action_get["PITCH"], action_get["THRUST"], True)
            df.update_scene()
            while True:
                flag = df.get_finish_flag()
                print('the flag is:', flag)
                if flag:
                    break

        # ______________obs stuff_____________________ #
        new_plane_state = df.get_plane_state(self.planeID)
        new_plan_loc = new_plane_state['position']

        new_enemy_loc = df.get_plane_state(self.enemyID)['position']
        new_enemy_state = df.get_plane_state(self.enemyID)

        terminate_value = self.terminate(new_plane_state['health_level'], new_enemy_state['health_level'])
        terminate = True if terminate_value else False
        reward = self.reward(new_plane_state['linear_speed'], terminate_value, new_plane_state['health_level'],
                             new_enemy_state['health_level'], new_plane_state['target_locked'],
                             new_enemy_state['target_locked'])

        Euler_plane = new_plane_state['Euler_angles']
        plane_roll = Euler_plane[2] / np.pi
        plane_pitch = Euler_plane[0] / np.pi
        plane_yaw = Euler_plane[1] / np.pi

        self.facing = self.facing_angle(plane_roll, new_plane_state['heading'], new_plan_loc[0], new_plan_loc[2],
                                        new_enemy_loc[0],
                                        new_enemy_loc[2])
        self.aming = self.angle_attacking(new_plane_state['heading'], new_plane_state['pitch_attitude'], new_plan_loc,
                                          new_enemy_loc) / 180
        self.buffer.add(self.last_obs, action, reward, int(terminate))
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
            self.getDistance() / 10000,
        ]
        if self.step_game < 50:
            df.set_plane_yaw(self.enemyID, 0.5)
        else:
            df.set_plane_yaw(self.enemyID, -0.5)


        self.last_obs = new_ob
        return new_ob, reward, terminate, {}

    def terminate(self, health, enemy_health):

        if self.step_game >= 256:
            return 2
        if health <= .8:
            return -1
        elif enemy_health <= 0.9:
            return 1
        else:
            return 0

    def reset(self):
        # 首先保存当前episode的数据（如果不是第一次reset）
        if self.step_game > 0:
            # 从buffer中获取所有数据
            data = self.buffer.get()
            if len(data['obs']) > 0:  # 确保有数据
                # 构建字典并保存
                episode_data = {
                    'observations': np.array(data['obs'][:-1]),  # 移除最后一个
                    'next_observations': np.array(data['obs'][1:]),  # 移除第一个
                    'actions': np.array(data['acts']),
                    'rewards': np.array(data['rews']),
                    'terminals': np.array(data['done'])
                }

                # 创建时间戳用于唯一文件名
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # 保存路径
                save_dir = f'human_demo/{self.dir}' if self.dir else 'human_demo'
                os.makedirs(save_dir, exist_ok=True)

                # 保存为pickle文件
                import pickle
                with open(f'{save_dir}/human_actions_{timestamp}.pkl', 'wb') as f:
                    pickle.dump(episode_data, f)

                print(f"Episode data saved with shape: {episode_data['observations'].shape}")

                # 清空buffer为下一个episode准备
                self.buffer.clear()

                # 原有的reset代码

        print(self.total_step)
        self.step_game = 0  # 给本局设定结束条件，初定500步
        self.missle_count = 0  # 记录导弹的数量，发射的越多罚分越多
        for i in self.planes:
            df.reset_machine(i)
        df.set_plane_thrust(self.planeID, 1)
        df.set_plane_thrust(self.enemyID, 1)
        df.set_client_update_mode(True)
        if self.rendering:
            df.set_renderless_mode(False)
        else:
            df.set_renderless_mode(True)
            # 收起起落架
        df.retract_gear(self.planes[0])

        new_plane_state = df.get_plane_state(self.planeID)
        new_plan_loc = new_plane_state['position']
        new_enemy_loc = df.get_plane_state(self.enemyID)['position']
        new_enemy_state = df.get_plane_state(self.enemyID)

        df.get_targets_list(self.planeID)
        Euler_plane = new_plane_state['Euler_angles']
        plane_roll = Euler_plane[2] / np.pi
        plane_pitch = Euler_plane[0] / np.pi
        plane_yaw = Euler_plane[1] / np.pi
        self.facing = self.facing_angle(plane_roll, new_plane_state['heading'], new_plan_loc[0], new_plan_loc[2],
                                        new_enemy_loc[0], new_enemy_loc[2])
        self.aming = self.angle_attacking(new_plane_state['heading'], new_plane_state['pitch_attitude'], new_plan_loc,
                                          new_enemy_loc) / 180
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
            self.getDistance() / 10000,
        ]
        self.last_obs = new_ob
        self.missle_count = 0
        self.step_game = 0

        center_enemy = hg.Vec3(1000, 4000, 4000)
        range_enemy = hg.Vec3(0, 0, 0)
        circle_range = 0  # np.pi * 0.75
        y_orientations_range_enemy = hg.Vec2(circle_range, circle_range)
        uni_enemy_x = uniform(center_enemy.x - range_enemy.x / 2, center_enemy.x + range_enemy.x / 2)
        uni_enemy_y = uniform(center_enemy.y - range_enemy.y / 2, center_enemy.y + range_enemy.y / 2)
        uni_enemy_z = uniform(center_enemy.z - range_enemy.z / 2, center_enemy.z + range_enemy.z / 2)
        uni_enemy_rad = uniform(y_orientations_range_enemy.x, y_orientations_range_enemy.y)
        df.reset_machine_matrix(self.enemyID, uni_enemy_x, uni_enemy_y, uni_enemy_z, 0, uni_enemy_rad, 0)

        # range_plane = hg.Vec3(500, 500, 500)
        range_plane = hg.Vec3(0, 0, 0)
        center_plane = hg.Vec3(1000, 4000, 1000)
        y_orientations_range_plane = hg.Vec2(0, 0)
        uni_plane_x = uniform(center_plane.x - range_plane.x / 2, center_plane.x + range_plane.x / 2)
        uni_plane_y = uniform(center_plane.y - range_plane.y / 2, center_plane.y + range_plane.y / 2)
        uni_plane_z = uniform(center_plane.z - range_plane.z / 2, center_plane.z + range_plane.z / 2)
        uni_plane_rad = uniform(y_orientations_range_plane.x, y_orientations_range_plane.y)
        current_loc = [uni_enemy_x, uni_enemy_y, uni_enemy_z, uni_enemy_rad, uni_plane_x, uni_plane_y, uni_plane_z, uni_plane_rad]
        df.reset_machine_matrix(self.planeID, uni_plane_x, uni_plane_y, uni_plane_z, 0, uni_plane_rad, 0)

        df.set_plane_linear_speed(self.planeID, 400)
        df.set_plane_linear_speed(self.enemyID, 300)
        self.reset_start_point.append(current_loc)

        return new_ob
    def getDistance(self):
        return ((df.get_plane_state(self.planeID)['position'][0] - df.get_plane_state(self.enemyID)['position'][0]) ** 2 +\
        (df.get_plane_state(self.planeID)['position'][1] - df.get_plane_state(self.enemyID)['position'][1]) ** 2 +\
        (df.get_plane_state(self.planeID)['position'][2] - df.get_plane_state(self.enemyID)['position'][2]) ** 2) ** .5
    def reward(self, speed, terminate, health, enemy_health, lock, enemy_lock):
        if terminate == 1:
            reward = 100
        elif terminate == -1:
            reward = -50
        elif terminate == 2:
            reward = -20
        else:
            reward = 0
        if enemy_lock:
            reward -= 10
        if lock:
            reward += 0.2
        reward += (3-self.missle_count)*0.01
        if self.facing < 30 or self.facing > 330:
            reward += 0.02
        elif abs(self.facing) < 60 or self.facing > 300:
            reward += 0.01
        elif abs(self.facing) > 90 or self.facing < 270:
            reward -= 0.02
        if self.aming < 30:
            reward += 0.01
        if speed < 83:
            reward -= 0.05
        return reward

    def angle_attacking(self, heading, pitch, plane_loc, enemy_loc):
        x1 = enemy_loc[2] - plane_loc[2]
        y1 = enemy_loc[0] - plane_loc[0]
        z1 = enemy_loc[1] - plane_loc[1]
        z = np.sin(pitch/180*np.pi)
        y = np.cos(pitch/180*np.pi)*np.sin(heading/180*np.pi)
        x = np.cos(pitch/180*np.pi)*np.cos(heading/180*np.pi)
        angle = np.arccos((x1*x + y1*y + z1*z)/np.sqrt(x1**2+y1**2+z1**2))
        return angle/np.pi*180

    def facing_angle(self, roll, heading, x, y, x_e, y_e):
        x1 = x_e - x
        y1 = y_e - y
        v1 = np.array([x1, y1])
        east = np.array([0, 1])
        cos_heading = np.dot(east, v1)/(np.linalg.norm(east) * np.linalg.norm(v1))
        angle_e = np.arccos(cos_heading) * 180 / np.pi
        if x_e < x:
            result =  -( heading - (360 - angle_e))
        else:
            result = 360 - heading + angle_e
        if result > 180:
            result = -(360 - result)
        if roll*180 > 90 or roll*180 < -90:
            result =- result
        return result