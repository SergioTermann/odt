import pickle
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.io

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
        self.states_total = []
        self.actions_total = []
        self.rewards_total = []
        self.is_terminals_total = []
        self.round_limit = 0
        self.colect_action = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

    def save_to_file(self):
        with open('C:\\Users\\bafs\\.conda\\envs\\ppo\\Lib\\site-packages\\gym\\envs\\dogfightEnv\\trajectory\\human_states.pkl', 'wb') as f:
            pickle.dump(self.states_total, f)
        with open('C:\\Users\\bafs\\.conda\\envs\\ppo\\Lib\\site-packages\\gym\\envs\\dogfightEnv\\trajectory\\human_actions.pkl', 'wb') as f:
            pickle.dump(self.actions_total, f)
        # with open('C:\\Users\\bafs\\.conda\\envs\\ppo\\Lib\\site-packages\\gym\\envs\\dogfightEnv\\human_demo\\human_rewards.pkl', 'wb') as f:
        #     pickle.dump(self.rewards_total, f)
        # with open('C:\\Users\\bafs\\.conda\\envs\\ppo\\Lib\\site-packages\\gym\\envs\\dogfightEnv\\human_demo\\human_terminal.pkl', 'wb') as f:
        #     pickle.dump(self.is_terminals_total, f)
    # def save_action(self):
    #     with open('C:\\Users\\bafs\\.conda\\envs\\ppo\\Lib\\site-packages\\gym\\envs\\dogfightEnv\\trajectory\\collect_states.pkl', 'wb') as f:
    #         pickle.dump(self.states_total, f)
    def add(self, state, action, reward, done):
        self.states.append(state)
        self.actions.append(action)
        # self.rewards.append(reward)
        # self.is_terminals.append(done)
        if done:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.is_terminals.pop(0)
            self.states_total.append(self.states.copy())
            self.actions_total.append(self.actions.copy())
            self.rewards_total.append(self.rewards.copy())
            self.is_terminals_total.append(self.is_terminals.copy())
            self.clear()
            self.round_limit += 1
            self.save_action()
            if self.round_limit >= 60:
                self.save_to_file()
                # self.save_action()
                print('save to file')
                print('===========================================')
                # self.plot_trajectories(self.states_total)
                time.sleep(2000)



    def plot_trajectories(self, trajectories):
        """绘制轨迹图"""
        # 存储还原后的坐标
        plane_locs = []
        enemy_locs = []
        # 提取并还原坐标
        for trajectory in trajectories:
            plan_traj = []
            enemy_traj = []

            for ob in trajectory:
                plan_loc = self.inverse_normalize(ob[:3], ob[3], ob[4], ob[5])  # 前三个值是 new_plan_loc 的归一化坐标
                enemy_loc = self.inverse_normalize(ob[11:14], 0, 0, 0)  # 接下来的三个值是 new_enemy_loc 的归一化坐标
                plan_traj.append(plan_loc)
                enemy_traj.append(enemy_loc)

            plane_locs.append(plan_traj)
            enemy_locs.append(enemy_traj)

        # 绘制轨迹图
        # fig = plt.figure(figsize=(10, 10))
        # ax = fig.add_subplot(111, projection='3d')
        trajectory_count = 0
        for plan_traj, enemy_traj in zip(plane_locs, enemy_locs):
            plan_traj = np.array(plan_traj)
            enemy_traj = np.array(enemy_traj)
            scipy.io.savemat('C:\\Users\\bafs\\Documents\\MATLAB\\flypath3d\\examples\\plane_trajectory_{}.mat'.format(trajectory_count), {'data': plan_traj})
            scipy.io.savemat('C:\\Users\\bafs\\Documents\\MATLAB\\flypath3d\\examples\\enemy_trajectory_{}.mat'.format(trajectory_count), {'data': enemy_traj})
            trajectory_count += 1

        #     # 绘制我方飞机的轨迹
        #     ax.plot(plan_traj[:, 0], plan_traj[:, 1], plan_traj[:, 2], label='Plane')
        #
        #     # 绘制敌方飞机的轨迹
        #     ax.plot(enemy_traj[:, 0], enemy_traj[:, 1], enemy_traj[:, 2], label='Enemy')
        #     # 在轨迹上添加箭头和时序信息
        # ax.set_xlim([0, 3000])
        # ax.set_ylim([500, 10000])
        # ax.set_zlim([2000, 5000])
        # ax.set_title("3D Flight Trajectories")
        # ax.set_xlabel("X Position")
        # ax.set_ylabel("Y Position")
        # ax.set_zlabel("Z Position")
        # ax.legend()
        # plt.show()




