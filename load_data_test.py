import pickle
import os

# 构建文件路径
file_path = os.path.join('human_demo', 'down', 'human_actions.pkl')

# 加载pickle文件
with open(file_path, 'rb') as f:
    human_actions = pickle.load(f)

print("human_actions.pkl 已成功加载")