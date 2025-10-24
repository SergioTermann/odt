import pickle


def read_pkl_file(file_path):
    """
    读取 Pickle 文件并返回文件中的对象

    参数:
    file_path (str): Pickle 文件的路径

    返回:
    object: 从 Pickle 文件中读取的对象
    """
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 未找到")
        return None
    except pickle.UnpicklingError:
        print(f"错误：无法解析文件 {file_path}")
        return None
    except Exception as e:
        print(f"读取文件时发生未知错误：{e}")
        return None


# 使用示例
if __name__ == "__main__":
    # 替换为你的 .pkl 文件路径
    file_path = 'episodes_20250731-000040.pkl'

    # 读取文件
    loaded_data = read_pkl_file(file_path)

    # 如果成功读取，打印数据
    if loaded_data is not None:
        print("成功读取 Pickle 文件:")
        print(loaded_data)

        # 根据数据类型进行进一步处理
        if isinstance(loaded_data, dict):
            print("数据是字典，键包括:", list(loaded_data.keys()))
        elif isinstance(loaded_data, list):
            print("数据是列表，长度为:", len(loaded_data))
        # 可以根据需要添加更多类型的处理
