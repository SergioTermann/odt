from datetime import datetime
import os
import utils
import numpy as np


class Logger:
    def __init__(self, variant):

        self.log_path = self.create_log_path(variant)
        utils.mkdir(self.log_path)
        print(f"Experiment log path: {self.log_path}")

    def log_metrics(self, outputs, iter_num, total_transitions_sampled, writer, is_online=False):
        print("=" * 80)
        print(f"{'Online' if is_online else 'Offline'} Iteration {iter_num}")
        for k, v in outputs.items():
            print(f"{k}: {v}")
            if writer:
                # 为online和offline训练添加不同的标签前缀
                prefix = "online/" if is_online else "offline/"
                # 添加类型检查，只记录标量值（数值类型）
                if isinstance(v, (int, float, bool, np.number)):
                    # 记录所有指标到TensorBoard
                    writer.add_scalar(prefix + k, v, iter_num)
                    
                    # 对于评估指标，同时记录与采样数量的关系
                    if k == "evaluation/return_mean_gm" or k.startswith("aug_traj/") or k.startswith("train/"):
                        writer.add_scalar(
                            prefix + k + "_vs_samples",
                            v,
                            total_transitions_sampled,
                        )
                        
                    # 对于online tuning特有的指标，添加额外的可视化
                    if is_online and (k.startswith("aug_traj/") or k.startswith("train/")):
                        # 记录训练过程中的关键指标
                        writer.add_scalar(
                            "online_tuning/" + k,
                            v,
                            iter_num,
                        )

    def create_log_path(self, variant):
        now = datetime.now().strftime("%Y.%m.%d/%H%M%S")
        exp_name = variant["exp_name"]
        prefix = variant["save_dir"]
        return f"{prefix}/{now}-{exp_name}"
