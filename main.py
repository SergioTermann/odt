from tensorboardX import SummaryWriter
import argparse
import pickle
import random
import time
import gym_dogfight
import torch
import numpy as np
# from replay_buffer import ReplayBuffer
from lamb import Lamb
from pathlib import Path
from data import create_dataloader
from decision_transformer.models.decision_transformer import DecisionTransformer
from evaluation import create_vec_eval_episodes_fn, vec_evaluate_episode_rtg
from trainer import SequenceTrainer
from causal_dt_trainer import CausalSequenceTrainer
from logger import Logger
import os
import subprocess
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
MAX_EPISODE_LEN = 1025


class Experiment:
    def __init__(self, variant):
        self.state_dim, self.act_dim, self.action_range = self._get_env_spec(variant)
        loc = r'episodes_20250915-102016'
        self.offline_trajs, self.state_mean, self.state_std = self._load_dataset(loc)
        # self.replay_buffer = ReplayBuffer(variant["replay_size"], self.offline_trajs)
        self.aug_trajs = []
        self.device = variant.get("device", "cuda")
        self.target_entropy = -self.act_dim
        self.model = DecisionTransformer(
            state_dim=self.state_dim,
            act_dim=self.act_dim,
            action_range=self.action_range,
            max_length=variant["K"],
            eval_context_length=variant["eval_context_length"],
            max_ep_len=MAX_EPISODE_LEN,
            hidden_size=variant["embed_dim"],
            n_layer=variant["n_layer"],
            n_head=variant["n_head"],
            n_inner=4 * variant["embed_dim"],
            activation_function=variant["activation_function"],
            n_positions=1024,
            resid_pdrop=variant["dropout"],
            attn_pdrop=variant["dropout"],
            stochastic_policy=True,
            ordering=variant["ordering"],
            init_temperature=variant["init_temperature"],
            target_entropy=self.target_entropy,
            num_tasks=4,  # 设置任务数量，这里假设有4种任务：爬升、下降、转弯和跟踪6
            use_causal_graph=variant.get("use_causal_graph", True),  # 是否使用因果图网络
        ).to(device=self.device)
        # self.model = torch.nn.DataParallel(self.model)

        self.optimizer = Lamb(self.model.parameters(), lr=variant["learning_rate"], weight_decay=variant["weight_decay"], eps=1e-8)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda steps: min((steps + 1) / variant["warmup_steps"], 1))

        self.log_temperature_optimizer = torch.optim.Adam([self.model.log_temperature], lr=1e-4, betas=[0.9, 0.999])

        self.pretrain_iter = 0
        self.online_iter = 0
        self.total_transitions_sampled = 0
        self.variant = variant
        self.reward_scale = 1.0 if "antmaze" in variant["env"] else 0.001
        self.logger = Logger(variant)

    def _get_env_spec(self, variant):
        env = gym_dogfight.make(variant["env"], host=variant["host"], port=variant["port"], rendering=True)
        state_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        action_range = [float(env.action_space.low.min()) + 1e-6, float(env.action_space.high.max()) - 1e-6]
        env.close()
        return state_dim, act_dim, action_range

    def _save_model(self, path_prefix, is_pretrain_model=False):
        to_save = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "pretrain_iter": self.pretrain_iter,
            "online_iter": self.online_iter,
            "args": self.variant,
            "total_transitions_sampled": self.total_transitions_sampled,
            "np": np.random.get_state(),
            "python": random.getstate(),
            "pytorch": torch.get_rng_state(),
            "log_temperature_optimizer_state_dict": self.log_temperature_optimizer.state_dict(),
        }

        with open(f"{path_prefix}/model.pt", "wb") as f:
            torch.save(to_save, f)
        print(f"\nModel saved at {path_prefix}/model.pt")

        if is_pretrain_model:
            with open(f"{path_prefix}/pretrain_model.pt", "wb") as f:
                torch.save(to_save, f)
            print(f"Model saved at {path_prefix}/pretrain_model.pt")

    def _load_model(self, path_prefix):
        if Path(f"{path_prefix}/model.pt").exists():
            with open(f"{path_prefix}/model.pt", "rb") as f:
                checkpoint = torch.load(f)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.log_temperature_optimizer.load_state_dict(checkpoint["log_temperature_optimizer_state_dict"])
            self.pretrain_iter = checkpoint["pretrain_iter"]
            self.online_iter = checkpoint["online_iter"]
            self.total_transitions_sampled = checkpoint["total_transitions_sampled"]
            np.random.set_state(checkpoint["np"])
            random.setstate(checkpoint["python"])
            torch.set_rng_state(checkpoint["pytorch"])
            print(f"Model loaded at {path_prefix}/model.pt")

    def _load_dataset(self, env_name):
        dataset_path = f"C:/Users/bafs/Desktop/odt/collected_data/{env_name}.pkl"
        with open(dataset_path, "rb") as f:
            trajectories_dict = pickle.load(f)
        trajectories = []
        for idx in sorted(trajectories_dict.keys()):  # 排序以确保顺序一致
            trajectories.append(trajectories_dict[idx])
        states, traj_lens, returns = [], [], []
        for path in trajectories:
            states.append(path["observations"])
            traj_lens.append(len(path["observations"]))
            returns.append(path["rewards"].sum())
        traj_lens, returns = np.array(traj_lens), np.array(returns)

        # used for input normalization
        states = np.concatenate(states, axis=0)
        state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
        num_timesteps = sum(traj_lens)

        print("=" * 50)
        print(f"Starting new experiment: {env_name}")
        print(f"{len(traj_lens)} trajectories, {num_timesteps} timesteps found")
        print(f"Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}")
        print(f"Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}")
        print(f"Average length: {np.mean(traj_lens):.2f}, std: {np.std(traj_lens):.2f}")
        print(f"Max length: {np.max(traj_lens):.2f}, min: {np.min(traj_lens):.2f}")
        print("=" * 50)

        sorted_inds = np.argsort(returns)  # lowest to highest
        num_trajectories = 1
        timesteps = traj_lens[sorted_inds[-1]]
        ind = len(trajectories) - 2
        while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] < num_timesteps:
            timesteps += traj_lens[sorted_inds[ind]]
            num_trajectories += 1
            ind -= 1
        sorted_inds = sorted_inds[-num_trajectories:]
        trajectories = [trajectories[ii] for ii in sorted_inds]

        return trajectories, state_mean, state_std

    def _augment_trajectories(self, online_envs, target_explore, n, randomized=False):
        max_ep_len = MAX_EPISODE_LEN
        with torch.no_grad():
            # generate init state
            target_return = [target_explore * self.reward_scale]
            # 注意：vec_evaluate_episode_rtg已经修改为处理双头输出
            returns, lengths, trajs = vec_evaluate_episode_rtg(online_envs, self.state_dim, self.act_dim, self.model, max_ep_len=max_ep_len, reward_scale=self.reward_scale, target_return=target_return, mode="normal", state_mean=self.state_mean, state_std=self.state_std, device=self.device, use_mean=False)

        # self.replay_buffer.add_new_trajs(trajs)
        self.aug_trajs += trajs
        self.total_transitions_sampled += np.sum(lengths)

        # 记录更详细的轨迹增强指标
        # 检查returns的类型，确保正确处理
        return_std = 0.0
        length_std = 0.0
        return_max = 0.0
        return_min = 0.0
        
        # 将标量值转换为数组以统一处理
        if np.isscalar(returns) or isinstance(returns, (np.int32, np.int64, np.float32, np.float64)):
            returns_array = np.array([returns])
        else:
            returns_array = np.array(returns)
            
        if np.isscalar(lengths) or isinstance(lengths, (np.int32, np.int64, np.float32, np.float64)):
            lengths_array = np.array([lengths])
        else:
            lengths_array = np.array(lengths)
        
        # 计算统计值
        if len(returns_array) > 1:
            return_std = np.std(returns_array)
        if len(returns_array) > 0:
            return_max = np.max(returns_array)
            return_min = np.min(returns_array)
            
        if len(lengths_array) > 1:
            length_std = np.std(lengths_array)
        
        return {
            "aug_traj/return": np.mean(returns_array), 
            "aug_traj/length": np.mean(lengths_array),
            "aug_traj/return_std": return_std,
            "aug_traj/length_std": length_std,
            "aug_traj/return_max": return_max,
            "aug_traj/return_min": return_min,
            "aug_traj/total_transitions": np.sum(lengths_array),
            "aug_traj/num_trajectories": len(trajs)
        }

    def pretrain(self, eval_envs):
        # 根据是否使用因果图网络选择不同的训练器
        if hasattr(self.model, 'use_causal_graph') and self.model.use_causal_graph:
            trainer = CausalSequenceTrainer(
                model=self.model,
                optimizer=self.optimizer,
                log_temperature_optimizer=self.log_temperature_optimizer,
                scheduler=self.scheduler,
                device=self.device,
                offline_data=self.offline_trajs
            )
        else:
            trainer = SequenceTrainer(
                model=self.model,
                optimizer=self.optimizer,
                log_temperature_optimizer=self.log_temperature_optimizer,
                scheduler=self.scheduler,
                device=self.device,
                offline_data=self.offline_trajs
            )

        writer = (SummaryWriter(self.logger.log_path) if self.variant["log_to_tb"] else None)
        while self.pretrain_iter < self.variant["max_pretrain_iters"]:
            train_outputs = trainer.train_iteration(variant=self.variant)
            outputs = {"time/total": time.time() - self.start_time}
            outputs.update(train_outputs)
            # outputs.update(eval_outputs)
            self.logger.log_metrics(outputs, iter_num=self.pretrain_iter, total_transitions_sampled=self.total_transitions_sampled, writer=writer, is_online=False)
            self._save_model(path_prefix=self.logger.log_path, is_pretrain_model=True)
            self.pretrain_iter += 1

    def evaluate(self, eval_fns):
        eval_start = time.time()
        self.model.eval()
        outputs = {}
        for eval_fn in eval_fns:
            o = eval_fn(self.model)
            outputs.update(o)
        outputs["time/evaluation"] = time.time() - eval_start

        eval_reward = outputs["evaluation/return_mean_gm"]
        return outputs, eval_reward

    def online_tuning(self, online_envs, eval_envs):

        print("\n\n\n*** Online Finetuning ***")

        # 根据是否使用因果图网络选择不同的训练器
        if hasattr(self.model, 'use_causal_graph') and self.model.use_causal_graph:
            trainer = CausalSequenceTrainer(model=self.model, optimizer=self.optimizer, log_temperature_optimizer=self.log_temperature_optimizer, scheduler=self.scheduler, device=self.device, offline_data=self.offline_trajs)
        else:
            trainer = SequenceTrainer(model=self.model, optimizer=self.optimizer, log_temperature_optimizer=self.log_temperature_optimizer, scheduler=self.scheduler, device=self.device, offline_data=self.offline_trajs)
        eval_fns = [
            create_vec_eval_episodes_fn(
                vec_env=eval_envs,
                eval_rtg=self.variant["eval_rtg"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                device=self.device,
                use_mean=True,
                reward_scale=self.reward_scale,
            )
        ]
        writer = (SummaryWriter(self.logger.log_path) if self.variant["log_to_tb"] else None)
        online_start_time = time.time()
        while self.online_iter < self.variant["max_online_iters"]:
            iter_start_time = time.time()
            outputs = {}
            
            # 记录当前迭代的基本信息
            outputs["online_tuning/iteration"] = self.online_iter
            outputs["online_tuning/total_transitions"] = self.total_transitions_sampled
            
            # 轨迹增强
            augment_start_time = time.time()
            augment_outputs = self._augment_trajectories(online_envs, self.variant["online_rtg"], n=self.variant["num_online_rollouts"])
            augment_time = time.time() - augment_start_time
            outputs.update(augment_outputs)
            outputs["time/augmentation"] = augment_time

            # 确定是否进行评估
            is_last_iter = self.online_iter == self.variant["max_online_iters"] - 1
            if (self.online_iter + 1) % self.variant["eval_interval"] == 0 or is_last_iter:
                evaluation = True
            else:
                evaluation = False

            # 训练迭代
            train_start_time = time.time()
            train_outputs = trainer.train_iteration(self.variant)
            train_time = time.time() - train_start_time
            outputs.update(train_outputs)
            outputs["time/training"] = train_time

            # 评估（如果需要）
            if evaluation:
                eval_start_time = time.time()
                eval_outputs, eval_reward = self.evaluate(eval_fns)
                eval_time = time.time() - eval_start_time
                outputs.update(eval_outputs)
                outputs["time/evaluation"] = eval_time

            # 记录总时间
            outputs["time/iteration"] = time.time() - iter_start_time
            outputs["time/total"] = time.time() - self.start_time
            outputs["time/since_online_start"] = time.time() - online_start_time

            # 记录学习进度
            outputs["online_tuning/progress"] = self.online_iter / self.variant["max_online_iters"]
            
            # 记录指标到日志和TensorBoard
            self.logger.log_metrics(outputs, iter_num=self.pretrain_iter + self.online_iter, total_transitions_sampled=self.total_transitions_sampled, writer=writer, is_online=True)
            
            # 保存模型
            self._save_model(path_prefix=self.logger.log_path, is_pretrain_model=False)
            self.online_iter += 1

    def __call__(self):
        def get_env_builder(env_name, host, port):
            def make_env_fn(env_name, host, port):
                env = gym_dogfight.make(env_name, host=host, port=port, rendering=True)
                return env
            return make_env_fn(env_name, host, port)

        target_goal = None
        eval_envs = get_env_builder(env_name='onevsone_ap-v0', host=self.variant['host'], port='57805')
        online_envs = get_env_builder(env_name='onevsone_ap-v0', host=self.variant['host'], port='57805')
        self.start_time = time.time()
        if self.variant["max_pretrain_iters"]:
            self.pretrain(eval_envs)

        if self.variant["max_online_iters"]:
            print("\n\nMaking Online Env.....")
            self.online_tuning(online_envs, eval_envs)
            online_envs.close()
        eval_envs.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default='data_collection-v0')
    parser.add_argument("--online_env", type=str, default='onevsone_ap-v0')
    # parser.add_argument("--host", type=str, default='192.168.43.180')
    parser.add_argument("--host", type=str, default='172.27.240.1')
    parser.add_argument("--port", type=str, default='57805')
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--total_timestep", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--activation_function", type=str, default="relu")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--eval_context_length", type=int, default=20)
    # 0: no pos embedding others: absolute ordering
    parser.add_argument("--ordering", type=int, default=0)
    # 每步执行的环境步数
    parser.add_argument("--inner_steps", type=int, default=10)

    # shared evaluation options
    parser.add_argument("--eval_rtg", type=int, default=3600)
    parser.add_argument("--num_eval_episodes", type=int, default=1)

    # shared training options
    parser.add_argument("--init_temperature", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", "-wd", type=float, default=5e-4)
    parser.add_argument("--warmup_steps", type=int, default=10000)

    # pretraining options
    parser.add_argument("--max_pretrain_iters", type=int, default=2) # 预训练的轮数
    parser.add_argument("--num_updates_per_pretrain_iter", type=int, default=5000)

    # finetuning options
    parser.add_argument("--max_online_iters", type=int, default=1500)
    parser.add_argument("--online_rtg", type=int, default=7200)
    parser.add_argument("--num_online_rollouts", type=int, default=100)
    parser.add_argument("--replay_size", type=int, default=1000)
    parser.add_argument("--num_updates_per_online_iter", type=int, default=300)
    parser.add_argument("--eval_interval", type=int, default=10)

    # environment options
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log_to_tb", "-w", type=bool, default=True)
    parser.add_argument("--save_dir", type=str, default="./exp")
    parser.add_argument("--exp_name", type=str, default="default")

    args = parser.parse_args()

    experiment = Experiment(vars(args))

    print("=" * 50)
    experiment()
