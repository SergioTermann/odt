import numpy as np
import torch
import time
import random


class SequenceTrainer:
    def __init__(self, model, optimizer, log_temperature_optimizer, scheduler=None, device="cuda", offline_data=None):
        self.model = model
        self.offline_traj = offline_data
        self.optimizer = optimizer
        self.log_temperature_optimizer = log_temperature_optimizer
        self.scheduler = scheduler
        self.device = device
        self.start_time = time.time()
        self.step_total = 0

    def train_iteration(self, variant):
        losses, nlls, entropies = [], [], []
        logs = dict()
        train_start = time.time()

        self.model.train()
        while self.step_total < variant['total_timestep']:
            self.step_total += variant.get('inner_steps', 10)  # 使用配置中的inner_steps参数，默认为10
            trajs = self.sample_trajectories()
            # train_step_stochastic已修改为处理双头输出
            loss, nll, entropy = self.train_step_stochastic(trajs)
            losses.append(loss)
            nlls.append(nll)
            entropies.append(entropy)

        # 记录时间相关指标
        logs["time/training"] = time.time() - train_start
        logs["time/steps_per_second"] = self.step_total / (time.time() - train_start)
        
        # 记录训练损失相关指标
        logs["training/train_loss_mean"] = np.mean(losses)
        logs["training/train_loss_std"] = np.std(losses)
        logs["training/train_loss_min"] = np.min(losses)
        logs["training/train_loss_max"] = np.max(losses)
        
        # 记录负对数似然和熵相关指标
        logs["training/nll_mean"] = np.mean(nlls)
        logs["training/nll_std"] = np.std(nlls)
        logs["training/nll_current"] = nlls[-1]
        logs["training/entropy_mean"] = np.mean(entropies)
        logs["training/entropy_std"] = np.std(entropies)
        logs["training/entropy_current"] = entropies[-1]
        
        # 记录温度参数
        logs["training/temp_value"] = self.model.temperature().detach().cpu().item()
        
        # 记录学习率
        if self.scheduler is not None:
            logs["training/learning_rate"] = self.scheduler.get_last_lr()[0]
        
        # 记录梯度信息
        total_grad_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.data.norm(2).item() ** 2
        logs["training/gradient_norm"] = np.sqrt(total_grad_norm)
        
        self.step_total = 0

        return logs

    def sample_trajectories(self, batch_size=32):
        # Ensure batch size doesn't exceed available trajectories
        batch_size = min(batch_size, len(self.offline_traj))

        # Randomly sample trajectory indices
        sampled_indices = np.random.choice(
            len(self.offline_traj),
            size=batch_size,
            replace=False  # Ensure unique sampling
        )
        # # Extract sampled trajectories
        # sampled_trajs = {
        #     'observations': [self.offline_traj[idx]['observations'] for idx in sampled_indices],
        #     'actions': [self.offline_traj[idx]['actions'] for idx in sampled_indices],
        #     'next_observations': [self.offline_traj[idx]['next_observations'] for idx in sampled_indices],
        #     'rewards': [self.offline_traj[idx]['rewards'] for idx in sampled_indices],
        #     'terminals': [self.offline_traj[idx]['terminals'] for idx in sampled_indices],
        #     'rtgs': [self.offline_traj[idx]['rtgs'] for idx in sampled_indices]
        # }
        # 预分配内存以提高性能
        observations = []
        actions = []
        next_observations = []
        rewards = []
        terminals = []
        rtgs = []
        max_trajectory_length = 1024
        # for idx, item in enumerate(self.offline_traj):
        #     if 'next_observations' not in item:
        #         print(item)

        for idx in sampled_indices:
            traj = self.offline_traj[idx]

            observations.append(traj['observations'][:max_trajectory_length])
            actions.append(traj['actions'][:max_trajectory_length])
            next_observations.append(traj['next_observations'][:max_trajectory_length])
            rewards.append(traj['rewards'][:max_trajectory_length])
            terminals.append(traj['terminals'][:max_trajectory_length])
            rtgs.append(traj['rtgs'][:max_trajectory_length])

            # 构建返回字典
            sampled_trajs = {'observations': observations, 'actions': actions, 'next_observations': next_observations, 'rewards': rewards, 'terminals': terminals, 'rtgs': rtgs}

        return sampled_trajs

    def safe_log_prob(self, action_preds, action_target):

        # 对目标值进行限制（通常在 [-1, 1] 范围）
        action_target = torch.clamp(action_target, min=-1.0+0.001, max=1.0-0.001)

        # 计算对数概率
        log_prob = action_preds.log_prob(action_target)

        return log_prob

    def loss_fn(self, action_preds, action_target, temperature, entropy_reg=0.01):
        # 安全计算对数概率
        log_prob = self.safe_log_prob(action_preds, action_target)

        # 平均对数似然
        log_likelihood = log_prob.mean()

        # 计算熵
        entropy = action_preds.entropy().mean()

        # 损失计算
        loss = -(log_likelihood + entropy_reg * entropy)

        return loss, -log_likelihood, entropy

    def train_step_stochastic(self, trajs):
        # states对应observations
        states = torch.tensor(trajs['observations'], dtype=torch.float32).to(self.device)
        actions = torch.tensor(trajs['actions'], dtype=torch.float32).to(self.device)
        rewards = torch.tensor(trajs['rewards'], dtype=torch.float32).to(self.device)
        dones = torch.tensor(trajs['terminals'], dtype=torch.bool).to(self.device)
        rtgs = torch.tensor(trajs['rtgs'], dtype=torch.float32).to(self.device)
        timesteps = torch.arange(len(states), device=self.device)
        ordering = torch.arange(len(states), device=self.device)
        padding_mask = torch.ones((len(states), states.shape[1]), dtype=torch.bool, device=self.device)
        action_target = torch.clone(actions)
        
        # 获取模型的双头输出
        _, action_preds, _, task_preds = self.model.forward(states, actions, rewards, rtgs, timesteps, ordering, padding_mask=padding_mask)
        
        # 计算任务参数（动作）的损失
        action_loss, nll, entropy = self.loss_fn(action_preds, action_target, self.model.temperature().detach())
        
        # 如果有任务标签，计算任务选择的损失
        # 注意：这里假设没有任务标签，所以我们暂时不计算任务选择的损失
        # 实际应用中，需要根据具体情况添加任务标签和相应的损失计算
        
        # 总损失为动作损失
        loss = action_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
        self.optimizer.step()

        self.log_temperature_optimizer.zero_grad()
        temperature_loss = (self.model.temperature() * (entropy - self.model.target_entropy).detach())
        temperature_loss.backward()
        self.log_temperature_optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()
        
        # 保存动作损失供外部使用
        self.action_loss = action_loss.detach().cpu().item()
        
        return loss.detach().cpu().item(), nll.detach().cpu().item(), entropy.detach().cpu().item()
