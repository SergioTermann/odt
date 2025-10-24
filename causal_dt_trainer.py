import numpy as np
import torch
import time
import random
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
from datetime import datetime
import networkx as nx
from decision_transformer.models.causal_trainer import CausalTrainer


class CausalSequenceTrainer:
    def __init__(self, model, optimizer, log_temperature_optimizer, scheduler=None, device="cuda", offline_data=None,
                 causal_config=None, adaptive_training=True, log_dir="./logs"):
        self.model = model
        self.offline_traj = offline_data
        self.optimizer = optimizer
        self.log_temperature_optimizer = log_temperature_optimizer
        self.scheduler = scheduler
        self.device = device
        self.start_time = time.time()
        self.step_total = 0
        
        # 训练配置
        self.adaptive_training = adaptive_training
        self.training_phase = "exploration"  # 初始阶段为探索阶段
        self.phase_counter = 0
        self.performance_history = deque(maxlen=10)  # 保存最近10次评估的性能
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 因果训练配置
        self.causal_config = {
            "causal_weight": 0.5,           # 因果训练权重
            "cf_diversity_weight": 0.2,    # 反事实多样性权重
            "sparsity_weight": 0.1,        # 稀疏性权重
            "consistency_weight": 0.3,     # 一致性权重
            "intervention_steps": 5,       # 干预步骤数
            "update_interval": 100,        # 因果图更新间隔
            "causal_discovery_algo": "notears",  # 因果发现算法
            "temporal_window": 3,          # 时序窗口大小
        }
        
        # 更新配置（如果提供）
        if causal_config is not None:
            self.causal_config.update(causal_config)
        
        # 初始化因果训练器（如果模型启用了因果图网络）
        if hasattr(model, 'causal_graph') and model.use_causal_graph:
            self.causal_trainer = CausalTrainer(
                model, 
                optimizer, 
                device,
                causal_discovery_method=self.causal_config["causal_discovery_algo"],
                sparsity_weight=self.causal_config["sparsity_weight"],
                consistency_weight=self.causal_config["consistency_weight"]
            )
        else:
            self.causal_trainer = None
            
        # 训练统计信息
        self.stats = {
            "iterations": 0,
            "total_loss": [],
            "action_loss": [],
            "task_loss": [],
            "causal_loss": [],
            "cf_diversity": [],
            "task_accuracy": [],
            "learning_rates": []
        }

    def train_iteration(self, variant):
        losses, nlls, entropies = [], [], []
        action_losses, task_losses = [], []
        causal_losses = []
        logs = dict()
        train_start = time.time()

        # 更新训练阶段和参数
        if self.adaptive_training:
            self._update_training_phase(variant)

        self.model.train()
        batch_size = variant.get('batch_size', 32)
        inner_steps = variant.get('inner_steps', 10)
        total_timestep = variant.get('total_timestep', 1000)
        eval_interval = variant.get('eval_interval', 100)
        
        # 训练循环
        while self.step_total < total_timestep:
            self.step_total += inner_steps
            self.stats["iterations"] += 1
            
            # 根据当前训练阶段调整采样策略
            if self.training_phase == "exploration":
                # 探索阶段：更多样化的轨迹采样
                trajs = self.sample_trajectories(batch_size=batch_size, diverse_sampling=True)
            elif self.training_phase == "refinement":
                # 精细化阶段：更有针对性的轨迹采样
                trajs = self.sample_trajectories(batch_size=batch_size, diverse_sampling=False)
            else:  # exploitation
                # 利用阶段：聚焦于高回报轨迹
                trajs = self.sample_trajectories(batch_size=batch_size, high_reward_bias=True)
            
            # 标准训练步骤
            loss, nll, entropy = self.train_step_stochastic(trajs)
            losses.append(loss)
            action_losses.append(self.action_loss)
            if self.task_loss is not None:
                task_losses.append(self.task_loss)
            nlls.append(nll)
            entropies.append(entropy)
            
            # 更新统计信息
            self.stats["total_loss"].append(loss)
            self.stats["action_loss"].append(self.action_loss)
            if self.task_loss is not None:
                self.stats["task_loss"].append(self.task_loss)
            
            # 如果启用了因果图网络，进行因果训练
            if self.causal_trainer is not None and variant.get('use_causal_training', True):
                # 根据训练阶段调整因果训练参数
                if self.training_phase == "exploration":
                    # 探索阶段：增大稀疏性权重，学习基本因果结构
                    self.causal_trainer.sparsity_weight = self.causal_config["sparsity_weight"] * 2.0
                    self.causal_trainer.consistency_weight = self.causal_config["consistency_weight"] * 0.5
                elif self.training_phase == "refinement":
                    # 精细化阶段：平衡稀疏性和一致性，优化因果结构
                    self.causal_trainer.sparsity_weight = self.causal_config["sparsity_weight"]
                    self.causal_trainer.consistency_weight = self.causal_config["consistency_weight"]
                else:  # exploitation
                    # 利用阶段：增大一致性权重，稳定因果结构
                    self.causal_trainer.sparsity_weight = self.causal_config["sparsity_weight"] * 0.5
                    self.causal_trainer.consistency_weight = self.causal_config["consistency_weight"] * 2.0
                
                # 从轨迹中提取任务标签和结果标签
                if 'task_labels' in trajs and 'outcome_labels' in trajs:
                    causal_loss = self.train_causal_step(trajs)
                    causal_losses.append(causal_loss['total_loss'])
                    self.stats["causal_loss"].append(causal_loss['total_loss'])
            
            # 定期评估和可视化
            if self.step_total % eval_interval == 0 and 'task_labels' in trajs:
                # 从当前批次中抽取一部分数据进行评估
                eval_indices = np.random.choice(len(trajs['observations']), min(10, len(trajs['observations'])), replace=False)
                eval_states = [trajs['observations'][i] for i in eval_indices]
                eval_task_labels = [trajs['task_labels'][i] for i in eval_indices]
                
                # 评估因果模型
                metrics = self.evaluate_causal(eval_states, eval_task_labels)
                
                # 更新性能历史记录
                if 'task_accuracy' in metrics and 'cf_diversity' in metrics:
                    # 综合指标：准确率 + 多样性
                    performance = metrics['task_accuracy'] + 0.5 * metrics['cf_diversity']
                    self.performance_history.append(performance)
                    
                    # 更新统计信息
                    self.stats["cf_diversity"].append(metrics['cf_diversity'])
                    self.stats["task_accuracy"].append(metrics['task_accuracy'])
                    
                    # 记录到日志
                    logs["eval/task_accuracy"] = metrics['task_accuracy']
                    logs["eval/cf_diversity"] = metrics['cf_diversity']
                    logs["eval/performance"] = performance

        # 记录时间相关指标
        logs["time/training"] = time.time() - train_start
        logs["time/steps_per_second"] = self.step_total / (time.time() - train_start)
        logs["time/total_iterations"] = self.stats["iterations"]
        
        # 记录训练损失相关指标
        logs["training/train_loss_mean"] = np.mean(losses)
        logs["training/train_loss_std"] = np.std(losses)
        logs["training/train_loss_min"] = np.min(losses)
        logs["training/train_loss_max"] = np.max(losses)
        
        # 记录动作和任务损失
        logs["training/action_loss_mean"] = np.mean(action_losses)
        if task_losses:
            logs["training/task_loss_mean"] = np.mean(task_losses)
        
        # 记录负对数似然和熵相关指标
        logs["training/nll_mean"] = np.mean(nlls)
        logs["training/nll_std"] = np.std(nlls)
        logs["training/nll_current"] = nlls[-1]
        logs["training/entropy_mean"] = np.mean(entropies)
        logs["training/entropy_std"] = np.std(entropies)
        logs["training/entropy_current"] = entropies[-1]
        
        # 记录因果训练相关指标
        if causal_losses:
            logs["training/causal_loss_mean"] = np.mean(causal_losses)
            logs["training/causal_loss_std"] = np.std(causal_losses)
            logs["training/causal_loss_current"] = causal_losses[-1]
        
        # 记录温度参数
        logs["training/temp_value"] = self.model.temperature().detach().cpu().item()
        
        # 记录学习率
        if self.scheduler is not None:
            logs["training/learning_rate"] = self.scheduler.get_last_lr()[0]
            self.stats["learning_rates"].append(self.scheduler.get_last_lr()[0])
        
        # 记录梯度信息
        total_grad_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.data.norm(2).item() ** 2
        logs["training/gradient_norm"] = np.sqrt(total_grad_norm)
        
        # 记录训练阶段
        logs["training/phase"] = self.training_phase
        logs["training/phase_counter"] = self.phase_counter
        
        # 可视化训练进度
        if variant.get('visualize', False):
            self.visualize_training_progress()
        
        self.step_total = 0

        return logs
        
    def _update_training_phase(self, variant):
        """根据训练进度和性能更新训练阶段"""
        if not self.adaptive_training:
            return
            
        # 获取配置参数
        phase_length = variant.get('phase_length', 5)  # 每个阶段的迭代次数
        
        # 更新阶段计数器
        self.phase_counter += 1
        
        # 根据阶段计数器和性能历史决定是否切换阶段
        if self.phase_counter >= phase_length:
            self.phase_counter = 0
            
            # 根据当前阶段决定下一阶段
            if self.training_phase == "exploration":
                # 从探索阶段切换到精细化阶段
                self.training_phase = "refinement"
                print(f"Training phase changed: exploration -> refinement")
            elif self.training_phase == "refinement":
                # 从精细化阶段切换到利用阶段
                self.training_phase = "exploitation"
                print(f"Training phase changed: refinement -> exploitation")
            else:  # exploitation
                # 根据性能历史决定是否需要回到探索阶段
                if len(self.performance_history) >= 5:
                    # 计算性能趋势
                    recent_perf = list(self.performance_history)[-5:]
                    if recent_perf[-1] < recent_perf[0] * 0.95:  # 性能下降超过5%
                        # 回到探索阶段
                        self.training_phase = "exploration"
                        print(f"Training phase changed: exploitation -> exploration (performance decline detected)")
                    else:
                        # 继续利用阶段
                        print(f"Continuing exploitation phase")
                else:
                    # 数据不足，继续当前阶段
                    print(f"Continuing {self.training_phase} phase (insufficient performance data)")
        
        # 记录当前阶段
        print(f"Current training phase: {self.training_phase}, counter: {self.phase_counter}")
        return self.training_phase

    def sample_trajectories(self, batch_size=32, diverse_sampling=False, high_reward_bias=False):
        """采样轨迹，支持多种采样策略
        
        Args:
            batch_size: 批次大小
            diverse_sampling: 是否使用多样化采样（增加轨迹多样性）
            high_reward_bias: 是否偏向高回报轨迹
        
        Returns:
            采样的轨迹字典
        """
        # Ensure batch size doesn't exceed available trajectories
        batch_size = min(batch_size, len(self.offline_traj))

        # 根据不同的采样策略选择轨迹
        if diverse_sampling:
            # 多样化采样：使用聚类或其他方法确保轨迹多样性
            # 这里使用一个简单的启发式方法：按照轨迹长度和平均回报进行排序，然后均匀采样
            traj_features = []
            for idx, traj in enumerate(self.offline_traj):
                traj_len = len(traj['observations'])
                avg_reward = np.mean(traj['rewards']) if traj_len > 0 else 0
                traj_features.append((idx, traj_len, avg_reward))
            
            # 按长度排序
            traj_features.sort(key=lambda x: x[1])
            
            # 均匀采样
            stride = max(1, len(traj_features) // batch_size)
            sampled_indices = [traj_features[i][0] for i in range(0, len(traj_features), stride)][:batch_size]
            
            # 如果样本不足，随机补充
            if len(sampled_indices) < batch_size:
                remaining = batch_size - len(sampled_indices)
                all_indices = set(range(len(self.offline_traj)))
                available = list(all_indices - set(sampled_indices))
                if available:
                    sampled_indices.extend(np.random.choice(available, size=min(remaining, len(available)), replace=False))
        
        elif high_reward_bias:
            # 高回报偏好：优先选择高回报轨迹
            traj_rewards = []
            for idx, traj in enumerate(self.offline_traj):
                avg_reward = np.mean(traj['rewards']) if len(traj['rewards']) > 0 else 0
                traj_rewards.append((idx, avg_reward))
            
            # 按回报降序排序
            traj_rewards.sort(key=lambda x: x[1], reverse=True)
            
            # 从高回报轨迹中采样，使用指数衰减概率
            probs = np.exp([i / -10 for i in range(len(traj_rewards))])
            probs = probs / np.sum(probs)
            
            # 根据概率采样
            indices = np.random.choice(
                [tr[0] for tr in traj_rewards], 
                size=batch_size, 
                replace=False, 
                p=probs[:len(traj_rewards)]
            )
            sampled_indices = indices.tolist()
        
        else:
            # 标准随机采样
            sampled_indices = np.random.choice(len(self.offline_traj), size=batch_size, replace=False)
        
        # 预分配内存以提高性能
        observations = []
        actions = []
        next_observations = []
        rewards = []
        terminals = []
        rtgs = []
        # 添加任务标签和结果标签的列表（如果有）
        task_labels = []
        outcome_labels = []
        
        max_trajectory_length = 1024

        for idx in sampled_indices:
            traj = self.offline_traj[idx]

            observations.append(traj['observations'][:max_trajectory_length])
            actions.append(traj['actions'][:max_trajectory_length])
            next_observations.append(traj['next_observations'][:max_trajectory_length])
            rewards.append(traj['rewards'][:max_trajectory_length])
            terminals.append(traj['terminals'][:max_trajectory_length])
            rtgs.append(traj['rtgs'][:max_trajectory_length])
            
            # 如果轨迹中包含任务标签和结果标签，则添加
            if 'task_labels' in traj:
                task_labels.append(traj['task_labels'][:max_trajectory_length])
            if 'outcome_labels' in traj:
                outcome_labels.append(traj['outcome_labels'][:max_trajectory_length])

        # 构建返回字典
        sampled_trajs = {
            'observations': observations, 
            'actions': actions, 
            'next_observations': next_observations, 
            'rewards': rewards, 
            'terminals': terminals, 
            'rtgs': rtgs
        }
        
        # 如果有任务标签和结果标签，则添加到返回字典中
        if task_labels:
            sampled_trajs['task_labels'] = task_labels
        if outcome_labels:
            sampled_trajs['outcome_labels'] = outcome_labels

        return sampled_trajs

    def safe_log_prob(self, action_preds, action_target):
        # 对目标值进行限制（通常在 [-1, 1] 范围）
        action_target = torch.clamp(action_target, min=-1.0+0.001, max=1.0-0.001)

        # 检查action_preds是否为分布对象或张量
        if hasattr(action_preds, 'log_prob'):
            # 如果是分布对象，使用log_prob方法
            log_prob = action_preds.log_prob(action_target)
        else:
            # 如果是张量，计算均方误差的负值作为对数概率的近似
            # 这里我们使用MSE的负值，因为对数概率越大越好，而MSE越小越好
            mse = torch.mean((action_preds - action_target) ** 2, dim=-1)
            log_prob = -mse

        return log_prob

    def loss_fn(self, action_preds, action_target, temperature, entropy_reg=0.01):
        # 安全计算对数概率
        log_prob = self.safe_log_prob(action_preds, action_target)

        # 平均对数似然
        log_likelihood = log_prob.mean()

        # 计算熵
        if hasattr(action_preds, 'entropy'):
            # 如果是分布对象，使用entropy方法
            entropy = action_preds.entropy().mean()
        else:
            # 如果是张量，我们使用一个固定的熵值或者基于张量的方差计算一个近似熵
            # 这里我们简单地使用一个固定值
            entropy = torch.tensor(0.0, device=action_preds.device)

        # 损失计算
        loss = -(log_likelihood + entropy_reg * entropy)

        return loss, -log_likelihood, entropy

    def train_step_stochastic(self, trajs):
        """执行一步随机训练
        
        Args:
            trajs: 轨迹数据字典
            
        Returns:
            tuple: (总损失, 负对数似然, 熵)
        """
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
        _, action_preds, _, task_preds = self.model.forward(
            states, actions, rewards, rtgs, timesteps, ordering, padding_mask=padding_mask
        )
        
        # 计算任务参数（动作）的损失
        action_loss, nll, entropy = self.loss_fn(action_preds, action_target, self.model.temperature().detach())
        
        # 如果有任务标签，计算任务选择的损失
        task_loss = None
        if 'task_labels' in trajs:
            task_labels = torch.tensor(trajs['task_labels'], dtype=torch.long).to(self.device)
            task_loss = torch.nn.functional.cross_entropy(
                task_preds.reshape(-1, self.model.num_tasks), 
                task_labels.reshape(-1)
            )
            # 总损失为动作损失和任务损失的加权和
            loss = action_loss + 0.5 * task_loss
            task_loss = task_loss.detach().cpu().item()  # 转换为标量以便返回
        else:
            # 总损失为动作损失
            loss = action_loss
        
        # 梯度清零
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
        
        # 保存动作损失和任务损失供外部使用
        self.action_loss = action_loss.detach().cpu().item()
        self.task_loss = task_loss
        
        return loss.detach().cpu().item(), nll.detach().cpu().item(), entropy.detach().cpu().item()
    
    def train_causal_step(self, trajs):
        """使用因果训练器进行因果关系学习和反事实训练
        
        Args:
            trajs: 轨迹数据字典，包含观测、动作、奖励等信息
            
        Returns:
            dict: 包含各种损失值的字典
        """
        if self.causal_trainer is None:
            return {'total_loss': 0.0}
        
        # 准备数据
        states = torch.tensor(trajs['observations'], dtype=torch.float32).to(self.device)
        actions = torch.tensor(trajs['actions'], dtype=torch.float32).to(self.device)
        rewards = torch.tensor(trajs['rewards'], dtype=torch.float32).to(self.device)
        rtgs = torch.tensor(trajs['rtgs'], dtype=torch.float32).to(self.device)
        timesteps = torch.arange(len(states), device=self.device)
        task_labels = torch.tensor(trajs['task_labels'], dtype=torch.long).to(self.device)
        outcome_labels = torch.tensor(trajs['outcome_labels'], dtype=torch.float32).to(self.device)
        
        # 根据当前训练阶段调整训练参数
        intervention_steps = self.causal_config["intervention_steps"]
        
        # 在探索阶段，增加干预步骤以更好地探索因果关系
        if self.training_phase == "exploration":
            intervention_steps = max(intervention_steps, 7)  # 增加干预步骤
        elif self.training_phase == "refinement":
            intervention_steps = intervention_steps  # 保持默认值
        else:  # exploitation
            intervention_steps = max(3, intervention_steps - 2)  # 减少干预步骤，专注于利用已学习的因果关系
        
        # 计算批次权重（基于轨迹回报）
        # 高回报轨迹在因果学习中获得更高权重
        batch_weights = None
        if 'rewards' in trajs:
            # 计算每个轨迹的平均回报
            avg_rewards = [np.mean(r) for r in trajs['rewards']]
            # 归一化为权重（使用softmax）
            exp_rewards = np.exp(np.array(avg_rewards) * 0.1)  # 温度参数0.1控制权重分布
            batch_weights = torch.tensor(exp_rewards / np.sum(exp_rewards), dtype=torch.float32).to(self.device)
        
        # 使用因果训练器进行训练，传入额外参数
        causal_loss = self.causal_trainer.train_counterfactual(
            states, 
            actions, 
            rewards, 
            rtgs, 
            timesteps, 
            task_labels, 
            outcome_labels,
            intervention_steps=intervention_steps,
            batch_weights=batch_weights
        )
        
        # 记录训练指标
        if 'cf_diversity' in causal_loss:
            self.stats["cf_diversity"].append(causal_loss['cf_diversity'])
        
        return causal_loss
    
    def evaluate_causal(self, states, task_labels, outcome_labels=None):
        """评估因果图网络和反事实推理的效果
        
        Args:
            states: 状态数据
            task_labels: 任务标签
            outcome_labels: 结果标签（可选）
            
        Returns:
            dict: 包含多种评估指标的字典
        """
        if self.causal_trainer is None:
            return {'cf_diversity': 0.0, 'task_accuracy': 0.0}
        
        # 准备数据
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        task_labels = torch.tensor(task_labels, dtype=torch.long).to(self.device)
        
        # 如果提供了结果标签，也将其转换为张量
        if outcome_labels is not None:
            outcome_labels = torch.tensor(outcome_labels, dtype=torch.float32).to(self.device)
        
        # 使用因果训练器进行评估
        metrics = self.causal_trainer.evaluate_counterfactual(states, task_labels, outcome_labels)
        
        # 计算额外的评估指标
        if hasattr(self.model, 'causal_graph') and self.model.use_causal_graph:
            # 获取因果图结构
            causal_matrix = self.model.causal_graph.get_causal_matrix().detach().cpu().numpy()
            
            # 计算因果图稀疏度（非零元素比例）
            if causal_matrix.size > 0:
                sparsity = 1.0 - (np.count_nonzero(causal_matrix) / causal_matrix.size)
                metrics['causal_sparsity'] = float(sparsity)
            
            # 计算因果图强度（非零元素的平均绝对值）
            if np.count_nonzero(causal_matrix) > 0:
                strength = np.abs(causal_matrix[causal_matrix != 0]).mean()
                metrics['causal_strength'] = float(strength)
            
            # 计算因果图的结构特性
            if hasattr(self.model.causal_graph, 'adjacency_matrix'):
                adj_matrix = self.model.causal_graph.adjacency_matrix.detach().cpu().numpy()
                # 计算入度和出度的均值和方差
                if adj_matrix.size > 0:
                    in_degrees = np.sum(adj_matrix, axis=0)
                    out_degrees = np.sum(adj_matrix, axis=1)
                    metrics['avg_in_degree'] = float(np.mean(in_degrees))
                    metrics['avg_out_degree'] = float(np.mean(out_degrees))
                    metrics['var_in_degree'] = float(np.var(in_degrees))
                    metrics['var_out_degree'] = float(np.var(out_degrees))
        
        # 如果有反事实多样性指标，计算归一化多样性
        if 'cf_diversity' in metrics and metrics['cf_diversity'] > 0:
            # 归一化到[0,1]范围
            metrics['cf_diversity_norm'] = min(1.0, metrics['cf_diversity'] / 0.5)  # 假设0.5是一个合理的多样性上限
        
        # 如果有任务准确率指标，计算加权性能分数
        if 'task_accuracy' in metrics and 'cf_diversity' in metrics:
            # 综合性能分数：准确率 + 多样性权重 * 多样性
            cf_weight = 0.3  # 多样性权重
            metrics['performance_score'] = metrics['task_accuracy'] + cf_weight * metrics.get('cf_diversity_norm', metrics['cf_diversity'])
        
        return metrics
        
    def visualize_training_progress(self, save_dir=None):
        """可视化训练进度，包括损失、准确率、因果指标等
        
        Args:
            save_dir: 保存图表的目录，如果为None则使用默认日志目录
        """
        if save_dir is None:
            save_dir = self.log_dir
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建图表
        plt.figure(figsize=(20, 15))
        
        # 1. 绘制动作损失和任务损失
        plt.subplot(2, 2, 1)
        if hasattr(self, 'action_losses') and len(self.action_losses) > 0:
            plt.plot(self.action_losses, label='Action Loss')
        if hasattr(self, 'task_losses') and len(self.task_losses) > 0:
            plt.plot(self.task_losses, label='Task Loss')
        plt.title('Training Losses')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # 2. 绘制因果损失和反事实损失
        plt.subplot(2, 2, 2)
        if hasattr(self, 'causal_losses') and len(self.causal_losses) > 0:
            plt.plot(self.causal_losses, label='Causal Loss')
        if hasattr(self, 'counterfactual_losses') and len(self.counterfactual_losses) > 0:
            plt.plot(self.counterfactual_losses, label='Counterfactual Loss')
        plt.title('Causal Training Losses')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # 3. 绘制评估指标
        plt.subplot(2, 2, 3)
        if hasattr(self, 'task_accuracies') and len(self.task_accuracies) > 0:
            plt.plot(self.task_accuracies, label='Task Accuracy')
        if hasattr(self, 'cf_diversities') and len(self.cf_diversities) > 0:
            plt.plot(self.cf_diversities, label='CF Diversity')
        if hasattr(self, 'performance_scores') and len(self.performance_scores) > 0:
            plt.plot(self.performance_scores, label='Performance Score')
        plt.title('Evaluation Metrics')
        plt.xlabel('Evaluation Points')
        plt.ylabel('Metric Value')
        plt.legend()
        plt.grid(True)
        
        # 4. 绘制因果图指标
        plt.subplot(2, 2, 4)
        if hasattr(self, 'causal_sparsities') and len(self.causal_sparsities) > 0:
            plt.plot(self.causal_sparsities, label='Causal Sparsity')
        if hasattr(self, 'causal_strengths') and len(self.causal_strengths) > 0:
            plt.plot(self.causal_strengths, label='Causal Strength')
        plt.title('Causal Graph Metrics')
        plt.xlabel('Evaluation Points')
        plt.ylabel('Metric Value')
        plt.legend()
        plt.grid(True)
        
        # 保存图表
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'training_progress_{timestamp}.png'))
        plt.close()
        
        print(f"训练进度可视化已保存到: {os.path.join(save_dir, f'training_progress_{timestamp}.png')}")
    
    def visualize_causal_graph(self, save_dir=None):
        """可视化因果图结构
        
        Args:
            save_dir: 保存图表的目录，如果为None则使用默认日志目录
        """
        if not hasattr(self.model, 'causal_graph') or not self.model.use_causal_graph:
            print("模型没有因果图组件，无法可视化")
            return
            
        if save_dir is None:
            save_dir = self.log_dir
            
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 获取因果矩阵
        causal_matrix = self.model.causal_graph.get_causal_matrix().detach().cpu().numpy()
        
        # 1. 绘制因果矩阵热力图
        plt.figure(figsize=(10, 8))
        sns.heatmap(causal_matrix, cmap='coolwarm', center=0, annot=False)
        plt.title('Causal Matrix Heatmap')
        plt.xlabel('Effect Variables')
        plt.ylabel('Cause Variables')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'causal_matrix_heatmap_{timestamp}.png'))
        plt.close()
        
        # 2. 如果有邻接矩阵，绘制因果网络图
        if hasattr(self.model.causal_graph, 'adjacency_matrix'):
            adj_matrix = self.model.causal_graph.adjacency_matrix.detach().cpu().numpy()
            
            # 创建有向图
            G = nx.DiGraph()
            
            # 添加节点
            n_nodes = adj_matrix.shape[0]
            for i in range(n_nodes):
                G.add_node(i)
            
            # 添加边（只添加权重大于阈值的边）
            threshold = 0.1
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if abs(adj_matrix[i, j]) > threshold:
                        G.add_edge(i, j, weight=float(adj_matrix[i, j]))
            
            # 绘制网络图
            plt.figure(figsize=(12, 10))
            pos = nx.spring_layout(G, seed=42)  # 布局算法
            
            # 绘制节点
            nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
            
            # 绘制边，边的宽度与权重成正比
            edges = G.edges()
            weights = [G[u][v]['weight'] * 2 for u, v in edges]  # 调整边的宽度
            nx.draw_networkx_edges(G, pos, width=weights, alpha=0.7, 
                                  edge_color='gray', arrows=True, arrowsize=15)
            
            # 绘制节点标签
            nx.draw_networkx_labels(G, pos, font_size=10)
            
            plt.title('Causal Network Graph')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'causal_network_graph_{timestamp}.png'))
            plt.close()
        
        # 3. 如果有多层次因果矩阵，绘制每一层的热力图
        if hasattr(self.model.causal_graph, 'multi_level_matrices'):
            multi_level_matrices = self.model.causal_graph.multi_level_matrices
            n_levels = len(multi_level_matrices)
            
            if n_levels > 0:
                fig, axes = plt.subplots(1, n_levels, figsize=(5 * n_levels, 4))
                if n_levels == 1:
                    axes = [axes]  # 确保axes是列表
                
                for i, matrix in enumerate(multi_level_matrices):
                    matrix_np = matrix.detach().cpu().numpy()
                    sns.heatmap(matrix_np, cmap='coolwarm', center=0, annot=False, ax=axes[i])
                    axes[i].set_title(f'Level {i+1} Causal Matrix')
                    axes[i].set_xlabel('Effect Variables')
                    if i == 0:
                        axes[i].set_ylabel('Cause Variables')
                
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'multi_level_causal_matrices_{timestamp}.png'))
                plt.close()
        
        print(f"因果图可视化已保存到目录: {save_dir}")
        
    def visualize_counterfactual_samples(self, states, task_labels, n_samples=5, save_dir=None):
        """可视化反事实样本
        
        Args:
            states: 原始状态数据
            task_labels: 任务标签
            n_samples: 生成的反事实样本数量
            save_dir: 保存图表的目录，如果为None则使用默认日志目录
        """
        if self.causal_trainer is None or not hasattr(self.model, 'causal_graph'):
            print("模型没有因果图或反事实组件，无法可视化")
            return
            
        if save_dir is None:
            save_dir = self.log_dir
            
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 准备数据
        states = torch.tensor(states[:n_samples], dtype=torch.float32).to(self.device)
        task_labels = torch.tensor(task_labels[:n_samples], dtype=torch.long).to(self.device)
        
        # 生成反事实样本
        with torch.no_grad():
            cf_samples = self.causal_trainer.generate_counterfactuals(states, task_labels, n_samples=3)
        
        # 可视化原始样本和反事实样本
        n_features = states.shape[1]
        n_rows = min(n_samples, states.shape[0])
        
        plt.figure(figsize=(15, 3 * n_rows))
        
        for i in range(n_rows):
            # 原始样本
            plt.subplot(n_rows, 4, i * 4 + 1)
            plt.bar(range(n_features), states[i].cpu().numpy())
            plt.title(f'Original Sample {i+1}\nTask: {task_labels[i].item()}')
            plt.xlabel('Feature Index')
            plt.ylabel('Feature Value')
            
            # 反事实样本
            for j in range(min(3, len(cf_samples[i]))):
                plt.subplot(n_rows, 4, i * 4 + j + 2)
                plt.bar(range(n_features), cf_samples[i][j].cpu().numpy())
                plt.title(f'Counterfactual {j+1} for Sample {i+1}')
                plt.xlabel('Feature Index')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'counterfactual_samples_{timestamp}.png'))
        plt.close()
        
        print(f"反事实样本可视化已保存到: {os.path.join(save_dir, f'counterfactual_samples_{timestamp}.png')}")
        
    def visualize_all(self, states=None, task_labels=None, save_dir=None):
        """可视化所有内容，包括训练进度、因果图和反事实样本
        
        Args:
            states: 用于反事实可视化的状态数据
            task_labels: 用于反事实可视化的任务标签
            save_dir: 保存图表的目录，如果为None则使用默认日志目录
        """
        # 创建保存目录
        if save_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = os.path.join(self.log_dir, f'visualization_{timestamp}')
            
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 可视化训练进度
        self.visualize_training_progress(save_dir)
        
        # 可视化因果图
        self.visualize_causal_graph(save_dir)
        
        # 如果提供了数据，可视化反事实样本
        if states is not None and task_labels is not None:
            self.visualize_counterfactual_samples(states, task_labels, save_dir=save_dir)
            
        print(f"所有可视化结果已保存到目录: {save_dir}")
        
        return save_dir