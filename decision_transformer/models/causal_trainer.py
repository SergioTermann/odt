import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import networkx as nx
from sklearn.metrics import mutual_info_score, adjusted_mutual_info_score
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
from datetime import datetime
from sklearn.preprocessing import KBinsDiscretizer

class CausalTrainer:
    """
    增强版因果关系学习和反事实训练器
    用于训练因果图网络和反事实决策模块
    支持高级因果发现算法、多阶段训练和自适应学习
    """
    def __init__(self, model, optimizer, device, causal_discovery_method='pc', 
                 sparsity_weight=0.1, consistency_weight=0.2):
        """
        初始化因果训练器
        
        Args:
            model: 包含因果图网络的决策转换器模型
            optimizer: 优化器
            device: 训练设备
            causal_discovery_method: 因果发现算法 ('pc', 'granger', 'score')
            sparsity_weight: 因果图稀疏性权重
            consistency_weight: 因果一致性权重
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.causal_discovery_method = causal_discovery_method
        self.sparsity_weight = sparsity_weight
        self.consistency_weight = consistency_weight
        
        # 确保模型启用了因果图网络
        assert hasattr(model, 'causal_graph') and hasattr(model, 'counterfactual_decision_maker'), \
            "模型必须包含因果图网络和反事实决策模块"
        
        # 任务转换统计
        self.task_transition_counts = torch.zeros((model.num_tasks, model.num_tasks), device=device)
        
        # 任务结果统计（成功/失败）
        self.task_outcome_counts = defaultdict(lambda: {'success': 0, 'failure': 0})
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # 因果图结构
        self.causal_graph_nx = None         # NetworkX格式的因果图
        
        # 训练状态跟踪
        self.training_stage = 'structure_learning'  # 'structure_learning', 'counterfactual_training', 'fine_tuning'
        self.best_validation_loss = float('inf')
        self.patience_counter = 0
        self.max_patience = 10
        
    def update_causal_statistics(self, task_sequences, outcome_sequences, state_features=None):
        """
        更新任务转换统计和结果统计，支持多层次和时序因果关系
        
        Args:
            task_sequences: 任务序列，形状为 [batch_size, seq_length]
                           每个元素是任务索引 (0 ~ num_tasks-1)
            outcome_sequences: 结果序列，形状为 [batch_size, seq_length]
                              每个元素是布尔值，表示任务是否成功
            state_features: 状态特征序列，形状为 [batch_size, seq_length, hidden_size]，可选
        """
        batch_size, seq_length = task_sequences.shape
        
        # 初始化时序任务关联（如果需要）
        if not hasattr(self, 'temporal_task_counts'):
            max_delay = 5  # 最多考虑5个时间步的延迟效应
            self.temporal_task_counts = torch.zeros(
                max_delay, self.model.num_tasks, self.model.num_tasks, 
                device=self.device
            )
        
        # 更新任务转换统计
        for b in range(batch_size):
            for t in range(seq_length - 1):
                curr_task = task_sequences[b, t].item()
                next_task = task_sequences[b, t + 1].item()
                self.task_transition_counts[curr_task, next_task] += 1
                
                # 更新时序任务关联（考虑多个时间步的延迟效应）
                for delay in range(1, min(t+1, self.temporal_task_counts.shape[0]) + 1):
                    if t-delay+1 >= 0:
                        past_task = task_sequences[b, t-delay+1].item()
                        self.temporal_task_counts[delay-1, past_task, curr_task] += 1
        
        # 更新任务结果统计
        for b in range(batch_size):
            for t in range(seq_length):
                task = task_sequences[b, t].item()
                outcome = outcome_sequences[b, t].item()
                
                if outcome > 0:
                    self.task_outcome_counts[task]['success'] += 1
                else:
                    self.task_outcome_counts[task]['failure'] += 1
        
        # 如果提供了状态特征，计算条件互信息以发现潜在的因果关系
        if state_features is not None and state_features.shape[0] > 10:  # 确保有足够的样本
            try:
                # 将任务和状态特征转换为CPU NumPy数组以使用sklearn
                tasks_np = task_sequences.cpu().numpy().reshape(-1)
                
                # 使用PCA降维状态特征以简化计算（如果维度太高）
                if state_features.shape[-1] > 10:
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=10)
                    states_np = state_features.reshape(-1, state_features.shape[-1]).cpu().numpy()
                    states_np = pca.fit_transform(states_np)
                else:
                    states_np = state_features.reshape(-1, state_features.shape[-1]).cpu().numpy()
                
                # 计算任务与状态特征之间的互信息
                mi_scores = []
                for feature_idx in range(states_np.shape[1]):
                    feature = states_np[:, feature_idx]
                    # 将连续特征离散化为10个bin
                    discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
                    feature_discrete = discretizer.fit_transform(feature.reshape(-1, 1)).flatten()
                    
                    # 计算互信息
                    mi = adjusted_mutual_info_score(tasks_np, feature_discrete)
                    mi_scores.append(mi)
                    
                # 记录最高互信息的特征
                top_features = np.argsort(mi_scores)[-5:]  # 取前5个最相关的特征
                print(f"Top 5 features with highest mutual information: {top_features}")
                print(f"MI scores: {[mi_scores[i] for i in top_features]}")
                
            except Exception as e:
                print(f"Error computing mutual information: {e}")
        
        # 更新NetworkX因果图（用于可视化和分析）
        self._update_causal_graph_nx()
    
    def _update_causal_graph_nx(self):
        """
        更新NetworkX格式的因果图，用于可视化和分析
        """
        G = nx.DiGraph()
        
        # 添加节点
        for i in range(self.model.num_tasks):
            G.add_node(i, name=f"Task_{i}")
        
        # 添加边（因果关系）
        transition_probs = self.task_transition_counts.clone()
        row_sums = transition_probs.sum(dim=1, keepdim=True)
        row_sums[row_sums == 0] = 1  # 避免除零
        transition_probs = transition_probs / row_sums
        
        for i in range(self.model.num_tasks):
            for j in range(self.model.num_tasks):
                if transition_probs[i, j] > 0.05:  # 只添加概率大于阈值的边
                    G.add_edge(i, j, weight=transition_probs[i, j].item())
        
        self.causal_graph_nx = G
    
    def learn_causal_structure(self, threshold=0.1, method=None):
        """
        从统计数据中学习因果结构，支持多种因果发现算法
        
        Args:
            threshold: 因果关系强度阈值
            method: 因果发现方法，如果为None则使用初始化时指定的方法
            
        Returns:
            learned: 是否学习到了新的因果结构
            metrics: 因果结构学习的相关指标
        """
        if method is None:
            method = self.causal_discovery_method
        
        # 获取任务数量
        num_tasks = self.model.num_tasks
        
        # 初始化新的因果矩阵列表（对应多层次因果结构）
        new_causal_matrices = []
        
        # 根据不同方法学习因果结构
        if method == 'pc':
            # PC算法启发式实现（简化版）
            # 1. 从完全连接的图开始
            causal_matrix = torch.ones(num_tasks, num_tasks).to(self.device)
            
            # 2. 移除独立的边（基于条件独立性测试）
            transition_probs = self.task_transition_counts.clone()
            row_sums = transition_probs.sum(dim=1, keepdim=True)
            row_sums[row_sums == 0] = 1  # 避免除零
            transition_probs = transition_probs / row_sums
            
            # 简化的条件独立性测试：如果转换概率低于阈值，认为是独立的
            causal_matrix = (transition_probs > threshold).float() * transition_probs
            
            # 3. 确定边的方向（这里简化为使用转换概率）
            new_causal_matrices.append(causal_matrix)
            
        elif method == 'granger':
            # Granger因果性启发式实现
            # 使用时序任务关联来确定因果关系
            if hasattr(self, 'temporal_task_counts'):
                for delay in range(self.temporal_task_counts.shape[0]):
                    delay_counts = self.temporal_task_counts[delay]
                    row_sums = delay_counts.sum(dim=1, keepdim=True)
                    row_sums[row_sums == 0] = 1  # 避免除零
                    delay_probs = delay_counts / row_sums
                    
                    # 应用阈值
                    delay_causal = (delay_probs > threshold).float() * delay_probs
                    
                    # 添加到因果矩阵列表
                    new_causal_matrices.append(delay_causal)
            else:
                # 如果没有时序数据，退回到基本方法
                transition_probs = self.task_transition_counts.clone()
                row_sums = transition_probs.sum(dim=1, keepdim=True)
                row_sums[row_sums == 0] = 1
                transition_probs = transition_probs / row_sums
                causal_matrix = (transition_probs > threshold).float() * transition_probs
                new_causal_matrices.append(causal_matrix)
                
        elif method == 'score':
            # 基于评分的方法（使用互信息作为评分）
            # 计算任务之间的互信息矩阵
            mi_matrix = torch.zeros(num_tasks, num_tasks).to(self.device)
            
            # 将计数转换为联合分布
            joint_counts = self.task_transition_counts.clone()
            joint_prob = joint_counts / (joint_counts.sum() + 1e-10)
            
            # 计算边缘分布
            row_marginal = joint_prob.sum(dim=1)
            col_marginal = joint_prob.sum(dim=0)
            
            # 计算互信息: MI(X,Y) = sum_x,y P(x,y) * log(P(x,y)/(P(x)P(y)))
            for i in range(num_tasks):
                for j in range(num_tasks):
                    if joint_prob[i, j] > 0:
                        mi_matrix[i, j] = joint_prob[i, j] * torch.log(
                            joint_prob[i, j] / (row_marginal[i] * col_marginal[j] + 1e-10)
                        )
            
            # 应用阈值
            causal_matrix = (mi_matrix > threshold).float() * mi_matrix
            new_causal_matrices.append(causal_matrix)
            
        else:  # 默认方法
            # 计算任务转换概率
            transition_probs = self.task_transition_counts.clone()
            row_sums = transition_probs.sum(dim=1, keepdim=True)
            row_sums[row_sums == 0] = 1  # 避免除零
            transition_probs = transition_probs / row_sums
            
            # 计算任务成功率
            success_rates = torch.zeros(num_tasks, device=self.device)
            for task in range(num_tasks):
                if task in self.task_outcome_counts:
                    success = self.task_outcome_counts[task]['success']
                    failure = self.task_outcome_counts[task]['failure']
                    total = success + failure
                    if total > 0:
                        success_rates[task] = success / total
            
            # 更新因果矩阵
            # 转换概率高且成功率高的任务对具有强因果关系
            causal_strength = transition_probs * success_rates.unsqueeze(0)
            
            # 应用sigmoid函数将值映射到合理范围
            causal_strength = torch.sigmoid(causal_strength * 5)  # 缩放因子5使分布更加明显
            
            new_causal_matrices.append(causal_strength)
        
        # 确保我们有足够的因果矩阵（对应多层次因果结构）
        if hasattr(self.model.causal_graph, 'causal_matrices'):
            while len(new_causal_matrices) < len(self.model.causal_graph.causal_matrices):
                # 复制最后一个矩阵并添加随机噪声以创建不同层次
                last_matrix = new_causal_matrices[-1].clone()
                noise = torch.randn_like(last_matrix) * 0.1
                noisy_matrix = torch.clamp(last_matrix + noise, 0, 1)
                new_causal_matrices.append(noisy_matrix)
            
            # 如果有多余的矩阵，只保留需要的数量
            new_causal_matrices = new_causal_matrices[:len(self.model.causal_graph.causal_matrices)]
            
            # 计算变化程度
            diff_sum = 0
            for i, (old_matrix, new_matrix) in enumerate(zip(self.model.causal_graph.causal_matrices, new_causal_matrices)):
                # 更新模型的因果矩阵
                old_data = old_matrix.data.clone()
                self.model.causal_graph.causal_matrices[i].data = new_matrix
                
                # 累加差异
                diff = torch.abs(old_data - new_matrix).sum().item()
                diff_sum += diff
            
            # 更新时序因果矩阵（如果有时序数据）
            if hasattr(self.model.causal_graph, 'temporal_causal_matrices') and hasattr(self, 'temporal_task_counts'):
                for i in range(min(len(self.model.causal_graph.temporal_causal_matrices), self.temporal_task_counts.shape[0])):
                    delay_counts = self.temporal_task_counts[i]
                    row_sums = delay_counts.sum(dim=1, keepdim=True)
                    row_sums[row_sums == 0] = 1
                    delay_probs = delay_counts / row_sums
                    
                    # 应用阈值
                    delay_causal = (delay_probs > threshold).float() * delay_probs
                    
                    # 更新模型的时序因果矩阵
                    self.model.causal_graph.temporal_causal_matrices[i].data = delay_causal
            
            # 计算稀疏度（非零元素比例）
            sparsity = sum([(m > 0.1).float().mean().item() for m in new_causal_matrices]) / len(new_causal_matrices)
            
            # 检查是否有显著变化
            total_elements = sum([m.numel() for m in new_causal_matrices])
            learned = diff_sum > 0.1 * total_elements
            
            # 收集指标
            metrics = {
                'diff_sum': diff_sum,
                'sparsity': sparsity,
                'learned': learned
            }
            
            return learned, metrics
        else:
            # 如果模型只有单一因果矩阵
            old_matrix = self.model.causal_graph.causal_matrix.data.clone()
            self.model.causal_graph.causal_matrix.data = new_causal_matrices[0]
            
            # 计算变化程度
            diff = torch.abs(old_matrix - new_causal_matrices[0]).sum().item()
            sparsity = (new_causal_matrices[0] > 0.1).float().mean().item()
            learned = diff > 0.1 * new_causal_matrices[0].numel()
            
            metrics = {
                'diff': diff,
                'sparsity': sparsity,
                'learned': learned
            }
            
            return learned, metrics
    
    def train_counterfactual(self, states, actions, rewards, returns_to_go, timesteps, task_labels, outcome_labels, batch_size=64):
        """
        训练反事实决策模块，支持多层次因果结构和时序因果效应
        
        Args:
            states: 状态序列
            actions: 动作序列
            rewards: 奖励序列
            returns_to_go: 未来回报序列
            timesteps: 时间步序列
            task_labels: 任务标签序列
            outcome_labels: 结果标签序列
            batch_size: 批次大小
        
        Returns:
            metrics: 训练指标字典
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # 获取状态特征
        state_features = self.model.transformer(inputs_embeds=self.model.embed_state(states))["last_hidden_state"]
        
        # 更新因果统计（包括状态特征）
        self.update_causal_statistics(task_labels, outcome_labels, state_features)
        
        # 学习因果结构
        learned, structure_metrics = self.learn_causal_structure()
        
        # 前向传播获取模型预测
        state_preds, action_preds, return_preds, task_preds = self.model(
            states, actions, rewards, returns_to_go, timesteps, ordering=None
        )
        
        # 计算任务预测损失
        task_loss = F.cross_entropy(task_preds.reshape(-1, self.model.num_tasks), task_labels.reshape(-1).long())
        
        # 反事实训练
        # 1. 对每个样本，随机选择多个任务进行干预
        batch_size = states.shape[0]
        num_interventions = min(3, self.model.num_tasks)  # 每个样本最多干预3个任务
        
        # 获取最后一个时间步的状态特征
        last_state_features = state_features[:, -1]
        
        # 2. 获取反事实预测
        cf_task_preds = []
        cf_uncertainties = []
        
        # 使用增强版的反事实决策模块
        if hasattr(self.model.counterfactual_decision_maker, 'adaptive_decision'):
            # 使用自适应决策
            refined_task_logits, decision_info = self.model.counterfactual_decision_maker.adaptive_decision(
                task_preds, last_state_features
            )
            task_preds_refined = refined_task_logits
            uncertainty = decision_info.get('uncertainty', torch.zeros(batch_size).to(self.device))
            explanations = decision_info.get('explanations', None)
        else:
            # 使用标准前向传播
            task_preds_refined = task_preds
            uncertainty = torch.zeros(batch_size).to(self.device)
        
        # 对每个样本，进行多次干预
        for intervention_idx in range(num_interventions):
            # 随机选择干预任务
            random_tasks = torch.randint(0, self.model.num_tasks, (batch_size,), device=self.device)
            
            # 进行反事实推理
            if hasattr(self.model.causal_graph, 'counterfactual_reasoning_with_uncertainty'):
                cf_logits, cf_uncertainty = self.model.causal_graph.counterfactual_reasoning_with_uncertainty(
                    task_preds_refined, last_state_features, random_tasks,
                    intervention_strength=0.8 + 0.2 * torch.rand(1).item()  # 随机干预强度
                )
                cf_uncertainties.append(cf_uncertainty)
            else:
                cf_logits = self.model.causal_graph.counterfactual_reasoning(
                    task_preds_refined, last_state_features, random_tasks
                )
            
            cf_task_preds.append(cf_logits)
        
        # 3. 计算反事实损失
        cf_losses = []
        diversity_losses = []
        intervention_losses = []
        
        for i, cf_logits in enumerate(cf_task_preds):
            cf_task_probs = F.softmax(cf_logits, dim=-1)
            task_label_onehot = F.one_hot(task_labels.reshape(-1).long(), self.model.num_tasks).float()
            
            # 干预后的预测应与原始标签不同（多样性损失）
            diversity_loss = -F.kl_div(cf_task_probs.log(), task_label_onehot, reduction='batchmean')
            diversity_losses.append(diversity_loss)
            
            # 如果有不确定性估计，根据不确定性加权损失
            if cf_uncertainties and len(cf_uncertainties) > i:
                # 不确定性高的样本权重较低
                certainty_weight = 1.0 - cf_uncertainties[i].detach()
                weighted_diversity_loss = diversity_loss * certainty_weight.mean()
                cf_losses.append(weighted_diversity_loss)
            else:
                cf_losses.append(diversity_loss)
        
        # 计算反事实多样性（不同干预应产生不同结果）
        cf_diversity = 0
        if len(cf_task_preds) > 1:
            for i in range(len(cf_task_preds) - 1):
                for j in range(i + 1, len(cf_task_preds)):
                    # 计算不同干预之间的差异
                    diversity = F.mse_loss(cf_task_preds[i], cf_task_preds[j])
                    cf_diversity += diversity
            
            # 归一化多样性
            cf_diversity = cf_diversity / (len(cf_task_preds) * (len(cf_task_preds) - 1) / 2)
            
            # 多样性越高越好，所以使用负值
            diversity_loss = -0.2 * cf_diversity
        else:
            diversity_loss = torch.tensor(0.0).to(self.device)
        
        # 计算稀疏性损失（鼓励因果图稀疏）
        sparsity_loss = 0
        if hasattr(self.model.causal_graph, 'causal_matrices'):
            for causal_matrix in self.model.causal_graph.causal_matrices:
                # L1正则化促进稀疏性
                sparsity_loss += self.sparsity_weight * torch.abs(causal_matrix).mean()
        else:
            # 单一因果矩阵
            sparsity_loss = self.sparsity_weight * torch.abs(self.model.causal_graph.causal_matrix).mean()
        
        # 计算一致性损失（鼓励不同层次的因果结构保持一定一致性）
        consistency_loss = 0
        if hasattr(self.model.causal_graph, 'causal_matrices') and len(self.model.causal_graph.causal_matrices) > 1:
            for i in range(len(self.model.causal_graph.causal_matrices) - 1):
                # 相邻层次的因果矩阵应该有一定相似性
                diff = F.mse_loss(self.model.causal_graph.causal_matrices[i], 
                                  self.model.causal_graph.causal_matrices[i+1])
                consistency_loss += self.consistency_weight * diff
        
        # 不确定性正则化（鼓励模型在适当情况下表达不确定性）
        if uncertainty.numel() > 0:
            uncertainty_loss = -0.1 * torch.mean(uncertainty * torch.log(uncertainty + 1e-10) + 
                                               (1 - uncertainty) * torch.log(1 - uncertainty + 1e-10))
        else:
            uncertainty_loss = torch.tensor(0.0).to(self.device)
        
        # 平均反事实损失
        cf_loss = torch.stack(cf_losses).mean() if cf_losses else torch.tensor(0.0).to(self.device)
        
        # 总损失
        total_loss = task_loss + 0.5 * cf_loss + diversity_loss + sparsity_loss + consistency_loss + uncertainty_loss
        
        # 反向传播
        total_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # 更新学习率调度器
        self.scheduler.step(total_loss)
        
        # 计算指标
        with torch.no_grad():
            task_preds_prob = F.softmax(task_preds, dim=-1)
            task_accuracy = (task_preds_prob.reshape(-1, self.model.num_tasks).argmax(dim=-1) == 
                            task_labels.reshape(-1)).float().mean()
        
        metrics = {
            'task_loss': task_loss.item(),
            'cf_loss': cf_loss.item() if isinstance(cf_loss, torch.Tensor) else cf_loss,
            'diversity_loss': diversity_loss.item() if isinstance(diversity_loss, torch.Tensor) else diversity_loss,
            'sparsity_loss': sparsity_loss.item() if isinstance(sparsity_loss, torch.Tensor) else sparsity_loss,
            'consistency_loss': consistency_loss.item() if isinstance(consistency_loss, torch.Tensor) else consistency_loss,
            'uncertainty_loss': uncertainty_loss.item() if isinstance(uncertainty_loss, torch.Tensor) else uncertainty_loss,
            'task_accuracy': task_accuracy.item(),
            'uncertainty': uncertainty.mean().item() if uncertainty.numel() > 0 else 0.0,
            'cf_diversity': cf_diversity.item() if isinstance(cf_diversity, torch.Tensor) else cf_diversity,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        # 添加因果结构学习指标
        metrics.update(structure_metrics)
        
        return metrics
    
    def evaluate_counterfactual(self, states, task_labels, outcome_labels=None):
        """
        评估反事实决策，支持多层次因果结构和不确定性估计
        
        Args:
            states: 状态序列
            task_labels: 任务标签序列
            outcome_labels: 结果标签序列（可选）
        
        Returns:
            metrics: 评估指标字典
        """
        self.model.eval()
        
        with torch.no_grad():
            # 获取状态特征
            state_features = self.model.transformer(inputs_embeds=self.model.embed_state(states))["last_hidden_state"]
            last_state_features = state_features[:, -1]
            
            # 获取任务预测
            task_logits = self.model.predict_task(last_state_features)
            task_probs = F.softmax(task_logits, dim=-1)
            
            # 使用增强版的反事实决策模块
            if hasattr(self.model.counterfactual_decision_maker, 'adaptive_decision'):
                # 使用自适应决策
                refined_task_logits, decision_info = self.model.counterfactual_decision_maker.adaptive_decision(
                    task_logits, last_state_features
                )
                task_logits_refined = refined_task_logits
                uncertainty = decision_info.get('uncertainty', torch.zeros(task_logits.shape[0]).to(self.device))
                explanations = decision_info.get('explanations', None)
                
                # 计算不确定性指标
                uncertainty_mean = uncertainty.mean().item()
                uncertainty_std = uncertainty.std().item()
            else:
                # 使用标准前向传播
                task_logits_refined = task_logits
                uncertainty_mean = 0.0
                uncertainty_std = 0.0
            
            # 反事实推理
            cf_task_probs_list = []
            cf_uncertainties_list = []
            
            for task_idx in range(self.model.num_tasks):
                # 对每个任务进行干预
                if hasattr(self.model.causal_graph, 'counterfactual_reasoning_with_uncertainty'):
                    cf_logits, cf_uncertainty = self.model.causal_graph.counterfactual_reasoning_with_uncertainty(
                        task_logits_refined, last_state_features, 
                        torch.tensor([task_idx] * states.shape[0], device=self.device)
                    )
                    cf_uncertainties_list.append(cf_uncertainty)
                else:
                    cf_logits = self.model.causal_graph.counterfactual_reasoning(
                        task_logits_refined, last_state_features, task_idx
                    )
                
                cf_probs = F.softmax(cf_logits, dim=-1)
                cf_task_probs_list.append(cf_probs)
            
            # 计算反事实多样性（不同干预应产生不同结果）
            cf_task_probs_stack = torch.stack(cf_task_probs_list, dim=1)  # [batch_size, num_tasks, num_tasks]
            cf_diversity = torch.std(cf_task_probs_stack, dim=1).mean().item()
            
            # 计算任务预测准确率
            task_accuracy = (task_probs.argmax(dim=-1) == task_labels).float().mean().item()
            
            # 计算反事实一致性（相似状态下的反事实效应应该相似）
            cf_consistency = 0.0
            if states.shape[0] > 1:
                # 计算状态特征的相似度矩阵
                state_sim = torch.matmul(last_state_features, last_state_features.transpose(0, 1))
                state_sim = state_sim / torch.norm(last_state_features, dim=1, keepdim=True)
                state_sim = state_sim / torch.norm(last_state_features, dim=1, keepdim=True).transpose(0, 1)
                
                # 计算反事实效应的相似度矩阵
                cf_effects = cf_task_probs_stack - task_probs.unsqueeze(1).expand_as(cf_task_probs_stack)
                cf_effects_flat = cf_effects.reshape(cf_effects.shape[0], -1)
                cf_sim = torch.matmul(cf_effects_flat, cf_effects_flat.transpose(0, 1))
                cf_sim = cf_sim / (torch.norm(cf_effects_flat, dim=1, keepdim=True) + 1e-8)
                cf_sim = cf_sim / (torch.norm(cf_effects_flat, dim=1, keepdim=True).transpose(0, 1) + 1e-8)
                
                # 计算状态相似度和反事实效应相似度的相关性
                # 相关性越高，说明反事实一致性越好
                mask = ~torch.eye(states.shape[0], dtype=torch.bool, device=self.device)  # 排除对角线
                if mask.any():
                    cf_consistency = torch.corrcoef(
                        torch.stack([state_sim[mask], cf_sim[mask]]))[0, 1].item()
            
            # 计算反事实效应强度（干预前后的变化程度）
            cf_effect_magnitude = torch.mean(torch.abs(cf_task_probs_stack - task_probs.unsqueeze(1))).item()
            
            # 如果提供了结果标签，计算因果效应准确性
            causal_effect_accuracy = 0.0
            if outcome_labels is not None and hasattr(self.model.causal_graph, 'causal_matrices'):
                # 获取预测的因果效应
                pred_effects = []
                for i in range(self.model.num_tasks):
                    # 计算干预i对结果的影响
                    effect = cf_task_probs_list[i].argmax(dim=-1)
                    pred_effects.append(effect)
                
                pred_effects = torch.stack(pred_effects, dim=1)  # [batch_size, num_tasks]
                
                # 计算真实的因果效应（如果可用）
                if outcome_labels.dim() > 1 and outcome_labels.shape[1] == self.model.num_tasks:
                    # 假设outcome_labels包含每个干预的真实结果
                    true_effects = outcome_labels
                    causal_effect_accuracy = (pred_effects == true_effects).float().mean().item()
            
            # 计算不确定性校准（不确定性与预测错误的相关性）
            uncertainty_calibration = 0.0
            if hasattr(self.model.counterfactual_decision_maker, 'adaptive_decision'):
                # 预测错误
                pred_errors = (task_probs.argmax(dim=-1) != task_labels).float()
                if pred_errors.sum() > 0 and pred_errors.sum() < pred_errors.numel():
                    # 计算不确定性与预测错误的相关性
                    uncertainty_calibration = torch.corrcoef(
                        torch.stack([uncertainty, pred_errors]))[0, 1].item()
            
            # 可视化因果图（如果可用）
            if hasattr(self, 'visualize_causal_graph') and hasattr(self, 'training_step') and self.training_step % 100 == 0:
                self.visualize_causal_graph()
            
            metrics = {
                'task_accuracy': task_accuracy,
                'cf_diversity': cf_diversity,
                'cf_consistency': cf_consistency,
                'cf_effect_magnitude': cf_effect_magnitude,
                'uncertainty_mean': uncertainty_mean,
                'uncertainty_std': uncertainty_std,
                'uncertainty_calibration': uncertainty_calibration,
                'causal_effect_accuracy': causal_effect_accuracy
            }
            
            return metrics
    
    def visualize_causal_graph(self, save_dir='./causal_graphs'):
        """
        可视化因果图结构
        
        Args:
            save_dir: 保存图像的目录
        """
        if not hasattr(self.model, 'causal_graph'):
            return
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 获取当前时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建自定义颜色映射（从蓝色到红色）
        colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]  # 蓝色 -> 白色 -> 红色
        cmap = LinearSegmentedColormap.from_list('causal_cmap', colors, N=100)
        
        # 可视化多层次因果矩阵（如果存在）
        if hasattr(self.model.causal_graph, 'causal_matrices'):
            for i, causal_matrix in enumerate(self.model.causal_graph.causal_matrices):
                plt.figure(figsize=(10, 8))
                
                # 获取因果矩阵数据
                matrix_data = causal_matrix.detach().cpu().numpy()
                
                # 绘制热力图
                im = plt.imshow(matrix_data, cmap=cmap, vmin=-1, vmax=1)
                plt.colorbar(im, label='因果强度')
                
                # 设置标题和标签
                plt.title(f'层次 {i+1} 因果关系矩阵')
                plt.xlabel('目标任务')
                plt.ylabel('源任务')
                
                # 添加网格线
                plt.grid(False)
                
                # 保存图像
                plt.savefig(os.path.join(save_dir, f'causal_matrix_level{i+1}_{timestamp}.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
                # 创建NetworkX图
                G = nx.DiGraph()
                
                # 添加节点
                for j in range(matrix_data.shape[0]):
                    G.add_node(j, label=f'Task {j}')
                
                # 添加边（只添加强度大于阈值的边）
                threshold = 0.2
                for src in range(matrix_data.shape[0]):
                    for dst in range(matrix_data.shape[1]):
                        weight = matrix_data[src, dst]
                        if abs(weight) > threshold:
                            G.add_edge(src, dst, weight=weight)
                
                # 绘制网络图
                plt.figure(figsize=(12, 10))
                pos = nx.spring_layout(G, seed=42)  # 固定布局种子以保持一致性
                
                # 绘制节点
                nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue')
                
                # 绘制边（正向关系为红色，负向关系为蓝色）
                edges_pos = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 0]
                edges_neg = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] < 0]
                
                # 计算边的宽度（基于权重的绝对值）
                edge_widths_pos = [2 * G[u][v]['weight'] for (u, v) in edges_pos]
                edge_widths_neg = [2 * abs(G[u][v]['weight']) for (u, v) in edges_neg]
                
                nx.draw_networkx_edges(G, pos, edgelist=edges_pos, width=edge_widths_pos, edge_color='red', arrows=True)
                nx.draw_networkx_edges(G, pos, edgelist=edges_neg, width=edge_widths_neg, edge_color='blue', arrows=True)
                
                # 添加标签
                nx.draw_networkx_labels(G, pos)
                
                plt.title(f'层次 {i+1} 因果网络图')
                plt.axis('off')
                
                # 保存网络图
                plt.savefig(os.path.join(save_dir, f'causal_network_level{i+1}_{timestamp}.png'), dpi=300, bbox_inches='tight')
                plt.close()
        else:
            # 可视化单一因果矩阵
            plt.figure(figsize=(10, 8))
            
            # 获取因果矩阵数据
            matrix_data = self.model.causal_graph.causal_matrix.detach().cpu().numpy()
            
            # 绘制热力图
            im = plt.imshow(matrix_data, cmap=cmap, vmin=-1, vmax=1)
            plt.colorbar(im, label='因果强度')
            
            # 设置标题和标签
            plt.title('因果关系矩阵')
            plt.xlabel('目标任务')
            plt.ylabel('源任务')
            
            # 添加网格线
            plt.grid(False)
            
            # 保存图像
            plt.savefig(os.path.join(save_dir, f'causal_matrix_{timestamp}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 创建NetworkX图
            G = nx.DiGraph()
            
            # 添加节点
            for j in range(matrix_data.shape[0]):
                G.add_node(j, label=f'Task {j}')
            
            # 添加边（只添加强度大于阈值的边）
            threshold = 0.2
            for src in range(matrix_data.shape[0]):
                for dst in range(matrix_data.shape[1]):
                    weight = matrix_data[src, dst]
                    if abs(weight) > threshold:
                        G.add_edge(src, dst, weight=weight)
            
            # 绘制网络图
            plt.figure(figsize=(12, 10))
            pos = nx.spring_layout(G, seed=42)  # 固定布局种子以保持一致性
            
            # 绘制节点
            nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue')
            
            # 绘制边（正向关系为红色，负向关系为蓝色）
            edges_pos = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 0]
            edges_neg = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] < 0]
            
            # 计算边的宽度（基于权重的绝对值）
            edge_widths_pos = [2 * G[u][v]['weight'] for (u, v) in edges_pos]
            edge_widths_neg = [2 * abs(G[u][v]['weight']) for (u, v) in edges_neg]
            
            nx.draw_networkx_edges(G, pos, edgelist=edges_pos, width=edge_widths_pos, edge_color='red', arrows=True)
            nx.draw_networkx_edges(G, pos, edgelist=edges_neg, width=edge_widths_neg, edge_color='blue', arrows=True)
            
            # 添加标签
            nx.draw_networkx_labels(G, pos)
            
            plt.title('因果网络图')
            plt.axis('off')
            
            # 保存网络图
            plt.savefig(os.path.join(save_dir, f'causal_network_{timestamp}.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 如果有时序因果矩阵，也进行可视化
        if hasattr(self.model.causal_graph, 'temporal_causal_matrix'):
            plt.figure(figsize=(10, 8))
            
            # 获取时序因果矩阵数据
            matrix_data = self.model.causal_graph.temporal_causal_matrix.detach().cpu().numpy()
            
            # 绘制热力图
            im = plt.imshow(matrix_data, cmap=cmap, vmin=-1, vmax=1)
            plt.colorbar(im, label='时序因果强度')
            
            # 设置标题和标签
            plt.title('时序因果关系矩阵')
            plt.xlabel('目标任务')
            plt.ylabel('源任务 (t-1)')
            
            # 添加网格线
            plt.grid(False)
            
            # 保存图像
            plt.savefig(os.path.join(save_dir, f'temporal_causal_matrix_{timestamp}.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def adaptive_training(self, states, actions, rewards, returns_to_go, timesteps, task_labels, outcome_labels, batch_size=64):
        """
        自适应训练方法，根据当前模型状态动态调整训练策略
        
        Args:
            states: 状态序列
            actions: 动作序列
            rewards: 奖励序列
            returns_to_go: 未来回报序列
            timesteps: 时间步序列
            task_labels: 任务标签序列
            outcome_labels: 结果标签序列
            batch_size: 批次大小
        
        Returns:
            metrics: 训练指标字典
        """
        # 检查训练状态，决定训练策略
        if not hasattr(self, 'training_phase'):
            self.training_phase = 'initial'  # 初始阶段
            self.phase_steps = 0
            self.best_metrics = {}
            self.patience = 0
            self.max_patience = 5
        
        # 更新阶段步数
        self.phase_steps += 1
        
        # 根据不同训练阶段采用不同策略
        if self.training_phase == 'initial':
            # 初始阶段：专注于学习基本因果结构
            # 增大稀疏性权重，鼓励发现主要因果关系
            original_sparsity_weight = self.sparsity_weight
            self.sparsity_weight *= 1.5
            
            # 训练
            metrics = self.train_counterfactual(states, actions, rewards, returns_to_go, timesteps, task_labels, outcome_labels, batch_size)
            
            # 恢复原始权重
            self.sparsity_weight = original_sparsity_weight
            
            # 检查是否应该进入下一阶段
            if self.phase_steps > 100 and metrics['task_accuracy'] > 0.7:
                self.training_phase = 'refinement'
                self.phase_steps = 0
                print("训练进入精细化阶段")
        
        elif self.training_phase == 'refinement':
            # 精细化阶段：平衡因果发现和反事实训练
            # 增大一致性权重，鼓励不同层次因果结构的一致性
            original_consistency_weight = self.consistency_weight
            self.consistency_weight *= 1.2
            
            # 训练
            metrics = self.train_counterfactual(states, actions, rewards, returns_to_go, timesteps, task_labels, outcome_labels, batch_size)
            
            # 恢复原始权重
            self.consistency_weight = original_consistency_weight
            
            # 检查是否应该进入下一阶段
            if self.phase_steps > 200 or (metrics['task_accuracy'] > 0.85 and metrics['cf_diversity'] > 0.3):
                self.training_phase = 'exploitation'
                self.phase_steps = 0
                print("训练进入利用阶段")
        
        else:  # exploitation阶段
            # 利用阶段：专注于反事实推理和决策
            # 使用标准训练
            metrics = self.train_counterfactual(states, actions, rewards, returns_to_go, timesteps, task_labels, outcome_labels, batch_size)
            
            # 检查是否需要重新进入精细化阶段（如果性能下降）
            if 'task_accuracy' in self.best_metrics and metrics['task_accuracy'] < 0.9 * self.best_metrics['task_accuracy']:
                self.patience += 1
                if self.patience >= self.max_patience:
                    self.training_phase = 'refinement'
                    self.phase_steps = 0
                    self.patience = 0
                    print("性能下降，重新进入精细化阶段")
            else:
                self.patience = 0
        
        # 更新最佳指标
        for key, value in metrics.items():
            if key not in self.best_metrics or value > self.best_metrics[key]:
                self.best_metrics[key] = value
        
        # 添加当前训练阶段到指标
        metrics['training_phase'] = self.training_phase
        metrics['phase_steps'] = self.phase_steps
        
        return metrics