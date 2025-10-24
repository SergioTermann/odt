import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CausalGraph(nn.Module):
    """
    因果图网络模块，用于建模任务之间的因果关系
    增强版：支持多层次因果结构和时序因果效应
    """
    def __init__(self, num_tasks, hidden_size, num_levels=3, max_delay=5):
        super().__init__()
        self.num_tasks = num_tasks
        self.hidden_size = hidden_size
        self.num_levels = num_levels
        self.max_delay = max_delay
        
        # 多层次因果矩阵，每层表示不同时间尺度或抽象层次的因果关系
        self.causal_matrices = nn.ParameterList([
            nn.Parameter(torch.zeros(num_tasks, num_tasks))
            for _ in range(num_levels)
        ])
        
        # 层次权重，决定各层的重要性
        self.level_weights = nn.Parameter(torch.ones(num_levels) / num_levels)
        
        # 时序因果矩阵，表示不同延迟时间的因果效应
        self.temporal_causal_matrices = nn.ParameterList([
            nn.Parameter(torch.zeros(num_tasks, num_tasks))
            for _ in range(max_delay)
        ])
        
        # 任务表示转换
        self.task_encoder = nn.Linear(num_tasks, hidden_size)
        self.task_decoder = nn.Linear(hidden_size, num_tasks)
        
        # 因果推理网络（增强版）
        self.causal_inference = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        
        # 层次整合网络
        self.level_integration = nn.Sequential(
            nn.Linear(num_tasks * num_levels, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_tasks)
        )
        
        # 时序整合网络
        self.temporal_integration = nn.LSTM(
            input_size=num_tasks,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True
        )
        
        # 注意力机制，用于整合不同层次和时序的因果效应
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
    
    def forward(self, task_logits, state_features, task_history=None):
        """
        前向传播，根据当前状态和任务逻辑，进行多层次和时序因果推理
        
        Args:
            task_logits: 任务选择的logits，形状为 [batch_size, num_tasks] 或 [batch_size, seq_len, num_tasks]
            state_features: 状态特征，形状为 [batch_size, hidden_size] 或 [batch_size, seq_len, hidden_size]
            task_history: 历史任务选择，形状为 [batch_size, seq_len, num_tasks]，可选
            
        Returns:
            refined_task_logits: 经过因果推理后的任务logits
        """
        # 确保输入是2D张量 [batch_size, feature_dim]
        original_shape = task_logits.shape
        if len(original_shape) > 2:
            # 如果是3D或4D张量，取最后一个时间步或压缩多余维度
            task_logits = task_logits.reshape(-1, original_shape[-1])
            state_features = state_features.reshape(-1, state_features.shape[-1])
        
        batch_size = task_logits.shape[0]
        
        # 计算任务概率分布
        task_probs = torch.softmax(task_logits, dim=-1)
        
        # 编码任务表示
        task_encoding = self.task_encoder(task_probs)
        
        # 因果推理
        combined_features = torch.cat([task_encoding, state_features], dim=-1)
        causal_features = self.causal_inference(combined_features)
        
        # 多层次因果影响
        multi_level_influences = []
        for level in range(self.num_levels):
            # 应用当前层次的因果矩阵
            level_influence = torch.matmul(task_probs, self.causal_matrices[level])  # [batch_size, num_tasks]
            level_influence = torch.softmax(level_influence, dim=-1)
            multi_level_influences.append(level_influence)
        
        # 整合多层次因果影响
        multi_level_tensor = torch.cat(multi_level_influences, dim=-1)  # [batch_size, num_tasks * num_levels]
        integrated_level_influence = self.level_integration(multi_level_tensor)  # [batch_size, num_tasks]
        integrated_level_influence = torch.softmax(integrated_level_influence, dim=-1)
        
        # 时序因果效应（如果有历史数据）
        temporal_influence = torch.zeros(batch_size, self.num_tasks, device=task_logits.device)
        if task_history is not None and task_history.size(1) > 1:
            # 计算每个时间步的因果效应
            temporal_effects = []
            for delay in range(1, min(task_history.size(1), self.max_delay) + 1):
                # 获取delay步之前的任务选择
                past_tasks = task_history[:, -delay, :]
                
                # 应用对应延迟的因果矩阵
                effect = torch.matmul(past_tasks, self.temporal_causal_matrices[delay-1])
                temporal_effects.append(effect)
            
            if temporal_effects:
                # 整合不同时间步的因果效应
                effects_tensor = torch.stack(temporal_effects, dim=1)  # [batch_size, delays, num_tasks]
                
                # 使用LSTM整合时序信息
                effects_features, _ = self.temporal_integration(effects_tensor)
                temporal_influence = torch.softmax(effects_features[:, -1, :], dim=-1)  # 使用最后一个时间步的输出
        
        # 使用注意力机制整合不同来源的因果影响
        task_encoding_expanded = task_encoding.unsqueeze(1)  # [batch_size, 1, hidden_size]
        causal_features_expanded = causal_features.unsqueeze(1)  # [batch_size, 1, hidden_size]
        
        # 创建查询、键和值
        query = task_encoding_expanded  # [batch_size, 1, hidden_size]
        key_value = torch.cat([task_encoding_expanded, causal_features_expanded], dim=1)  # [batch_size, 2, hidden_size]
        
        # 确保query是2D或3D张量
        assert len(query.shape) == 3, f"query should be unbatched 2D or batched 3D tensor but received {len(query.shape)}-D query tensor with shape {query.shape}"
        
        # 应用注意力
        attn_output, _ = self.attention(query, key_value, key_value)
        attn_output = attn_output.squeeze(1)  # [batch_size, hidden_size]
        
        # 解码回任务空间
        refined_task_logits = self.task_decoder(attn_output)
        
        # 结合多层次和时序因果影响
        if task_history is not None and task_history.size(1) > 1:
            # 如果有历史数据，结合时序影响
            combined_influence = 0.7 * integrated_level_influence + 0.3 * temporal_influence
        else:
            # 否则只使用多层次影响
            combined_influence = integrated_level_influence
        
        # 应用因果影响
        refined_task_logits = refined_task_logits * combined_influence
        
        # 如果原始输入是高维张量，恢复原始形状
        if len(original_shape) > 2:
            # 保持最后一个维度（num_tasks），其他维度与原始形状一致
            new_shape = list(original_shape[:-1]) + [refined_task_logits.shape[-1]]
            refined_task_logits = refined_task_logits.reshape(new_shape)
        
        return refined_task_logits
    
    def counterfactual_reasoning(self, task_logits, state_features, intervention_task_idx, task_history=None, intervention_strength=1.0):
        """
        增强版反事实推理：如果选择了特定任务，会对其他任务产生什么影响
        支持多层次因果结构、时序因果效应和干预强度调节
        
        Args:
            task_logits: 任务选择的logits，形状为 [batch_size, num_tasks]
            state_features: 状态特征，形状为 [batch_size, hidden_size]
            intervention_task_idx: 干预的任务索引
            task_history: 历史任务选择，形状为 [batch_size, seq_len, num_tasks]，可选
            intervention_strength: 干预强度，范围[0,1]，1.0表示完全干预，0.0表示无干预
            
        Returns:
            counterfactual_logits: 反事实推理后的任务logits
            uncertainty: 反事实推理的不确定性估计
        """
        batch_size = task_logits.shape[0]
        
        # 创建干预后的任务分布（软干预）
        intervention = torch.zeros_like(task_logits)
        intervention[:, intervention_task_idx] = 1.0
        
        # 应用干预强度（软干预）
        task_probs = torch.softmax(task_logits, dim=-1)
        soft_intervention = intervention_strength * intervention + (1 - intervention_strength) * task_probs
        
        # 编码干预后的任务表示
        intervention_encoding = self.task_encoder(soft_intervention)
        
        # 因果推理
        combined_features = torch.cat([intervention_encoding, state_features], dim=-1)
        counterfactual_features = self.causal_inference(combined_features)
        
        # 多层次因果影响
        multi_level_influences = []
        level_uncertainties = []
        
        for level in range(self.num_levels):
            # 应用当前层次的因果矩阵
            level_influence = torch.matmul(soft_intervention, self.causal_matrices[level])  # [batch_size, num_tasks]
            level_influence = torch.softmax(level_influence, dim=-1)
            multi_level_influences.append(level_influence)
            
            # 计算每个层次的不确定性（熵）
            entropy = -torch.sum(level_influence * torch.log(level_influence + 1e-10), dim=-1)  # [batch_size]
            level_uncertainties.append(entropy)
        
        # 整合多层次因果影响
        multi_level_tensor = torch.cat(multi_level_influences, dim=-1)  # [batch_size, num_tasks * num_levels]
        integrated_level_influence = self.level_integration(multi_level_tensor)  # [batch_size, num_tasks]
        integrated_level_influence = torch.softmax(integrated_level_influence, dim=-1)
        
        # 时序因果效应（如果有历史数据）
        temporal_influence = torch.zeros(batch_size, self.num_tasks, device=task_logits.device)
        if task_history is not None and task_history.size(1) > 1:
            # 计算每个时间步的因果效应
            temporal_effects = []
            for delay in range(1, min(task_history.size(1), self.max_delay) + 1):
                # 获取delay步之前的任务选择
                past_tasks = task_history[:, -delay, :]
                
                # 应用对应延迟的因果矩阵
                effect = torch.matmul(past_tasks, self.temporal_causal_matrices[delay-1])
                temporal_effects.append(effect)
            
            if temporal_effects:
                # 整合不同时间步的因果效应
                effects_tensor = torch.stack(temporal_effects, dim=1)  # [batch_size, delays, num_tasks]
                
                # 使用LSTM整合时序信息
                effects_features, _ = self.temporal_integration(effects_tensor)
                temporal_influence = torch.softmax(effects_features[:, -1, :], dim=-1)  # 使用最后一个时间步的输出
        
        # 使用注意力机制整合不同来源的因果影响
        intervention_encoding_expanded = intervention_encoding.unsqueeze(1)  # [batch_size, 1, hidden_size]
        counterfactual_features_expanded = counterfactual_features.unsqueeze(1)  # [batch_size, 1, hidden_size]
        
        # 创建查询、键和值
        query = intervention_encoding_expanded  # [batch_size, 1, hidden_size]
        key_value = torch.cat([intervention_encoding_expanded, counterfactual_features_expanded], dim=1)  # [batch_size, 2, hidden_size]
        
        # 确保query是2D或3D张量
        assert len(query.shape) == 3, f"query should be unbatched 2D or batched 3D tensor but received {len(query.shape)}-D query tensor with shape {query.shape}"
        
        # 应用注意力
        attn_output, attn_weights = self.attention(query, key_value, key_value)
        attn_output = attn_output.squeeze(1)  # [batch_size, hidden_size]
        
        # 解码回任务空间
        counterfactual_logits = self.task_decoder(attn_output)
        
        # 结合多层次和时序因果影响
        if task_history is not None and task_history.size(1) > 1:
            # 如果有历史数据，结合时序影响
            combined_influence = 0.7 * integrated_level_influence + 0.3 * temporal_influence
        else:
            # 否则只使用多层次影响
            combined_influence = integrated_level_influence
        
        # 应用因果影响
        counterfactual_logits = counterfactual_logits * combined_influence
        
        # 计算总体不确定性
        level_uncertainty = torch.stack(level_uncertainties, dim=1)  # [batch_size, num_levels]
        weighted_uncertainty = torch.sum(level_uncertainty * F.softmax(self.level_weights, dim=0), dim=1)  # [batch_size]
        
        return counterfactual_logits, weighted_uncertainty


class CounterfactualDecisionMaker(nn.Module):
    """
    增强版反事实决策模块，用于评估不同任务选择的潜在结果
    支持不确定性评估、多样性探索和决策解释
    """
    def __init__(self, num_tasks, hidden_size, uncertainty_threshold=0.5, diversity_weight=0.3):
        super().__init__()
        self.num_tasks = num_tasks
        self.hidden_size = hidden_size
        self.uncertainty_threshold = uncertainty_threshold
        self.diversity_weight = diversity_weight
        
        # 因果图网络
        self.causal_graph = CausalGraph(num_tasks, hidden_size)
        
        # 状态特征投影层
        self.state_projection = nn.Linear(hidden_size, num_tasks)
        
        # 反事实评估网络（增强版）
        self.counterfactual_evaluator = nn.Sequential(
            nn.Linear(num_tasks * 3, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_tasks)
        )
        
        # 不确定性估计网络
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # 决策解释生成器
        self.explanation_generator = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_tasks * 2)  # 每个任务的正负影响因素
        )
        
        # 多样性探索参数
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
        self.exploration_factor = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(self, task_logits, state_features, task_history=None, exploration_mode=False):
        """
        前向传播，进行反事实决策
        
        Args:
            task_logits: 任务选择的logits，形状为 [batch_size, num_tasks] 或 [batch_size, seq_len, num_tasks]
            state_features: 状态特征，形状为 [batch_size, hidden_size] 或 [batch_size, seq_len, hidden_size]
            task_history: 历史任务选择，形状为 [batch_size, seq_len, num_tasks]，可选
            exploration_mode: 是否启用探索模式，增加决策多样性
            
        Returns:
            refined_task_logits: 经过反事实决策后的任务logits
            uncertainty: 决策的不确定性，形状为 [batch_size]
            explanations: 决策解释，形状为 [batch_size, num_tasks, 2]
        """
        # 确保输入是2D张量 [batch_size, feature_dim]
        original_shape = task_logits.shape
        if len(original_shape) > 2:
            # 如果是3D或4D张量，取最后一个时间步或压缩多余维度
            task_logits = task_logits.reshape(-1, original_shape[-1])
            state_features = state_features.reshape(-1, state_features.shape[-1])
            
        batch_size = task_logits.shape[0]
        
        # 获取因果推理结果
        causal_task_logits = self.causal_graph(task_logits, state_features, task_history)
        
        # 计算任务概率分布（可选择性地应用温度参数）
        if exploration_mode:
            # 探索模式：使用更高的温度增加随机性
            effective_temperature = self.temperature * (1.0 + self.exploration_factor)
            task_probs = torch.softmax(causal_task_logits / effective_temperature, dim=-1)
        else:
            # 利用模式：使用标准温度
            task_probs = torch.softmax(causal_task_logits / self.temperature, dim=-1)
        
        # 对每个可能的任务进行反事实推理
        counterfactual_results = []
        all_uncertainties = []
        all_explanations = []
        
        for task_idx in range(self.num_tasks):
            # 进行反事实推理（获取不确定性）
            cf_logits, uncertainty = self.causal_graph.counterfactual_reasoning(
                task_logits, state_features, task_idx, task_history,
                intervention_strength=0.9  # 使用软干预
            )
            counterfactual_results.append(cf_logits)
            all_uncertainties.append(uncertainty)
            
            # 生成决策解释
            task_encoding = self.causal_graph.task_encoder(torch.softmax(cf_logits, dim=-1))
            explanation_features = torch.cat([task_encoding, state_features, task_encoding * state_features], dim=-1)
            explanation_logits = self.explanation_generator(explanation_features)  # [batch_size, num_tasks*2]
            explanation = explanation_logits.view(batch_size, self.num_tasks, 2)  # [batch_size, num_tasks, 2]
            all_explanations.append(explanation)
        
        # 将所有反事实结果拼接
        all_cf_logits = torch.stack(counterfactual_results, dim=1)  # [batch_size, num_tasks, num_tasks]
        all_cf_probs = torch.softmax(all_cf_logits, dim=-1)
        all_uncertainties = torch.stack(all_uncertainties, dim=1)  # [batch_size, num_tasks]
        all_explanations = torch.stack(all_explanations, dim=1)  # [batch_size, num_tasks, num_tasks, 2]
        
        # 计算每个任务的反事实价值（平均概率）
        cf_values = all_cf_probs.mean(dim=2)  # [batch_size, num_tasks]
        
        # 结合原始任务logits和反事实评估
        # 将state_features投影到与任务维度相同的空间
        projected_state = self.state_projection(state_features)
        
        combined_features = torch.cat([
            projected_state, 
            torch.softmax(task_logits, dim=-1),
            cf_values
        ], dim=-1)
        
        # 最终决策
        refined_task_logits = self.counterfactual_evaluator(combined_features)
        
        # 计算总体不确定性
        uncertainty = (all_uncertainties * task_probs).sum(dim=1)  # [batch_size]
        
        # 选择最可能任务的解释作为整体解释
        best_task_indices = torch.argmax(task_probs, dim=1)  # [batch_size]
        explanations = torch.zeros(batch_size, self.num_tasks, 2, device=task_logits.device)
        for b in range(batch_size):
            best_idx = best_task_indices[b]
            explanations[b] = all_explanations[b, best_idx]
        
        # 如果原始输入是高维张量，恢复原始形状
        if len(original_shape) > 2:
            # 保持最后一个维度（num_tasks），其他维度与原始形状一致
            new_shape = list(original_shape[:-1]) + [refined_task_logits.shape[-1]]
            refined_task_logits = refined_task_logits.reshape(new_shape)
            
            # 调整uncertainty和explanations的形状
            uncertainty = uncertainty.reshape(original_shape[0], -1)
            explanations = explanations.reshape(original_shape[0], -1, self.num_tasks, 2)
        
        return refined_task_logits, uncertainty, explanations
    
    def adaptive_decision(self, task_logits, state_features, task_history=None):
        """
        自适应决策：根据不确定性调整决策策略
        
        Args:
            task_logits: 任务选择的logits，形状为 [batch_size, num_tasks]
            state_features: 状态特征，形状为 [batch_size, hidden_size]
            task_history: 历史任务选择，形状为 [batch_size, seq_len, num_tasks]，可选
            
        Returns:
            refined_task_logits: 经过反事实决策后的任务logits
            decision_info: 决策相关信息字典
        """
        # 首先进行标准前向传播
        refined_task_logits, uncertainty, explanations = self.forward(
            task_logits, state_features, task_history, exploration_mode=False
        )
        
        # 根据不确定性决定是否需要探索
        batch_size = task_logits.shape[0]
        exploration_mask = (uncertainty > self.uncertainty_threshold).float().unsqueeze(-1)  # [batch_size, 1]
        
        # 如果不确定性高，进行探索模式的前向传播
        if exploration_mask.sum() > 0:
            explore_task_logits, _, _ = self.forward(
                task_logits, state_features, task_history, exploration_mode=True
            )
            
            # 根据不确定性混合标准决策和探索决策
            refined_task_logits = (1 - exploration_mask) * refined_task_logits + exploration_mask * explore_task_logits
        
        # 计算决策多样性（熵）
        task_probs = torch.softmax(refined_task_logits, dim=-1)
        entropy = -torch.sum(task_probs * torch.log(task_probs + 1e-10), dim=-1)  # [batch_size]
        
        # 收集决策相关信息
        decision_info = {
            'uncertainty': uncertainty,
            'entropy': entropy,
            'exploration_rate': exploration_mask.mean().item(),
            'explanations': explanations
        }
        
        return refined_task_logits, decision_info