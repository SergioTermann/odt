import torch
import torch.nn as nn
import torch.optim as optim
import decision_transformer


class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha

        # 低秩分解矩阵
        self.A = nn.Parameter(torch.randn(in_features, rank))
        self.B = nn.Parameter(torch.zeros(rank, out_features))

        # 原始权重矩阵
        self.weight = nn.Parameter(torch.zeros(in_features, out_features))

    def forward(self, x):
        # LoRA更新公式: W_new = W + (BA * alpha/rank)
        lora_update = torch.matmul(torch.matmul(x, self.A), self.B) * (self.alpha / self.rank)
        return torch.matmul(x, self.weight) + lora_update


class DecisionTransformerLoRA(nn.Module):
    def __init__(self, base_model, config):
        super().__init__()
        self.base_model = base_model
        self.config = config

        # 在记忆模块应用LoRA
        self._inject_lora_to_memory_module()

    def _inject_lora_to_memory_module(self):
        # 为查询、键、值矩阵注入LoRA层
        memory_module = self.base_model.memory_module

        memory_module.query_lora = LoRALayer(
            in_features=memory_module.query.in_features,
            out_features=memory_module.query.out_features,
            rank=self.config.lora_rank,
            alpha=self.config.lora_alpha
        )

        memory_module.key_lora = LoRALayer(
            in_features=memory_module.key.in_features,
            out_features=memory_module.key.out_features,
            rank=self.config.lora_rank,
            alpha=self.config.lora_alpha
        )

        memory_module.value_lora = LoRALayer(
            in_features=memory_module.value.in_features,
            out_features=memory_module.value.out_features,
            rank=self.config.lora_rank,
            alpha=self.config.lora_alpha
        )

    def forward(self, x, return_to_go, timestep):
        # 冻结主模型参数
        for param in self.base_model.parameters():
            param.requires_grad = False

        # 仅更新LoRA参数
        for param in self.parameters():
            if 'lora' in param.name:
                param.requires_grad = True

        return self.base_model(x, return_to_go, timestep)

    def fine_tune(self, dataset, config):
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=config.learning_rate, weight_decay=config.weight_decay)

        for epoch in range(config.epochs):
            for batch in dataset:
                optimizer.zero_grad()

                # 前向传播
                loss = self.compute_loss(batch)

                # 反向传播
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.parameters()), max_norm=1.0)

                optimizer.step()

    def compute_loss(self, batch):
        predicted_actions = self(batch.states, batch.return_to_go, batch.timestep)
        action_loss = nn.MSELoss()(predicted_actions, batch.actions)
        reward_loss = nn.MSELoss()(self.predicted_rewards, batch.rewards)
        return_loss = nn.MSELoss()(self.predicted_returns, batch.returns)
        total_loss = action_loss + self.config.reward_loss_weight * reward_loss + self.config.return_loss_weight * return_loss

        return total_loss


# 配置类
class LoRAConfig:
    def __init__(self):
        self.lora_rank = 4  # 论文推荐的秩大小
        self.lora_alpha = 16
        self.lora_dropout = 0.1
        self.learning_rate = 1e-4
        self.weight_decay = 1e-4
        self.epochs = 10
        self.reward_loss_weight = 1.0
        self.return_loss_weight = 1.0


# 使用示例
config = LoRAConfig()
base_dt_model = decision_transformer()  # 您的基础决策变压器模型
lora_dt_model = DecisionTransformerLoRA(base_model=base_dt_model, config=config)

# 在特定任务数据集上进行微调
lora_dt_model.fine_tune(task_dataset, config)
