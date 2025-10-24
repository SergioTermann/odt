import torch
import torch.nn as nn

import transformers

from decision_transformer.models.model import TrajectoryModel
from decision_transformer.models.trajectory_gpt2 import GPT2Model
from decision_transformer.models.causal_graph import CausalGraph, CounterfactualDecisionMaker
import math
import numpy as np
import torch.nn.functional as F
from torch import distributions as pyd


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, std):
        self.loc = loc
        self.std = std
        self.base_dist = pyd.Normal(loc, std)

        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

    def entropy(self, N=1):
        # sample from the distribution and then compute
        # the empirical entropy:
        x = self.rsample((N,))
        log_p = self.log_prob(x)

        # log_p: (batch_size, context_len, action_dim),
        return -log_p.mean(axis=0).sum(axis=2)

    def log_likelihood(self, x):
        # log_prob(x): (batch_size, context_len, action_dim)
        # sum up along the action dimensions
        # Return tensor shape: (batch_size, context_len)
        return self.log_prob(x).sum(axis=2)


class DiagGaussianActor(nn.Module):
    def __init__(self, hidden_dim, act_dim, log_std_bounds=[-5.0, 2.0]):
        super().__init__()

        self.mu = torch.nn.Linear(hidden_dim, act_dim)
        self.log_std = torch.nn.Linear(hidden_dim, act_dim)
        self.log_std_bounds = log_std_bounds

        def weight_init(m):
            """Custom weight init for Conv2D and Linear layers."""
            if isinstance(m, torch.nn.Linear):
                nn.init.orthogonal_(m.weight.data)
                if hasattr(m.bias, "data"):
                    m.bias.data.fill_(0.0)

        self.apply(weight_init)

    def forward(self, obs):
        mu, log_std = self.mu(obs), self.log_std(obs)
        # 直接返回mu作为结果，不再创建SquashedNormal分布
        return mu


class DecisionTransformer(TrajectoryModel):
    def __init__(self, state_dim, act_dim, hidden_size, action_range, ordering=0, max_length=None, eval_context_length=None, max_ep_len=4096, action_tanh=True, stochastic_policy=False, init_temperature=0.1, target_entropy=None, num_tasks=5, use_causal_graph=True, **kwargs):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        self.num_tasks = num_tasks  # 任务数量
        self.use_causal_graph = use_causal_graph  # 是否使用因果图网络
        config = transformers.GPT2Config(vocab_size=1, n_embd=hidden_size, **kwargs)

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        if ordering:
            self.embed_ordering = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        self.predict_return = torch.nn.Linear(hidden_size, 1)
        
        # 第一个头：任务选择
        self.predict_task = nn.Linear(hidden_size, num_tasks)
        
        # 第二个头：任务参数（原始动作预测）
        if stochastic_policy:
            self.predict_action = DiagGaussianActor(hidden_size, self.act_dim)
        else:
            self.predict_action = nn.Sequential(*([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else [])))
        
        # 因果图网络和反事实推理
        if use_causal_graph:
            self.causal_graph = CausalGraph(num_tasks, hidden_size)
            self.counterfactual_decision_maker = CounterfactualDecisionMaker(num_tasks, hidden_size)
        
        self.stochastic_policy = stochastic_policy
        self.eval_context_length = eval_context_length
        self.ordering = ordering
        self.action_range = action_range

        if stochastic_policy:
            self.log_temperature = torch.tensor(np.log(init_temperature))
            self.log_temperature.requires_grad = True
            self.target_entropy = target_entropy

    def temperature(self):
        if self.stochastic_policy:
            return self.log_temperature.exp()
        else:
            return None

    def forward(self, states, actions, rewards, returns_to_go, timesteps, ordering, padding_mask=None):
        batch_size, seq_length = states.shape[0], states.shape[1]

        if padding_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            padding_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)

        if self.ordering:
            order_embeddings = self.embed_ordering(timesteps)
        else:
            order_embeddings = 0.0

        state_embeddings = state_embeddings + order_embeddings
        action_embeddings = action_embeddings + order_embeddings
        returns_embeddings = returns_embeddings + order_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = (torch.stack((returns_embeddings, state_embeddings, action_embeddings), dim=1).permute(0, 2, 1, 3).reshape(batch_size, 3 * seq_length, self.hidden_size))
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_padding_mask = (torch.stack((padding_mask, padding_mask, padding_mask), dim=1).permute(0, 2, 1).reshape(batch_size, 3 * seq_length))

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(inputs_embeds=stacked_inputs, attention_mask=stacked_padding_mask)
        x = transformer_outputs["last_hidden_state"]

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        # predict next return given state and action
        return_preds = self.predict_return(x[:, 2])
        # predict next state given state and action
        state_preds = self.predict_state(x[:, 2])
        
        # 第一个头：预测任务选择
        task_logits = self.predict_task(x[:, 1])
        
        # 应用因果图网络和反事实推理优化任务选择
        if self.use_causal_graph:
            # 使用状态特征进行因果推理
            state_features = x[:, 1]  # 使用状态的隐藏表示
            
            # 确保输入维度正确
            # 如果task_logits或state_features是4D张量，需要调整为3D
            if len(task_logits.shape) > 3:
                task_logits = task_logits.reshape(task_logits.shape[0], -1, task_logits.shape[-1])
            if len(state_features.shape) > 3:
                state_features = state_features.reshape(state_features.shape[0], -1, state_features.shape[-1])
            
            # 基础因果推理
            causal_task_logits = self.causal_graph(task_logits, state_features)
            
            # 反事实决策
            refined_task_logits = self.counterfactual_decision_maker(task_logits, state_features)
            
            # 最终任务预测结果
            task_preds = refined_task_logits
        else:
            task_preds = task_logits
        
        # 第二个头：预测任务参数（原始动作）
        action_preds = self.predict_action(x[:, 1])

        return state_preds, action_preds, return_preds, task_preds

    def get_predictions(self, states, actions, rewards, returns_to_go, timesteps, num_envs=1, **kwargs):
        # we don't care about the past rewards in this model
        # tensor shape: batch_size, seq_length, variable_dim
        states = states.reshape(num_envs, -1, self.state_dim)
        actions = actions.reshape(num_envs, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(num_envs, -1, 1)

        # tensor shape: batch_size, seq_length
        timesteps = timesteps.reshape(num_envs, -1)

        # max_length is the DT context length (should be input length of the subsequence)
        # eval_context_length is the how long you want to use the history for your prediction
        if self.max_length is not None:
            states = states[:, -self.eval_context_length:]
            actions = actions[:, -self.eval_context_length:]
            returns_to_go = returns_to_go[:, -self.eval_context_length:]
            timesteps = timesteps[:, -self.eval_context_length:]

            ordering = torch.tile(torch.arange(timesteps.shape[1], device=states.device), (num_envs, 1))
            # pad all tokens to sequence length
            padding_mask = torch.cat([torch.zeros(self.max_length - states.shape[1]), torch.ones(states.shape[1])])
            padding_mask = padding_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            padding_mask = padding_mask.repeat((num_envs, 1))

            states = torch.cat([torch.zeros((states.shape[0], self.max_length - states.shape[1], self.state_dim), device=states.device), states], dim=1).to(dtype=torch.float32)
            actions = torch.cat([torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim), device=actions.device), actions], dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat([torch.zeros((returns_to_go.shape[0], self.max_length - returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go], dim=1).to(dtype=torch.float32)
            timesteps = torch.cat([torch.zeros((timesteps.shape[0], self.max_length - timesteps.shape[1]), device=timesteps.device), timesteps], dim=1).to(dtype=torch.long)
            ordering = torch.cat([torch.zeros((ordering.shape[0], self.max_length - ordering.shape[1]), device=ordering.device), ordering], dim=1).to(dtype=torch.long)
        else:
            padding_mask = None

        state_preds, action_preds, return_preds, task_preds = self.forward(states, actions, None, returns_to_go, timesteps, ordering, padding_mask=padding_mask, **kwargs)
        
        # 确保task_preds是张量而不是元组
        if isinstance(task_preds, tuple):
            # 如果是元组，取最后一个元素
            task_pred = task_preds[-1]
        else:
            task_pred = task_preds
            
        # 确保返回的是二维张量 [batch_size, num_classes]
        # 如果task_pred是三维或更高维度，将其转换为二维
        if len(task_pred.shape) > 2:
            # 保留批次维度和最后一个维度
            task_pred = task_pred.reshape(task_pred.shape[0], -1)
        
        # 无论stochastic_policy如何，都返回相同的结果
        return task_pred

    def clamp_action(self, action):
        return action.clamp(*self.action_range)
