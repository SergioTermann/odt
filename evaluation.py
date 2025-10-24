import numpy as np
import torch
MAX_EPISODE_LEN = 1000


def create_vec_eval_episodes_fn(vec_env, eval_rtg, state_dim, act_dim, state_mean, state_std, device, use_mean=False, reward_scale=0.001):
    def eval_episodes_fn(model):
        target_return = [eval_rtg * reward_scale]
        # vec_evaluate_episode_rtg已修改为处理双头输出
        returns, lengths, _ = vec_evaluate_episode_rtg(vec_env, state_dim,act_dim, model, max_ep_len=MAX_EPISODE_LEN, reward_scale=reward_scale, target_return=target_return, mode="normal", state_mean=state_mean, state_std=state_std, device=device, use_mean=use_mean)
        suffix = "_gm" if use_mean else ""
        return {f"evaluation/return_mean{suffix}": np.mean(returns), f"evaluation/return_std{suffix}": np.std(returns), f"evaluation/length_mean{suffix}": np.mean(lengths), f"evaluation/length_std{suffix}": np.std(lengths)}
    return eval_episodes_fn


@torch.no_grad()
def vec_evaluate_episode_rtg(vec_env, state_dim, act_dim, model, target_return: list, max_ep_len=1000, reward_scale=0.001, state_mean=0.0, state_std=1.0, device="cuda", mode="normal", use_mean=False):
    model.eval()
    model.to(device=device)
    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)
    num_envs = 1
    state = vec_env.reset()
    state = np.array(state)
    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = (torch.from_numpy(state).reshape(num_envs, state_dim).to(device=device, dtype=torch.float32)).reshape(num_envs, -1, state_dim)
    actions = torch.zeros(0, device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(num_envs, -1, 1)
    timesteps = torch.tensor([0] * num_envs, device=device, dtype=torch.long).reshape(num_envs, -1)

    # episode_return, episode_length = 0.0, 0
    episode_return = 0
    episode_length = np.full(num_envs, np.inf)

    unfinished = np.ones(num_envs).astype(bool)
    for t in range(max_ep_len):
        # add padding
        actions = torch.cat([actions, torch.zeros((num_envs, act_dim), device=device).reshape(num_envs, -1, act_dim)], dim=1)
        rewards = torch.cat([rewards, torch.zeros((num_envs, 1), device=device).reshape(num_envs, -1, 1)], dim=1)

        # 获取模型的任务预测输出
        task_pred = model.get_predictions(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
            num_envs=num_envs,
        )
        # 使用任务预测来控制行为
        # 这里我们假设task_pred是一个张量，表示任务选择的logits
        # 我们可以将其转换为动作
        action = torch.zeros((num_envs, act_dim), device=device)
        
        # 根据任务预测选择动作
        # 这里简单地将任务预测映射到动作空间
        # 实际应用中可能需要更复杂的映射逻辑
        
        # 确保task_pred是二维张量 [batch_size, num_classes]
        if len(task_pred.shape) > 2:
            # 如果维度过多，取最后两个维度
            task_pred = task_pred.reshape(-1, task_pred.shape[-1])
        
        task_indices = torch.argmax(task_pred, dim=1)
        
        # 为每个任务设置不同的动作
        for i in range(num_envs):
            # 检查task_indices[i]是否为标量张量
            if task_indices[i].numel() > 1:
                # 如果是多元素张量，取第一个元素
                task_idx = task_indices[i][0].item()
            else:
                # 如果是标量张量，直接转换
                task_idx = task_indices[i].item()
            
            # 根据任务索引设置动作
            # 这里使用简单的映射，实际应用中可能需要更复杂的逻辑
            action[i, task_idx % act_dim] = 1.0
        
        action = action.clamp(*model.action_range)

        state, reward, done, _ = vec_env.steps(action.detach().cpu().numpy())

        # eval_env.step() will execute the action for all the sub-envs, for those where
        # the episodes have terminated, the envs will be reset. Hence we use
        # "unfinished" to track whether the first episode we roll out for each sub-env is
        # finished. In contrast, "done" only relates to the current episode
        episode_return += reward

        actions[:, -1] = action
        state = (torch.from_numpy(state).to(device=device).reshape(num_envs, -1, state_dim))
        states = torch.cat([states, state], dim=1)
        reward = torch.from_numpy(reward).to(device=device).reshape(num_envs, 1)
        rewards[:, -1] = reward

        if mode != "delayed":
            pred_return = target_return[:, -1] - (reward * reward_scale)
        else:
            pred_return = target_return[:, -1]
        target_return = torch.cat([target_return, pred_return.reshape(num_envs, -1, 1)], dim=1)

        timesteps = torch.cat([timesteps, torch.ones((num_envs, 1), device=device, dtype=torch.long).reshape(num_envs, 1) * (t + 1)], dim=1)

        if np.any(done):
            ind = np.where(done)[0]
            unfinished[ind] = False
            episode_length[ind] = np.minimum(episode_length[ind], t + 1)

        if not np.any(unfinished):
            break

    trajectories = []
    for ii in range(num_envs):
        ep_len = episode_length[ii].astype(int)
        terminals = np.zeros(ep_len)
        terminals[-1] = 1
        traj = {
            "observations": states[ii].detach().cpu().numpy()[:ep_len],
            "actions": actions[ii].detach().cpu().numpy()[:ep_len],
            "rewards": rewards[ii].detach().cpu().numpy()[:ep_len],
            "terminals": terminals,
        }
        trajectories.append(traj)

    return episode_return, episode_length.reshape(num_envs), trajectories
