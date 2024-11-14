import torch
import torch.nn.functional as F
import torch.optim as optim
import gym
import time

from envs import create_atari_env
from model import ActorCritic


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                               shared_model.parameters()):
        if shared_param.grad is None:
            shared_param._grad = param.grad
        else:
            shared_param._grad.data.add_(param.grad.data)


def local_training_step(model, state, done, cx, hx, args, lock, counter, env):
    values = []
    log_probs = []
    rewards = []
    entropies = []
    episode_length = 0
    episode_reward = 0

    for step in range(args.num_steps):
        episode_length += 1
        value, logit, (hx, cx) = model((state.unsqueeze(0), (hx, cx)))
        prob = F.softmax(logit, dim=-1)
        log_prob = F.log_softmax(logit, dim=-1)
        entropy = -(log_prob * prob).sum(1, keepdim=True)
        entropies.append(entropy)

        action = prob.multinomial(num_samples=1).detach()
        log_prob = log_prob.gather(1, action)

        state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated or episode_length >= args.max_episode_length
        reward = max(min(reward, 1), -1)

        with lock:
            counter.value += 1

        if done:
            episode_length = 0
            state, _ = env.reset()

        state = torch.from_numpy(state)
        values.append(value)
        log_probs.append(log_prob)
        rewards.append(reward)
        episode_reward += reward

        if done:
            break

    R = torch.zeros(1, 1)
    if not done:
        value, _, _ = model((state.unsqueeze(0), (hx, cx)))
        R = value.detach()

    values.append(R)
    return values, log_probs, rewards, entropies, state, done, cx, hx, episode_reward


def compute_loss(values, log_probs, rewards, entropies, args, model, shared_model):
    R = torch.zeros(1, 1)
    policy_loss = 0
    value_loss = 0
    gae = torch.zeros(1, 1)
    
    for i in reversed(range(len(rewards))):
        R = args.gamma * R + rewards[i]
        advantage = R - values[i]
        value_loss = value_loss + 0.5 * advantage.pow(2)

        delta_t = rewards[i] + args.gamma * values[i + 1] - values[i]
        gae = gae * args.gamma * args.gae_lambda + delta_t
        policy_loss = policy_loss - log_probs[i] * gae.detach() - args.entropy_coef * entropies[i]

    # 计算L2正则化
    l2_reg = 0
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        l2_reg += ((param - shared_param.detach()) ** 2).sum()

    total_loss = policy_loss + args.value_loss_coef * value_loss
    return total_loss, policy_loss, value_loss


def train(rank, args, shared_model, counter, lock, optimizer=None, weight_allocator=None):
    torch.manual_seed(args.seed + rank)

    # 为每个进程生成一个独特的学习率
    process_lr = max(0, torch.normal(mean=args.lr, std=args.lr * 0.5, size=(1,)).item())  # 以args.lr为均值，10%为标准差
    print(f"Process {rank} using learning rate: {process_lr}")

    start_time = time.time()
    training_duration = args.train_time

    env = create_atari_env(args.env_name)
    state, _ = env.reset()
    state = torch.from_numpy(state)

    model = ActorCritic(env.observation_space.shape[0], env.action_space)

    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=process_lr)

    model.train()

    done = True
    episode_length = 0
    episode_reward = 0  # 记录每个episode的总奖励
    while True:
        # 检查是否超过5分钟
        if time.time() - start_time > training_duration:
            print(f"Training finished after {training_duration} seconds")
            break

        # 加载共享模型
        model.load_state_dict(shared_model.state_dict())
        
        if done:
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
        else:
            cx = cx.detach()
            hx = hx.detach()

        # 执行两次本地训练
        accumulated_loss = 0
        for _ in range(2):
            values, log_probs, rewards, entropies, state, done, cx, hx, episode_reward = \
                local_training_step(model, state, done, cx, hx, args, lock, counter, env)
            
            total_loss, policy_loss, value_loss = compute_loss(
                values, log_probs, rewards, entropies, args, model, shared_model)
            accumulated_loss += total_loss

        # 计算累积梯度
        optimizer.zero_grad()
        accumulated_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        # 应用权重
        if weight_allocator is not None:
            weight = weight_allocator.get_weight(rank)
            if (weight > 0.5 or weight < 0.0001):
                print(f"Process {rank} using weight: {weight}")
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(weight)

        # 更新共享模型
        with lock:
            ensure_shared_grads(model, shared_model)
            optimizer.step()
            optimizer.zero_grad()

        # 更新权重分配器
        if done and weight_allocator is not None:
            with lock:
                weight_allocator.update_performance(rank, episode_reward)
                weight_allocator.update_weights()

    env.close()
    return
