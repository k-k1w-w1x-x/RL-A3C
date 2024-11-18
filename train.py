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


def train(rank, args, shared_model, counter, lock, optimizer=None, weight_allocator=None):
    torch.manual_seed(args.seed + rank)

    # 为每个进程生成一个独特的学习率
    process_lr = max(0, torch.normal(mean=args.lr, std=args.lr * 0.1, size=(1,)).item())  # 以args.lr为均值，10%为标准差
    print(f"Process {rank} using learning rate: {process_lr}")

    start_time = time.time()
    training_duration = args.train_time

    env = create_atari_env(args.env_name)
    state, _ = env.reset()
    state = torch.from_numpy(state)

    model = ActorCritic(env.observation_space.shape[0], env.action_space)

    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=process_lr)

    # 添加KL相关参数
    beta = torch.tensor(args.beta_prime, requires_grad=False)  # 初始化为tensor,default=0.01
    target_kl = args.kl_target  # KL散度的目标值,default=0.001
    kl_max = 1.2  # KL散度的最大值
    kl_min = 0.8  # KL散度的最小值
    beta_up = 1.5  # beta的增加系数
    beta_down = 0.5  # beta的减少系数
    beta_min = 1e-6  # beta的最小值
    beta_max = 0.01  # beta的最大值
    # adaptive_coef = 0.1  # 自适应系数，控制beta调整的幅度
    
    model.train()
    done = True
    episode_length = 0
    episode_reward = 0

    while True:
        # 检查是否超过5分钟
        if time.time() - start_time > training_duration:
            print(f"Training finished after {training_duration} seconds")
            break

        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())
        if done:
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
        else:
            cx = cx.detach()
            hx = hx.detach()

        # 存储old policy的概率分布
        old_probs = []
        values = []
        log_probs = []
        rewards = []
        entropies = []
        
        for step in range(args.num_steps):
            episode_length += 1
            value, logit, (hx, cx) = model((state.unsqueeze(0), (hx, cx)))
            prob = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)
            
            # 保存old policy的概率分布
            old_probs.append(prob.detach())
            
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

            if done:
                break

        R = torch.zeros(1, 1)
        if not done:
            value, _, _ = model((state.unsqueeze(0), (hx, cx)))
            R = value.detach()

        values.append(R)
        policy_loss = 0
        value_loss = 0
        kl_loss = 0
        gae = torch.zeros(1, 1)
        
        # 计算总的KL散度
        total_kl = 0
        
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimation
            delta_t = rewards[i] + args.gamma * values[i + 1] - values[i]
            gae = gae * args.gamma * args.gae_lambda + delta_t

            # 计算新旧策略的KL散度
            value, logit, _ = model((state.unsqueeze(0), (hx, cx)))
            new_prob = F.softmax(logit, dim=-1)
            old_prob = old_probs[i]
            
            # 计算KL散度
            kl_div = torch.sum(old_prob * (torch.log(old_prob + 1e-10) - torch.log(new_prob + 1e-10)), dim=-1).mean()
            kl_loss += kl_div
            total_kl += kl_div.item()

            policy_loss = policy_loss - log_probs[i] * gae.detach() - args.entropy_coef * entropies[i]

        # 计算平均KL散度
        avg_kl = total_kl / len(rewards)
        
        # 修改beta的更新逻辑
        kl_ratio = avg_kl / target_kl
        if kl_ratio > kl_max:
            # KL过大，增加beta
            beta *= beta_up
        elif kl_ratio < kl_min:
            # KL过小，减少beta
            beta *= beta_down

        # 使用torch.clamp替代max函数
        # 确保beta在合理范围内
        beta = torch.clamp(beta, min=beta_min, max=beta_max)  # 确保beta保持tensor类型

        # 使用自适应的beta计算总损失
        total_loss = policy_loss + args.value_loss_coef * value_loss + beta * kl_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        # 应用权重
        if weight_allocator is not None:
            weight = weight_allocator.get_weight(rank)
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(weight)

        # 使用锁保护整个更新过程
        with lock:
            ensure_shared_grads(model, shared_model)
            optimizer.step()
            # 清除shared_model的梯度，为下一次更新做准备
            optimizer.zero_grad()

        # 添加模型参数检查
        def check_model_state():
            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    print(f"NaN found in {name}")
                if torch.isinf(param).any():
                    print(f"Inf found in {name}")

        check_model_state()  # 训练开始时
        check_model_state()  # 每个episode结束时
        if done:
            print(f"Process {rank} Beta: {beta}, KL Loss: {kl_loss.item():.4f}")
        # 在训练循环中添加损失值打印
        # print(f"Episode {counter.value}")
        # print(f"Policy Loss: {policy_loss.item()}")
        # print(f"Value Loss: {value_loss.item()}")
        # print(f"Total Loss: {(policy_loss + args.value_loss_coef * value_loss).item()}")

        # 在每个episode结束时也可以打印剩余时间
        # elapsed_time = time.time() - start_time
        # remaining_time = max(0, training_duration - elapsed_time)
        # print(f"Time remaining: {remaining_time:.1f} seconds")

    # 训练结束后关闭环境
    env.close()
    return
