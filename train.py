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
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def train(rank, args, shared_model, counter, lock, optimizer=None):
    torch.manual_seed(args.seed + rank)

    start_time = time.time()
    training_duration = 1800

    env = create_atari_env(args.env_name)
    state, _ = env.reset()
    state = torch.from_numpy(state)

    model = ActorCritic(env.observation_space.shape[0], env.action_space)

    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)

    model.train()

    done = True
    # print("model.train()")
    episode_length = 0
    while True:
        # 检查是否超过5分钟
        if time.time() - start_time > training_duration:
            print(f"Training finished after {training_duration} seconds")
            break

        # Sync with the shared model
        # print("train : model.load_state_dict()")
        model.load_state_dict(shared_model.state_dict())
        if done:
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
        else:
            cx = cx.detach()
            hx = hx.detach()

        values = []
        log_probs = []
        rewards = []
        entropies = []
        # print("args.num_steps", args.num_steps)
        for step in range(args.num_steps):
            episode_length += 1
            value, logit, (hx, cx) = model((state.unsqueeze(0),
                                            (hx, cx)))
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

            if done:
                break

        R = torch.zeros(1, 1)
        if not done:
            value, _, _ = model((state.unsqueeze(0), (hx, cx)))
            R = value.detach()

        values.append(R)
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimation
            delta_t = rewards[i] + args.gamma * \
                values[i + 1] - values[i]
            gae = gae * args.gamma * args.gae_lambda + delta_t

            policy_loss = policy_loss - \
                log_probs[i] * gae.detach() - args.entropy_coef * entropies[i]

        optimizer.zero_grad()

        (policy_loss + args.value_loss_coef * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        ensure_shared_grads(model, shared_model)
        optimizer.step()

        # 添加模型参数检查
        def check_model_state():
            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    print(f"NaN found in {name}")
                if torch.isinf(param).any():
                    print(f"Inf found in {name}")

        check_model_state()  # 训练开始时
        check_model_state()  # 每个episode结束时

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
