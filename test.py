import time
from collections import deque

import torch
import torch.nn.functional as F

from envs import create_atari_env
from model import ActorCritic

import matplotlib.pyplot as plt
from datetime import datetime, timedelta


def test(rank, args, shared_model, counter):
    print("test()")
    torch.manual_seed(args.seed + rank)

    env = create_atari_env(args.env_name)
    env.seed(args.seed + rank)
    # print("Environment created")
    model = ActorCritic(env.observation_space.shape[0], env.action_space)

    model.eval()
    # print("Model loaded")
    state, _ = env.reset()
    state = torch.from_numpy(state)
    # print("Initial state shape:", state.shape)
    reward_sum = 0
    done = True

    start_time = time.time()

    # a quick hack to prevent the agent from stucking
    actions = deque(maxlen=100)
    episode_length = 0

    # 新增：记录测试结果
    test_results = []
    start_test_time = datetime.now()
    test_duration = timedelta(minutes=30)

    while True:
        episode_length += 1
        # Sync with the shared model
        if done:
            # print("test : model.load_state_dict()")
            model.load_state_dict(shared_model.state_dict())
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)        
        else:
            cx = cx.detach()
            hx = hx.detach()

        with torch.no_grad():
            value, logit, (hx, cx) = model((state.unsqueeze(0), (hx, cx)))
        prob = F.softmax(logit, dim=-1)
        action = prob.max(1, keepdim=True)[1].numpy()

        state, reward, terminated, truncated, info = env.step(action[0, 0])
        # print("State shape after step:", state.shape)
        done = terminated or truncated
        reward_sum += reward

        # a quick hack to prevent the agent from stucking
        actions.append(action[0, 0])
        if actions.count(actions[0]) == actions.maxlen:
            done = True

        if done:
            # 记录每集的结果
            test_results.append({
                'reward': reward_sum,
                'length': episode_length,
                'steps': counter.value
            })
            
            print("Time {}, num steps {}, FPS {:.0f}, episode reward {}, episode length {}".format(
                time.strftime("%Hh %Mm %Ss",
                              time.gmtime(time.time() - start_time)),
                counter.value, counter.value / (time.time() - start_time),
                reward_sum, episode_length))
                
            # 检查是否达到5分钟
            if datetime.now() - start_test_time >= test_duration:
                # 绘制结果
                plot_results(test_results, args)
                break
                
            reward_sum = 0
            episode_length = 0
            actions.clear()
            state, _ = env.reset()
            time.sleep(60)

        state = torch.from_numpy(state)

def plot_results(results, args):
    """绘制测试结果"""
    episodes = range(len(results))
    rewards = [r['reward'] for r in results]
    lengths = [r['length'] for r in results]
    print("rewards:", rewards)
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(episodes, rewards, 'b-')
    plt.title(f'Episode Rewards\n{args.env_name} ({args.num_processes} processes)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(1, 2, 2)
    plt.plot(episodes, lengths, 'r-')
    plt.title(f'Episode Lengths\n{args.env_name} ({args.num_processes} processes)')
    plt.xlabel('Episode')
    plt.ylabel('Length')
    
    plt.tight_layout()
    plt.savefig(f'test_results_{args.env_name}_{args.num_processes}proc_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png')
    plt.close()
