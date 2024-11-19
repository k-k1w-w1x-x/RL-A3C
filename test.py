import time
from collections import deque

import torch
import torch.nn.functional as F

from envs import create_atari_env
# from model import ActorCritic
from model import A3CWithAttention  # 引入新模型

import matplotlib.pyplot as plt
from datetime import datetime, timedelta
# from gymnasium.wrappers import RecordVideo
import gym



def test(rank, args, shared_model, counter):
    print("test()")
    torch.manual_seed(args.seed + rank)

    # 创建环境
    env = create_atari_env("PongDeterministic-v4")
    # env = create_atari_env(args.env_name)
    
    # 获取输入维度和动作空间大小
    input_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n
    
    # 实例化模型
    model = A3CWithAttention(
        input_dim=input_dim,  # 使用 input_dim
        num_actions=num_actions,  # 使用动作空间大小
        num_heads=4  # 你想要的注意力头数
    )

    model.eval()
    state, _ = env.reset()
    state = torch.from_numpy(state)
    reward_sum = 0
    done = True

    start_time = time.time()

    # a quick hack to prevent the agent from stucking
    actions = deque(maxlen=100)
    episode_length = 0

    # 新增：记录测试结果
    test_results = []
    start_test_time = datetime.now()
    test_duration = timedelta(seconds=args.test_time)

    while True:
        episode_length += 1
        # Sync with the shared model
        if done:
            model.load_state_dict(shared_model.state_dict(), strict=False)
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)        
        else:
            cx = cx.detach()
            hx = hx.detach()

        with torch.no_grad():
            policy, value, (hx, cx) = model((state.unsqueeze(0), (hx, cx)))
        prob = F.softmax(policy, dim=-1)

        action = prob.max(1, keepdim=True)[1].numpy()

        state, reward, terminated, truncated, info = env.step(action[0, 0])
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
                
            if datetime.now() - start_test_time >= test_duration:
                # 绘制结果
                plot_results(test_results, args)
                # 进行最终测试并显示游戏画面
                evaluate_with_render(shared_model, args)
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

def evaluate_with_render(shared_model, args):
    """测试模型并录制游戏画面"""

    # 定义录像保存路径（包括环境名称和时间戳以保证唯一性）
    output_dir = f"videos/{args.env_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    # 确保使用 gym.make() 创建环境
    env = gym.make(args.env_name)
    print(type(env))  # 打印环境类型
    assert isinstance(env, gym.Env), "Environment is not a valid gym.Env instance"
    
    # # 使用 RecordVideo 包装器
    # env = RecordVideo(env, video_folder=output_dir, episode_trigger=lambda x: True)

    # 创建模型并加载共享的训练参数
     # 创建环境
    env = create_atari_env("PongDeterministic-v4")
    
    # 获取输入维度和动作空间大小
    input_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n
    
    # 实例化模型
    model = A3CWithAttention(
        input_dim=input_dim,  # 使用 input_dim
        num_actions=num_actions,  # 使用动作空间大小
        num_heads=4  # 你想要的注意力头数
    )

    model.load_state_dict(shared_model.state_dict(), strict=False)
    model.eval()  # 切换为评估模式

    # 初始化状态和 LSTM 隐藏状态
    state, _ = env.reset()
    done = False
    total_reward = 0
    cx = torch.zeros(1, 256)  # LSTM 的细胞状态
    hx = torch.zeros(1, 256)  # LSTM 的隐藏状态

    while not done:
        state = torch.from_numpy(state)  # 转换状态为 PyTorch 张量
        with torch.no_grad():  # 禁用梯度计算以提升推理效率
            # 传递状态和 LSTM 隐藏状态的元组
            policy, value, (hx, cx) = model((state.unsqueeze(0), (hx, cx)))
        
        # 从动作分布中选择概率最高的动作
        prob = F.softmax(policy, dim=-1)
        action = prob.max(1, keepdim=True)[1].numpy()

        # 执行动作并获取反馈
        state, reward, terminated, truncated, _ = env.step(action[0, 0])
        done = terminated or truncated  # 检查是否结束
        total_reward += reward

    # # 输出总奖励并保存视频路径
    # print(f"Final evaluation - Total reward: {total_reward}")
    # env.close()
    # print(f"Video saved to: {output_dir}")

# 修复 RecordVideo 的 __del__ 方法
def safe_del(self):
    if hasattr(self, 'recorded_frames') and len(self.recorded_frames) > 0:
        # 处理 recorded_frames
        pass

# # 将 __del__ 方法替换为安全版本
# RecordVideo.__del__ = safe_del

