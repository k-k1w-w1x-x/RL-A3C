import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)

class AttentionNetwork(nn.Module):
    def __init__(self, input_dim, num_heads):  
        super(AttentionNetwork, self).__init__()
        self.conv = nn.Conv2d(input_dim, 32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32, num_heads)  
        self.softmax = nn.Softmax(dim=1)



    def forward(self, shared_features):
        x = F.relu(self.conv(shared_features))
        x = torch.mean(x, dim=(2, 3))  # Global Average Pooling
        attention_weights = self.softmax(self.fc(x))
        return attention_weights
    
class Subnet(nn.Module):
    def __init__(self, flatten_size, output_dim):
        super(Subnet, self).__init__()
        # print(f"Initializing Subnet with flatten_size: {flatten_size}")
        self.fc1 = nn.Linear(flatten_size, 128)  # 动态设置输入尺寸
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        # print(f"Input shape to fc1: {x.shape}")
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# 注意力机制
class A3CWithAttention(nn.Module):
    def __init__(self, input_dim, num_actions, num_heads):
        super(A3CWithAttention, self).__init__()
        self.shared_features = nn.Conv2d(input_dim, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(112896, 256)
        self.attention = AttentionNetwork(input_dim=64, num_heads=num_heads)
        self.subnets = [nn.Linear(256, num_actions) for _ in range(num_heads)]
        self.value_heads = [nn.Linear(256, 1) for _ in range(num_heads)]

        self.flatten_size = None
        self.num_heads = num_heads
        self.num_actions = num_actions

        self.lstm_input_size = 64 * 42 * 42  # 根据实际的特征图大小调整
        self.lstm = nn.LSTMCell(self.lstm_input_size, 256)

        

    def forward(self, inputs):
        state, (hx, cx) = inputs
        # print("Initial state shape:", state.shape)

        shared_features = F.relu(self.shared_features(state))
        # print("Shared features shape:", shared_features.shape)

        flat_features = shared_features.view(shared_features.size(0), -1)
        # print("Flattened features size:", flat_features.size())

        x = flat_features
        # print("Input to fc1 size:", x.size())

        x = F.relu(self.fc1(x))
        # print("Output from fc1 size:", x.size())

        hx, cx = self.lstm(flat_features, (hx, cx))
        # print("LSTM output hx size:", hx.size())
        # print("LSTM output cx size:", cx.size())

        policies = []
        values = []
        for i, (policy_head, value_head) in enumerate(zip(self.subnets, self.value_heads)):
            policy_output = policy_head(hx)
            value_output = value_head(hx)
            # print(f"Policy head {i} output size:", policy_output.size())
            # print(f"Value head {i} output size:", value_output.size())
            policies.append(policy_output)
            values.append(value_output)

        if not policies:
            print("Error: No policies generated.")
        if not values:
            print("Error: No values generated.")

        final_policy = torch.stack(policies, dim=0).sum(dim=0)
        final_value = torch.stack(values, dim=0).sum(dim=0)
        # print("Final policy size:", final_policy.size())
        # print("Final value size:", final_value.size())

        return final_policy, final_value, (hx, cx)





    
# class ActorCritic(torch.nn.Module):
#     def __init__(self, input_dim, action_space):
#         super(ActorCritic, self).__init__()
#         self.conv1 = nn.Conv2d(input_dim, 32, 3, stride=2, padding=1)
#         self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
#         self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
#         self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

#         self.lstm = nn.LSTMCell(32 * 3 * 3, 256)

#         num_outputs = action_space.n
#         self.critic_linear = nn.Linear(256, 1)
#         self.actor_linear = nn.Linear(256, num_outputs)

#         self.apply(weights_init)
#         self.actor_linear.weight.data = normalized_columns_initializer(
#             self.actor_linear.weight.data, 0.01)
#         self.actor_linear.bias.data.fill_(0)
#         self.critic_linear.weight.data = normalized_columns_initializer(
#             self.critic_linear.weight.data, 1.0)
#         self.critic_linear.bias.data.fill_(0)

#         self.lstm.bias_ih.data.fill_(0)
#         self.lstm.bias_hh.data.fill_(0)

#         self.train()

#     def forward(self, inputs):
#         inputs, (hx, cx) = inputs
#         x = F.elu(self.conv1(inputs))
#         x = F.elu(self.conv2(x))
#         x = F.elu(self.conv3(x))
#         x = F.elu(self.conv4(x))

#         x = x.view(-1, 32 * 3 * 3)
#         hx, cx = self.lstm(x, (hx, cx))
#         x = hx

#         return self.critic_linear(x), self.actor_linear(x), (hx, cx)
# # python main.py --env-name "PongDeterministic-v4" --num-processes 16、




