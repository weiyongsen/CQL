import torch
import torch.nn as nn
import torch.distributions as dist
import numpy as np


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)  # 初始化偏置为0


class SharedExpert(nn.Module):
    """共享专家，学习基础知识"""
    def __init__(self, state_dim, hidden_dim=128, hidden_dims=[256, 128, 64]):
        super(SharedExpert, self).__init__()
        layers = []
        current_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.LeakyReLU())
            current_dim = hidden_dim
        self.network = nn.Sequential(*layers)
        
        # 参数初始化
        self.network.apply(weights_init)
    
    def forward(self, state):
        # 确保输入是PyTorch张量
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        return self.network(state)  # 输出隐藏层特征


class Expert(nn.Module):
    """专家网络，负责生成隐藏层特征"""
    def __init__(self, state_dim, hidden_dims=[256, 128, 64]):
        super(Expert, self).__init__()
        layers = []
        current_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.LeakyReLU())
            current_dim = hidden_dim
        self.network = nn.Sequential(*layers)
        
        # 参数初始化
        self.network.apply(weights_init)
    
    def forward(self, state):
        return self.network(state)  # 输出隐藏层特征


class GatingNetwork(nn.Module):
    """门控网络，负责决定每个专家的权重"""
    def __init__(self, state_dim, num_experts, hidden_dims=[128, 64]):
        super(GatingNetwork, self).__init__()
        layers = []
        current_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.LeakyReLU())
            current_dim = hidden_dim
        self.network = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_dims[-1], num_experts)
        self.network.apply(weights_init)
        self.output.apply(weights_init)
    def forward(self, state):
        x = self.network(state)
        temperature = 1
        logits = self.output(x) / temperature
        weights = torch.softmax(logits, dim=-1)
        return weights


class Actor(nn.Module):
    """混合专家Actor网络"""
    def __init__(self, state_dim=20*4, action_dim=4, num_experts=3, hidden_dims=[256, 128, 64], default_std=0.5):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_experts = num_experts
        self.std = default_std
        
        # 共享专家
        self.shared_expert = SharedExpert(state_dim, hidden_dims=hidden_dims)
        # 路由专家
        self.experts = nn.ModuleList([Expert(state_dim, hidden_dims=hidden_dims) for _ in range(num_experts)])
        # 门控网络
        self.router = GatingNetwork(state_dim, num_experts, hidden_dims=[128, 64])  # 门控网络一般不需要太深
        # 共享权重网络
        self.shared_weight_net = nn.Sequential(
            nn.Linear(state_dim + 2 * hidden_dims[-1], 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        # 动作头和标准差头
        self.action_head = nn.Linear(hidden_dims[-1], action_dim)
        self.std_head = nn.Linear(hidden_dims[-1], action_dim)
        
        # 参数初始化
        self.shared_weight_net.apply(weights_init)
        self.action_head.apply(weights_init)
        self.std_head.apply(weights_init)

    def forward(self, state):
        # 如果输入是单个状态，添加batch维度
        if len(state.shape) == 1:
            state = state.unsqueeze(0)  # [1, state_dim]

        # 专家输出
        shared_output = self.shared_expert(state)  # [batch_size, hidden_dim]
        expert_outputs = torch.stack([expert(state) for expert in self.experts], dim=1)  # [batch_size, num_experts, hidden_dim]

        # 路由器计算专家权重
        router_weights = self.router(state)  # [batch_size, num_experts]

        # 路由专家软路由融合
        routed_output = (expert_outputs * router_weights.unsqueeze(-1)).sum(dim=1)  # [batch_size, hidden_dim]

        # 合并共享和路由专家输出
        combined_input = torch.cat([state, shared_output, routed_output], dim=-1)  # [batch_size, state_dim + 2 * hidden_dim]
        shared_weight = self.shared_weight_net(combined_input)  # [batch_size, 1]
        combined_output = shared_weight * shared_output + (1 - shared_weight) * routed_output  # [batch_size, hidden_dim]

        # 动作均值与标准差
        action_mean = torch.tanh(self.action_head(combined_output))  # [batch_size, action_dim]
        action_std = torch.sigmoid(self.std_head(combined_output)) * 0.69 + 0.01  # [batch_size, action_dim]
        
        # 如果输入是单个状态，移除batch维度
        if len(state.shape) == 2 and state.shape[0] == 1:
            action_mean = action_mean.squeeze(0)
            action_std = action_std.squeeze(0)
        
        return action_mean, action_std
    
    def action_distribution(self, state):
        mean, std = self.forward(state)
        dist_gaussian = dist.Normal(mean, std)
        return dist_gaussian
    
    def sample_action(self, state, greedy=False):
        dist_gaussian = self.action_distribution(state)
        if greedy:
            action_gaussian = dist_gaussian.mean
        else:
            action_gaussian = dist_gaussian.sample()
        action_log_prob = dist_gaussian.log_prob(action_gaussian)
        action_log_prob = action_log_prob.sum(dim=-1)
        return action_gaussian, action_log_prob
    
    def log_prob(self, state, action):
        dist_gaussian = self.action_distribution(state)
        action_log_prob = dist_gaussian.log_prob(action)
        action_log_prob = action_log_prob.sum(dim=-1)
        return action_log_prob
    
    def set_std(self, std):
        self.std = std


class Critic(nn.Module):
    def __init__(self, state_dim=20*4, hidden_dims=[128, 64, 32]):
        super(Critic, self).__init__()
        layers = []
        current_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.GELU())
            current_dim = hidden_dim
        self.network = nn.Sequential(*layers)
        self.critic_fc = nn.Linear(hidden_dims[-1], 1)

        self.network.apply(weights_init)
        self.critic_fc.apply(weights_init)

    def forward(self, state):
        x = self.network(state)
        return self.critic_fc(x)


# 测试代码
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(2, 20).to(device)
    actor = Actor(20, 4).to(device)

    dist_gaussian = actor.action_distribution(x)
    print(f"Mean: {dist_gaussian.mean}, \nStd: {dist_gaussian.stddev}")

    y1, prob1 = actor.sample_action(x)
    print("action:", y1)
    print("log_prob:", prob1)
    probs = actor.log_prob(x, y1)
    print("log_prob2:", probs) 