import torch
import torch.nn as nn
import torch.distributions as dist
import numpy as np


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)  # 初始化偏置为0


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=None, default_std=0.5):
        super(Actor, self).__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        layers = []
        current_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            # if current_dim == state_dim:
            #     layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.LeakyReLU())
            # layers.append(nn.Dropout(p=0.1))
            current_dim = hidden_dim
        self.network = nn.Sequential(*layers)

        # 生成所有输出的mean和log_std参数
        self.actor_mean = nn.Linear(hidden_dims[-1], action_dim)
        self.actor_std = nn.Linear(hidden_dims[-1], action_dim)
        self.std = default_std

        # 参数初始化
        self.network.apply(weights_init)
        self.actor_mean.apply(weights_init)
        self.actor_std.apply(weights_init)

        # 将模型移动到GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        # 确保输入是PyTorch张量并移动到GPU
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        state = state.to(self.device)
        x = self.network(state)
        mean = torch.tanh(self.actor_mean(x))
        std = torch.sigmoid(self.actor_std(x)) + 1e-5
        return mean, std

    def action_distribution(self, state):
        mean, std = self.forward(state)
        dist_gaussian = dist.Normal(mean, std)  # std会自动广播，与mean维度重合
        return dist_gaussian

    def sample_action(self, state, greedy=False):
        dist_gaussian = self.action_distribution(state)
        if greedy:
            action_gaussian = dist_gaussian.mean
        else:
            action_gaussian = dist_gaussian.sample()
        action_log_prob = dist_gaussian.log_prob(action_gaussian)
        action_log_prob = action_log_prob.sum(dim=-1)
        # 将结果移回CPU
        action_gaussian = action_gaussian.cpu().detach().numpy()
        action_log_prob = action_log_prob.cpu().detach().numpy()
        return action_gaussian, action_log_prob

    def log_prob(self, state, action):
        dist_gaussian = self.action_distribution(state)
        action_log_prob = dist_gaussian.log_prob(action)   # 返回log概率的总和，求最后一维的和
        action_log_prob = action_log_prob.sum(dim=-1)
        return action_log_prob

    def set_std(self, std):
        self.std = std


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dims=None):
        super(Critic, self).__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        layers = []
        current_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
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
    x = torch.randn(2, 6).to(device)
    actor = Actor(6, 4).to(device)

    dist_gaussian = actor.action_distribution(x)
    print(f"Mean: {dist_gaussian.mean}, \nStd: {dist_gaussian.stddev}")

    y1, prob1 = actor.sample_action(x)
    print("action:", y1)
    print("log_prob:", prob1)
    probs = actor.log_prob(x, y1)
    print("log_prob2:", probs)


