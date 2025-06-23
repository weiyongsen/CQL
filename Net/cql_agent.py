import torch
import torch.nn.functional as F
import numpy as np
from .cql_networks import Actor, Critic
from .replay_buffer import ReplayBuffer

class CQLAgent:
    def __init__(self,
                 state_dim,
                 action_dim,
                 device,
                 hidden_dim=256,
                 discount=0.99,
                 tau=0.005,
                 min_q_weight=1.0,
                 num_random_actions=10,
                 bc_iters=500,
                 lr=3e-4):
        
        self.device = device
        self.discount = discount
        self.tau = tau
        self.min_q_weight = min_q_weight
        self.num_random_actions = num_random_actions
        self.bc_iters = bc_iters
        
        # 创建网络
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
        # 初始化目标网络
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # 创建经验回放缓冲区
        self.replay_buffer = ReplayBuffer()
        
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()
    
    def train(self, batch_size=256):
        if len(self.replay_buffer) < batch_size:
            return
        
        # 从经验回放中采样
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)
        
        # 计算目标Q值
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1 - done) * self.discount * target_q
        
        # 更新Critic
        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # CQL正则化项
        random_actions = torch.FloatTensor(
            batch_size * self.num_random_actions, action.shape[-1]
        ).uniform_(-1, 1).to(self.device)
        
        random_states = state.unsqueeze(1).repeat(1, self.num_random_actions, 1).view(
            batch_size * self.num_random_actions, state.shape[-1]
        )
        
        random_q1, random_q2 = self.critic(random_states, random_actions)
        random_q1 = random_q1.view(batch_size, self.num_random_actions, 1)
        random_q2 = random_q2.view(batch_size, self.num_random_actions, 1)
        
        random_q1 = random_q1.mean(dim=1)
        random_q2 = random_q2.mean(dim=1)
        
        cql_loss = torch.logsumexp(torch.cat([random_q1, random_q2], dim=1), dim=1).mean() - \
                  torch.cat([current_q1, current_q2], dim=1).mean()
        
        critic_loss = critic_loss + self.min_q_weight * cql_loss
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 更新Actor
        actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 软更新目标网络
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'cql_loss': cql_loss.item()
        }
    
    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict()
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer']) 