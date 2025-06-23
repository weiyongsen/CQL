import ray
import torch
import numpy as np
from Net.cql_agent import CQLAgent

@ray.remote
class CQLWorker:
    def __init__(self, state_dim, action_dim, device, config):
        self.agent = CQLAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            **config
        )
        self.env = None  # 环境将在set_env中设置
        
    def set_env(self, env):
        self.env = env
        
    def collect_data(self, num_episodes):
        data = []
        for _ in range(num_episodes):
            state = self.env.reset()
            done = False
            episode_data = []
            
            while not done:
                action = self.agent.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                episode_data.append((state, action, reward, next_state, done))
                state = next_state
                
            data.extend(episode_data)
        return data
    
    def train(self, batch_size):
        return self.agent.train(batch_size)
    
    def get_weights(self):
        return {
            'actor': self.agent.actor.state_dict(),
            'critic': self.agent.critic.state_dict()
        }
    
    def set_weights(self, weights):
        self.agent.actor.load_state_dict(weights['actor'])
        self.agent.critic.load_state_dict(weights['critic'])

def create_workers(num_workers, state_dim, action_dim, device, config):
    workers = []
    for _ in range(num_workers):
        worker = CQLWorker.remote(state_dim, action_dim, device, config)
        workers.append(worker)
    return workers

def train_parallel(workers, num_episodes, batch_size):
    # 并行收集数据
    data_futures = [worker.collect_data.remote(num_episodes) for worker in workers]
    collected_data = ray.get(data_futures)
    
    # 合并所有数据
    all_data = []
    for data in collected_data:
        all_data.extend(data)
    
    # 并行训练
    train_futures = [worker.train.remote(batch_size) for worker in workers]
    train_results = ray.get(train_futures)
    
    return all_data, train_results

def average_weights(workers):
    # 获取所有worker的权重
    weight_futures = [worker.get_weights.remote() for worker in workers]
    weights = ray.get(weight_futures)
    
    # 计算平均权重
    avg_weights = {
        'actor': {},
        'critic': {}
    }
    
    for key in weights[0]['actor'].keys():
        avg_weights['actor'][key] = torch.stack([w['actor'][key] for w in weights]).mean(0)
    
    for key in weights[0]['critic'].keys():
        avg_weights['critic'][key] = torch.stack([w['critic'][key] for w in weights]).mean(0)
    
    # 设置平均权重到所有worker
    set_weight_futures = [worker.set_weights.remote(avg_weights) for worker in workers]
    ray.get(set_weight_futures) 