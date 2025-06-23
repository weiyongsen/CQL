import os
import sys
import ray
import numpy as np
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from custom_env.DogFight import Base_env
from Net.PPO import Actor
import torch


# 添加项目根目录到系统路径
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.append(project_root)


@ray.remote(num_cpus=1)  # 每个Actor分配1个CPU核心
class CollectActor():
    """
    数据收集Actor类
    负责与环境交互并收集轨迹数据
    """
    def __init__(self, agent_id, env_config, is_collect=True, sample_policy_path=None):
        """
        初始化Actor
        Args:
            agent_id: Actor的唯一标识符
            env_config: 环境配置参数
            is_collect: 是否收集数据
            sample_policy_path: 预训练策略的路径
        """
        self.is_collect = is_collect
        self.agent_id = agent_id
        self.env_config = env_config

        self.batch_builder_singleagent = SampleBatchBuilder()  # 用于构建轨迹批次
        self.env = Base_env(env_config)  # 创建环境实例
        self.mode = env_config['mode']
        # 创建网络并移动到CPU
        self.device = torch.device("cpu")
        self.actor = Actor(env_config['state_size'] * env_config['state_stack_num'], env_config['action_size']).to(self.device)  # 创建Actor网络
        # 加载预训练策略
        if sample_policy_path is not None:
            self.actor.load_state_dict(torch.load(sample_policy_path))

    def collect_one_episode(self, eps_id):
        """
        收集一个完整的轨迹
        Args:
            eps_id: 轨迹的唯一标识符
        Returns:
            dict: 包含轨迹数据和统计信息的字典
            int: Actor的ID
        """
        # 重置环境，设置初始状态
        obs = self.env.reset(h_initial=9000, h_setpoint=10000, psi_initial=180,
                           psi_setpoint=280, v_initial=1000, v_setpoint=1200)
        prev_action = np.zeros(4)  # 初始化前一个动作
        prev_reward = 0  # 初始化前一个奖励
        reward_total = 0  # 累计奖励
        done = False
        t = 0  # 时间步计数器

        while not done:
            # 使用Actor网络选择动作
            with torch.no_grad():
                state_tensor = torch.FloatTensor(obs).to(self.device)
                action_tensor, action_log_prob_tensor = self.actor.sample_action(state_tensor)
                action = action_tensor.cpu().numpy()
                action_log_prob = action_log_prob_tensor.cpu().item()
            # 与环境交互
            new_obs, reward, done, info = self.env.step(action)

            if self.is_collect:
                # 计算动作概率密度值，用于重要性采样
                action_prob = np.exp(action_log_prob)
                # 将轨迹数据添加到批次构建器中
                self.batch_builder_singleagent.add_values(
                    t=t,
                    eps_id=eps_id,
                    obs=obs,
                    actions=action,
                    rewards=reward,
                    dones=done,
                    new_obs=new_obs,
                )
            # 更新状态
            obs = new_obs
            prev_action = action
            prev_reward = reward
            reward_total += reward
            t += 1

        self.env.close()  # 关闭环境
        # 构建并重置批次
        batch = self.batch_builder_singleagent.build_and_reset() if self.is_collect else None
        return {'batch': batch,
                'reward_total': reward_total,
                'step': t
                }, self.agent_id


if __name__ == '__main__':
    # 测试代码：使用单个worker测试数据收集功能
    import json
    from ray.rllib.offline.json_writer import JsonWriter
    ray.init(local_mode=True)  # 本地模式初始化Ray
    writer = JsonWriter("D:\Desktop\CQL\collect\sample_save_folder\collect_actor_test", max_file_size=500 * 1024 * 1024)
    # 创建测试Actor
    actor = CollectActor.remote(agent_id=0,
                              env_config={
                                  'red_num': 1,
                                  'blue_num': 1,
                                  'state_size': 20,
                                  'action_size': 4,
                                  'render': 0,
                                  'ip': '127.0.0.1',
                                  'port': 8080,
                                  'mode': 'collect',
                                  'state_stack_num': 2,
                                  'excute_path': r'D:\Desktop\project_competition\platform\MM\windows\ZK.exe',
                                  'step_num_max': 300,
                              },
                              is_collect=True,
                              sample_policy_path=r'D:\Desktop\CQL\sample_policy\model_train_0518_192803\actor40000_184.7115')
    # 收集一条轨迹
    results, agent_id = ray.get(actor.collect_one_episode.remote(0))

    # 保存结果
    writer.write(results['batch'])
    print(results['reward_total'])
    ray.kill(actor)  # 清理资源
