import random
import time

import ray
from ray.rllib.algorithms.sac.sac_torch_policy import SACTorchPolicy
import gym
import numpy as np
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder, MultiAgentSampleBatchBuilder
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from custom_env.DogFight import Base_env
import math


@ray.remote(num_cpus=1)
class collect_actor():
    def __init__(self, agent_id, env_config, is_collect=True):
        self.is_collect = is_collect
        self.agent_id = agent_id
        self.env_config = env_config
        obs_space = gym.spaces.Box(low=-10, high=10, shape=(20,))
        action_space = gym.spaces.Box(-1, 1, shape=(4,))

        self.batch_builder_singleagent = SampleBatchBuilder()
        self.env = Base_env(config=env_config)


    def collect_one_episode(self, eps_id):

        obs = self.env.reset()
        prev_action = np.zeros_like(self.action_space.sample())
        prev_reward = 0
        done = False
        t = 0
        while not done:
            action = self.env.action_space.sample()
            # info
            new_obs, rew, done, info = self.env.step(action)
            
            if self.is_collect:
                self.batch_builder_singleagent.add_values(
                    t=t,
                    eps_id=eps_id,
                    agent_index=0,
                    obs=obs,
                    actions=action,
                    action_prob=1.0,  # put the true action probability here
                    action_logp=0.0,
                    rewards=rew,
                    prev_actions=prev_action,
                    prev_rewards=prev_reward,
                    dones=done,
                    infos=info,
                    new_obs=obs,
                )
            obs = new_obs
            prev_action = action
            prev_reward = rew
            t += 1

        self.env.close()
        batch = self.batch_builder_multiagent.build_and_reset() if self.is_collect else None
        return {'batch': batch,
                'reward': rew,
                'count': t
                }, self.agent_id


