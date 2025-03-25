import random
import time

import ray
from ray.rllib.algorithms.sac.sac_torch_policy import SACTorchPolicy
import gym
import numpy as np
from util.util_actor import switch_actor_policy, actor_policy, switch_actor_multiagent_policy_for_collect
from util.util import get_min_distance_index
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder, MultiAgentSampleBatchBuilder
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from custom_env.DogFight_mappo import DogFight
import math


@ray.remote(num_cpus=1)
class collect_actor():
    def __init__(self, agent_id, env_config, is_collect=True, policy='expendable',
                 switch_policy_path=None, attdef_policy_path=None):
        self.is_collect = is_collect
        self.agent_id = agent_id
        self.env_config = env_config
        self.stack_size = env_config['stack_size']
        self.policy = policy
        self.control_side = env_config['control_side']
        self.opponent_side = 'blue' if self.control_side == 'red' else 'red'
        obs_space = gym.spaces.Box(low=-10, high=10, dtype=np.float32,
                                   shape=(env_config['frame_feature_size'] * self.stack_size,))
        action_space = gym.spaces.Discrete(27)
        policies = {
            f'red_{i}': SACTorchPolicy(obs_space=obs_space,
                                       action_space=action_space,
                                       config=
                                       {
                                         'custom_model_config': env_config['custom_model_config'],
                                         'hiddens': [256, 256], 'fcnet_hiddens': [256, 256], 'n_step': 8
                                       }) for i in range(env_config['custom_model_config']['entity_teammate_num'] + 1)
        }
        model_config = {
            'custom_model_config': env_config['custom_model_config'],
            'hiddens': [256, 256],
            'fcnet_hiddens': [256, 256],
            'activation': 'tanh',
            'fcnet_activation': 'tanh',
        }
        self.switch_net_multiagent = switch_actor_multiagent_policy_for_collect(
            obs_space=gym.spaces.Box(low=-10, high=10, dtype=np.float32,
                                     shape=(env_config['frame_feature_size'] * self.stack_size,)),
            action_space=gym.spaces.Discrete(4),
            model_config=model_config
        )
        self.switch_net_multiagent.restore(switch_policy_path)
        self.policy_intention = env_config['policy_intention']
        multiagent = True
        self.batch_builder_multiagent = MultiAgentSampleBatchBuilder(policies, False, DefaultCallbacks())
        self.batch_builder_singleagent = SampleBatchBuilder()
        env_config['multiagent'] = multiagent
        self.env = DogFight(config=env_config)

    def collect_one_episode(self, eps_id, policy_name=None):
        if self.env_config['collect_usage'] == 'mid_policy':
            return self.collect_one_episode_for_mid_policy(eps_id, policy_name)
        else:
            return self.collect_one_episode_for_top_policy(eps_id, policy_name)

    def collect_one_episode_for_mid_policy(self, eps_id, policy_name=None):
        control_side = self.control_side
        opponent_side = self.opponent_side
        obs = self.env.reset()
        red_num = self.env_config['custom_model_config']['entity_teammate_num'] + 1
        prev_reward = {f'red_{i}': 0 for i in range(red_num)}
        prev_reward['__all__'] = 0
        rew = {f'red_{i}': 0 for i in range(red_num)}
        rew['__all__'] = 0
        t = 0
        while not self.env.is_done['__all__']:
            action, action_info_dict = self.switch_net_multiagent.compute_action(obs)
            # info
            new_obs, rew, done, info = self.env.step(action)
            if self.is_collect:
                for _, action_info_dict in info.items():
                    for iid_, key_ in enumerate([f'red_{i}' for i in range(red_num)]):
                        self.batch_builder_multiagent.add_values(
                            policy_id=key_,
                            t=t,
                            eps_id=eps_id,
                            agent_index=iid_,
                            obs=obs[key_],
                            actions=action_info_dict[key_]['move'],
                            actions_move=action_info_dict[key_]['move'],
                            action_logp=0.0,
                            action_prob=1.0,
                            actions_weapon=action_info_dict[key_]['weapon'],
                            action_logp_weapon=0.0,
                            action_prob_weapon=1.0,
                            actions_index=action_info_dict[key_]['index'],
                            action_logp_index=0.0,
                            action_prob_index=1.0,
                            rewards=rew[key_],
                            dones=self.env.is_done['__all__'],
                            infos=info,
                            new_obs=new_obs[key_],
                            agent_id=key_,
                            prev_rewards=prev_reward[key_],
                        )
            obs = new_obs
            prev_reward = rew
            t += 1
        # print('Env step is', t)
        reward_tot = 0
        for key, rew_tmp in rew.items():
            reward_tot += rew_tmp
        self.env.close()
        batch = self.batch_builder_multiagent.build_and_reset() if self.is_collect else None
        return {'batch': batch,
                'reward': reward_tot,
                'count': t
                }, self.agent_id

    def collect_one_episode_for_top_policy(self, eps_id, policy_name=None):
        control_side = self.control_side
        opponent_side = self.opponent_side
        obs = self.env.reset()
        red_num = self.env_config['custom_model_config']['entity_teammate_num'] + 1
        prev_action_index = {f'red_{i}': 0 for i in range(red_num)}
        prev_action_strategy = {f'red_{i}': 0 for i in range(red_num)}
        prev_reward = {f'red_{i}': 0 for i in range(red_num)}
        prev_reward['__all__'] = 0
        rew = {f'red_{i}': 0 for i in range(red_num)}
        rew['__all__'] = 0
        t = 0
        while not self.env.is_done['__all__']:
            # TODO 测试: 己方为switch_net获取网络输出的动作, 己方为其他策略获取对应策略输出的动作
            if self.policy_intention == 'switch_net':
                action, action_info_dict = self.switch_net_multiagent.compute_action(obs)
            else:
                action = {}
                used_opponent_num = self.env.used_opponent_num
                used_control_num = self.env.used_control_num
                for control_index in range(used_control_num):
                    attack_index = get_min_distance_index(self.env.last_obs_tot, self.control_side, control_index,
                                                          used_opponent_num)
                    action_single , _ = self.env.expert_policy.get_policy(self.policy)(
                        obs=self.env.last_obs_tot, is_done=self.env.is_done,
                        flag=self.env.control_side, value_oracle=[],
                        step_num=self.env.step_num, control_side=self.env.control_side,
                        switch_action=0, flag_index=control_index, flag_opponent_index=attack_index,
                        is_change_target=True)
                    action.update(action_single)
            # info
            new_obs, rew, done, info = self.env.step(action)
            # print(1, action_info_dict['action_index'])
            # target_index_chosen = info["target_index_chosen"]
            # strategy_chosen = info["strategy_chosen"]
            # for red_name in [f'red_{i}' for i in range(red_num)]:
            #     if target_index_chosen[red_name] != -1:
            #         action_info_dict['action_index'][red_name] = target_index_chosen[red_name]
            #     if strategy_chosen[red_name] != -1:
            #         action_info_dict['action_strategy'][red_name] = strategy_chosen[red_name]
            # print(2, action_info_dict['action_index'])

            if self.is_collect:
                for iid_, key_ in enumerate([f'red_{i}' for i in range(red_num)]):
                    self.batch_builder_multiagent.add_values(
                        policy_id=key_,
                        t=t,
                        eps_id=eps_id,
                        agent_index=iid_,
                        obs=obs[key_],
                        actions=action_info_dict['action_index'][key_],
                        actions_index=action_info_dict['action_index'][key_],
                        actions_strategy=action_info_dict['action_strategy'][key_],
                        action_logp=action_info_dict['action_logp_strategy'][key_],
                        action_dist_inputs=action_info_dict['action_dist_strategy'][key_],
                        opponent_action=action_info_dict['opponent_action'][key_],
                        rewards=rew[key_],
                        dones=self.env.is_done['__all__'],
                        infos=info,

                        new_obs=new_obs[key_],
                        agent_id=key_,
                        action_prob=action_info_dict['action_prob_strategy'][key_],
                        action_prob_strategy=action_info_dict['action_prob_strategy'][key_],
                        action_prob_index=action_info_dict['action_prob_index'][key_],
                        action_logp_strategy=action_info_dict['action_logp_strategy'][key_],
                        action_logp_index=action_info_dict['action_logp_index'][key_],
                        action_dist_inputs_strategy=action_info_dict['action_dist_strategy'][key_],
                        action_dist_inputs_index=action_info_dict['action_dist_index'][key_],
                        prev_actions_index=prev_action_index[key_],
                        prev_action_strategy=prev_action_strategy[key_],
                        prev_rewards=prev_reward[key_],
                    )
            obs = new_obs
            if self.is_collect:
                prev_action_index = action_info_dict['action_index']
                prev_action_strategy = action_info_dict['action_strategy']
            prev_reward = rew
            t += 1
        # print('Env step is', t)
        reward_tot = 0
        for key, rew_tmp in rew.items():
            reward_tot += rew_tmp
        self.env.close()
        batch = self.batch_builder_multiagent.build_and_reset() if self.is_collect else None
        return {'batch': batch,
                'reward': reward_tot,
                'count': t
                }, self.agent_id


