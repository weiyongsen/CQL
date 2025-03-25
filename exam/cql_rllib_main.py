
from typing import Any, Dict
import gym
import ray
import numpy as np
from ray.rllib.algorithms import ppo
from ray.rllib.algorithms import cql
from ray.tune.logger import pretty_print
from cql.custom_cql_policy import CustomCQLPolicy
from ray.rllib.algorithms.cql.cql_torch_policy import CQLTorchPolicy
from ray.rllib.algorithms.cql import CQLConfig
from envs.zhikong import param
from envs.zhikong.custom_env.DogFight_mappo_cql_with_save_update import DogFight
import os
import atexit


@atexit.register
def exit():
    os.system('taskkill /f /im %s' % 'ZK.exe')
    os.system("ps -ef|grep ZK.x86_64|grep -v grep |awk '{print $2}'|xargs kill -9")


def train(args, train_config: Dict[str, Any], train_num: int, save_folder=None, checkpoint_path=None, train_class=None) -> None:
    """Train the agent with n_iters iterations."""
    if train_class is None:
        train_class = ppo.PPO
    agent = train_class(config=train_config, env=train_config["env"])
    if checkpoint_path:
        agent.restore(checkpoint_path)
        print('restore from:', checkpoint_path)
    for _ in range(10000):
    # while True:
        result = agent.train()
        train_num += 1
        if train_num % 1 == 0:
            print(pretty_print(result))
            if train_num % args.evaluation_interval == 0:
                try:
                    reward_mean = result['evaluation']['episode_reward_mean']
                except:
                    reward_mean = result['episode_reward_mean']
                checkpoint_path = agent.save(os.path.join(
                    save_folder, 'checkpoint_%06d_%.1f' % (train_num, reward_mean)))
                print(f"Checkpoint saved in {checkpoint_path}")


if __name__ == "__main__":
    args = param.parser.parse_args()
    args.excute_path = '/home/wys/projects/platform/1022/nolinux/ZK.x86_64' # server8 server3 server9
    # initialize ray
    # ray.init(local_mode=True)
    ray.init()
    red_num = args.red_num
    blue_num = args.blue_num
    d_model = 256
    body_model = 31 + 28
    entity_opponent_num = blue_num
    entity_opponent_model = 13
    entity_teammate_num = red_num - 1
    entity_teammate_model = 34 + 28
    weapon_num = 4
    weapon_model = 5
    weapon_num_state = 10
    weapon_model_state = 12
    obs_feature_size = body_model + entity_opponent_num * entity_opponent_model + \
                       entity_teammate_num * entity_teammate_model + weapon_num * weapon_model + \
                       entity_teammate_num + entity_opponent_num + weapon_num
    state_feature_size = body_model * (red_num + blue_num) + weapon_num_state * weapon_model_state + \
                         entity_teammate_num + entity_opponent_num + weapon_num_state
    args.frame_feature_size = obs_feature_size + state_feature_size
    observation_space = {f'red_{i}': gym.spaces.Box(low=-10, high=10, dtype=np.float32,
                                                    shape=(args.frame_feature_size * args.stack_size,))
                         for i in range(red_num)}
    if args.control_role == 'hierarchical_emergency':
        action_space = {f'red_{i}': gym.spaces.Discrete(32) for i in range(red_num)}  # Combined action space
    else:
        action_space = {f'red_{i}': gym.spaces.Discrete(32) for i in range(red_num)}
    # from envs.zhikong.agent.hier_top_mappo.mappo_torch_policy_attention_index import MAPPOTorchWithAttentionIndexPolicy as ppo_policy
    # from envs.zhikong.agent.hier_top_mappo.mappo_torch_policy_attention_index import MAPPOWithAttentionIndex as train_class
    total_opponent_policy_name = ['expert_attack', 'expert_defense', 'expendable', 'circuity', 'run_off']
    total_opponent_policy_rate = [0, 0, 1, 0, 0]
    policies = dict()
    if args.is_same_model:
        policies['default_policy'] = (CustomCQLPolicy, observation_space['red_0'], action_space['red_0'], {})
        def policy_mapping_fn(agent_id, episode, worker, **kwargs):
            return 'default_policy'
        mul_config = {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn,
            "policies_to_train": ['default_policy'],
        }
    else:
        for key, obs_space in observation_space.items():
            policies[key] = (CustomCQLPolicy, obs_space, action_space[key],
                             {})
        def policy_mapping_fn(agent_id, episode, worker, **kwargs):
            return agent_id
        mul_config = {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn,
            "policies_to_train": [f"red_{i}" for i in range(red_num)],
        }
    custom_model_config = {
        'd_model': d_model,
        'body_model': body_model,
        'entity_opponent_num': entity_opponent_num,
        'entity_opponent_model': entity_opponent_model,
        'entity_teammate_num': entity_teammate_num,
        'entity_teammate_model': entity_teammate_model,
        'weapon_num': weapon_num,
        'weapon_model': weapon_model,
        'weapon_num_state': weapon_num_state,
        'weapon_model_state': weapon_model_state,
        'dropout': 0,
        'obs_feature_size': obs_feature_size,
        'state_feature_size': state_feature_size,
        'supervised_checkpoint': args.supervised_model_path,
        'level': 'top'
    }

    config = {
        "env": DogFight,
        "input": 'data/zhikong_with_save_actions_trans',
        "env_config": {
            "red_num": red_num,
            "blue_num": blue_num,
            "render": args.render,
            'ip': '127.0.0.1',
            'port': 8310,
            'mode': 'train',
            'human_ai_reinforce': 1,
            'usage': 'train',
            "excute_path": args.excute_path,
            'stack_size': args.stack_size,
            "frame_feature_size": args.frame_feature_size,
            "multiagent": param.MultiAgent,
            'policy_intention': args.policy_intention,
            'min_trigger_time': 50,
            'max_trigger_time': 100,
            'step_method': 'event_triggered',
            'run_off_checkpoint': args.run_off_checkpoint,
            'excute_step': 50,
            'control_side': 'red',
            'is_used_rule': args.is_used_rule,
            'is_used_ai_assigned': args.is_used_ai_assigned,
            'is_used_flee_reward': args.is_used_flee_reward,
            'is_update_opponent_model': args.is_update_opponent_model,
            'total_opponent_policy_name': total_opponent_policy_name,
            'total_opponent_policy_rate': total_opponent_policy_rate,
            'save_model_path': args.save_folder,
            'init_model_path': args.init_model_path,
            'custom_model_config': custom_model_config,
            'is_oracle': args.is_oracle,
            'confront_scene': args.confront_scene,
            'control_role': args.control_role,
        },
        "num_gpus": args.num_gpus,
        "lr": 5e-4,
        # "n_step": 8,
        "num_workers": args.num_workers,
        'model': {'fcnet_activation': 'tanh',
                  'custom_model_config': custom_model_config},
        'evaluation_sample_timeout_s': 800,
        'evaluation_num_workers': args.evaluation_num_workers,
        'evaluation_interval': args.evaluation_interval,
        'evaluation_config': {
            'explore': False,
            "env_config": {
                "red_num": red_num,
                "blue_num": blue_num,
                "render": args.render,
                'control_role': args.control_role,
                'ip': '127.0.0.1',
                'port': 8380,
                'mode': 'eval',
                'usage': 'train',
                'stack_size': args.stack_size,
                "frame_feature_size": args.frame_feature_size,
                "excute_path": args.excute_path,
                "multiagent": param.MultiAgent,
                'policy_intention': args.policy_intention,
                'human_ai_reinforce': 1,
                'min_trigger_time': 50,
                'max_trigger_time': 100,
                'step_method': 'time_triggered',
                'excute_step': 50,
                'run_off_checkpoint': args.run_off_checkpoint,
                'control_side': 'red',
                'total_opponent_policy_name': total_opponent_policy_name,
                'total_opponent_policy_rate': total_opponent_policy_rate,
                'custom_model_config': custom_model_config,
                'save_model_path': args.save_folder,
                'init_model_path': args.init_model_path,
                'is_used_rule': args.is_used_rule,
                'is_used_ai_assigned': args.is_used_ai_assigned,
                'is_used_flee_reward': args.is_used_flee_reward,
                'is_update_opponent_model': args.is_update_opponent_model_evaluation,
                'is_oracle': args.is_oracle,
                'confront_scene': args.confront_scene,

            },
            'input': 'sampler',
        },
        # "evaluation_parallel_to_training": True,
        'evaluation_duration': args.evaluation_duration,
        "evaluation_duration_unit": "episodes",

        "batch_mode": "truncate_episodes",

        'seed': args.randomseed,
        "multiagent": mul_config,
        "framework": "torch",
        # 训练相关参数
        "ignore_worker_failures":args.ignore_worker_failures,
        "recreate_failed_workers":args.recreate_failed_workers,
        # CQL相关参数
        'temperature': 10,
        'min_q_weight': 0.01,
        'num_actions': 32,
        'bc_iters': 1,
        # 'lagrangian': True,
        "train_batch_size": 32,
        # 学习率
        "optimization": {
          "actor_learning_rate": 0.0001,
          "critic_learning_rate": 0.0003,
          "entropy_learning_rate": 0.0001
        }

    }


    # evaluate
    print("Start training.")
    print(f"当前训练场景：{args.confront_scene}")
    train(args, config, train_num=args.train_num,
          save_folder=args.save_folder,
          checkpoint_path=args.checkpoint,
          train_class=cql.CQL
          )
