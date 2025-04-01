import ray
import ray.rllib.algorithms.cql as cql
from ray.tune.logger import pretty_print
from custom_env.DogFight import Base_env
import param
from datetime import datetime
import os

ray.init(ignore_reinit_error=True)
args = param.parser.parse_args()

# 创建保存路径
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_dir = os.path.join("save_model", timestamp)
os.makedirs(save_dir, exist_ok=True)

# 配置CQL算法
config = cql.DEFAULT_CONFIG.copy()
config.update({
    "num_workers": 4,
    "framework": "torch",
    "input": "collect/sample_save_folder/PPO/2025-04-01_09-45",
    "env": Base_env,
    "env_config": {
        'red_num': 1,
        'blue_num': 1,
        'state_size': 20,
        'action_size': 4,
        'render': 0,
        'ip': '127.0.0.1',
        'port': 8560,
        'mode': 'collect',
        'excute_path': r'D:\Desktop\project_competition\platform\MM\windows\ZK.exe',
        'step_num_max': 300,
    },
    "lr": 5e-4,
    "temperature": 10,
    "min_q_weight": 0.01,
    "num_actions": 10,
    "bc_iters": 1,
    "train_batch_size": 256,
    "optimization": {
        "actor_learning_rate": 0.0001,
        "critic_learning_rate": 0.0003,
        "entropy_learning_rate": 0.0001
    },
    # 新增配置
    "evaluation_interval": 50,
    "evaluation_duration": 10,
    "evaluation_config": {
        'red_num': 1,
        'blue_num': 1,
        'state_size': 20,
        'action_size': 4,
        'render': 0,
        'ip': '127.0.0.1',
        'port': 8620,
        'mode': 'eval',
        'excute_path': r'D:\Desktop\project_competition\platform\MM\windows\ZK.exe',
        'step_num_max': 300,
    },
})

# 创建CQL算法实例
algo = cql.CQL(config=config, env=Base_env)

# 训练循环
for i in range(1001):
    result = algo.train()
    print(pretty_print(result))

    if i % 50 == 0:
        checkpoint = algo.save(save_dir)
        print("checkpoint saved at", checkpoint)

ray.shutdown()