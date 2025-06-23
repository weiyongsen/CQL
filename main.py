import ray
import os
import platform
import subprocess
import atexit
import ray.rllib.algorithms.cql as cql
from ray.tune.logger import pretty_print
from datetime import datetime
from custom_env.DogFight import Base_env
import param


def kill_process_by_name(process_name):
    """
    跨平台终止进程函数
    Args:
        process_name: 要终止的进程名称
    """
    system = platform.system().lower()
    try:
        if system == 'windows':
            # Windows系统使用taskkill命令
            subprocess.run(['taskkill', '/F', '/IM', process_name],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
        elif system == 'linux':
            # Linux系统使用pkill命令
            subprocess.run(['pkill', '-f', process_name],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
        else:
            print(f"不支持的操作系统: {system}")
    except Exception as e:
        print(f"终止进程 {process_name} 时出错: {str(e)}")


def get_executable_path():
    """
    根据操作系统返回正确的可执行文件路径
    Returns:
        str: 可执行文件的路径
    Raises:
        ValueError: 如果操作系统不支持
    """
    system = platform.system().lower()
    if system == 'windows':
        return r'D:\Desktop\project_competition\platform\MM\windows\ZK.exe'
    elif system == 'linux':
        return './ZK.x86_64'  # Linux下的相对路径
    else:
        raise ValueError(f"不支持的操作系统: {system}")


@atexit.register
def exit():
    """程序退出时清理资源，确保所有ZK进程都被终止"""
    kill_process_by_name('ZK.exe')
    kill_process_by_name('ZK.x86_64')


ray.init(ignore_reinit_error=True)
args = param.parser.parse_args()
env_config = args.env_config
# 创建保存路径
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_dir = os.path.join("save_model", timestamp)
os.makedirs(save_dir, exist_ok=True)

# 配置CQL算法
config = {
    "num_workers": 0,
    "framework": "torch",
    "input": "collect/sample_save_folder/PPO/2025-06-09_17-13",     # 离线数据
    "input_config": {"format": "json"},
    "env": Base_env,
    "env_config": {
        'red_num': env_config['red_num'],
        'blue_num': env_config['blue_num'],
        'state_size': args.state_size,
        'action_size': args.action_size,
        'render': env_config['render'],
        'ip': env_config['ip'],
        'port': 8560,  # 为每个worker分配不同的端口
        'mode': 'train',
        'state_stack_num': env_config['state_stack_num'],
        'excute_path': get_executable_path(),
        'step_num_max': env_config['step_num_max'],
    },
    "lr": 5e-4,
    # eval配置
    'evaluation_num_workers': 0,
    "evaluation_interval": 10,      # 评估间隔
    "evaluation_duration": 5,      # 评估时采取几个episode
    "evaluation_config": {
        "input": "sampler",
        "explore": False,
        'env_config': {
            'red_num': env_config['red_num'],
            'blue_num': env_config['blue_num'],
            'state_size': args.state_size,
            'action_size': args.action_size,
            'render': env_config['render'],
            'ip': env_config['ip'],
            'port': 8620,
            'mode': 'eval',
            'state_stack_num': env_config['state_stack_num'],
            'excute_path': get_executable_path(),
            'step_num_max': env_config['step_num_max'],
        },
    },
    'normalize_actions': True,
    # CQL
    "min_q_weight": 1,          # 增强保守性
    "train_batch_size": 512,     # 增大批大小
    "temperature": 1,           # 调整温度参数
    "bc_iters": 500,                # 增加BC轮次
    "optimization": {
        "actor_learning_rate": 3e-3,  # 降低学习率
        "critic_learning_rate": 3e-3,
        "entropy_learning_rate": 1e-6
    },
}

# 创建CQL算法实例
algo = cql.CQL(config=config, env=Base_env)

# 训练循环
for i in range(1001):
    result = algo.train()
    print(pretty_print(result))
    # print(i, result)

    if i % 50 == 0:
        checkpoint = algo.save(save_dir)
        print("checkpoint saved at", checkpoint)

ray.shutdown()