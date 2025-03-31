import argparse
import os
# 定义全局变量
train_path = os.path.join(os.getcwd(), "checkpoints", "F_test_555_1560")
parser = argparse.ArgumentParser()

######################### 分割线 ##########################
parser.add_argument('--env_config', default=
                    {
                        'red_num': 1,
                        'blue_num': 1,
                        'render': 0,
                        'ip': '127.0.0.1',
                        'port': 8080,
                        'mode': 'train',
                        'excute_path': r'D:\Desktop\project_competition\platform\MM\windows\ZK.exe',
                        'step_num_max': 300
                    }, type=dict)


parser.add_argument('--state_size', default=20, type=int)
parser.add_argument('--action_size', default=4, type=int)
# PPO更新过程参数
parser.add_argument('--optimization_epochs', default=5, type=int)
parser.add_argument('--ac_delay', default=1, type=int)
parser.add_argument('--clamp', default=0.2, type=float)
parser.add_argument('--update_method', default='MC', type=str)
parser.add_argument('--tau', default=0.1, type=float)
parser.add_argument('--entropy_coef', default=0.00001, type=float)
parser.add_argument('--gamma', default=0.99, type=float)
parser.add_argument('--lamda', default=0.98, type=float)
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--actor_lr', default=1e-4, type=float)
parser.add_argument('--critic_lr', default=2e-4, type=float)
parser.add_argument('--min_actor_lr', default=1e-5, type=int)
parser.add_argument('--min_critic_lr', default=1e-5, type=int)
parser.add_argument('--lr_decay_T_max', default=5, type=int)
parser.add_argument('--default_std', default=0.5, type=float)
parser.add_argument('--min_std', default=0.3, type=float)
parser.add_argument('--std_decay_interval', default=3000, type=float)
# 数据采集
parser.add_argument('--tune', default=False, type=int)
parser.add_argument('--num_workers', default=1, type=int)
parser.add_argument('--per_collect_num', default=1, type=int)
parser.add_argument('--buffer_size', default=1e6, type=int)
parser.add_argument('--task_range_increment_interval', default=200, type=int)
# 训练与保存
parser.add_argument('--total_episode_num', default=60000, type=int)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--save_interval', default=100, type=int)
parser.add_argument('--test_interval', default=40, type=int)
parser.add_argument('--per_test_num', default=1, type=int)


# 尝试抗波动损失，惩罚相邻动作之间的剧烈变化
# 尝试添加标准差正则化损失，熵正则化损失
# 初期使用动作滤波、 逐步转向奖励平滑


# 对mean做处理而不是最后的动作值
# sigma值限制
# 叠加历史动作
