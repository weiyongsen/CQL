import math
import random
import numpy as np
import torch
from pylab import mpl

# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
# 设置正常显示符号
mpl.rcParams["axes.unicode_minus"] = False


def save_test(actor, critic, model_path, episode_train, reward_mean):
    torch.save(actor.state_dict(), model_path + '/actor{}_{}'.format(episode_train, reward_mean))
    torch.save(critic.state_dict(), model_path + '/critic{}_{}'.format(episode_train, reward_mean))


def angle_difference(current_angle, target_angle):
    diff = current_angle - target_angle
    diff = (diff + 180) % 360 - 180
    return diff


# 将action的list转为dict形式
def action_ltod(action_list, mode=0):  # mode=0 代表使用动作指令控制，2是内置控制器
    # 设置为不使用武器 副翼、升降舵、方向舵、油门
    action_dict = {
        'red':
            {'red_0': {'mode': mode, "fcs/aileron-cmd-norm": action_list[0],
                       "fcs/elevator-cmd-norm": action_list[1],
                       "fcs/rudder-cmd-norm": action_list[2], "fcs/throttle-cmd-norm": action_list[3],
                       "fcs/weapon-launch": 0, "change-target": 99,
                       "switch-missile": 0,
                       },
             },
        'blue':
            {'blue_0': {'mode': mode, "fcs/aileron-cmd-norm": 0,
                        "fcs/elevator-cmd-norm": 0,
                        "fcs/rudder-cmd-norm": 0, "fcs/throttle-cmd-norm": 0,
                        "fcs/weapon-launch": 0, "change-target": 99,
                        "switch-missile": 0,
                        },
             }
    }

    return action_dict


def action_ltod_mode2(action_list):  # mode=0 代表使用动作指令控制，2是内置控制器
    action_dict = {
        'red':
            {'red_0': {'mode': 2,
                       'target_altitude_ft': action_list[0],
                       'target_track_deg': action_list[1],
                       'target_velocity': action_list[2],
                       "fcs/weapon-launch": 0,
                       "switch-missile": 0,
                       },
             },
        'blue':
            {'blue_0': {'mode': 0, "fcs/aileron-cmd-norm": 0,
                        "fcs/elevator-cmd-norm": 0,
                        "fcs/rudder-cmd-norm": 0, "fcs/throttle-cmd-norm": 0,
                        "fcs/weapon-launch": 0, "change-target": 99,
                        "switch-missile": 0,
                        },
             }
    }

    return action_dict


def linear_scale(value, min_val, max_val):
    """线性归一化到[-1, 1]"""
    return 2 * (value - min_val) / (max_val - min_val) - 1


def sqrt_scale(value, scale_factor=500):
    """非线性归一化（平方根缩放）"""
    return 10 * np.sign(value) * np.sqrt(abs(value)) / np.sqrt(scale_factor)


# 将obs的dict转为list形式
def obs_process(obs_list):
    # 创建一个obs_list的副本
    processed_obs_list = np.array(obs_list)
    # 在副本上进行操作
    # 非线性归一化（sqrt_scale）：高度差、航向差、速度差等
    # processed_obs_list[0] = linear_scale(processed_obs_list[0], -2500, 2500)  # 高度差
    # processed_obs_list[1] = linear_scale(processed_obs_list[1], -100, 100)  # 航向差180
    # processed_obs_list[2] = linear_scale(processed_obs_list[2], -400, 400)  # 速度差
    # # 线性归一化：俯仰角、翻滚角、侧滑角
    # processed_obs_list[3] = linear_scale(processed_obs_list[3], -math.pi / 2, math.pi / 2)  # 俯仰角
    # processed_obs_list[4] = linear_scale(processed_obs_list[4], -math.pi, math.pi)  # 翻滚角
    # processed_obs_list[5] = linear_scale(processed_obs_list[5], -180, 180)  # 侧滑角
    # # 速度（u, v, w）线性归一化
    # processed_obs_list[6] = linear_scale(processed_obs_list[6], -400,400)  # u
    # processed_obs_list[7] = linear_scale(processed_obs_list[7], -100, 100)  # v
    # processed_obs_list[8] = linear_scale(processed_obs_list[8], -100, 100)  # w
    # # 角速度（pqr）线性归一化
    # processed_obs_list[9] = linear_scale(processed_obs_list[9],  -math.pi / 5, math.pi / 5)  # p
    # processed_obs_list[10] = linear_scale(processed_obs_list[10],  -math.pi / 5, math.pi / 5)  # q
    # processed_obs_list[11] = linear_scale(processed_obs_list[11],  -math.pi / 5, math.pi / 5)  # r

    processed_obs_list[0] = linear_scale(processed_obs_list[0], -1000, 1000)  # 高度差
    processed_obs_list[1] = linear_scale(processed_obs_list[1], -50, 50)  # 航向差180
    processed_obs_list[2] = linear_scale(processed_obs_list[2], -200, 200)  # 速度差
    # 线性归一化：俯仰角、翻滚角、侧滑角
    processed_obs_list[3] = linear_scale(processed_obs_list[3], -math.pi / 4, math.pi / 4)  # 俯仰角
    processed_obs_list[4] = linear_scale(processed_obs_list[4], -math.pi / 2, math.pi / 2)  # 翻滚角
    processed_obs_list[5] = linear_scale(processed_obs_list[5], -100, 100)  # 侧滑角
    # 速度（u, v, w）线性归一化
    processed_obs_list[6] = linear_scale(processed_obs_list[6], -400, 400)  # u
    processed_obs_list[7] = linear_scale(processed_obs_list[7], -50, 50)  # v
    processed_obs_list[8] = linear_scale(processed_obs_list[8], -50, 50)  # w
    # 角速度（pqr）线性归一化
    processed_obs_list[9] = linear_scale(processed_obs_list[9], -math.pi / 10, math.pi / 10)  # p
    processed_obs_list[10] = linear_scale(processed_obs_list[10], -math.pi / 10, math.pi / 10)  # q
    processed_obs_list[11] = linear_scale(processed_obs_list[11], -math.pi / 10, math.pi / 10)  # r

    # 油门[0,2]
    processed_obs_list[15] /= 2

    return np.clip(processed_obs_list, -5, 5)

    # processed_obs_list[0] = linear_scale(processed_obs_list[0], -5000, 5000)  # 高度差
    # processed_obs_list[1] = linear_scale(processed_obs_list[1], -150, 150)  # 航向差180
    # # 线性归一化：俯仰角、翻滚角、侧滑角
    # processed_obs_list[2] = linear_scale(processed_obs_list[2], -math.pi / 2, math.pi / 2)  # 俯仰角
    # processed_obs_list[3] = linear_scale(processed_obs_list[3], -math.pi, math.pi)  # 翻滚角
    # processed_obs_list[4] = linear_scale(processed_obs_list[4], -180, 180)  # 侧滑角
    # # 速度（u, v, w）线性归一化
    # processed_obs_list[5] = linear_scale(processed_obs_list[5], -1000, 1000)  # u
    # processed_obs_list[6] = linear_scale(processed_obs_list[6], -200, 200)  # v
    # processed_obs_list[7] = linear_scale(processed_obs_list[7], -200, 200)  # w
    # # 角速度（pqr）线性归一化
    # processed_obs_list[8] = linear_scale(processed_obs_list[8], -math.pi, math.pi)  # p
    # processed_obs_list[9] = linear_scale(processed_obs_list[9], -math.pi / 5, math.pi / 5)  # q
    # processed_obs_list[10] = linear_scale(processed_obs_list[10], -math.pi / 5, math.pi / 5)  # r



def clamp_list(input_list, min_value, max_value):
    return [max(min_value, min(x, max_value)) for x in input_list]


# 归一化优势函数
def normalize(x):
    mean = x.mean()
    std = x.std() + 1e-8
    return (x - mean) / std


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def data_process_complex(self, x):
    # 只有在训练的时候才会更新main和std
    if self.mode == "train" and self.cnt <= 50000:
        x = np.asarray(x)
        self.n += 1
        if self.n == 1:
            self.mean_value = x
        else:
            # 更新样本均值和方差
            old_mean = self.mean_value.copy()
            self.mean_value = old_mean + (x - old_mean) / self.n
            self.var_value = self.var_value + (x - old_mean) * (x - self.mean_value)
            # 状态归一化
        if self.n > 1:
            self.std_value = np.sqrt(self.var_value / (self.n - 1))
        else:
            self.std_value = self.mean_value

    # 归一化处理
    x = x - self.mean_value
    x = x / (self.std_value + 1e-8)
    x = np.clip(x, -10, +10)
    # print(self.n, x)

    return x


def get_gae(rewards, values, dones, gamma=0.99, lam=0.98):
    length_batch = len(rewards)
    # 初始化数组，用来存储MC目标和优势
    MC_target = torch.zeros_like(rewards)
    advantages = torch.zeros_like(rewards)
    # 初始时的运行返回值和运行优势
    running_returns = 0
    running_advantage = 0
    # 倒序计算GAE
    for t in reversed(range(length_batch)):
        # 下一状态的值估计 V(s_{t+1})
        next_value = values[t + 1] if t + 1 < length_batch else 0  # 如果是最后一个状态或done为1，则next_value为0
        # 计算当前时间步的TD误差 (delta_t)
        delta_t = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        # 更新优势 (Advantage)
        running_advantage = delta_t + gamma * lam * running_advantage * (1 - dones[t])
        # 将计算出的优势存储在advantages数组中
        advantages[t] = running_advantage
        # 计算当前状态的回报 (return_t)
        running_returns = rewards[t] + gamma * running_returns * (1 - dones[t])
        # 将回报存储在MC_target数组中
        MC_target[t] = running_returns
    # 归一化优势（避免方差过大）
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return MC_target, advantages


def func_coef(index, a=1, b=22.12586, c=0.00054, d=-0.035):
    # develop_a = 1
    # develop_b = 35.3087  # 22.12586#35.3087
    # develop_c = 0.00085  # 0.00054#0.00085
    # develop_d = 0
    index = index / 2
    coef = (a / (1 + b * np.exp(-c * index)) + d) / 0.9
    return min(coef, 1)
