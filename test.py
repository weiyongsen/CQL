import ray
import ray.rllib.algorithms.cql as cql
from ray.tune.logger import pretty_print
from custom_env.DogFight import Base_env
import param
from datetime import datetime
import os
import psutil
import matplotlib.pyplot as plt
import numpy as np


# 初始化Ray
ray.init(ignore_reinit_error=True, local_mode=True)
args = param.parser.parse_args()

# 配置CQL算法（需要与训练时保持一致）
config = cql.DEFAULT_CONFIG.copy()
config.update({
    "num_workers": 0,  # 测试时只需要一个worker
    "framework": "torch",
    "env": Base_env,
    "env_config": {
        'red_num': 1,
        'blue_num': 1,
        'state_size': 20,
        'action_size': 4,
        'render': 0,  # 测试时可以打开可视化
        'ip': '127.0.0.1',
        'port': 8630,  # CQL算法使用的端口
        'mode': 'test',
        'excute_path': r'D:\Desktop\project_competition\platform\MM\windows\ZK.exe',
        'step_num_max': 300,
    },
    "explore": False,  # 关闭探索，使用确定性策略
})

# 创建CQL算法实例
algo = cql.CQL(config=config, env=Base_env)

# 创建环境实例（使用不同的端口）
test_env_config = config["env_config"].copy()
test_env_config['port'] = 8631  # 测试环境使用不同的端口
env = Base_env(test_env_config)

# 加载训练好的模型
checkpoint_path = r"save_model/2025-06-09_19-58-25/checkpoint_000751"  # 修改为您的checkpoint路径
algo.restore(checkpoint_path)

# checkpoint_path = r"save_model/2025-05-25_19-31-39/checkpoint_000851"  # 修改为您的checkpoint路径
# algo.restore(checkpoint_path)
print(f"✅ 成功加载模型：{checkpoint_path}")

# 评估循环
episode_data = []  # 存储每个episode的数据
for episode in range(1):  # 评估3个episode
    obs = env.reset(h_initial=9000, h_setpoint=10000,
                    psi_initial=180, psi_setpoint=280,
                    v_initial=1000, v_setpoint=1200)
    total_reward = 0
    done = False
    
    while not done:
        # action = algo.compute_single_action(obs, explore=False)
        action = algo.compute_single_action(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
    
    print(f"Episode {episode + 1} total reward: {total_reward:.2f}")
    
    # 保存每个episode的数据
    episode_data.append({
        'height': env.h_list,
        'heading': env.psi_list,
        'velocity': env.v_list,
        'target_height': env.h_setpoint,
        'target_heading': env.psi_setpoint,
        'target_velocity': env.v_setpoint
    })

# 设置matplotlib的全局样式
plt.style.use('seaborn-v0_8-paper')  # 使用学术论文风格
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['axes.edgecolor'] = 'black'

# 创建图表
fig = plt.figure(figsize=(12, 8))
fig.suptitle('save_model/2025-06-08_16-33-09/checkpoint_000201', fontsize=10, y=0.95)


# 定义颜色方案
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 使用专业的配色方案
target_color = '#d62728'  # 目标线使用红色

# 高度子图
ax1 = plt.subplot(3, 1, 1)
for i, data in enumerate(episode_data):
    ax1.plot(data['height'], color=colors[i], linewidth=2, label=f'Episode {i+1}')
    ax1.axhline(y=data['target_height'], color=target_color, linestyle='--', linewidth=1.5, label=f'Target (Ep {i+1})')
ax1.set_title('Height Control', pad=10)
ax1.set_xlabel('Time Steps')
ax1.set_ylabel('Height (ft)')
ax1.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black')
ax1.grid(True, linestyle='--', alpha=0.3)

# 航向角子图
ax2 = plt.subplot(3, 1, 2)
for i, data in enumerate(episode_data):
    ax2.plot(data['heading'], color=colors[i], linewidth=2, label=f'Episode {i+1}')
    ax2.axhline(y=data['target_heading'], color=target_color, linestyle='--', linewidth=1.5, label=f'Target (Ep {i+1})')
ax2.set_title('Heading Control', pad=10)
ax2.set_xlabel('Time Steps')
ax2.set_ylabel('Heading (deg)')
ax2.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black')
ax2.grid(True, linestyle='--', alpha=0.3)

# 速度子图
ax3 = plt.subplot(3, 1, 3)
for i, data in enumerate(episode_data):
    ax3.plot(data['velocity'], color=colors[i], linewidth=2, label=f'Episode {i+1}')
    ax3.axhline(y=data['target_velocity'], color=target_color, linestyle='--', linewidth=1.5, label=f'Target (Ep {i+1})')
ax3.set_title('Velocity Control', pad=10)
ax3.set_xlabel('Time Steps')
ax3.set_ylabel('Velocity (ft/s)')
ax3.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black')
ax3.grid(True, linestyle='--', alpha=0.3)

# 调整子图之间的间距
plt.tight_layout()

# 保存图片（高DPI以确保清晰度）
plt.savefig('test_results.png', dpi=300, bbox_inches='tight')
plt.close()

# 清理资源
print("正在清理资源...")
env.close()
algo.stop()  # 停止CQL算法
ray.shutdown()  # 关闭Ray

# 清理端口
def kill_process_by_port(port):
    for proc in psutil.process_iter(['pid', 'name', 'connections']):
        try:
            for conn in proc.connections():
                if conn.laddr.port == port:
                    print(f"正在终止端口 {port} 的进程 (PID: {proc.pid})")
                    proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

# 清理两个环境使用的端口
kill_process_by_port(8630)  # CQL算法环境端口
kill_process_by_port(8631)  # 测试环境端口

print("✅ 资源清理完成")