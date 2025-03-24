import gym
import numpy as np
import ray
import ray.rllib.algorithms.cql as cql

# ==== 你需要填写的路径 ====
checkpoint_path = "save_model/checkpoint_000091"  # 修改为你保存的 checkpoint 路径
env_name = "Pendulum-v1"

# ==== 初始化 Ray ====
ray.init(ignore_reinit_error=True)

# ==== 配置必须与训练时一致 ====
config = cql.DEFAULT_CONFIG.copy()
config["framework"] = "torch"
config["explore"] = False  # 关闭探索，使用确定性策略评估
config["num_workers"] = 0  # 只在主进程评估
config["env"] = env_name

# ==== 恢复算法 ====
algo = cql.CQL(config=config, env=env_name)
algo.restore(checkpoint_path)
print(f"✅ 成功加载模型：{checkpoint_path}")

# ==== 评估循环 ====
env = gym.make(env_name)

for episode in range(5):  # 跑 5 个 episode 看看效果
    obs = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = algo.compute_single_action(obs, explore=False)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        env.render()  # 可视化，如果在本地支持图形界面

    print(f"Episode {episode + 1} total reward: {total_reward:.2f}")

env.close()
ray.shutdown()
