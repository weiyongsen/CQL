import gym
import ray
import ray.rllib.algorithms.cql as cql

ray.init(ignore_reinit_error=True)
config = cql.DEFAULT_CONFIG.copy()
config["num_gpus"] = 0
config["num_workers"] = 1
config["framework"] = "torch"
algo = cql.CQL(config=config, env="Pendulum-v1")

# 加载之前保存的checkpoint（请将此路径替换为实际的checkpoint路径）
checkpoint_path = "save_model/your_checkpoint_directory_or_file"  
algo.restore(checkpoint_path)

# 使用加载的模型在环境中进行测试
env = gym.make("Pendulum-v1")
obs = env.reset()
done = False
while not done:
    action = algo.compute_action(obs)
    obs, reward, done, info = env.step(action)
    env.render()

ray.shutdown()
