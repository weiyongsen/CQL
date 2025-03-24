import ray
import ray.rllib.algorithms.cql as cql
from ray.tune.logger import pretty_print

ray.init(ignore_reinit_error=True)

# 配置CQL算法
config = cql.DEFAULT_CONFIG.copy()
config["num_workers"] = 1
config["framework"] = "torch"
config["input"] = "D:\\Desktop\\CQL\\jsonwriter\\pendulum-out"  # 使用您生成的数据

# 创建CQL算法实例
algo = cql.CQL(config=config, env="Pendulum-v1")

# 训练循环
for i in range(10):
    result = algo.train()
    print(pretty_print(result))

    if i % 10 == 0:
        checkpoint = algo.save("save_model")
        print("checkpoint saved at", checkpoint)

ray.shutdown()