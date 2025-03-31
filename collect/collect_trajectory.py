"""
Author: Chenxu Qian
Email: qianchenxu@mail.nankai.edu.cn
用于收集训练数据，为后期模仿学习做铺垫
此外还可以用于测试胜率和负率
"""
import os
import sys

# # 在终端中运行代码前，先设置PYTHONPATH
# # 添加项目根目录到系统路径
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
# sys.path.append(project_root)
import time
import param
from ray.rllib.offline.json_writer import JsonWriter
from .collect_actor import collect_actor
import ray
import atexit
import datetime
import pandas as pd


@atexit.register
def exit():
    os.system('taskkill /f /im %s' % 'ZK.exe')
    os.system("ps -ef|grep ZK.x86_64|grep -v grep |awk '{print $2}'|xargs kill -9")


if __name__ == '__main__':
    # 以下几个是用来控制程序运行的变量
    num_workers = 100
    ray.init(num_cpus=num_workers, num_gpus=0)
    
    is_collect = True
    current_collect_num = 0
    save_path = 'sample_save_folder'

    args = param.parser.parse_args()
    
    # custom_model_config = {
        
    # }
    policy_name = 'PPO'

    save_folder = os.path.join(save_path, policy_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    save_folder = os.path.join(save_folder, current_time)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    writer = JsonWriter(save_folder, max_file_size=500*1024*1024)
    collect_num = current_collect_num
    truelly_collect_num = 0
    reward = 0
    count = 0
    
    policy_path = 'checkpoint/checkpoint_001700_0.8/checkpoint_001700'  # 8V8

    # 创建结果列表用于保存数据
    results = []
    
    # 为不同场景和模型组合设置不同的端口范围
    port_base = 8560
    model_index = 0
    
    # 循环测试每个场景和模型组合

    model_name = os.path.basename(os.path.dirname(policy_path))
    print(f"\n正在测试模型: {model_name}")
    print(f"使用端口范围: {current_port_start} - {current_port_start + num_workers - 1}")
    
    try:
        # 重置统计变量
        collect_num = 0
        truelly_collect_num = 0
        reward = 0
        count = 0
        
        actor_list = [collect_actor.remote(
            agent_id=i,
            env_config={
                "collect_usage": collect_usage,
                "excute_path": excute_path,
                "GFT_model_path": GFT_model_path,
                'red_num': red_num,
                'blue_num': blue_num,
                'port': i + current_port_start,
                'control_side': control_side,
                'mode': mode,
                'human_ai_reinforce': 1,
                'excute_step': excute_step,
                'stack_size': args.stack_size,
                'frame_feature_size': args.frame_feature_size,
                'excute_step_opponent': excute_step_opponent,
                'usage': 'collect', 'action_input_type': action_input_type,
                "render": render, 'policy_intention': policy_intention,
                'total_opponent_policy_name': total_opponent_policy_name,
                'total_opponent_policy_rate': total_opponent_policy_rate,
                # 'run_off_checkpoint': 'checkpoint\\checkpoint_007540_9.0\\checkpoint_007540',
                'history_opponent_model_num': history_opponent_model_num,
                'custom_model_config': custom_model_config,
                'init_model_path': init_model_path,
                'confront_scene': scene,
            },
            is_collect=is_collect,
            policy=policy,
            switch_policy_path=switch_policy_path) for i in range(num_workers)]

        task_list = []
        for actor in actor_list:
            task_list.append(actor.collect_one_episode.remote(collect_num))
            collect_num += 1
        
        begin = time.time()
        while truelly_collect_num < total_collect_num:
            done_id, task_list = ray.wait(task_list)
            data_dict, worker_id = ray.get(done_id)[0]
            if is_collect and data_dict["reward"] > 0:
                writer.write(data_dict['batch'])

            collect_num += 1
            truelly_collect_num += 1
            reward += data_dict['reward']
            count += data_dict['count']
            end = time.time()
            if data_dict['reward'] > 0:
                win_num += 1
            elif data_dict['reward'] == 0:
                tie_num += 1
            else:
                loss_num += 1
            # TODO 攻击改为攻击成功率，防御改为防御成功率
            print(f'Trully collect {truelly_collect_num} episodes. Cost time is {end - begin} s. '
                    f'Current Reward is {data_dict["reward"]} Average Reward is {reward / truelly_collect_num} '
                    f'Current Count is {data_dict["count"]} Average Count is {count / truelly_collect_num} '
                    f'Win_rate is {win_num/truelly_collect_num} Loss_rate is {loss_num/truelly_collect_num} '
                    f'Tie_rate is {tie_num/truelly_collect_num} Accelerate rate is {count * 0.2 * excute_step / (end - begin)}')
            # print(f'已测试 {truelly_collect_num} 局. 花费时间 {end - begin} s. '
            #       f'当前奖励 {data_dict["reward"]} 平均奖励 {reward / truelly_collect_num} '
            #       f'本局步数 {data_dict["count"]} 平均步数 {count / truelly_collect_num} '
            #       f'攻击成功率 {win_num / truelly_collect_num} 攻击失败率 {loss_num / truelly_collect_num} '
            #       f'加速比 {count * 0.2 * excute_step / (end - begin)}')
            # print(f'已测试 {truelly_collect_num} 局. 花费时间 {end - begin} s. '
            #       f'当前奖励 {data_dict["reward"]} 平均奖励 {reward / truelly_collect_num} '
            #       f'本局步数 {data_dict["count"]} 平均步数 {count / truelly_collect_num} '
            #       f'防御成功率 {win_num / truelly_collect_num} 防御失败率 {loss_num / truelly_collect_num} '
            #       f'加速比 {count * 0.2 * excute_step / (end - begin)}')
            task_list.append(actor_list[worker_id].collect_one_episode.remote(collect_num))
        
        # 计算并保存结果
        avg_reward = reward / truelly_collect_num
        win_rate = win_num / truelly_collect_num
        loss_rate = loss_num / truelly_collect_num
        tie_rate = tie_num / truelly_collect_num
        
        results.append({
            '模型': model_name,
            '场景': scene,
            '平均奖励': avg_reward,
            '胜率': win_rate,
            '负率': loss_rate,
            '平率': tie_rate
        })
        
        print(f"\n场景 {scene} 测试结果:")
        print(f"平均奖励: {avg_reward:.4f}")
        print(f"胜率: {win_rate:.4f}")
        print(f"负率: {loss_rate:.4f}")
        print(f"平率: {tie_rate:.4f}")
    
    except Exception as e:
        print(f"测试失败 - 模型: {model_name}, 场景: {scene}")
        print(f"错误信息: {str(e)}")
        results.append({
            '模型': model_name,
            '场景': scene,
            '平均奖励': None,
            '胜率': None,
            '负率': None,
            '平率': None,
            '错误': str(e)
        })
    
    finally:
        # 清理资源
        # 逐个终止 actor
        for actor in actor_list:
            ray.kill(actor)
        time.sleep(5)
        
        # # 根据操作系统执行不同的清理命令
        # if os.name == 'nt':  # Windows
        #     os.system('taskkill /f /im %s' % 'ZK.exe')
        # else:  # Linux
        #     os.system("ps -ef|grep ZK.x86_64|grep -v grep |awk '{print $2}'|xargs kill -9 2>/dev/null || true")
        # time.sleep(10)
    
    
    # 将结果写入CSV文件
    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    df = pd.DataFrame(results)
    csv_path = f'test_results_{current_time}.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n测试结果已保存至: {csv_path}")
