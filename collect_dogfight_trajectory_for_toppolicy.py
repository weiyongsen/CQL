"""
Author: Chenxu Qian
Email: qianchenxu@mail.nankai.edu.cn
用于收集训练数据，为后期模仿学习做铺垫
此外还可以用于测试胜率和负率
"""
import os
import sys

# 在终端中运行代码前，先设置PYTHONPATH
# 添加项目根目录到系统路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)
import random
import time
import param
from ray.rllib.offline.json_writer import JsonWriter
from util.collect_actor_for_top import collect_actor
from util.util import kill_process
import ray
import argparse
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
    # ray.init(local_mode=True)
    # TODO 测试:修改动作类型, action_input_type智能体为index,专家策略为dict,  policy和 policy_intension为对应策略
    action_input_type = 'dict'
    # action_input_type = 'index'
    current_collect_num = 0
    total_collect_num = 3000
    policy = 'switch_net' # 'expendable' 'circuity' 'run_off' 'expert_defense' 'switch_net' 'expert_attack'
    policy_intention = 'switch_net' # 'expendable' 'circuity' 'run_off' 'expert_defense' 'switch_net' 'expert_attack'
    confront_scene = 'common' # 'emergency_1'
    render = 0
    mode = 'train'
    is_collect = True
    collect_usage = 'top_policy'
    # excute_path = 'D:/learning_source/graduate/RLEnv/ZhiKong/class3_1021/1022/MMDome/windows/ZK.exe'
    excute_path = '/home/wys/projects/platform/1022/nolinux/ZK.x86_64' # server8
    # excute_path = '/home/qcx/projects/platform/20231019/nolinux/ZK.x86_64' # server7
    # 1V1 GFT
    # GFT_model_path = r"D:\learning_source\graduate\RLEnv\ZhiKong\nvn_human_ai_0918\models_1010\OptimalIndividuals\[Epoch_321]Individual(1.0).json"
    # GFT_model_path = "/home/wys/projects/code/gft/models_1010/OptimalIndividuals/[Epoch_321]Individual(1.0).json"
    # 2V2 GFT
    GFT_model_path = r"D:\learning_source\graduate\RLEnv\ZhiKong\nvn_human_ai_0918\models_11_10_2V2\OptimalIndividuals\[Epoch_466]Individual(2.0).json"
    # GFT_model_path = "/home/wys/projects/code/gft/models_11_10_2V2/OptimalIndividuals/[Epoch_466]Individual(2.0).json"
    # 4V4 GFT
    # GFT_model_path = "/home/wys/projects/code/gft/models_11_11_4V4/OptimalIndividuals/[Epoch_333]Individual(3.0).json"
    # GFT_model_path = "/home/wys/project/nvn_human_ai_0918/models_11_11_4V4/OptimalIndividuals/[Epoch_333]Individual(3.0).json"
    # 6V6 GFT
    # GFT_model_path = "/home/wys/projects/code/gft/models_11_12_6V6/OptimalIndividuals/[Epoch_397]Individual(4.0).json"
    # GFT_model_path = "/home/wys/project/nvn_human_ai_0918/models_11_12_6V6/OptimalIndividuals/[Epoch_397]Individual(4.0).json"
    # 8V8 GFT
    # GFT_model_path = r"D:\learning_source\graduate\RLEnv\ZhiKong\nvn_human_ai_0918\models_1014第二次训练\OptimalIndividuals\[Epoch_497]Individual(8.0).json"
    # GFT_model_path = "/home/wys/project/nvn_human_ai_0918/models_1014第二次训练/OptimalIndividuals/[Epoch_395]Individual(7.0).json"
    excute_step = 50
    if policy != 'switch_net':
        excute_step = 1

    excute_step_opponent = 50
    control_side = 'red'
    save_path = 'D:\\experiment\\sample_save_folder'

    args = param.parser.parse_args()
    red_num = 8
    blue_num = 8
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
    # TODO 测试:修改对手
    total_opponent_policy_name = ['expert_attack', 'expert_defense', 'expendable', 'circuity', 'run_off', 'GFT']
    # total_opponent_policy_rate = [0, 0, 1/3, 1/3, 1/3, 0]
    total_opponent_policy_rate = [0, 1, 0, 0, 0, 0]

    if total_opponent_policy_rate[-1] == 1:
        excute_step_opponent = 10
    policy_name = f"{policy}_vs_{total_opponent_policy_name}"

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
    win_num = 0
    loss_num = 0
    tie_num = 0
    history_opponent_model_num = 10
    init_model_path = None
    # switch_policy_path = 'checkpoint/model/1v1/checkpoint_000500_1.0/checkpoint_000500'  # 1V1
    # switch_policy_path = 'checkpoint/model/2v2/checkpoint_003000_1.0/checkpoint_003000'  # 2V2
    # switch_policy_path = 'checkpoint/model/4v4/checkpoint_000700_2.9/checkpoint_000700'  # 4V4
    # switch_policy_path = 'checkpoint/model/6v6/checkpoint_000920_4.2/checkpoint_000920'  # 6V6
    switch_policy_path = 'checkpoint/checkpoint_001700_0.8/checkpoint_001700'  # 8V8

    # 定义要测试的场景和模型
    scenes = ['emergency_1']
    model_paths = [
        # '/home/wys/projects/code/prompt_dt_mujoco/checkpoint/8v8/checkpoint_001700_0.8/checkpoint_001700',  # 8V8
        # '/home/wys/projects/code/prompt_dt_mujoco/checkpoint/突发MAPPOcheckpoint_000200_-0.3/checkpoint_000200', #突发8V8
        '/home/wys/projects/code/prompt_dt_mujoco/checkpoint/0304突发MAPPOcheckpoint_000100_0.7/checkpoint_000100', #突发1_8V8

    ]
    # 创建结果列表用于保存数据
    results = []
    
    # 为不同场景和模型组合设置不同的端口范围
    port_base = 8280
    model_index = 0
    
    # 循环测试每个场景和模型组合
    for switch_policy_path in model_paths:
        model_name = os.path.basename(os.path.dirname(switch_policy_path))
        print(f"\n正在测试模型: {model_name}")
        
        for scene in scenes:
            print(f"\n测试场景: {scene}")
            current_port_start = port_base + (model_index * len(scenes) + scenes.index(scene)) * num_workers
            print(f"使用端口范围: {current_port_start} - {current_port_start + num_workers - 1}")
            
            try:
                # 重置统计变量
                collect_num = 0
                truelly_collect_num = 0
                reward = 0
                count = 0
                win_num = 0
                loss_num = 0
                tie_num = 0
                
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
        
        model_index += 1
    
    # 将结果写入CSV文件
    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    df = pd.DataFrame(results)
    csv_path = f'test_results_{current_time}.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n测试结果已保存至: {csv_path}")
