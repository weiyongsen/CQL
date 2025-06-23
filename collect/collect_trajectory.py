"""
Author: Chenxu Qian
Email: qianchenxu@mail.nankai.edu.cn
用于收集训练数据，为后期模仿学习做铺垫
此外还可以用于测试胜率和负率
"""
import os
import sys
import time
import param
import platform
from ray.rllib.offline.json_writer import JsonWriter
from collect_actor import CollectActor
import ray
import atexit
import datetime
import pandas as pd
import subprocess

# 在终端中运行代码前，先设置PYTHONPATH
# 添加项目根目录到系统路径
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
# sys.path.append(project_root)

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

if __name__ == '__main__':
    # 配置参数
    num_workers = 5  # 并行收集的worker数量
    ray.init()  # 初始化Ray，分配CPU资源
    
    is_collect = True  # 是否收集数据
    total_collect_num = 500  # 要收集的总轨迹数
    save_path = 'sample_save_folder'  # 数据保存路径

    args = param.parser.parse_args()
    env_config = args.env_config
    policy_name = 'PPO'  # 使用的策略名称

    # 创建保存目录
    save_folder = os.path.join(save_path, policy_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    save_folder = os.path.join(save_folder, current_time)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    writer = JsonWriter(save_folder, max_file_size=500*1024*1024)  # 创建JSON写入器
    
    # 初始化统计变量
    truelly_collect_num = 0  # 实际收集的轨迹数
    reward = 0  # 总奖励
    step = 0  # 总步数
    
    policy_path = r'D:\Desktop\CQL\sample_policy\model_train_0518_192803\actor40000_184.7115'
    results = []  # 存储测试结果
    
    # 端口配置
    port_base = 8560  # 基础端口号
    model_index = 0
    
    model_name = os.path.dirname(policy_path)
    print(f"\n正在测试模型: {model_name}")
    
    try:
        # 重置统计变量
        collect_num = 0
        truelly_collect_num = 0
        reward = 0
        step = 0
        success_count = 0  # 成功次数计数器
        
        # 创建多个Actor进行并行数据收集
        actor_list = [CollectActor.remote(
            agent_id=i,
            env_config={
                'red_num': env_config['red_num'],
                'blue_num': env_config['blue_num'],
                'state_size': args.state_size,
                'action_size': args.action_size,
                'render': env_config['render'],
                'ip': env_config['ip'],
                'port': port_base + i,  # 为每个worker分配不同的端口
                'mode': 'collect',
                'state_stack_num': env_config['state_stack_num'],
                'excute_path': get_executable_path(),
                'step_num_max': env_config['step_num_max'],
            },
            is_collect=is_collect,
            sample_policy_path=policy_path) for i in range(num_workers)]

        # 初始化任务列表
        task_list = []
        for actor in actor_list:
            task_list.append(actor.collect_one_episode.remote(collect_num))
            collect_num += 1
        
        begin = time.time()
        end = begin
        # 主循环：收集轨迹直到达到目标数量
        while truelly_collect_num < total_collect_num:
            # 等待任意一个任务完成
            done_id, task_list = ray.wait(task_list)
            data_dict, worker_id = ray.get(done_id)[0]
            
            # 只收集reward大于0的数据
            if is_collect and data_dict["reward_total"] > 0:
                writer.write(data_dict['batch'])
                success_count += 1

            # 更新统计信息
            collect_num += 1
            truelly_collect_num += 1
            reward += data_dict['reward_total']
            step += data_dict['step']
            end = time.time()

            # 打印进度信息
            print(f'Trully collect {truelly_collect_num}/{total_collect_num} episodes. '
                  f'Cost time is {end - begin:.2f} s. '
                  f'Current Reward is {data_dict["reward_total"]:.2f} '
                  f'Average Reward is {reward / truelly_collect_num:.2f} '
                  f'Current step is {data_dict["step"]} '
                  f'Average step is {step / truelly_collect_num:.2f} '
                  f'Success Rate is {success_count / truelly_collect_num:.2%}')

            # 添加新的收集任务
            task_list.append(actor_list[worker_id].collect_one_episode.remote(collect_num))
        
        # 计算最终统计结果
        avg_reward = reward / truelly_collect_num
        avg_step = step / truelly_collect_num
        success_rate = success_count / truelly_collect_num
        total_time = end - begin
        
        # 保存结果
        results.append({
            '模型': model_name,
            '采集局数': truelly_collect_num,
            '平均奖励': avg_reward,
            '平均步数': avg_step,
            '成功率': success_rate,
            '总奖励': reward,
            '总步数': step,
            '成功次数': success_count,
            '总耗时(秒)': total_time,
            '平均每局耗时(秒)': total_time / truelly_collect_num
        })
        
    except Exception as e:
        print(f"错误信息: {str(e)}")
        # 发生错误时保存当前结果
        results.append({
            '模型': model_name,
            '采集局数': truelly_collect_num,
            '平均奖励': avg_reward if 'avg_reward' in locals() else 0,
            '平均步数': avg_step if 'avg_step' in locals() else 0,
            '成功率': success_rate if 'success_rate' in locals() else 0,
            '总奖励': reward if 'reward' in locals() else 0,
            '总步数': step if 'step' in locals() else 0,
            '成功次数': success_count if 'success_count' in locals() else 0,
            '总耗时(秒)': total_time if 'total_time' in locals() else 0,
            '平均每局耗时(秒)': total_time / truelly_collect_num if 'total_time' in locals() and truelly_collect_num > 0 else 0
        })
    
    finally:
        # 清理资源
        print("\n正在清理资源...")
        # 终止所有Actor
        for actor in actor_list:
            try:
                ray.kill(actor)
            except Exception as e:
                print(f"终止actor时出错: {str(e)}")
        
        # 等待资源释放
        time.sleep(5)
        
        # 确保所有ZK进程都已终止
        kill_process_by_name('ZK.exe')
        kill_process_by_name('ZK.x86_64')
        
        print("资源清理完成")

    # 将结果保存到CSV文件
    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    df = pd.DataFrame(results)
    csv_path = f'sample_save_folder/csv/test_results_{current_time}.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n测试结果已保存至: {csv_path}")
