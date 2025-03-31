import io
import math
import socket
import json
import random
import os
import time
import zipfile

import numpy as np
import gym
import pandas as pd

from custom_env import Reward
from util.util import angle_difference, action_ltod, obs_process, action_ltod_mode2

base_class = gym.Env


class Base_env(base_class):
    """
    (1) 动作输入指令: 字典格式
        Action_input: Dict
    {
        'red':{
            'red_0':{
                'fcs/aileron-cmd-norm':  Float   副翼指令      [-1, 1]
                'fcs/elevator-cmd-norm': Float   升降舵指令    [-1, 1]
                'fcs/rudder-cmd-norm':   Float   方向舵指令    [-1, 1]
                'fcs/throttle-cmd-norm': Float   油门指令      [ 0, 1]
                'fcs/weapon-launch':     Enum    导弹发射指令
                                        0-不发射 (Don't launch)
                                        1-发射导弹 (Launch missile)
                                        2-发射子弹 (Launch bullet)
                'fcs/change-target':     Enum     切换目标指令
                                        99-不变 (Don't change)
                                        88-由内部程序控制 (Controlled by procedure)
                                        0/1/12/012/0134-优先锁定目标机编号
                'fcs/switch-missile':    Bool    导弹切换指令
                        0-不变 (Don't switch)
                        1-切换 (Switch)
            }
        }
        'blue':{
            'blue_0':{
                与上述格式相同 The same as above
            }
        }
    }
    (2) 初始化指令(init 与 reset)
        {
        'flag': {
            'init': {
                'render':                   Bool        是否显示可视化界面
                }},
        'red': {
            'red_0': {
                "ic/h-sl-ft":               Float       初始高度 [ft]
                "ic/terrain-elevation-ft":  Float       初始地形高度 [ft]
                "ic/long-gc-deg":           Float       初始经度
                "ic/lat-geod-deg":          Float       初始纬度
                "ic/u-fps":                 Float       机体坐标系x轴速度 [ft/s]
                "ic/v-fps":                 Float       机体坐标系y轴速度 [ft/s]
                "ic/w-fps":                 Float       机体坐标系z轴速度 [ft/s]
                "ic/p-rad_sec":             Float       翻滚速率 [rad/s]
                "ic/q-rad_sec":             Float       俯仰速率 [rad/s]
                "ic/r-rad_sec":             Float       偏航速率 [rad/s]
                "ic/roc-fpm":               Float       初始爬升速率 [ft/min]
                "ic/psi-true-deg":          Float       初始航向 [度]
                }},
        'blue': {
            'blue_0': the same as above}

    """
    reset_data_example = {
        'red': {
            'red_0': {
                "ic/h-sl-ft": 5000, "ic/terrain-elevation-ft": 0,
                "ic/long-gc-deg": 0, "ic/lat-geod-deg": 0,
                "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0,
                "ic/p-rad_sec": 0, "ic/q-rad_sec": 0, "ic/r-rad_sec": 0,
                "ic/roc-fpm": 0, "ic/psi-true-deg": 0},
        },
        'blue': {
            'blue_0': {
                "ic/h-sl-ft": 10000, "ic/terrain-elevation-ft": 0,
                "ic/long-gc-deg": 0, "ic/lat-geod-deg": 0,
                "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0,
                "ic/p-rad_sec": 0, "ic/q-rad_sec": 0, "ic/r-rad_sec": 0,
                "ic/roc-fpm": 0, "ic/psi-true-deg": 0},
        }}

    IP = '127.0.0.1'
    PORT = 8000
    INITIAL = False
    RENDER = 0

    def __init__(self, config=None, render=0):
        """
        :param config: 从RLlib中传输过来的参数，在这个config里面可以传递希望定制的环境变量，譬如ip，render等
        :param ip:     开启软件的ip地址
        :param port:   开启软件的端口号，注意端口号应该是四位的
        :param render: 是否可视化
        :param excute_path: 主要是根据软件位置的不同
        """
        # 默认配置参数
        default_config = {
            'ip': '127.0.0.1',
            'port': 8000,
            'red_num': 1,
            'blue_num': 1,
            'state_size': 20,
            'action_size': 4,
            'scenes': 3,
            'mode': 'train',
            'step_num_max': 300,
            'excute_path': r'C:\Users\Absol\Desktop\ZK\ZK_v2.6\build\windows\ZK.exe',
            'render': render
        }

        # 更新配置参数
        if config is not None:
            # 使用字典的get方法更新配置，如果config中没有对应的键，则使用默认值
            for key in default_config:
                default_config[key] = config.get(key, default_config[key])
            
            # 处理worker_index
            try:
                default_config['port'] += config.worker_index
            except:
                pass

        # 将配置参数赋值给实例变量
        for key, value in default_config.items():
            setattr(self, key.upper() if key in ['ip', 'port', 'render'] else key, value)

        # 初始化其他变量
        self.data = None  # 用于调试
        self.INITIAL = False
        self.obs_tot = None

        # 创建环境实体
        self.create_entity()

        # 添加数据，高度, 航向, 速度任务值，需要在reset中修改
        self.h_initial = 5000  # 初始高度
        self.h_setpoint = 8000  # 目标高度
        self.psi_initial = 90
        self.psi_setpoint = 200
        self.v_initial = 500
        self.v_setpoint = 400

        # 记录数据，都需要在reset中重置
        self.step_num = 0
        self.h_list = []
        self.psi_list = []
        self.v_list = []
        self.v_true = 0
        self.last_action = np.zeros(4)
        self.last_action_input = np.zeros(4)
        self.obs_list_episode = []
        self.w_list = []
        self.prev_action_list = []
        self.process_action_list = []

    def _build_execute_cmd(self):
        """构建执行命令"""
        return f'''{self.excute_path} Ip={self.IP} Port={self.PORT} PlayMode={self.RENDER} RedNum={self.red_num} BlueNum={self.blue_num} Scenes={self.scenes}'''

    # 创建游戏环境
    def create_entity(self):
        is_success = False
        while not is_success:
            try:
                self.excute_cmd = self._build_execute_cmd()
                print('Creating Env', self.excute_cmd)
                self.unity = os.popen(self.excute_cmd)
                time.sleep(20)
                self._connect()
                is_success = True
                print('Env Created')
            except Exception as e:
                print('Create failed and the reason is ', e)
                time.sleep(5)

    # 发送action，为dict形式
    def _send_condition(self, data):
        self.socket.send(bytes(data.encode('utf-8')))
        self.data = data

    def _connect(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(20)
        print(f'Connecting {self.IP}:{self.PORT}')
        self.socket.connect((self.IP, self.PORT))

    def reconstruct(self):
        print('Reconstruct Env')
        # 暂时不修改，看看能不能接收数据
        self.PORT = self.PORT
        self.create_entity()
        self.INITIAL = False

    def kill_env(self):
        print('Kill Env')
        output = os.popen(f'netstat -ano | findstr {self.IP}:{self.PORT}')
        output = output.read()
        output = output.split("\n")
        pid = None
        print('output', output)
        for out_tmp in output:
            out = out_tmp.split(' ')
            out_msg = []
            for msg_tmp in out:
                if msg_tmp != '':
                    out_msg.append(msg_tmp)
            try:
                if out_msg[1] == f'{self.IP}:{self.PORT}':
                    pid = out_msg[-1]
                    break
            except Exception as e:
                print('out_msg', out_msg)
        if pid is not None:
            os.system('taskkill /f /im %s' % pid)
            # os.system('kill -9 %s' % pid)
        self.socket.shutdown(socket.SHUT_RDWR)
        self.socket.close()
        # 暂时不修改，看看能不能接收数据
        self.PORT = self.PORT + 10

    def obs_dtol(self, obs_dict):
        # 误差相关
        obs_dict["red"]["red_0"]["altitude_error_ft"] = obs_dict["red"]["red_0"]["position/h-sl-ft"] - self.h_setpoint
        obs_dict["red"]["red_0"]["track_error_deg"] = angle_difference(obs_dict["red"]["red_0"]["attitude/psi-deg"],
                                                                       self.psi_setpoint)
        self.v_true = math.sqrt(
            obs_dict["red"]["red_0"]["velocities/u-fps"] ** 2 + obs_dict["red"]["red_0"]["velocities/v-fps"] ** 2 +
            obs_dict["red"]["red_0"]["velocities/w-fps"] ** 2)
        obs_dict["red"]["red_0"]["delta_velocity"] = self.v_true - self.v_setpoint
        # 需要的特征，注意顺序
        feature_select = [
            "altitude_error_ft", "track_error_deg", "delta_velocity",    # 控制模型信息: 高度误差, 航向角误差, 速度误差 [ft, degree, ft/s]
            "attitude/pitch-rad", "attitude/roll-rad", "aero/beta-deg",  # 位姿信息：俯仰角, 翻滚角, 侧滑角[rad]
            "velocities/u-fps", "velocities/v-fps", "velocities/w-fps",  # 速度信息：u,v,w线性 [ft/s]
            "velocities/p-rad_sec", "velocities/q-rad_sec", "velocities/r-rad_sec",  # p,q,r角度 [rad/s]
            "fcs/left-aileron-pos-norm", "fcs/elevator-pos-norm",
            "fcs/rudder-pos-norm", "fcs/throttle-pos-norm"  # 控制状态：左副翼位置，升降舵位置，方向舵位置，油门位置
        ]
        # 选择需要的特征
        obs_vlist = [obs_dict["red"]["red_0"][key] for key in feature_select]

        # 归一化在step中
        return obs_vlist

    # 接收飞机状态，dict形式
    def _accept_from_socket(self):
        msg_receive = None
        try:
            # msg_receive = json.loads(str(self.socket.recv(81920), encoding='utf-8'))
            load_data = self.socket.recv(8192 * 16)
            zip_data = io.BytesIO(load_data)
            zip_file = zipfile.ZipFile(zip_data)
            msg_receive = zip_file.read(zip_file.namelist()[0])
            msg_receive = json.loads(str(msg_receive, encoding='utf-8'))
        except Exception as e:
            if e == socket.timeout:
                print('out of time')
            print("fail to recieve message from unity")
            print("the last sent data is {}", self.data)
            print(e)
            self.kill_env()
            self.create_entity()
            self.INITIAL = False
        return msg_receive

    def reset(self, red_number: int = 1, blue_number: int = 1, reset_attribute: dict = reset_data_example, **kwargs):
        if kwargs:
            self.h_initial = kwargs['h_initial']
            self.h_setpoint = kwargs['h_setpoint']
            self.psi_initial = kwargs['psi_initial']
            self.psi_setpoint = kwargs['psi_setpoint']
            self.v_initial = kwargs['v_initial']
            self.v_setpoint = kwargs['v_setpoint']
        self.h_list = []
        self.psi_list = []
        self.v_list = []
        self.prev_action_list = []
        self.process_action_list = []
        self.last_action = np.zeros(4)
        self.obs_list_episode = []
        self.step_num = 0
        self.w_list = []

        if self.mode == 'test':
            print(f'h_initial= {self.h_initial}, h_setpoint= {self.h_setpoint}')
            print(f'psi_initial= {self.psi_initial}, psi_setpoint= {self.psi_setpoint}')
            print(f'v_initial= {self.v_initial}, v_setpoint= {self.v_setpoint}')

        # 修改初始高度、航向、速度
        reset_attribute['red']['red_0']["ic/h-sl-ft"] = self.h_initial
        reset_attribute['red']['red_0']["ic/psi-true-deg"] = self.psi_initial
        reset_attribute['red']['red_0']["ic/u-fps"] = self.v_initial

        init_info = {'red': reset_attribute['red'],
                     'blue': reset_attribute['blue']}
        if self.INITIAL is False:
            self.INITIAL = True
            init_info['flag'] = {"init": {'render': self.RENDER}}
        else:
            init_info['flag'] = {'reset': {'render': self.RENDER}}
        data = json.dumps(init_info)
        self._send_condition(data)
        obs_tot = self._accept_from_socket()
        obs_list = obs_process(self.obs_dtol(obs_tot))
        self.last_action = np.zeros(4)
        # 将last_action作为观察的一部分返回
        return np.concatenate([obs_list, self.last_action])

    def step(self, action_list, default_control=False):
        action_list = np.clip(action_list, -1, 1)
        if default_control:
            # 内置控制器模式
            action_attribute = action_ltod_mode2(
                [self.h_setpoint, self.psi_setpoint, self.v_setpoint])
        else:
            # 自定义控制器模式
            if self.step_num == 0:
                action_new = np.zeros_like(action_list)
            else:
                action_new = np.clip(1 * self.last_action + 0.2 * action_list, [-1, -1, -1, 0], [1, 1, 1, 1])

            self.last_action = action_new.copy()
            self.last_action_input = action_list.copy()

            if self.mode == "test":
                self.prev_action_list.append(action_list)
                self.process_action_list.append(action_new)

            action_attribute = action_ltod(action_new.tolist())

        data = json.dumps(action_attribute)
        self._send_condition(data)
        obs_tot = self._accept_from_socket()
        self.step_num += 1

        info = {}
        self.obs_tot = obs_tot

        # 判断终止
        if obs_tot is None:
            return [], 0, True, info
        if obs_tot['red']['red_0']['LifeCurrent'] == 0 or self.step_num >= self.step_num_max \
                or obs_tot['red']['red_0']['position/h-sl-ft'] < 1000 or obs_tot['red']['red_0'][
            'position/h-sl-ft'] > 22000:
            done = True
        else:
            done = False

        # 这里得到的obs需要改为list形式
        obs_list = self.obs_dtol(obs_tot)
        [reward_h_scale, reward_psi_scale, reward_v_scale], [reward_h_linear, reward_psi_linear, reward_v_linear], w = (
            Reward.get_reward_complex(obs_list[0], obs_list[1], obs_list[2], self.h_initial - self.h_setpoint,
                                      self.psi_initial - self.psi_setpoint, self.v_initial - self.v_setpoint,
                                      self.step_num))
        # reward_linear = w[0] * reward_h_linear + w[1] * reward_psi_linear + w[2] * reward_v_linear
        # reward_scale = w[0] * reward_h_scale + w[1] * reward_psi_scale + w[2] * reward_v_scale
        # reward = (reward_scale + 0.1 * reward_linear) / 1.1
        # 速度权重随航向角误差变化
        # reward = w[0] * reward_h_scale + w[1] * reward_psi_scale + w[2] * reward_v_scale
        # 奖励值使用乘积形式
        reward = reward_h_scale * reward_psi_scale * reward_v_scale
        completion_degree = [reward_h_scale, reward_psi_scale, reward_v_scale]

        obs_list = obs_process(obs_list)
        if not default_control and self.mode == 'test':
            print(f'step: {self.step_num}\n'
                  f'obs: {obs_list}\n'
                  f'net_action: {action_list}, current_action: {action_new}')
            # print('h_reward', reward_h_linear, reward_h_scale)
            # print('psi_reward', reward_psi_linear, reward_psi_scale)
            # print('v_reward', reward_v_linear, reward_v_scale)
            # print(f'w: {w}, reward: {reward}')

        # 测试模式下记录数据
        if self.mode == "test":
            self.h_list.append(obs_tot["red"]["red_0"]["position/h-sl-ft"])
            self.psi_list.append(obs_tot["red"]["red_0"]["attitude/psi-deg"])
            self.v_list.append(self.v_true)
            self.w_list.append(w)
            self.obs_list_episode.append(obs_list)

        return np.concatenate([obs_list, self.last_action]), reward, done, completion_degree

    def change_target(self, h_setpoint, psi_setpoint):
        if self.mode == 'test':
            print(f'target change, current step:{self.step_num}')
            print(f'h_initial= {self.h_setpoint}, h_setpoint= {h_setpoint}')
            print(f'psi_initial= {self.psi_setpoint}, psi_setpoint= {psi_setpoint}')

        self.h_initial = self.h_setpoint // 20 * 20
        self.psi_initial = self.psi_setpoint // 5 * 5
        self.h_setpoint = h_setpoint
        self.psi_setpoint = psi_setpoint
