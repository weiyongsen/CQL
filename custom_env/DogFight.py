import io
import math
import socket
import json
import random
import os
import time
import zipfile

import numpy as np
import gymnasium as gym
import pandas as pd
from util.util import angle_difference, action_ltod, obs_process

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
        ip = '127.0.0.1'
        port = 8000
        excute_path = r'C:\Users\Absol\Desktop\ZK\ZK_v2.6\build\windows\ZK.exe'
        # excute_path = 'game/nolinux/ZK.x86_64'
        # 更改飞机数量，原为3
        red_num = 1
        blue_num = 1
        scenes = 3
        mode = 'train'
        step_num_max = 300
        state_size = 20
        action_size = 4
        state_stack_num = 2


        if config is not None:
            ip = config.get('ip', ip)
            port = config.get('port', port)
            red_num = config.get('red_num', red_num)
            blue_num = config.get('blue_num', blue_num)
            scenes = config.get('scenes', scenes)
            excute_path = config.get('excute_path', excute_path)
            render = config.get('render', render)
            state_size = config.get('state_size', state_size)
            action_size = config.get('action_size', action_size)
            step_num_max = config.get('step_num_max', step_num_max)
            mode = config.get('mode', mode)
            state_stack_num = config.get('state_stack_num', state_stack_num)
        try:
            port = config.worker_index + port
        except:
            port = port
        self.IP = ip
        self.PORT = port
        self.RENDER = int(render)
        self.red_num = red_num
        self.blue_num = blue_num
        self.scenes = scenes
        self.excute_path = excute_path
        self.mode = mode
        self.state_size = state_size
        self.action_size = action_size
        self.state_stack_num = state_stack_num
        self.step_num_max = step_num_max
        print(f'all config: ip = {self.IP}, port = {self.PORT}, red_num = {self.red_num}, blue_num = {self.blue_num}, scenes = {self.scenes}, excute_path = {self.excute_path}, mode = {self.mode}, state_stack_num = {self.state_stack_num}, step_num_max = {self.step_num_max}')
        self.data = None  # set for debug
        self.INITIAL = False
        self.excute_cmd = f'{self.excute_path} ip={self.IP} port={self.PORT} ' \
                          f'PlayMode={self.RENDER} ' \
                          f'RedNum={self.red_num} BlueNum={self.blue_num} ' \
                          f'Scenes={self.scenes}'
        
        self.observation_space = gym.spaces.Box(low=-10, high=10, shape=(self.state_size*self.state_stack_num,))
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.action_size,))

        self.create_entity()
        self.obs_tot = None

        # 添加数据， 高度, 航向, 速度任务值, 需要在reset中修改
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
        self.obs_list = []
        self.obs_list_episode = []
        self.delta_coef_list = []
        self.last_state = None  # 添加变量存储上一时刻的完整状态(obs+action)

        self.prev_action_list = []  # 存储原始动作序列
        self.process_action_list = []  # 存储处理后的动作序列

        self.state_history = None
        self.last_h = None
        self.last_psi = None
        self.last_v = None
        # 任务完成度缓冲区
        self.completion_buffer = [0] * 10  # 初始化为零分，长度可调
        # 稳定计数器和失败计数器
        self.h_stable_counter, self.h_failure_count = 0, 0
        self.psi_stable_counter, self.psi_failure_count = 0, 0
        self.v_stable_counter, self.v_failure_count = 0, 0
        self.delta_action = np.zeros(4)  # 记录当前动作改变量
        # 记录初始误差
        self.initial_h_error = 0
        self.initial_psi_error = 0
        self.initial_v_error = 0
        # 设置动态阈值
        self.h_threshold = 0
        self.psi_threshold = 0
        self.v_threshold = 0
        

    # 创建游戏环境
    def create_entity(self):
        is_success = False
        while not is_success:
            try:
                self.excute_cmd = f'{self.excute_path} Ip={self.IP} Port={self.PORT} ' \
                                  f'PlayMode={self.RENDER} ' \
                                  f'RedNum={self.red_num} BlueNum={self.blue_num} ' \
                                  f'Scenes={self.scenes}'
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
            "altitude_error_ft", "track_error_deg", "delta_velocity",  # 控制模型信息: 高度误差, 航向角误差, 速度误差 [ft, degree, ft/s]
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

        self.h_initial = 9000
        self.h_setpoint = 10000
        self.psi_initial = 180
        self.psi_setpoint = 280
        self.v_initial = 1000
        self.v_setpoint = 1200

        self.h_list = []
        self.psi_list = []
        self.v_list = []
        self.prev_action_list = []
        self.process_action_list = []
        self.last_action = np.zeros(4)
        self.obs_list = []
        self.obs_list_episode = []
        self.step_num = 0
        self.delta_coef_list = []
        self.v_true = 0
        self.last_state = None
        self.completion_buffer = [0] * 10
        self.delta_action = np.zeros(4) 

        # 记录初始误差
        self.initial_h_error = abs(self.h_initial - self.h_setpoint)
        self.initial_psi_error = abs(angle_difference(self.psi_initial, self.psi_setpoint))
        self.initial_v_error = abs(self.v_initial - self.v_setpoint)

        # 设置动态阈值（初始误差的10%，但有最小和最大限制）
        self.h_threshold = min(max(self.initial_h_error * 0.1, 4), 50)
        self.psi_threshold = min(max(self.initial_psi_error * 0.1, 0.5), 3)
        self.v_threshold = min(max(self.initial_v_error * 0.1, 1), 5)

        # 重置稳定计数器
        self.h_stable_counter, self.h_failure_count = 0, 0
        self.psi_stable_counter, self.psi_failure_count = 0, 0
        self.v_stable_counter, self.v_failure_count = 0, 0

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
        
        # 初始化状态和动作历史
        current_state = np.concatenate([obs_list, self.last_action])
        self.state_history = [current_state.copy() for _ in range(self.state_stack_num)]  # 可变堆叠帧数
        
        # 初始化上一时刻状态值
        self.last_h = obs_tot["red"]["red_0"]["position/h-sl-ft"]
        self.last_psi = obs_tot["red"]["red_0"]["attitude/psi-deg"]
        self.last_v = self.v_true
        
        return np.concatenate(self.state_history)

    def step(self, action_list, default_control=False):
        # if self.step_num %10 == 0:
        #     print('action_list', self.step_num, action_list)
        if default_control:
            action_attribute = action_ltod([self.h_setpoint, self.psi_setpoint, self.v_setpoint], mode=2)
            action_new = np.zeros_like(action_list)
            delta_coef = 0.2
            completion_avg = 0
        else:
            if self.step_num == 0:
                action_new = np.zeros_like(action_list)
                delta_coef = 0.2
                completion_avg = 0

            else:
                # 添加噪声
                # if self.mode == 'train':
                #     action_list = self._add_action_noise(action_list, noise_std=0.1)
                action_list = np.clip(action_list, -1, 1)
                # 计算动作改变量
                completion_avg = np.mean(self.completion_buffer)
                delta_coef = self._get_delta_coef(completion_avg, base_delta=0.2, min_delta=0.1, high_completion=0.98)
                self.delta_action = delta_coef * action_list
                
                # 动作裁剪
                # action_threshold = 0.02  # 你可以根据实际情况调整
                # self.delta_action = np.where(np.abs(self.delta_action) < action_threshold, 0, self.delta_action)
                action_new = self._clip_action(self.last_action + self.delta_action, [-1, -1, -1, 0], [1, 1, 1, 1])

            action_attribute = action_ltod(action_new.tolist(), mode=0)
            self.last_action = action_new.copy()    # 更新动作历史
            
            # 记录动作序列
            if self.mode == 'test':
                self.prev_action_list.append(action_list.copy())  # 原始动作
                self.process_action_list.append(action_new.copy())  # 处理后的动作

        data = json.dumps(action_attribute)
        self._send_condition(data)
        obs_tot = self._accept_from_socket()
        self.step_num += 1

        info = {}
        self.obs_tot = obs_tot
        done = self._check_done(obs_tot)    # 判断终止

        # 这里得到的obs需要改为list形式
        self.obs_list = self.obs_dtol(obs_tot)
        
        # 计算奖励
        reward_scales, reward_converges, reward_penalties, action_penalty, cross_target_penalty = self.get_reward_complex()
        # 整合全部的奖励， 分开获得是因为存储和输出只需要reward_scales就可以，辅助的奖励不看
        reward_h = reward_scales[0] + reward_converges[0] + reward_penalties[0]
        reward_psi = reward_scales[1] + reward_converges[1] + reward_penalties[1]
        reward_v = reward_scales[2] + reward_converges[2] + reward_penalties[2]
        reward_single_max = 1.2
        reward_norm = (reward_h * reward_psi * reward_v + reward_h + reward_psi + reward_v) / 4
        reward = reward_norm - action_penalty - cross_target_penalty
        # 更新任务完成度缓冲区
        completion_degree = reward_scales
        completion_mean = np.mean(completion_degree)
        self._update_completion_buffer(completion_mean)
        # 更新上一时刻的状态值（添加这部分）
        self.last_h = obs_tot['red']['red_0']['position/h-sl-ft']
        self.last_psi = obs_tot['red']['red_0']['attitude/psi-deg']
        self.last_v = self.v_true
        # 构建当前状态并更新状态历史
        obs_list = obs_process(self.obs_list)
        stacked_state = self._update_state_history(obs_list, action_new)
        # 记录数据
        self._record_test_data(obs_tot, obs_list, reward_scales, delta_coef, default_control)
        return stacked_state, reward, done, info

    def get_reward_complex(self):
        """计算复合奖励，包含基础奖励、收敛奖励和惩罚项"""
        # 1. 获取当前误差和上一时刻误差
        h_error = self.obs_list[0]
        psi_error = self.obs_list[1]
        v_error = self.obs_list[2]
        h_diff = abs(h_error)
        psi_diff = abs(psi_error)
        v_diff = abs(v_error)

        last_h_error = self.last_h - self.h_setpoint
        last_psi_error = angle_difference(self.last_psi, self.psi_setpoint)
        last_v_error = self.last_v - self.v_setpoint
        last_h_diff = abs(last_h_error)
        last_psi_diff = abs(last_psi_error)
        last_v_diff = abs(last_v_error)

        # 2. 计算基础奖励（使用非线性奖励函数）
        k_h, k_psi, k_v = 200, 10, 20
        reward_h_scale = self.second_function(h_diff, k_h)
        reward_psi_scale = self.second_function(psi_diff, k_psi)
        reward_v_scale = self.second_function(v_diff, k_v)

        # 3. 计算接近目标的程度（用于收敛奖励和惩罚）
        h_close = np.exp(- (h_diff / self.h_threshold) ** 2)
        psi_close = np.exp(- (psi_diff / self.psi_threshold) ** 2)
        v_close = np.exp(- (v_diff / self.v_threshold) ** 2)          
        close_factor = (h_close + psi_close + v_close) / 3

        # 4. 分别计算各维度的收敛奖励
        # 更新各维度的稳定计数器
        if h_diff <= 1.5 * self.h_threshold:
            self.h_stable_counter += 1
            self.h_failure_count = 0  # 成功一次，失败计数器归零
        else:
            self.h_failure_count += 1
            if self.h_failure_count >= 3:
                self.h_stable_counter = 0  # 连续两次失败，稳定计数器归零
            else:
                self.h_stable_counter = max(0, self.h_stable_counter - 2)  # 单次失败，计数器衰减
        if psi_diff <= 1.5 * self.psi_threshold:
            self.psi_stable_counter += 1
            self.psi_failure_count = 0
        else:
            self.psi_failure_count += 1
            if self.psi_failure_count >= 3:
                self.psi_stable_counter = 0
            else:
                self.psi_stable_counter = max(0, self.psi_stable_counter - 2)
        if v_diff <= 1.5 * self.v_threshold:
            self.v_stable_counter += 1
            self.v_failure_count = 0
        else:
            self.v_failure_count += 1
            if self.v_failure_count >= 3:
                self.v_stable_counter = 0
            else:
                self.v_stable_counter = max(0, self.v_stable_counter - 2)
        # 分别计算各维度的收敛奖励
        h_converge = 0.2 * (1 - np.exp(-self.h_stable_counter / 5)) * h_close  # 范围[0, 0.1]
        psi_converge = 0.2 * (1 - np.exp(-self.psi_stable_counter / 5)) * psi_close
        v_converge = 0.2 * (1 - np.exp(-self.v_stable_counter / 5)) * v_close

        # 5. 计算动作惩罚项（只在接近目标时生效）
        action_penalty = 10 * close_factor * np.mean(np.abs(self.delta_action)) if close_factor > 0.8 else 0

        # 6. 状态变化惩罚（每个维度独立承担）
        h_penalty, psi_penalty, v_penalty = 0, 0, 0
        if h_close > 0.8:
            h_penalty = 0.2 * self.dynamic_reward_penalty(h_diff, last_h_diff, self.h_threshold, h_close)
        if psi_close > 0.8:
            psi_penalty = 0.2 * self.dynamic_reward_penalty(psi_diff, last_psi_diff, self.psi_threshold, psi_close)
        if v_close > 0.8:
            v_penalty = 0.2 * self.dynamic_reward_penalty(v_diff, last_v_diff, self.v_threshold, v_close)

        # 7. 如果跨过目标，“接近目标且误差很小”时才惩罚，且误差越小惩罚越大
        cross_target_penalty = 0
        if np.sign(h_error) != np.sign(last_h_error) and h_close > 0.5 and abs(h_error) < self.h_threshold:
            cross_target_penalty += 0.05 * (1 - abs(h_error) / self.h_threshold)
        if np.sign(psi_error) != np.sign(last_psi_error) and psi_close > 0.5 and abs(psi_error) < self.psi_threshold:
            cross_target_penalty += 0.05 * (1 - abs(psi_error) / self.psi_threshold)
        if np.sign(v_error) != np.sign(last_v_error) and v_close > 0.5 and abs(v_error) < self.v_threshold:
            cross_target_penalty += 0.05 * (1 - abs(v_error) / self.v_threshold)

        if self.mode == 'test':
            print(f'h_diff: {h_diff}, psi_diff: {psi_diff}, v_diff: {v_diff}')
            print(f'last_h_diff: {last_h_diff}, last_psi_diff: {last_psi_diff}, last_v_diff: {last_v_diff}')
            print(f'h_threshold: {self.h_threshold}, psi_threshold: {self.psi_threshold}, v_threshold: {self.v_threshold}')
            print(f'h_close: {h_close}, psi_close: {psi_close}, v_close: {v_close}')
            print(f'h_stable_counter: {self.h_stable_counter}, psi_stable_counter: {self.psi_stable_counter}, v_stable_counter: {self.v_stable_counter}')
            print(f'reward_h_scale: {reward_h_scale}, reward_psi_scale: {reward_psi_scale}, reward_v_scale: {reward_v_scale}')
            print(f'h_converge: {h_converge}, psi_converge: {psi_converge}, v_converge: {v_converge}')
            print(f'h_penalty: {h_penalty}, psi_penalty: {psi_penalty}, v_penalty: {v_penalty}')
            print(f'action_penalty: {action_penalty}')
            print(f'cross_target_penalty: {cross_target_penalty}')
        reward_scales = [reward_h_scale, reward_psi_scale, reward_v_scale]
        reward_converges = [h_converge, psi_converge, v_converge]
        reward_penalties = [h_penalty, psi_penalty, v_penalty]

        return reward_scales, reward_converges, reward_penalties, action_penalty, cross_target_penalty

    def dynamic_reward_penalty(self, current_error, last_error, threshold, close):
        """
        根据误差变化和 close 动态计算奖励/惩罚
        """
        # 误差变化量
        error_change = current_error - last_error
        # 动态奖励/惩罚系数（close 越大，系数越大）
        reward_scale = close * 0.1  # 基础奖励系数
        penalty_scale = close * 0.2  # 惩罚系数（惩罚更强）

        if error_change < 0:  # 误差减小 → 奖励
            return reward_scale * (1 - current_error / threshold)
        else:  # 误差增大 → 惩罚
            return -penalty_scale * (current_error / threshold)

    def change_target(self, h_setpoint, psi_setpoint):
        """切换目标"""
        if self.mode == 'test':
            print(f'target change, current step:{self.step_num}')
            print(f'h_initial= {self.h_setpoint}, h_setpoint= {h_setpoint}')
            print(f'psi_initial= {self.psi_setpoint}, psi_setpoint= {psi_setpoint}')
            print(f'v_initial= {self.v_setpoint}, v_setpoint= {self.v_setpoint}')

        self.h_initial = self.h_setpoint // 20 * 20
        self.psi_initial = self.psi_setpoint // 5 * 5
        self.h_setpoint = h_setpoint
        self.psi_setpoint = psi_setpoint

        self.initial_h_error = abs(self.h_initial - self.h_setpoint)
        self.initial_psi_error = abs(angle_difference(self.psi_initial, self.psi_setpoint))
        self.initial_v_error = abs(self.v_initial - self.v_setpoint)

        # 设置动态阈值（初始误差的10%，但有最小和最大限制）
        self.h_threshold = min(max(self.initial_h_error * 0.1, 4), 50)
        self.psi_threshold = min(max(self.initial_psi_error * 0.1, 0.5), 3)
        self.v_threshold = min(max(self.initial_v_error * 0.1, 1), 5)

    # 奖励相关方法
    def first_function(self, error_now, k):
        """幂函数"""
        return 1 - (error_now / k) / (1 + error_now / k)

    def second_function(self, error_now, k):
        """指数函数"""
        scaled_error = error_now * 0.69 / k
        return np.exp(-scaled_error)

    def third_function(self, error_now, k, p=1.2):
        """广义幂函数"""
        return 1 / (1 + (error_now / k) ** p)

    def fourth_function(self, error_now, k, p=1.0, epsilon=0.1):
        """自适应幂函数"""
        normalized_error = abs(error_now) / (k + epsilon)  # 防止除零
        return 1 / (1 + normalized_error ** p) * (1 + np.tanh(1 / (normalized_error + 0.1)))

    def _get_delta_coef(self, completion_avg, base_delta, min_delta, high_completion):
        """
        分段自适应调整动作步长系数
        - completion_avg >= high_completion: min_delta
        - completion_avg <= high_completion/2: base_delta
        - 其余区间线性插值
        """
        if completion_avg >= high_completion:
            return min_delta
        elif completion_avg <= high_completion / 2:
            return base_delta
        else:
            # 线性插值
            ratio = (completion_avg - high_completion / 2) / (high_completion / 2)
            delta_coef = base_delta + (min_delta - base_delta) * ratio
            delta_coef = np.clip(delta_coef, min_delta, base_delta)
            return round(delta_coef, 3)

    def _clip_action(self, action, low, high):
        """裁剪动作到指定范围"""
        return np.clip(action, low, high)

    def _check_done(self, obs_tot):
        """判断当前回合是否终止"""
        if obs_tot is None:
            return True
        if obs_tot['red']['red_0']['LifeCurrent'] == 0 or self.step_num >= self.step_num_max \
                or obs_tot['red']['red_0']['position/h-sl-ft'] < 1000 or obs_tot['red']['red_0']['position/h-sl-ft'] > 22000:
            return True
        return False

    def _update_completion_buffer(self, completion_mean):
        """更新任务完成度的滑动窗口"""
        self.completion_buffer.pop(0)
        self.completion_buffer.append(completion_mean)

    def _update_state_history(self, obs_list, action_new):
        """更新状态堆叠历史，最新状态插入最前"""
        current_state = np.concatenate([obs_list, action_new])
        self.state_history.pop(-1)  # 移除最后一个状态
        self.state_history.insert(0, current_state)  # 新状态插入到开头
        return np.concatenate(self.state_history)

    def _record_test_data(self, obs_tot, obs_list, reward_scales, delta_coef, default_control):
        """测试模式下打印和记录关键数据"""
        if not default_control and self.mode == 'test':
            print(f'step: {self.step_num}')
            # print(f'obs: {obs_list}')
            # print(f'net_action: {self.last_action}, current_action: {obs_list[-4:] if len(obs_list) >= 4 else obs_list}')
        if self.mode == "test":
            self.h_list.append(obs_tot["red"]["red_0"]["position/h-sl-ft"])
            self.psi_list.append(obs_tot["red"]["red_0"]["attitude/psi-deg"])
            self.v_list.append(self.v_true)
            self.delta_coef_list.append(delta_coef)
            self.obs_list_episode.append(obs_list)