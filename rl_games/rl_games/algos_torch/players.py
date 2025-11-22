from rl_games.common.player import BasePlayer
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.common.tr_helpers import unsqueeze_obs
import gym
import torch 
from torch import nn
import numpy as np
import pickle
from datetime import datetime
from pathlib import Path
import json

def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action = action * d + m
    return scaled_action


"""
一个具体的强化学习智能体实现，用于 连续动作空间（Continuous Action Space） 的 PPO（Proximal Policy Optimization） 算法
继承自 BasePlayer，因此继承了环境管理、模型加载、checkpoint监控、运行逻辑等通用功能
这是一个“具体玩家（player）”类，定义了如何：
    - 构建模型
    - 选择动作
    - 加载模型权重
    - 在需要时重置内部状态（尤其是 RNN）
"""
class PpoPlayerContinuous(BasePlayer):
    
    """
    __init__ 是初始化方法
    """
    def __init__(self, params):
        
        # 调用父类 BasePlayer 的初始化方法
        BasePlayer.__init__(self, params)
        
        # train 配置文件的 network 字段
        self.network = self.config['network']
        
        # 读取动作空间的维度（即连续动作的数量）
        # Ant 任务中有 8 个连续动作维度
        self.actions_num = self.action_space.shape[0]
        
        # 获取环境允许的动作下限，并转换为 PyTorch 张量
        # Ant 任务是 -1 
        self.actions_low = torch.from_numpy(self.action_space.low.copy()).float().to(self.device)
        
        # 获取环境允许的动作上限，并转换为 PyTorch 张量
        # Ant 任务是 1
        self.actions_high = torch.from_numpy(self.action_space.high.copy()).float().to(self.device)
        
        # 定义一个简单的动作掩码
        # 对于连续动作空间，通常不需要掩码，因此这里设置为全 False
        self.mask = [False]

        # 是否对输入状态（observations）进行归一化
        self.normalize_input = self.config['normalize_input']
        
        # 是否对价值函数输出进行归一化；这有助于在奖励尺度变化较大时保持 critic 学习稳定
        # Ant 任务返回的是 True
        self.normalize_value = self.config.get('normalize_value', False)

        # 保存环境观测空间（observation space）的形状
        # Ant 任务是 (60,)
        obs_shape = self.obs_shape
        
        # 构建模型所需的配置信息字典
        config = {
            'actions_num': self.actions_num,
            'input_shape': obs_shape,
            'num_seqs': self.num_agents,
            'value_size': self.env_info.get('value_size', 1),
            'normalize_value': self.normalize_value,
            'normalize_input': self.normalize_input,
        } 
        
        # 调用网络构建器（ModelBuilder）的 build() 方法，创建模型实例
        self.model = self.network.build(config)
        
        # 将模型移动到指定的计算设备（CPU 或 GPU）
        self.model.to(self.device)
        
        # 将模型切换到“评估模式”
        self.model.eval()
        
        # 检查模型是否包含 RNN（循环神经网络）结构
        self.is_rnn = self.model.is_rnn()

    """
    get_action 方法用于根据当前观察选择动作
    """
    def get_action(self, obs, is_deterministic = False):
        
        # 如果当前观测没有 batch 维度（即只是一条观测）
        if self.has_batch_dimension == False:
            # 调用 unsqueeze_obs() 增加一个批次维度，例如从 (obs_dim,) → (1, obs_dim)
            obs = unsqueeze_obs(obs)
            
        # 对观测进行预处理：
        #  - 如果观测是图像（dtype = uint8），则转换为 float 并除以 255.0
        #  - 如果观测是 dict 类型，则递归处理每个键
        # 这样做是为了确保模型输入归一化到 [0, 1] 或合理的数值范围
        obs = self._preproc_obs(obs)
        
        # PPO 模型在 forward 时通常接收这种结构化输入
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs': obs,
            'rnn_states': self.states
        }
        
        # 在不计算梯度的上下文中执行前向传播
        with torch.no_grad():
            res_dict = self.model(input_dict)
        mu = res_dict['mus']
        action = res_dict['actions']
        self.states = res_dict['rnn_states']
        
        # 根据是否为确定性模式选择输出动作：
        # - 如果是确定性模式，直接使用策略均值 mu 作为动作
        # - 否则，使用模型输出的采样动作
        if is_deterministic:
            current_action = mu
        else:
            current_action = action
        
        # 如果原本输入没有批次维度，则去掉批次维度
        # 例如从 (1, action_dim) → (action_dim,)
        # .detach() 确保返回的张量不在计算图中（即不会产生梯度）
        if self.has_batch_dimension == False:
            current_action = torch.squeeze(current_action.detach())
    
        # 如果配置了动作裁剪（clip_actions），则将动作限制在 [-1, 1] 范围内
        if self.clip_actions:
            return rescale_actions(self.actions_low, self.actions_high, torch.clamp(current_action, -1.0, 1.0))
        else:
            return current_action

    """
    restore 方法用于从 checkpoint 文件加载模型权重
    """
    def restore(self, fn):
        
        # 从文件路径 fn（通常是 .pth 格式的 checkpoint 文件）加载模型检查点数据
        # 这里 torch_ext.load_checkpoint(fn) 是一个工具函数
        # 封装了 torch.load(fn, map_location=...) 的逻辑
        # 它返回一个 Python 字典，包含模型权重、归一化器状态、环境状态等
        checkpoint = torch_ext.load_checkpoint(fn)
        
        # 从 checkpoint 中取出 'model' 键对应的模型参数字典（state_dict）
        # 然后加载到当前的 self.model 中
        # 这一步会将神经网络的所有可学习参数恢复到保存时的状态
        self.model.load_state_dict(checkpoint['model'])
        
        # 如果训练过程中启用了输入归一化（normalize_input=True）
        # 并且 checkpoint 中包含 'running_mean_std' 这个键
        # 说明保存时也保存了输入归一化器的运行统计信息（均值和方差）
          # 需要一并恢复，以保证推理时输入分布一致
        if self.normalize_input and 'running_mean_std' in checkpoint:
            # 将 checkpoint 中保存的 running_mean_std 参数加载回模型内部的归一化层
            self.model.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
            
        # 尝试从 checkpoint 中取出环境状态（如果有保存）
        # 这在某些自定义环境中用于保存环境内部的随机数种子、物理仿真状态等
        env_state = checkpoint.get('env_state', None)
        
        # 如果当前 player 持有环境实例（self.env 存在），并且 checkpoint 中有环境状态
        # 那么调用环境的 set_env_state() 方法，将环境恢复到保存时的状态
        # 这样可以实现完全可复现的推理或继续训练
        if self.env is not None and env_state is not None:
            self.env.set_env_state(env_state)

    """
    reset 方法用于在需要时重置智能体的内部状态，尤其是 RNN 状态
    """
    def reset(self):
        # 如果模型包含 RNN 结构，则初始化 RNN 状态
        self.init_rnn()

    

class AntTrajectoryCollector(PpoPlayerContinuous):
    """
    轨迹收集器，继承自PpoPlayerContinuous
    在运行时收集固定数量的轨迹，每个轨迹对应一个完整的episode
    只收集指定的状态信息字段
    """
    
    def __init__(self, params):
        super().__init__(params)
        
        # 轨迹收集配置
        collector_config = params.get('collector_config', {})
        
        self.num_trajectories = collector_config.get('num_trajectories', 5)
        save_dir_str = collector_config.get('save_dir', './trajectories')
        self.save_filename = collector_config.get('save_filename', 'trajectories')

        # 确保使用绝对路径
        self.save_dir = Path(save_dir_str).resolve()
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 需要收集的字段列表
        self.required_fields = [
            'potentials',
            'prev_potentials',
            'up_vec',
            'heading_vec',
            'root_states',
            'targets',
            'inv_start_rot',
            'dof_pos',
            'dof_vel',
            'dof_limits_lower',
            'dof_limits_upper',
            'dof_vel_scale',
            'vec_sensor_tensor',
            'dt',
            'contact_force_scale',
            'basis_vec0',
            'basis_vec1',
            'up_axis_idx',
            'actions',
            'rewards'
        ]
        
        # 已收集的轨迹列表
        self.collected_trajectories = []
        
        # 当前episode的轨迹缓存
        self.current_trajectory = None
        
        self.max_steps = 100
        
        print(f"[TrajectoryCollector] Will collect {self.num_trajectories} trajectories")
        print(f"[TrajectoryCollector] Save directory: {self.save_dir}")
        print(f"[TrajectoryCollector] Required fields: {len(self.required_fields)} fields")
    
    def init_trajectory(self):
        """初始化一条新轨迹"""
        self.current_trajectory = {field: [] for field in self.required_fields}
    
    def add_step(self, state_info):
        """
        添加一个时间步的数据到当前轨迹
        
        Args:
            state_info: 包含所需字段的字典
        """
        if self.current_trajectory is None:
            self.init_trajectory()
        
        # 遍历所有需要的字段
        for field in self.required_fields:
            if field in state_info:
                value = state_info[field]
                
                # 转换tensor为numpy
                if isinstance(value, torch.Tensor):
                    value_np = value.cpu().numpy()
                else:
                    value_np = value
                
                self.current_trajectory[field].append(value_np)
            else:
                # 如果字段不存在，记录警告（仅第一次）
                if len(self.current_trajectory[field]) == 0:
                    print(f"[TrajectoryCollector] Warning: field '{field}' not found in state_info")
                self.current_trajectory[field].append(None)
    
    def finish_trajectory(self):
        """完成当前轨迹的收集"""
        if self.current_trajectory is None or len(self.current_trajectory['actions']) == 0:
            return
        
        # 将列表转换为numpy数组
        processed_trajectory = {}
        for field, value_list in self.current_trajectory.items():
            # 过滤掉None值
            valid_values = [v for v in value_list if v is not None]
            
            if len(valid_values) > 0:
                try:
                    # 尝试转换为numpy数组
                    processed_trajectory[field] = np.array(valid_values)
                except:
                    # 如果无法转换（例如shape不一致），保持为列表
                    processed_trajectory[field] = valid_values
        
        # 添加统计信息
        if 'rewards' in processed_trajectory:
            processed_trajectory['total_reward'] = np.sum(processed_trajectory['rewards'])
        processed_trajectory['length'] = len(processed_trajectory['actions']) if 'actions' in processed_trajectory else 0
        
        self.collected_trajectories.append(processed_trajectory)
        
        total_reward = processed_trajectory.get('total_reward', 0)
        length = processed_trajectory['length']
        
        print(f"[TrajectoryCollector] Trajectory {len(self.collected_trajectories)} collected: "
              f"length={length}, total_reward={total_reward:.2f}")
        
        self.current_trajectory = None
    
    def save_trajectories(self):
        """保存所有收集的轨迹到文件"""
        if len(self.collected_trajectories) == 0:
            print("[TrajectoryCollector] No trajectories to save.")
            return
        
        # 保存为字典格式，包含轨迹和元数据
        save_data = {
            'trajectories': self.collected_trajectories,
            'metadata': {
                'num_trajectories': len(self.collected_trajectories),
                'timestamp': datetime.now().isoformat()
            }
        }
        
        save_path = self.save_dir / self.save_filename
        # 保存为pickle文件
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)


        
        print(f"\n[TrajectoryCollector] Successfully saved {len(self.collected_trajectories)} trajectories to {save_path}")


        return save_path
    
    def get_state_info_from_env(self, action, reward):
        """
        从环境中提取所需的状态信息
        
        Args:
            action: 当前执行的动作
            reward: 当前步的奖励
            
        Returns:
            state_info: 包含所需字段的字典
        """
        state_info = {}
        
        # 从环境中获取状态信息
        # 假设环境有这些属性，需要根据实际环境调整
        env = self.env
        
        # 尝试获取各个字段
        field_mappings = {
            'potentials': 'potentials',
            'prev_potentials': 'prev_potentials',
            'up_vec': 'up_vec',
            'heading_vec': 'heading_vec',
            'root_states': 'root_states',
            'targets': 'targets',
            'inv_start_rot': 'inv_start_rot',
            'dof_pos': 'dof_pos',
            'dof_vel': 'dof_vel',
            'dof_limits_lower': 'dof_limits_lower',
            'dof_limits_upper': 'dof_limits_upper',
            'dof_vel_scale': 'dof_vel_scale',
            'vec_sensor_tensor': 'vec_sensor_tensor',
            'dt': 'dt',
            'contact_force_scale': 'contact_force_scale',
            'basis_vec0': 'basis_vec0',
            'basis_vec1': 'basis_vec1',
            'up_axis_idx': 'up_axis_idx',
        }
        
        for field, attr_name in field_mappings.items():
            if hasattr(env, attr_name):
                state_info[field] = getattr(env, attr_name)
            elif hasattr(env, 'task') and hasattr(env.task, attr_name):
                state_info[field] = getattr(env.task, attr_name)
        
        # 添加action和reward
        state_info['actions'] = action
        state_info['rewards'] = reward
        
        return state_info
    
    def get_action(self, obs, is_deterministic=False):
        """
        重写get_action方法，保持原有功能
        """
        if self.has_batch_dimension == False:
            obs_input = unsqueeze_obs(obs)
        else:
            obs_input = obs
        
        obs_input = self._preproc_obs(obs_input)
        
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs': obs_input,
            'rnn_states': self.states
        }
        
        with torch.no_grad():
            res_dict = self.model(input_dict)
        
        mu = res_dict['mus']
        action = res_dict['actions']
        self.states = res_dict['rnn_states']
        
        if is_deterministic:
            current_action = mu
        else:
            current_action = action
        
        if self.has_batch_dimension == False:
            current_action = torch.squeeze(current_action.detach())
        
        if self.clip_actions:
            return rescale_actions(self.actions_low, self.actions_high, 
                                 torch.clamp(current_action, -1.0, 1.0))
        else:
            return current_action
    
    def run(self):
        """
        重写run方法，收集指定数量的轨迹
        """
        is_deterministic = self.is_deterministic
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None

        if has_masks_func:
            has_masks = self.env.has_action_mask()

        self.wait_for_checkpoint()
        need_init_rnn = self.is_rnn
        
        print(f"\n[TrajectoryCollector] Starting trajectory collection...")
        
        # 收集指定数量的轨迹
        while len(self.collected_trajectories) < self.num_trajectories:
            # 初始化新轨迹
            self.init_trajectory()
            
            # 重置环境
            obs = self.env_reset(self.env)
            batch_size = self.get_batch_size(obs, 1)

            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            episode_reward = 0
            episode_steps = 0
            
            # 运行一个episode
            for step in range(self.max_steps):
                # 获取动作
                if has_masks:
                    masks = self.env.get_action_mask()
                    action = self.get_masked_action(obs, masks, is_deterministic)
                else:
                    action = self.get_action(obs, is_deterministic)
                
                # 执行动作
                next_obs, reward, done, info = self.env_step(self.env, action)
                
                # 从环境中提取状态信息
                state_info = self.get_state_info_from_env(action, reward)
                
                # 添加到轨迹
                self.add_step(state_info)
                
                episode_reward += reward.item() if isinstance(reward, torch.Tensor) else reward
                episode_steps += 1
                
                obs = next_obs
                
                # 检查是否完成
                if done.any() if isinstance(done, torch.Tensor) else done:
                    print(f"[TrajectoryCollector] Episode finished: steps={episode_steps}, reward={episode_reward:.2f}")
                    
                    # 完成当前轨迹
                    self.finish_trajectory()
                    
                    # 重置RNN状态
                    if self.is_rnn:
                        self.init_rnn()
                    
                    break
            
            # 如果episode没有正常结束（达到最大步数），也保存轨迹
            if self.current_trajectory is not None:
                print(f"[TrajectoryCollector] Episode reached max steps: steps={episode_steps}, reward={episode_reward:.2f}")
                self.finish_trajectory()
        
        # 保存所有收集的轨迹
        self.save_trajectories()
        
        print(f"\n[TrajectoryCollector] Collection complete! Total trajectories: {len(self.collected_trajectories)}")
    




class ShadowHandTrajectoryCollector(PpoPlayerContinuous):
    """
    轨迹收集器，继承自PpoPlayerContinuous
    在运行时收集固定数量的轨迹，每个轨迹对应一个完整的episode
    只收集指定的状态信息字段
    """
    
    def __init__(self, params):
        super().__init__(params)
        
        # 轨迹收集配置
        collector_config = params.get('collector_config', {})
        
        self.num_trajectories = collector_config.get('num_trajectories', 5)
        save_dir_str = collector_config.get('save_dir', './trajectories')
        self.save_filename = collector_config.get('save_filename', 'trajectories')

        # 确保使用绝对路径
        self.save_dir = Path(save_dir_str).resolve()
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 需要收集的字段列表
        self.required_fields = [
            'object_pose',
            'object_pos',
            'object_rot',
            'object_linvel',
            'object_angvel',
            'goal_pose',
            'goal_pos',
            'goal_rot',
            'fingertip_state',
            'fingertip_pos',
            'actions',
            'rewards'
        ]
        
        # 已收集的轨迹列表
        self.collected_trajectories = []
        
        # 当前episode的轨迹缓存
        self.current_trajectory = None
        
        # 最大步数限制
        self.max_steps = 100
        
        print(f"[TrajectoryCollector] Will collect {self.num_trajectories} trajectories")
        print(f"[TrajectoryCollector] Save directory: {self.save_dir}")
        print(f"[TrajectoryCollector] Required fields: {len(self.required_fields)} fields")
    
    def init_trajectory(self):
        """初始化一条新轨迹"""
        self.current_trajectory = {field: [] for field in self.required_fields}
    
    def add_step(self, state_info):
        """
        添加一个时间步的数据到当前轨迹
        
        Args:
            state_info: 包含所需字段的字典
        """
        if self.current_trajectory is None:
            self.init_trajectory()
        
        # 遍历所有需要的字段
        for field in self.required_fields:
            if field in state_info:
                value = state_info[field]
                
                # 转换tensor为numpy
                if isinstance(value, torch.Tensor):
                    value_np = value.cpu().numpy()
                else:
                    value_np = value
                
                self.current_trajectory[field].append(value_np)
            else:
                # 如果字段不存在，记录警告（仅第一次）
                if len(self.current_trajectory[field]) == 0:
                    print(f"[TrajectoryCollector] Warning: field '{field}' not found in state_info")
                self.current_trajectory[field].append(None)
    
    def finish_trajectory(self):
        """完成当前轨迹的收集"""
        if self.current_trajectory is None or len(self.current_trajectory['actions']) == 0:
            return
        
        # 将列表转换为numpy数组
        processed_trajectory = {}
        for field, value_list in self.current_trajectory.items():
            # 过滤掉None值
            valid_values = [v for v in value_list if v is not None]
            
            if len(valid_values) > 0:
                try:
                    # 尝试转换为numpy数组
                    processed_trajectory[field] = np.array(valid_values)
                except:
                    # 如果无法转换（例如shape不一致），保持为列表
                    processed_trajectory[field] = valid_values
        
        # 添加统计信息
        if 'rewards' in processed_trajectory:
            processed_trajectory['total_reward'] = np.sum(processed_trajectory['rewards'])
        processed_trajectory['length'] = len(processed_trajectory['actions']) if 'actions' in processed_trajectory else 0
        
        self.collected_trajectories.append(processed_trajectory)
        
        total_reward = processed_trajectory.get('total_reward', 0)
        length = processed_trajectory['length']
        
        print(f"[TrajectoryCollector] Trajectory {len(self.collected_trajectories)} collected: "
              f"length={length}, total_reward={total_reward:.2f}")
        
        self.current_trajectory = None
    
    def save_trajectories(self):
        """保存所有收集的轨迹到文件"""
        if len(self.collected_trajectories) == 0:
            print("[TrajectoryCollector] No trajectories to save.")
            return
        
        # 保存为字典格式，包含轨迹和元数据
        save_data = {
            'trajectories': self.collected_trajectories,
            'metadata': {
                'num_trajectories': len(self.collected_trajectories),
                'timestamp': datetime.now().isoformat()
            }
        }
        
        save_path = self.save_dir / self.save_filename
        # 保存为pickle文件
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"\n[TrajectoryCollector] Successfully saved {len(self.collected_trajectories)} trajectories to {save_path}")

        return save_path
    
    def get_state_info_from_env(self, action, reward):
        """
        从环境中提取所需的状态信息
        
        Args:
            action: 当前执行的动作
            reward: 当前步的奖励
            
        Returns:
            state_info: 包含所需字段的字典
        """
        state_info = {}
        
        # 从环境中获取状态信息
        # 假设环境有这些属性，需要根据实际环境调整
        env = self.env
        
        # 尝试获取各个字段
        field_mappings = {
            'object_pose': 'object_pose',
            'object_pos': 'object_pos',
            'object_rot': 'object_rot',
            'object_linvel': 'object_linvel',
            'object_angvel': 'object_angvel',
            'goal_pose': 'goal_pose',
            'goal_pos': 'goal_pos',
            'goal_rot': 'goal_rot',
            'fingertip_state': 'fingertip_state',
            'fingertip_pos': 'fingertip_pos',
        }
        
        for field, attr_name in field_mappings.items():
            if hasattr(env, attr_name):
                state_info[field] = getattr(env, attr_name)
            elif hasattr(env, 'task') and hasattr(env.task, attr_name):
                state_info[field] = getattr(env.task, attr_name)
        
        # 添加action和reward
        state_info['actions'] = action
        state_info['rewards'] = reward
        
        return state_info
    
    def get_action(self, obs, is_deterministic=False):
        """
        重写get_action方法，保持原有功能
        """
        if self.has_batch_dimension == False:
            obs_input = unsqueeze_obs(obs)
        else:
            obs_input = obs
        
        obs_input = self._preproc_obs(obs_input)
        
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs': obs_input,
            'rnn_states': self.states
        }
        
        with torch.no_grad():
            res_dict = self.model(input_dict)
        
        mu = res_dict['mus']
        action = res_dict['actions']
        self.states = res_dict['rnn_states']
        
        if is_deterministic:
            current_action = mu
        else:
            current_action = action
        
        if self.has_batch_dimension == False:
            current_action = torch.squeeze(current_action.detach())
        
        if self.clip_actions:
            return rescale_actions(self.actions_low, self.actions_high, 
                                 torch.clamp(current_action, -1.0, 1.0))
        else:
            return current_action
    
    def run(self):
        """
        重写run方法，收集指定数量的轨迹
        """
        is_deterministic = self.is_deterministic
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None

        if has_masks_func:
            has_masks = self.env.has_action_mask()

        self.wait_for_checkpoint()
        need_init_rnn = self.is_rnn
        
        print(f"\n[TrajectoryCollector] Starting trajectory collection...")
        
        # 收集指定数量的轨迹
        while len(self.collected_trajectories) < self.num_trajectories:
            # 初始化新轨迹
            self.init_trajectory()
            
            # 重置环境
            obs = self.env_reset(self.env)
            batch_size = self.get_batch_size(obs, 1)

            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            episode_reward = 0
            episode_steps = 0
            
            # 运行一个episode
            for step in range(self.max_steps):
                # 获取动作
                if has_masks:
                    masks = self.env.get_action_mask()
                    action = self.get_masked_action(obs, masks, is_deterministic)
                else:
                    action = self.get_action(obs, is_deterministic)
                
                # 执行动作
                next_obs, reward, done, info = self.env_step(self.env, action)
                
                # 从环境中提取状态信息
                state_info = self.get_state_info_from_env(action, reward)
                
                # 添加到轨迹
                self.add_step(state_info)
                
                episode_reward += reward.item() if isinstance(reward, torch.Tensor) else reward
                episode_steps += 1
                
                obs = next_obs
                
                # 检查是否完成
                if done.any() if isinstance(done, torch.Tensor) else done:
                    print(f"[TrajectoryCollector] Episode finished: steps={episode_steps}, reward={episode_reward:.2f}")
                    
                    # 完成当前轨迹
                    self.finish_trajectory()
                    
                    # 重置RNN状态
                    if self.is_rnn:
                        self.init_rnn()
                    
                    break
            
            # 如果episode没有正常结束（达到最大步数），也保存轨迹
            if self.current_trajectory is not None:
                print(f"[TrajectoryCollector] Episode reached max steps: steps={episode_steps}, reward={episode_reward:.2f}")
                self.finish_trajectory()
        
        # 保存所有收集的轨迹
        self.save_trajectories()
        
        print(f"\n[TrajectoryCollector] Collection complete! Total trajectories: {len(self.collected_trajectories)}")
        
# 辅助函数
def unsqueeze_obs(obs):
    """为观察添加batch维度"""
    if isinstance(obs, dict):
        return {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v 
                for k, v in obs.items()}
    else:
        return obs.unsqueeze(0) if isinstance(obs, torch.Tensor) else obs


def rescale_actions(low, high, action):
    """将[-1, 1]范围的动作重新缩放到[low, high]"""
    return low + (action + 1.0) * 0.5 * (high - low)
            
class PpoPlayerDiscrete(BasePlayer):

    def __init__(self, params):
        BasePlayer.__init__(self, params)

        self.network = self.config['network']
        if type(self.action_space) is gym.spaces.Discrete:
            self.actions_num = self.action_space.n
            self.is_multi_discrete = False
        if type(self.action_space) is gym.spaces.Tuple:
            self.actions_num = [action.n for action in self.action_space]
            self.is_multi_discrete = True

        self.mask = [False]
        self.normalize_input = self.config['normalize_input']
        self.normalize_value = self.config.get('normalize_value', False)

        obs_shape = self.obs_shape
        config = {
            'actions_num': self.actions_num,
            'input_shape': obs_shape,
            'num_seqs': self.num_agents,
            'value_size': self.env_info.get('value_size', 1),
            'normalize_value': self.normalize_value,
            'normalize_input': self.normalize_input,
        }

        self.model = self.network.build(config)
        self.model.to(self.device)
        self.model.eval()
        self.is_rnn = self.model.is_rnn()

    def get_masked_action(self, obs, action_masks, is_deterministic=True):
        if not self.has_batch_dimension:
            obs = unsqueeze_obs(obs)
        obs = self._preproc_obs(obs)
        action_masks = torch.Tensor(action_masks).to(self.device).bool()
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs': obs,
            'action_masks': action_masks,
            'rnn_states': self.states
        }
        self.model.eval()

        with torch.no_grad():
            res_dict = self.model(input_dict)
        logits = res_dict['logits']
        action = res_dict['actions']
        self.states = res_dict['rnn_states']
        if self.is_multi_discrete:
            if is_deterministic:
                action = [torch.argmax(logit.detach(), axis=-1).squeeze() for logit in logits]
                return torch.stack(action,dim=-1)
            else:    
                return action.squeeze().detach()
        else:
            if is_deterministic:
                return torch.argmax(logits.detach(), axis=-1).squeeze()
            else:
                return action.squeeze().detach()

    def get_action(self, obs, is_deterministic=False):
        if not self.has_batch_dimension:
            obs = unsqueeze_obs(obs)
        obs = self._preproc_obs(obs)

        self.model.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs': obs,
            'rnn_states': self.states
        }
        with torch.no_grad():
            res_dict = self.model(input_dict)
        logits = res_dict['logits']
        action = res_dict['actions']
        self.states = res_dict['rnn_states']
        if self.is_multi_discrete:
            if is_deterministic:
                action = [torch.argmax(logit.detach(), axis=1).squeeze() for logit in logits]
                return torch.stack(action, dim=-1)
            else:
                return action.squeeze().detach()
        else:
            if is_deterministic:
                return torch.argmax(logits.detach(), axis=-1).squeeze()
            else:
                return action.squeeze().detach()

    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.model.load_state_dict(checkpoint['model'])
        if self.normalize_input and 'running_mean_std' in checkpoint:
            self.model.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

        env_state = checkpoint.get('env_state', None)
        if self.env is not None and env_state is not None:
            self.env.set_env_state(env_state)

    def reset(self):
        self.init_rnn()


class SACPlayer(BasePlayer):
    """
    Player implementation for Soft Actor-Critic (SAC) algorithm.
    Handles agent inference for both training and evaluation.
    """

    def __init__(self, params):
        BasePlayer.__init__(self, params)
        self.network = self.config['network']
        self.actions_num = self.action_space.shape[0] 
        self.action_range = [
            float(self.env_info['action_space'].low.min()),
            float(self.env_info['action_space'].high.max())
        ]

        obs_shape = self.obs_shape
        self.normalize_input = self.config.get('normalize_input', False)
        config = {
            'obs_dim': self.env_info["observation_space"].shape[0],
            'action_dim': self.env_info["action_space"].shape[0],
            'actions_num': self.actions_num,
            'input_shape': obs_shape,
            'value_size': self.env_info.get('value_size', 1),
            'normalize_value': False,
            'normalize_input': self.normalize_input,
        }  
        self.model = self.network.build(config)
        self.model.to(self.device)
        self.model.eval()
        self.is_rnn = self.model.is_rnn()

    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.model.sac_network.actor.load_state_dict(checkpoint['actor'])
        self.model.sac_network.critic.load_state_dict(checkpoint['critic'])
        self.model.sac_network.critic_target.load_state_dict(checkpoint['critic_target'])
        if self.normalize_input and 'running_mean_std' in checkpoint:
            self.model.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

        env_state = checkpoint.get('env_state', None)
        if self.env is not None and env_state is not None:
            self.env.set_env_state(env_state)

    def get_action(self, obs, is_deterministic=False):
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        obs = self.model.norm_obs(obs)
        dist = self.model.actor(obs)
        actions = dist.sample() if not is_deterministic else dist.mean
        actions = actions.clamp(*self.action_range).to(self.device)
        if self.has_batch_dimension == False:
            actions = torch.squeeze(actions.detach())
        return actions

    def reset(self):
        pass
