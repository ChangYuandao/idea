import torch
import numpy as np
import pickle
import random
import aprel
from pathlib import Path
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
import logging
import importlib.util
import sys
import inspect
import re
import ast
from copy import deepcopy


class IsaacGymRewardFunction:
    """
    IsaacGym环境的可调参数奖励函数(动态加载)
    """
    def __init__(self, reward_file_path: str = None):
        """
        Args:
            reward_file_path: 奖励函数文件路径
        """
        self.reward_file_path = reward_file_path
        self.reward_module = None
        self.compute_reward_func = None
        self.func_signature = None
        self.param_names = []
        
        if reward_file_path:
            self.load_reward_function(reward_file_path)
    
    def __deepcopy__(self, memo):
        """
        自定义深拷贝方法,避免复制不可序列化的模块对象
        """
        # 创建新实例时重新加载模块
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        
        # 复制基本属性
        result.reward_file_path = self.reward_file_path
        result.param_names = self.param_names.copy()
        
        # 重新加载模块(而不是复制)
        if self.reward_file_path:
            result.load_reward_function(self.reward_file_path)
        else:
            result.reward_module = None
            result.compute_reward_func = None
            result.func_signature = None
        
        return result
    
    def extract_function_params_from_source(self, source_code: str) -> List[str]:
        """
        从源代码中提取 compute_reward 函数的参数列表
        
        Args:
            source_code: 源代码字符串
            
        Returns:
            参数名列表
        """
        # 方法1: 使用正则表达式
        pattern = r'def\s+compute_reward\s*\((.*?)\):'
        match = re.search(pattern, source_code, re.DOTALL)
        
        if match:
            params_str = match.group(1)
            # 清理参数字符串
            params_str = re.sub(r'\s+', ' ', params_str).strip()
            
            # 分割参数
            if not params_str or params_str == '':
                return []
            
            params = []
            for param in params_str.split(','):
                param = param.strip()
                # 移除类型注解
                param = re.sub(r':\s*\w+.*?(?=,|$)', '', param)
                # 移除默认值
                param = re.sub(r'=.*?(?=,|$)', '', param).strip()
                if param:
                    params.append(param)
            
            return params
        
        # 方法2: 使用 AST 解析(更可靠)
        try:
            tree = ast.parse(source_code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == 'compute_reward':
                    params = []
                    for arg in node.args.args:
                        params.append(arg.arg)
                    return params
        except Exception as e:
            logging.warning(f"Failed to parse source with AST: {e}")
        
        return []
    
    def load_reward_function(self, reward_file_path: str):
        """
        动态加载奖励函数
        
        Args:
            reward_file_path: 奖励函数文件路径
        """
        self.reward_file_path = reward_file_path
        
        # 先读取源代码提取参数信息
        with open(reward_file_path, 'r') as f:
            source_code = f.read()
        
        # 提取参数列表
        self.param_names = self.extract_function_params_from_source(source_code)
        
        print(f"Extracted function parameters from source: {self.param_names}")
        
        # 动态加载模块
        spec = importlib.util.spec_from_file_location("reward_module", reward_file_path)
        self.reward_module = importlib.util.module_from_spec(spec)
        sys.modules["reward_module"] = self.reward_module
        spec.loader.exec_module(self.reward_module)
        
        # 获取 compute_reward 函数
        if hasattr(self.reward_module, 'compute_reward'):
            self.compute_reward_func = self.reward_module.compute_reward
            logging.info(f"Successfully loaded compute_reward from {reward_file_path}")
            logging.info(f"Function signature: compute_reward({', '.join(self.param_names)})")
        else:
            raise ValueError(f"compute_reward function not found in {reward_file_path}")
    
    def compute_reward(self, **kwargs) -> Tuple[float, Dict[str, float]]:
        """
        计算奖励(调用动态加载的函数)
        
        Args:
            **kwargs: 所有可能的参数,根据函数签名自动匹配
        
        Returns:
            total_reward: 总奖励
            reward_components: 奖励分量字典
        """
        if self.compute_reward_func is None:
            raise ValueError("Reward function not loaded. Call load_reward_function first.")
        
        # 准备传递给函数的参数
        func_args = {}
        
        for param_name in self.param_names:
            if param_name in kwargs:
                value = kwargs[param_name]
                
                # 确保输入是 torch.Tensor(如果原值不是)
                if isinstance(value, np.ndarray):
                    value = torch.tensor(value, dtype=torch.float32)
                
                func_args[param_name] = value
            else:
                logging.warning(f"Missing parameter: {param_name}")
        
        # 调用加载的函数
        try:
            result = self.compute_reward_func(**func_args)
            
            # 处理返回值
            if isinstance(result, tuple):
                total_reward, reward_components = result
            else:
                # 如果只返回一个值,假设是总奖励
                total_reward = result
                reward_components = {}
            
            # 转换为 numpy
            if isinstance(total_reward, torch.Tensor):
                total_reward = total_reward.detach().cpu().numpy().astype(np.float32)
            
            return total_reward, reward_components
        except Exception as e:
            logging.error(f"Error executing compute_reward: {e}")
            logging.error(f"Expected params: {self.param_names}")
            logging.error(f"Provided params: {list(func_args.keys())}")
            raise

# ...existing code...


class IsaacGymTrajectoryStep(aprel.basics.Trajectory):
    """
    IsaacGym轨迹的单步表示（用于APReL）
    """
    def __init__(self, trajectory_data: Dict):
        """
        Args:
            trajectory_data: 包含轨迹所有信息的字典
        """
        self.trajectory_data = trajectory_data
        # APReL需要的trajectory格式
        self.trajectory = [(trajectory_data, None)]
        self.features = [(trajectory_data, None)]
        self.clip_path = None


class IsaacGymRewardSoftmaxUser(aprel.SoftmaxUser):
    """
    IsaacGym环境的自定义奖励Softmax用户类
    """
    
    def __init__(self, params, hp_ranges, reward_function):
        """
        初始化
        
        Args:
            params: 用户模型参数
            hp_ranges: 超参数范围字典
            reward_function: IsaacGymRewardFunction实例
        """
        params_dict_copy = params.copy()
        params_dict_copy.setdefault("beta", 1.0)
        params_dict_copy.setdefault("beta_D", 1.0)
        params_dict_copy.setdefault("delta", 0.1)
        
        self.hp_ranges = hp_ranges
        self.reward_function = reward_function
        
        super(IsaacGymRewardSoftmaxUser, self).__init__(params_dict_copy)
    
    def denormalize(self, normalized_hps):
        """
        将归一化的超参数转换为实际值
        
        Args:
            normalized_hps: 归一化超参数值（0~1）
        
        Returns:
            实际超参数字典
        """
        hps = {}
        for i, (k, v) in enumerate(self.hp_ranges.items()):
            min_v, max_v = self.hp_ranges[k]
            hps[k] = (max_v - min_v) * normalized_hps[i] + min_v
        return hps
    
    def reward(self, trajectories):
        """
        使用自定义奖励函数计算轨迹的奖励
        
        Args:
            trajectories: 轨迹或轨迹集合
        
        Returns:
            每条轨迹的奖励值数组
        """
        rewards = []
        
        for trajectory in trajectories:
            # 获取轨迹数据
            traj_data, _ = trajectory[0]
            
            # 从轨迹中提取最后一步的所有数据
            last_idx = -1
            
            # 构建参数字典：将轨迹中的所有 key-value 传递给 compute_reward
            func_kwargs = {}
            
            for key, value in traj_data.items():
                # 如果是数组类型，提取最后一个时间步
                if isinstance(value, (np.ndarray, list)):
                    if len(value) > 0:
                        func_kwargs[key] = value[last_idx]
                    else:
                        func_kwargs[key] = value
                else:
                    # 标量直接传递
                    func_kwargs[key] = value
            
            # 调用奖励函数（自动匹配参数）
            try:
                reward, _ = self.reward_function.compute_reward(**func_kwargs)
                
                # 如果返回的是数组，取平均值
                if isinstance(reward, np.ndarray):
                    reward = reward.mean()
                
                rewards.append(float(reward))
            except Exception as e:
                logging.error(f"Failed to compute reward for trajectory: {e}")
                logging.error(f"Available keys in trajectory: {list(traj_data.keys())}")
                # 返回一个失败的奖励值
                rewards.append(-1000.0)
        
        return np.array(rewards)


class IsaacGymPreferenceLearning:
    """
    IsaacGym环境的偏好学习主类
    """
    
    def __init__(self, hp_ranges: Dict[str, Tuple[float, float]], 
                 initial_values: Dict[str, float],
                 reward_file_path: str,
                 beta: float = 1.0):
        """
        初始化偏好学习
        
        Args:
            hp_ranges: 超参数范围字典
            initial_values: 初始参数值字典
            reward_file_path: 奖励函数文件路径
            beta: 理性系数
        """
        self.hp_ranges = hp_ranges
        self.beta = beta
        self.reward_function = IsaacGymRewardFunction(reward_file_path)
        
        # 使用提供的初始值
        self.hps = initial_values.copy()
        
        print(f"[IsaacGymPreferenceLearning] Initialized with parameters:")
        for k, v in self.hps.items():
            print(f"  {k}: {v:.4f} (range: {hp_ranges[k]})")
    
    def normalize_hps(self, hps: Dict) -> Dict:
        """归一化超参数到[0, 1]"""
        normalized = {}
        for k, v in hps.items():
            min_v, max_v = self.hp_ranges[k]
            normalized[k] = (v - min_v) / (max_v - min_v)
        return normalized
    
    def load_trajectories(self, filepath: str) -> List[Dict]:
        """
        加载轨迹文件
        
        Args:
            filepath: 轨迹文件路径
        
        Returns:
            轨迹列表
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        trajectories = data['trajectories']
        print(f"[IsaacGymPreferenceLearning] Loaded {len(trajectories)} trajectories")
        
        # 打印第一条轨迹的键，用于调试
        if len(trajectories) > 0:
            print(f"Trajectory keys: {list(trajectories[0].keys())}")
        
        return trajectories
    
    def generate_random_preferences(self, trajectories: List[Dict], n_pairs: int = 10) -> List[Tuple[int, int, int]]:
        """
        随机生成偏好对（用于验证）
        
        Args:
            trajectories: 轨迹列表
            n_pairs: 生成的偏好对数量
        
        Returns:
            偏好对列表，每个元素为 (traj1_idx, traj2_idx, preferred_idx)
            preferred_idx: 0表示traj1更好，1表示traj2更好
        """
        preferences = []
        n_trajs = len(trajectories)
        
        for _ in range(n_pairs):
            # 随机选择两条不同的轨迹
            idx1, idx2 = random.sample(range(n_trajs), 2)
            
            # 随机选择哪一个更好（或基于总奖励）
            total_reward1 = trajectories[idx1]['total_reward']
            total_reward2 = trajectories[idx2]['total_reward']
            
            # 基于总奖励决定偏好（也可以完全随机）
            if total_reward1 > total_reward2:
                preferred = 0
            else:
                preferred = 1
            
            preferences.append((idx1, idx2, preferred))
            
            print(f"[Preference] Trajectory {idx1} (reward: {total_reward1:.2f}) vs "
                  f"Trajectory {idx2} (reward: {total_reward2:.2f}) -> "
                  f"Preferred: {idx1 if preferred == 0 else idx2}")
        
        return preferences
    
    def update_reward_parameters(self, 
                                 trajectories: List[Dict],
                                 preferences: List[Tuple[int, int, int]],
                                ) -> Dict:
        """
        使用偏好学习更新奖励函数参数
        
        Args:
            trajectories: 轨迹列表
            preferences: 偏好对列表 [(traj1_idx, traj2_idx, preferred_idx), ...]
            visualize: 是否可视化更新过程
        
        Returns:
            更新后的参数字典
        """
        print(f"\n[IsaacGymPreferenceLearning] Starting parameter update with {len(preferences)} preferences")
        
        # 初始化参数
        params = {
            "beta": self.beta,
            "weights": None,
            "norm_hps": np.array([v for k, v in self.normalize_hps(self.hps).items()]),
        }
        
        # 定义用户模型
        user_model = IsaacGymRewardSoftmaxUser(
            params, self.hp_ranges, self.reward_function
        )
        
        # 初始化贝叶斯信念
        belief = aprel.SamplingBasedBelief(user_model, [], params)
        
        
        # 遍历偏好对进行贝叶斯更新
        for i, (idx1, idx2, preferred) in enumerate(preferences):
            # 构造偏好查询
            traj1 = IsaacGymTrajectoryStep(trajectories[idx1])
            traj2 = IsaacGymTrajectoryStep(trajectories[idx2])
            
            query = aprel.PreferenceQuery([traj1, traj2])
            
            # 更新信念（preferred是0或1，表示第几个轨迹更好）
            belief.update(aprel.Preference(query, [preferred]))
            
            print(f"[Update {i+1}/{len(preferences)}] Processed preference pair ({idx1}, {idx2}) -> {preferred}")
        
        # 获取更新后的参数
        updated_norm_hps = belief.mean['norm_hps']
        
        # 反归一化
        updated_hps = {}
        for i, (k, v) in enumerate(self.hp_ranges.items()):
            min_v, max_v = v
            updated_hps[k] = (max_v - min_v) * updated_norm_hps[i] + min_v
        
        
        # 打印更新结果
        print(f"\n[IsaacGymPreferenceLearning] Parameter update complete!")
        print("Initial parameters:")
        for k, v in self.hps.items():
            print(f"  {k}: {v:.4f}")
        print("\nUpdated parameters:")
        for k, v in updated_hps.items():
            print(f"  {k}: {v:.4f}")
        print("\nChanges:")
        for k in self.hps.keys():
            change = updated_hps[k] - self.hps[k]
            change_pct = (change / self.hps[k]) * 100 if self.hps[k] != 0 else 0
            print(f"  {k}: {change:+.4f} ({change_pct:+.2f}%)")
        
        # 更新内部参数
        self.hps = updated_hps
        
        return updated_hps

