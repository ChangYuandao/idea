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
                 beta: float = 1.0,
                 task_name: str = None):
        """
        初始化偏好学习
        
        Args:
            hp_ranges: 超参数范围字典
            initial_values: 初始参数值字典
            reward_file_path: 奖励函数文件路径
            beta: 理性系数
            task_name: 任务名称(用于加载评估函数)
        """
        self.hp_ranges = hp_ranges
        self.beta = beta
        self.reward_function = IsaacGymRewardFunction(reward_file_path)
        self.task_name = task_name
        
        # 使用提供的初始值
        self.hps = initial_values.copy()
        
        print(f"[IsaacGymPreferenceLearning] Initialized with parameters:")
        for k, v in self.hps.items():
            print(f"  {k}: {v:.4f} (range: {hp_ranges[k]})")
    
    def load_evaluation_functions(self, evaluate_dir: str) -> Dict[str, List]:
        """
        加载轨迹评估函数
        
        Args:
            evaluate_dir: 评估函数目录路径
        
        Returns:
            评估函数字典 {role: [func1, func2, ...]}
        """
        eval_funcs = {
            "GOAL": [],
            "SAFE": [],
            "EFFICIENCY": []
        }
        
        for role in eval_funcs.keys():
            for i in range(1, 6):
                func_file = Path(evaluate_dir) / f"{role}_{i}.txt"
                if func_file.exists():
                    try:
                        # 读取函数代码
                        with open(func_file, 'r') as f:
                            func_code = f.read()
                        
                        # 动态执行函数定义
                        local_namespace = {}
                        exec(func_code, globals(), local_namespace)
                        
                        if 'trajectory_evaluate' in local_namespace:
                            eval_funcs[role].append(local_namespace['trajectory_evaluate'])
                            logging.info(f"Loaded evaluation function: {role}_{i}")
                        else:
                            logging.warning(f"Function 'trajectory_evaluate' not found in {func_file}")
                    except Exception as e:
                        logging.error(f"Failed to load {func_file}: {e}")
                else:
                    logging.warning(f"Evaluation function file not found: {func_file}")
        
        return eval_funcs
    
    def compare_trajectories(self, traj_a: Dict, traj_b: Dict, 
                            eval_funcs: Dict[str, List]) -> List[int]:
        """
        使用多个评估函数比较两条轨迹
        
        Args:
            traj_a: 轨迹A
            traj_b: 轨迹B
            eval_funcs: 评估函数字典 {role: [func1, func2, ...]}
        
        Returns:
            最终的标签列表 (每个时间步: 1表示A优于B, -1表示A劣于B, 0表示无法区分)
        """
        # 存储所有评估结果
        role_labels = {
            "GOAL": [],
            "SAFE": [],
            "EFFICIENCY": []
        }
        
        # 对每个角色的所有函数进行评估
        for role, funcs in eval_funcs.items():
            for func_idx, func in enumerate(funcs):
                try:
                    label_list = func(traj_a, traj_b)
                    role_labels[role].append(np.array(label_list))
                except Exception as e:
                    import traceback
                    logging.error(f"Error evaluating with {role} function {func_idx}: {e}")
                    logging.error(f"Traceback: {traceback.format_exc()}")
                    logging.error(f"Trajectory A keys: {list(traj_a.keys())}")
                    logging.error(f"Trajectory B keys: {list(traj_b.keys())}")
                    # 打印轨迹A的第一个状态的形状信息
                    if len(traj_a) > 0:
                        first_state_a = traj_a[list(traj_a.keys())[0]]
                        logging.error(f"First state A type: {type(first_state_a)}")
                        if isinstance(first_state_a, np.ndarray):
                            logging.error(f"First state A shape: {first_state_a.shape}")
                    # 如果评估失败,添加全0标签
                    if len(role_labels[role]) > 0:
                        role_labels[role].append(np.zeros_like(role_labels[role][0]))
        
        # 计算每个角色的平均得分
        role_scores = {}
        for role, labels in role_labels.items():
            if len(labels) > 0:
                # 取所有函数的平均值
                role_scores[role] = np.mean(labels, axis=0)

        
        # 加权计算最终得分 (GOAL=0.6, SAFE=0.3, EFFICIENCY=0.1)
        weights = {
            "GOAL": 0.7,
            "SAFE": 0.15,
            "EFFICIENCY": 0.15
        }
        
        final_scores = (
            weights["GOAL"] * role_scores["GOAL"] +
            weights["SAFE"] * role_scores["SAFE"] +
            weights["EFFICIENCY"] * role_scores["EFFICIENCY"]
        )
        
        # 根据得分范围映射到 {-1, 0, 1}
        final_labels = np.zeros_like(final_scores, dtype=int)
        final_labels[final_scores <= -0.5] = -1
        final_labels[final_scores >= 0.5] = 1
        
        return final_labels.tolist()
    
    def check_consecutive_preference(self, labels: List[int], 
                                    min_consecutive: int = 5) -> Tuple[bool, List[Tuple[int, Tuple[int, int]]]]:
        """
        检查所有连续的偏好标签片段
        
        Args:
            labels: 标签列表
            min_consecutive: 最小连续步数
        
        Returns:
            (是否有连续偏好, [(偏好值, (起始索引, 结束索引)), ...])
            偏好值: 1表示A更好, -1表示B更好
        """
        if len(labels) < min_consecutive:
            return False, []
        
        segments = []  # 存储所有有效片段: [(preference_value, (start, end)), ...]
        
        current_label = 0
        current_start = -1
        consecutive_count = 0
        
        for i, label in enumerate(labels):
            if label == current_label and label != 0:
                # 继续当前连续片段
                consecutive_count += 1
            else:
                # 检查之前的片段是否满足要求
                if consecutive_count >= min_consecutive and current_label != 0:
                    segments.append((current_label, (current_start, i)))
                
                # 开始新的片段
                if label != 0:
                    current_label = label
                    current_start = i
                    consecutive_count = 1
                else:
                    current_label = 0
                    current_start = -1
                    consecutive_count = 0
        
        # 检查最后一个片段
        if consecutive_count >= min_consecutive and current_label != 0:
            segments.append((current_label, (current_start, len(labels))))
        
        has_preference = len(segments) > 0
        
        return has_preference, segments

    def generate_preference_buffer(self, trajectories: List[Dict], 
                                evaluate_dir: str,
                                min_consecutive: int = 5) -> List[Tuple[int, int, int, Tuple[int, int], Tuple[int, int]]]:
        """
        生成偏好对缓冲区（支持多个片段）
        
        Args:
            trajectories: 轨迹列表
            evaluate_dir: 评估函数目录路径
            min_consecutive: 最小连续偏好步数
        
        Returns:
            偏好对列表 [(traj_a_idx, traj_b_idx, preference, segment_a, segment_b), ...]
            preference: 0表示traj_a更好, 1表示traj_b更好
            segment_a, segment_b: 轨迹片段范围 (start, end)
        """
        logging.info(f"[Preference Buffer] Loading evaluation functions from {evaluate_dir}")
        
        # 加载评估函数
        eval_funcs = self.load_evaluation_functions(evaluate_dir)
        
        # 统计加载的函数数量
        total_funcs = sum(len(funcs) for funcs in eval_funcs.values())
        logging.info(f"[Preference Buffer] Loaded {total_funcs} evaluation functions")
        for role, funcs in eval_funcs.items():
            logging.info(f"  {role}: {len(funcs)} functions")
        
        preference_buffer = []
        n_trajs = len(trajectories)
        
        logging.info(f"[Preference Buffer] Comparing {n_trajs} trajectories...")
        
        # 两两比较所有轨迹
        comparison_count = 0
        total_segments = 0
        
        for i in range(n_trajs):
            for j in range(i + 1, n_trajs):
                comparison_count += 1
                
                
                # 截取轨迹到最小长度
                traj_a = self.reconstruct_trajectory(trajectories[i])
                traj_b = self.reconstruct_trajectory(trajectories[j])
                min_length = min(len(traj_a), len(traj_b))
                traj_a = traj_a[:min_length]
                traj_b = traj_b[:min_length]
                    
                try:
                    # 比较两条轨迹
                    labels = self.compare_trajectories(traj_a, traj_b, eval_funcs)
                    logging.info(f"  Compared Trajectory {i} vs Trajectory {j}: Labels = {labels}")
                    
                    # 检查是否有连续偏好（返回所有片段）
                    has_preference, segments = self.check_consecutive_preference(
                        labels, min_consecutive
                    )
                    
                    if has_preference:
                        logging.info(f"  ✅ Found {len(segments)} preference segment(s)")
                        
                        # 为每个片段创建偏好对
                        for seg_idx, (preference_value, segment_range) in enumerate(segments):
                            start, end = segment_range
                            
                            if preference_value == 1:
                                # A更好
                                preference_buffer.append((i, j, 0, segment_range, segment_range))
                                logging.info(
                                    f"    Segment {seg_idx+1}: Trajectory {i} > Trajectory {j}, "
                                    f"Steps [{start}:{end}] ({end-start} steps)"
                                )
                            elif preference_value == -1:
                                # B更好
                                preference_buffer.append((i, j, 1, segment_range, segment_range))
                                logging.info(
                                    f"    Segment {seg_idx+1}: Trajectory {j} > Trajectory {i}, "
                                    f"Steps [{start}:{end}] ({end-start} steps)"
                                )
                            
                            total_segments += 1
                    else:
                        logging.info(f"  ⚠️ No strong preference detected")
                        
                except Exception as e:
                    logging.error(f"  ❌ Comparison failed: {e}")
                    import traceback
                    logging.error(traceback.format_exc())
                    continue
        
        logging.info(
            f"\n[Preference Buffer] Summary:\n"
            f"  - Trajectory comparisons: {comparison_count}\n"
            f"  - Total preference segments: {total_segments}\n"
            f"  - Unique preference pairs: {len(preference_buffer)}"
        )
        
        return preference_buffer
    
    def extract_trajectory_segment(self, trajectory: Dict, segment: Tuple[int, int]) -> Dict:
        """
        从完整轨迹中提取指定片段
        
        Args:
            trajectory: 完整轨迹字典
            segment: 片段范围 (start, end)
        
        Returns:
            片段轨迹字典
        """
        start, end = segment
        
        if start < 0 or end > trajectory.get('length', 0):
            logging.warning(f"Invalid segment range ({start}, {end}) for trajectory length {trajectory.get('length', 0)}")
            # 调整范围
            start = max(0, start)
            end = min(trajectory.get('length', 0), end)
        
        # 创建新的轨迹字典
        segment_traj = {}
        
        for key, value in trajectory.items():
            if key in ['length', 'total_reward']:
                # 特殊字段处理
                if key == 'length':
                    segment_traj[key] = end - start
                # total_reward 不复制
                continue
            
            # 提取片段数据
            if isinstance(value, np.ndarray):
                if value.ndim > 0 and len(value) >= end:
                    segment_traj[key] = value[start:end]
                else:
                    # 标量或长度不足，直接复制
                    segment_traj[key] = value
            elif isinstance(value, list):
                if len(value) >= end:
                    segment_traj[key] = value[start:end]
                else:
                    segment_traj[key] = value
            else:
                # 标量直接复制
                segment_traj[key] = value
        
        return segment_traj
    
    def reconstruct_trajectory(self, traj_dict):
        """
        将保存的轨迹字典拆分为每个时间步的状态字典列表
        """
        length = traj_dict['length']
        traj_list = []
        for t in range(length):
            state = {k: (v[t] if np.ndim(v) > 0 and hasattr(v, '__getitem__') else v) 
                    for k, v in traj_dict.items() 
                    if k not in ['length', 'total_reward']}
            traj_list.append(state)
        return traj_list
    
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
    
    def normalize_hps(self, hps: Dict) -> Dict:
        """归一化超参数到[0, 1]"""
        normalized = {}
        for k, v in hps.items():
            min_v, max_v = self.hp_ranges[k]
            normalized[k] = (v - min_v) / (max_v - min_v)
        return normalized
    
    def update_reward_parameters(self, 
                                trajectories: List[Dict],
                                preferences: List[Tuple[int, int, int, Tuple[int, int], Tuple[int, int]]],
                                ) -> Dict:
        """
        使用偏好学习更新奖励函数参数（使用片段）
        
        Args:
            trajectories: 完整轨迹列表
            preferences: 偏好对列表 [(traj1_idx, traj2_idx, preferred_idx, segment1, segment2), ...]
        
        Returns:
            更新后的参数字典
        """
        print(f"\n[IsaacGymPreferenceLearning] Starting parameter update with {len(preferences)} preference segments")
        
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
        for i, (idx1, idx2, preferred, segment1, segment2) in enumerate(preferences):
            # 提取轨迹片段
            traj1_segment = self.extract_trajectory_segment(trajectories[idx1], segment1)
            traj2_segment = self.extract_trajectory_segment(trajectories[idx2], segment2)
            
            # 构造偏好查询（使用片段）
            traj1 = IsaacGymTrajectoryStep(traj1_segment)
            traj2 = IsaacGymTrajectoryStep(traj2_segment)
            
            query = aprel.PreferenceQuery([traj1, traj2])
            
            # 更新信念
            belief.update(aprel.Preference(query, [preferred]))
            
            start1, end1 = segment1
            start2, end2 = segment2
            print(
                f"[Update {i+1}/{len(preferences)}] "
                f"Traj {idx1}[{start1}:{end1}] vs Traj {idx2}[{start2}:{end2}] "
                f"-> Preferred: {idx1 if preferred == 0 else idx2}"
            )
        
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