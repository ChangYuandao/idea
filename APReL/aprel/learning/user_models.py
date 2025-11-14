"""Modules for user response models, including human users."""
from copy import deepcopy
from typing import Dict, List, Union
import numpy as np
import scipy.special as ssp
from copy import deepcopy

from aprel.basics import Trajectory, TrajectorySet
from aprel.learning import Query, PreferenceQuery, WeakComparisonQuery, FullRankingQuery
from aprel.learning import QueryWithResponse, Demonstration, Preference, WeakComparison, FullRanking


class User:
    """
    一个抽象类（Abstract Class），用于建模“用户”的行为——即我们希望学习其奖励函数的用户。

    参数:
        params_dict (Dict): 用户模型的参数字典，用来描述用户的偏好、噪声水平等属性。
    """
    def __init__(self, params_dict: Dict = None):
        # 构造函数：初始化用户参数
        if params_dict is not None:
            # 如果传入参数字典，则复制一份保存为内部属性
            self._params = params_dict.copy()
        else:
            # 如果未提供，则创建一个空参数字典
            self._params = {}

    @property
    def params(self):
        """返回当前用户模型的参数字典。"""
        return self._params

    @params.setter
    def params(self, params_dict: Dict):
        """
        当外部设置新的参数时更新用户参数。
        逻辑：新字典优先覆盖已有参数，但保留旧参数中未被新字典覆盖的键。
        """
        
        # 复制输入字典，避免直接修改外部数据
        params_dict_copy = params_dict.copy()
        # 若新字典中缺少某个键，则保留旧值
        for key, value in self._params.items():
            params_dict_copy.setdefault(key, value)
        # 更新内部参数字典
        self._params = params_dict_copy

    def copy(self):
        """返回当前用户对象的深拷贝（deepcopy）。"""
        return deepcopy(self)

    def response_logprobabilities(self, query: Query) -> np.array:
        """
        抽象方法（未实现）：计算在给定 query 下，
        用户对每个可能响应的对数概率（log-probabilities）。

        参数:
            query (Query): 包含一个问题和一组可能响应。

        返回:
            numpy.array: 每个响应的 log 概率。
        """
        
        # 子类必须实现该方法
        raise NotImplementedError

    def response_probabilities(self, query: Query) -> np.array:
        """
        计算用户对每个可能响应的概率（非对数形式）。

        参数:
            query (Query): 查询对象。

        返回:
            numpy.array: 每个响应的概率。
        """
        
        # 概率 = exp(对数概率)
        return np.exp(self.response_logprobabilities(query))

    def loglikelihood(self, data: QueryWithResponse) -> float:
        """
        计算给定的用户反馈（query + 实际响应）的对数似然（log-likelihood）。

        参数:
            data (QueryWithResponse): 包含查询与用户实际选择的响应。

        返回:
            float: 该反馈在当前用户模型下的 log-likelihood 值。
        """
        
        # 获取各响应的 log 概率
        logprobs = self.response_logprobabilities(data)
        
        # 根据 data 类型找到实际响应的索引
        if isinstance(data, Preference) or isinstance(data, WeakComparison):
            # 对于偏好或弱比较类型，用 == 匹配响应
            idx = np.where(data.query.response_set == data.response)[0][0]
        elif isinstance(data, FullRanking):
            # 对于排序类型，需要逐行比较整个响应数组
            idx = np.where((data.query.response_set == data.response).all(axis=1))[0][0]
        # 返回实际响应对应的 log 概率
        return logprobs[idx]

    def likelihood(self, data: QueryWithResponse) -> float:
        """
        返回给定用户反馈的似然（非对数形式）。
        即：likelihood = exp(loglikelihood)
        """
        return np.exp(self.loglikelihood(data))

    def loglikelihood_dataset(self, dataset: List[QueryWithResponse]) -> float:
        """
        计算整个数据集的总 log-likelihood（将各样本 log-likelihood 相加）。

        参数:
            dataset (List[QueryWithResponse]): 含多个 (query, response) 数据的列表。

        返回:
            float: 总 log-likelihood。
        """
        return np.sum([self.loglikelihood(data) for data in dataset])

    def likelihood_dataset(self, dataset: List[QueryWithResponse]) -> float:
        """
        计算整个数据集的总似然（各样本似然的乘积）。
        实现方式：exp(所有 log-likelihood 的和)。
        """
        return np.exp(self.loglikelihood_dataset(dataset))

    def respond(self, queries: Union[Query, List[Query]]) -> List:
        """
        模拟用户在面对给定 query 时的响应行为。

        参数:
            queries (Query 或 List[Query]): 单个或多个查询对象。

        返回:
            List: 用户的响应列表（即便输入是单个 query，输出仍为列表）。
        """
        
        # 如果输入是单个 query，则将其包装成列表
        if not isinstance(queries, list):
            queries = [queries]
        
        # 存储模拟的用户响应
        responses = []
        
        for query in queries:
            # 计算该 query 下的响应概率分布
            probs = self.response_probabilities(query)
            # 按概率分布随机选择一个响应索引
            idx = np.random.choice(len(probs), p=probs)
            # 将选中的响应加入响应列表
            responses.append(query.response_set[idx])
        
        # 返回所有模拟响应
        return responses


class SoftmaxUser(User):
    """
    Softmax 用户类，用户在面对多个轨迹（trajectories）时，
    会根据每个轨迹的奖励值的 softmax 概率分布进行选择。

    参数:
        params_dict (Dict): Softmax 用户模型的参数，包括：
            - `weights` (numpy.array): 奖励函数的线性权重参数。
            - `beta` (float): 比较和排序时的理性程度系数（rationality coefficient）。
            - `beta_D` (float): 用于示范任务的理性系数。
            - `delta` (float): 弱比较（weak comparison）中用户能感知的最小差异。

    异常:
        AssertionError: 若参数字典中未提供 `weights`，则抛出异常。
    """
    def __init__(self, params_dict: Dict):
        
        # 检查输入参数中是否包含必要的 "weights"
        assert('weights' in params_dict), 'weights is a required parameter for the softmax user model.'
        
        # 复制输入参数，避免修改外部变量
        params_dict_copy = params_dict.copy()
        
        # 若未指定其他参数，则设置默认值
        params_dict_copy.setdefault('beta', 1.0)
        params_dict_copy.setdefault('beta_D', 1.0)
        params_dict_copy.setdefault('delta', 0.1)
        
        # 调用父类构造函数，初始化用户参数
        super(SoftmaxUser, self).__init__(params_dict_copy)

    def response_logprobabilities(self, query: Query) -> np.array:
        """
        重写父类的抽象方法。
        计算用户在不同类型 query 下，对每个可能响应的 log 概率。
        """
        
        # ----------- 情况 1：偏好查询 (PreferenceQuery) -----------
        if isinstance(query, PreferenceQuery):
            # reward(query.slate) 计算每个轨迹的奖励
            # beta * reward 表示 rationality-scaled reward（理性调节奖励）
            rewards = self.params['beta'] * self.reward(query.slate)
            # Softmax log 概率公式：
            return rewards - ssp.logsumexp(rewards)
        
        # ----------- 情况 2：弱比较查询 (WeakComparisonQuery) -----------
        elif isinstance(query, WeakComparisonQuery):
            rewards = self.params['beta'] * self.reward(query.slate)
            
            # 三种可能响应：[相等, 选左, 选右]
            logprobs = np.zeros((3))
            
            # log P(选择左轨迹)
            logprobs[1] = -np.log(1 + np.exp(self.params['delta'] + rewards[1] - rewards[0]))
            
            # log P(选择右轨迹)
            logprobs[2] = -np.log(1 + np.exp(self.params['delta'] + rewards[0] - rewards[1]))
            
            # log P(无差异 / 平局)
            # 该项较复杂：使用 δ 控制“差异阈值”
            logprobs[0] = np.log(np.exp(2*self.params['delta']) - 1) + logprobs[1] + logprobs[2]
            return logprobs

        # ----------- 情况 3：完整排序查询 (FullRankingQuery) -----------
        elif isinstance(query, FullRankingQuery):
            rewards = self.params['beta'] * self.reward(query.slate)
            logprobs = np.zeros(len(query.response_set))
            
            # 对每种可能的排序（response）计算 log 概率
            for response_id in range(len(query.response_set)):
                
                # 排序，如 [2,0,1]
                response = query.response_set[response_id]
                
                # 根据排序排列奖励
                sorted_rewards = rewards[response]
                
                # Plackett-Luce 模型的 log 概率：
                logprobs[response_id] = np.sum([sorted_rewards[i] - ssp.logsumexp(sorted_rewards[i:]) for i in range(len(response))])
            return logprobs
        
        # ----------- 其他类型的查询尚未实现 -----------
        raise NotImplementedError("response_logprobabilities is not defined for demonstration queries.")

    def loglikelihood(self, data: QueryWithResponse) -> float:
        """
        重写父类方法：计算单个 (query, response) 的 log-likelihood。

        注意：
        - 对于 Demonstration 类型，返回“非归一化”的 log 概率（未使用 softmax）。
        - 其他类型则为精确的 log 概率。
        """
        
        # ----------- 演示任务 (Demonstration) -----------
        if isinstance(data, Demonstration):
            return self.params['beta_D'] * self.reward(data)
        
        # ----------- 偏好任务 (Preference) -----------
        elif isinstance(data, Preference):
            rewards = self.params['beta'] * self.reward(data.query.slate)
            return rewards[data.response] - ssp.logsumexp(rewards)

        # ----------- 弱比较任务 (WeakComparison) -----------
        elif isinstance(data, WeakComparison):
            rewards = self.params['beta'] * self.reward(data.query.slate)

            # 计算三种可能响应的 log 概率：
            logp0 = -np.log(1 + np.exp(self.params['delta'] + rewards[1] - rewards[0]))
            if data.response == 0: return logp0

            logp1 = -np.log(1 + np.exp(self.params['delta'] + rewards[0] - rewards[1]))
            if data.response == 1: return logp1

            if data.response == -1:
                return np.log(np.exp(2*self.params['delta']) - 1) + logp0 + logp1

        # ----------- 完整排序任务 (FullRanking) -----------
        elif isinstance(data, FullRanking):
            rewards = self.params['beta'] * self.reward(data.query.slate)
            sorted_rewards = rewards[data.response]
            return np.sum([sorted_rewards[i] - ssp.logsumexp(sorted_rewards[i:]) for i in range(len(data.response))])

        # 若遇到未支持的查询类型，抛出异常
        raise NotImplementedError("User response model for the given data is not implemented.")

    def reward(self, trajectories: Union[Trajectory, TrajectorySet]) -> Union[float, np.array]:
        """
        计算给定轨迹（或轨迹集合）的奖励。

        参数:
            trajectories (Trajectory 或 TrajectorySet): 输入轨迹（单个或多个）。

        返回:
            numpy.array 或 float: 对应奖励值（按线性函数计算）。
        """
        
        # 如果输入是多个轨迹的集合：
        if isinstance(trajectories, TrajectorySet):
            
            # 每个轨迹的特征矩阵 × 权重向量 → 奖励向量
            return np.dot(trajectories.features_matrix, self.params['weights'])
        
        # 如果是单条轨迹，则计算其线性奖励
        return np.dot(trajectories.features, self.params['weights'])


class CustomizedRewardSoftmaxUser(SoftmaxUser):
    """
    自定义奖励的 Softmax 用户类（继承自 SoftmaxUser）。

    模型假设：
        - 用户面对多个轨迹时，选择轨迹的概率遵循 softmax 规则：
          P(选择轨迹 i) ∝ exp(奖励(轨迹_i))

    参数:
        params_dict (Dict): 用户模型参数，包含：
            - `weights` (numpy.array)：线性奖励函数的权重（可选，如果不使用自定义奖励可保留）
            - `beta` (float)：偏好与排序的理性系数
            - `beta_D` (float)：演示任务的理性系数
            - `delta` (float)：弱比较任务的感知差异阈值

    抛出:
        AssertionError：如果没有提供 `weights` 参数（继承 SoftmaxUser 时可能会要求）
    """

    def __init__(self, params, hp_ranges, reward_fn):
        """
        初始化自定义奖励 Softmax 用户

        参数:
            params (dict): 用户模型参数
            hp_ranges (dict): 超参数范围字典，用于归一化到真实值
            reward_fn (callable): 自定义奖励函数
        """
        
        # 创建参数字典副本并设置默认参数
        params_dict_copy = params.copy()
        params_dict_copy.setdefault("beta", 1.0)
        params_dict_copy.setdefault("beta_D", 1.0)
        params_dict_copy.setdefault("delta", 0.1)

        # 保存超参数范围和自定义奖励函数
        self.hp_ranges = hp_ranges
        self.reward_fn = reward_fn

        # 调用父类构造函数，初始化 SoftmaxUser 参数
        super(CustomizedRewardSoftmaxUser, self).__init__(params_dict_copy)

    def denormalize(self, normalized_hps):
        """
        将归一化的超参数转换为实际值

        参数:
            normalized_hps (list 或 numpy.array)：归一化超参数值（0~1）

        返回:
            dict: 实际超参数字典
        """
        hps = {}
        
        # 遍历所有超参数，将归一化值映射到真实范围
        for i, (k, v) in enumerate(self.hp_ranges.items()):
            
            # 超参数 k 的最小值和最大值
            min_v, max_v = self.hp_ranges[k]
            
            # 线性映射
            hps[k] = (max_v - min_v) * normalized_hps[i] + min_v
            
        return hps

    def reward(self, trajectories: Union[Trajectory, TrajectorySet]) -> Union[float, np.array]:
        """
        使用自定义奖励函数计算轨迹的奖励

        参数:
            trajectories (Trajectory 或 TrajectorySet)：轨迹或轨迹集合

        返回:
            numpy.array: 每条轨迹的奖励值数组
        """
        rewards = []
        
        # 遍历每条轨迹计算奖励
        for trajectory in trajectories:
            
            # trajectory[0] 假设轨迹第一步包含 (obs, reward_vars)
            obs, reward_vars = trajectory[0]
            
            # 将归一化超参数 denormalize 成实际参数
            inputs = self.denormalize(self.params["norm_hps"])
            
            # 将轨迹观测值和奖励相关变量添加到输入
            inputs['obs'] = obs
            inputs["reward_vars"] = reward_vars
            
            # 调用自定义奖励函数计算奖励
            # 假设 reward_fn 返回数组，第一个元素就是奖励
            rewards.append(self.reward_fn(**inputs)[0])
        
        # 返回奖励数组
        return np.array(rewards)

class HumanUser(User):
    """
    Human user class whose response model is unknown. This class is useful for interactive runs, where
    a real human responds to the queries rather than simulated user models.

    Parameters:
        delay (float): The waiting time between each trajectory visualization during querying in seconds.

    Attributes:
        delay (float): The waiting time between each trajectory visualization during querying in seconds.
    """
    def __init__(self, delay: float = 0.):
        super(HumanUser, self).__init__()
        self.delay = delay

    def respond(self, queries: Union[Query, List[Query]]) -> List:
        """
        Interactively asks for the user's responses to the given queries.

        Args:
            queries (Query or List[Query]): A query or a list of queries for which the user's response(s)
                is/are requested.

        Returns:
            List: A list of user responses where each response corresponds to the query in the :py:attr:`queries`.
                :Note: The return type is always a list, even if the input is a single query.
        """
        if not isinstance(queries, list):
            queries = [queries]
        responses = []
        for query in queries:
            responses.append(query.visualize(self.delay))
        return responses
