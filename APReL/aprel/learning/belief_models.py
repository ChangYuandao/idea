"""
This file contains Belief classes, which store and update the belief distributions about the user whose reward function is being learned.

:TODO: GaussianBelief class will be implemented so that the library will include the following work:
    E. Biyik, N. Huynh, M. J. Kochenderger, D. Sadigh; "Active Preference-Based Gaussian Process Regression for Reward Learning", RSS'20.
"""
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
from aprel.learning import QueryWithResponse, User
from aprel.utils import gaussian_proposal, uniform_logprior


class Belief:
    """An abstract class for Belief distributions."""

    def __init__(self):
        pass

    def update(
        self, data: Union[QueryWithResponse, List[QueryWithResponse]], **kwargs
    ):
        """Updates the belief distribution with a given feedback or a list of feedbacks."""
        raise NotImplementedError


class LinearRewardBelief(Belief):
    """An abstract class for Belief distributions for the problems where reward function is assumed to be a linear function of the features."""

    def __init__(self):
        pass

    @property
    def mean(self) -> Dict:
        """Returns the mean parameters with respect to the belief distribution."""
        raise NotImplementedError


class SamplingBasedBelief(LinearRewardBelief):  # 采样式信念（基于线性奖励）的具体实现
    """
    一个基于采样的信念分布类。

    在该模型中，会保存并使用全部用户反馈数据来计算任意给定参数集合的真实后验值。
    然后使用 Metropolis-Hastings 算法从该真实后验中采样一组参数样本。

    参数:
        user_model (User): 假定的用户响应模型。
        dataset (List[QueryWithResponse]): 用户反馈列表。
        initial_point (Dict): Metropolis-Hastings 初始参数点。
        logprior (Callable): 参数的先验分布的对数，默认是超球均匀分布。
        num_samples (int): 需要的样本数量（采样后的有效样本数）。
        **kwargs: Metropolis-Hastings 的超参数，包括:
            - burnin (int): 初始需要丢弃的样本数量（去相关）。
            - thin (int): 间隔保留的步长（降低自相关）。
            - proposal_distribution (Callable): 提案分布函数。

    属性:
        user_model (User): 当前假设的用户模型。
        dataset (List[QueryWithResponse]): 当前累积的用户反馈数据。
        num_samples (int): 目标采样的样本数量（保留的）。
        sampling_params (Dict): 采样相关超参数（burnin, thin, proposal_distribution）。
    """  # 中文版类说明

    def __init__(  # 构造函数
        self,
        user_model: User,  # 用户模型（需要具备 copy 与 loglikelihood_dataset）
        dataset: List[QueryWithResponse],  # 初始数据集
        initial_point: Dict,  # 初始参数点
        logprior: Callable = uniform_logprior,  # 先验对数函数，默认均匀
        num_samples: int = 100,  # 需要保留的有效样本数量
        **kwargs  # 其它采样相关超参数
    ):
        super(SamplingBasedBelief, self).__init__()  # 调用父类初始化
        self.logprior = logprior  # 保存先验函数
        self.user_model = user_model  # 保存用户模型
        self.dataset = []  # 初始化内部数据集为空列表
        self.num_samples = num_samples  # 保存目标样本数

        kwargs.setdefault("burnin", 200)  # 若未提供则设置默认 burn-in
        kwargs.setdefault("thin", 20)  # 若未提供则设置默认 thin 间隔
        kwargs.setdefault("proposal_distribution", gaussian_proposal)  # 默认提案分布为高斯
        self.sampling_params = kwargs  # 保存采样参数字典
        self.update(dataset, initial_point)  # 用初始数据与初始点进行首次采样

    def update(  # 更新信念分布：添加数据并重新采样
        self,
        data: Union[QueryWithResponse, List[QueryWithResponse]],  # 新增反馈（单个或列表）
        initial_point: Dict = None,  # 若为 None 则用当前均值作为初始点
    ):
        """
        根据新反馈（查询-响应对）更新信念：追加到当前数据集并重新执行 MH 采样。
        参数:
            data: 一个或多个 QueryWithResponse 实例，包含多轨迹选项及用户选择索引。
            initial_point: MH 初始参数；若为 None 则使用当前分布均值。
        """  # 中文版方法说明
        if isinstance(data, list):  # 判断是否为列表
            self.dataset.extend(data)  # 批量追加数据
        else:
            self.dataset.append(data)  # 单条追加
        if initial_point is None:  # 若未指定初始点
            initial_point = self.mean  # 使用当前样本均值作为初始点

        self.create_samples(initial_point)  # 重新进行采样

    def create_samples(  # 进行 Metropolis-Hastings 采样
        self, initial_point: Dict  # 初始参数点
    ) -> Tuple[List[Dict], List[float]]:  # 返回样本与对应对数概率
        """使用 Metropolis-Hastings 从后验中采样 num_samples 个用户参数样本。
        参数:
            initial_point (Dict): 采样链初始位置。
        返回:
            (samples, logprobs):
                samples: 参数字典的列表，每个元素是一个采样点。
                logprobs: 每个采样点对应的后验对数概率。
        """  # 中文版采样函数说明
        burnin = self.sampling_params["burnin"]  # 提取 burn-in 步数
        thin = self.sampling_params["thin"]  # 提取 thin 步长
        proposal_distribution = self.sampling_params["proposal_distribution"]  # 提案分布函数

        samples = []  # 用于暂存全部原始链样本
        logprobs = []  # 用于暂存全部原始链样本的对数概率
        curr_point = initial_point.copy()  # 当前采样点（复制避免引用问题）
        sampling_user = self.user_model.copy()  # 复制用户模型用于计算似然
        sampling_user.params = curr_point  # 设置用户模型参数为当前点
        curr_logprob = self.logprior(curr_point) + sampling_user.loglikelihood_dataset(self.dataset)  # 计算当前点的后验对数（先验+似然）
        samples.append(curr_point)  # 保存第一个样本
        logprobs.append(curr_logprob)  # 保存第一个样本的对数概率
        for _ in range(burnin + thin * self.num_samples - 1):  # 迭代总步数（包含 burn-in 与之后采样区间）
            next_point = proposal_distribution(curr_point)  # 从提案分布生成下一个候选点
            sampling_user.params = next_point  # 设置模型参数为候选点
            next_logprob = self.logprior(next_point) + sampling_user.loglikelihood_dataset(self.dataset)  # 计算候选点后验对数
            if np.log(np.random.rand()) < next_logprob - curr_logprob:  # MH 接受判据（对数形式）
                curr_point = next_point.copy()  # 接受：更新当前点（复制避免引用）
                curr_logprob = next_logprob  # 更新当前对数概率
            samples.append(curr_point)  # 保存本步（可能是新点或重复旧点）
            logprobs.append(curr_logprob)  # 保存对应对数概率
        self.samples, self.logprobs = (samples[burnin::thin], logprobs[burnin::thin])  # 丢弃 burn-in 并按 thin 抽取，形成最终样本与概率

    @property
    def mean(self) -> Dict:  # 计算样本均值作为当前信念的均值估计
        """通过对已生成的 MH 样本做均值来返回参数均值。若参数名为 'weights' 则再归一化。"""  # 中文版均值说明
        mean_params = {}  # 存放均值结果
        for key in self.samples[0].keys():  # 遍历参数字典的全部键
            mean_params[key] = np.mean(  # 计算该键在所有样本中的逐元素均值
                [self.samples[i][key] for i in range(self.num_samples)], axis=0
            )
            if key == "weights":  # 若参数是权重向量
                mean_params[key] /= np.linalg.norm(mean_params[key])  # 进行 L2 归一化
        return mean_params  # 返回均值参数字典
