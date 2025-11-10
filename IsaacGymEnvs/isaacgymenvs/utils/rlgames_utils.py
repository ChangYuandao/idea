# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import os
from collections import deque
from typing import Callable, Dict, Tuple, Any
import isaacgym
import os
import gym
import numpy as np
import torch
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import AlgoObserver
from rl_games.algos_torch import torch_ext

from isaacgymenvs.tasks import isaacgym_task_map
from isaacgymenvs.utils.utils import set_seed, flatten_dict


def import_class_from_file(file_path, function_name):
    spec = importlib.util.spec_from_file_location("module.name", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    function = getattr(module, function_name)
    return function


def multi_gpu_get_rank(multi_gpu):
    if multi_gpu:
        rank = int(os.getenv("LOCAL_RANK", "0"))
        print("GPU rank: ", rank)
        return rank

    return 0


def get_rlgames_env_creator(
        # used to create the vec task
        seed: int,
        task_config: dict,
        task_name: str,
        sim_device: str,
        rl_device: str,
        graphics_device_id: int,
        headless: bool,
        env_path: str = '',
        # Used to handle multi-gpu case
        multi_gpu: bool = False,
        post_create_hook: Callable = None,
        virtual_screen_capture: bool = False,
        force_render: bool = False,
):
    """Parses the configuration parameters for the environment task and creates a VecTask

    Args:
        task_config: environment configuration.
        task_name: Name of the task, used to evaluate based on the imported name (eg 'Trifinger')
        sim_device: The type of env device, eg 'cuda:0'
        rl_device: Device that RL will be done on, eg 'cuda:0'
        graphics_device_id: Graphics device ID.
        headless: Whether to run in headless mode.
        multi_gpu: Whether to use multi gpu
        post_create_hook: Hooks to be called after environment creation.
            [Needed to setup WandB only for one of the RL Games instances when doing multiple GPUs]
        virtual_screen_capture: Set to True to allow the users get captured screen in RGB array via `env.render(mode='rgb_array')`. 
        force_render: Set to True to always force rendering in the steps (if the `control_freq_inv` is greater than 1 we suggest stting this arg to True)
    Returns:
        A VecTaskPython object.
    """
    def create_rlgpu_env():
        """
        Creates the task from configurations and wraps it using RL-games wrappers if required.
        """
        
        # 若启用了多 GPU（例如使用 Horovod 分布式训练），每个进程通过环境变量 LOCAL_RANK 获取 GPU ID
        # 否则使用用户指定的 sim_device 与 rl_device
        if multi_gpu:

            rank = int(os.getenv("LOCAL_RANK", "0"))
            
            print("Horovod rank: ", rank)

            _sim_device = f'cuda:{rank}'
            _rl_device = f'cuda:{rank}'

            task_config['rank'] = rank
            task_config['rl_device'] = 'cuda:' + str(rank)
        else:
            _sim_device = sim_device
            _rl_device = rl_device
            
        try:
            # task_caller = import_class_from_file(env_path, task_name)
            import importlib
            module_name = f"isaacgymenvs.tasks.{task_config['env']['env_name'].lower()}"
            module = importlib.import_module(module_name)
            # 创建对应的任务类
            task_caller = getattr(module, task_name)
        except:
            print("Could not import task from file, trying from isaacgym_task_map")
            task_caller = isaacgym_task_map[task_name]
        
        env = task_caller(
            cfg=task_config,
            rl_device=_rl_device,
            sim_device=_sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )

        if post_create_hook is not None:
            post_create_hook()

        return env
    return create_rlgpu_env


class RLGPUAlgoObserver(AlgoObserver):
    """Allows us to log stats from the env along with the algorithm running stats. """

    def __init__(self):
        super().__init__()
        self.algo = None
        self.writer = None

        # ep_infos 存储单个 episode 的信息
        self.ep_infos = []
        # 存储平铺的直接日志数据
        self.direct_info = {}
        # episode_cumulative 累积每个环境的 episode 信息
        self.episode_cumulative = dict()
        # episode_cumulative_avg 存储最近若干 episode 的平均值
        self.episode_cumulative_avg = dict()
        # new_finished_episodes 标记是否有新完成的 episode
        self.new_finished_episodes = False

    # 接受参数 algo
    # 将算法对象和 TensorBoard writer 保存到 observer 实例
    # 后续可以直接使用 self.algo 获取算法属性，self.writer 写日志
    def after_init(self, algo):
        self.algo = algo
        self.writer = self.algo.writer

    # infos：环境返回的信息，期望是字典类型
    # done_indices：当前 batch 中完成 episode 的索引（通常是张量或列表）
    def process_infos(self, infos, done_indices):
        # 确保 infos 是字典，否则无法使用 key 访问
        assert isinstance(infos, dict), 'RLGPUAlgoObserver expects dict info'
        if not isinstance(infos, dict):
            return

        if 'episode' in infos:
            self.ep_infos.append(infos['episode'])

        if 'episode_cumulative' in infos:
            # 累积每个环境的 episode 信息（例如奖励总和）
            for key, value in infos['episode_cumulative'].items():
                if key not in self.episode_cumulative:
                    self.episode_cumulative[key] = torch.zeros_like(value)
                self.episode_cumulative[key] += value

            # 对每个完成 episode 的环境
            # 将累积值记录到最近 N 个 episode 平均值队列中
            # 重置该环境的累积值为 0
            for done_idx in done_indices:
                self.new_finished_episodes = True
                done_idx = done_idx.item()

                for key, value in infos['episode_cumulative'].items():
                    if key not in self.episode_cumulative_avg:
                        self.episode_cumulative_avg[key] = deque([], maxlen=self.algo.games_to_track)

                    self.episode_cumulative_avg[key].append(self.episode_cumulative[key][done_idx].item())
                    self.episode_cumulative[key][done_idx] = 0

        # 将可直接记录的标量信息收集到 direct_info 中，以便写入日志
        if len(infos) > 0 and isinstance(infos, dict):  
            infos_flat = flatten_dict(infos, prefix='', separator='/')
            self.direct_info = {}
            for k, v in infos_flat.items():
                if isinstance(v, float) or isinstance(v, int) or (isinstance(v, torch.Tensor) and len(v.shape) == 0):
                    self.direct_info[k] = v

    # frame：当前训练帧或步数
    # epoch_num：当前训练 epoch
    # total_time：训练累计时间（本函数中未使用）
    def after_print_stats(self, frame, epoch_num, total_time):
        # 处理单个 episode 的信息
        if self.ep_infos:
            # 每个 ep_info 是一个字典，ep_infos[0] 获取第一个字典以获取所有 key
            for key in self.ep_infos[0]:
                # 创建一个空的 Tensor，用于收集所有环境的该 key 值
                infotensor = torch.tensor([], device=self.algo.device)
                for ep_info in self.ep_infos:
                    # 确保是张量，将标量或列表转换为 1D Tensor
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    # unsqueeze(0) 将 0 维张量变为 1D 张量，方便后续拼接
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    # 沿维度 0 拼接所有环境的值
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.algo.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, epoch_num)
            # 清空列表
            self.ep_infos.clear()
        # 处理累积 episode 信息
        if self.new_finished_episodes:
            for key in self.episode_cumulative_avg:
                self.writer.add_scalar(f'episode_cumulative/{key}', np.mean(self.episode_cumulative_avg[key]), frame)
                self.writer.add_scalar(f'episode_cumulative_min/{key}_min', np.min(self.episode_cumulative_avg[key]), frame)
                self.writer.add_scalar(f'episode_cumulative_max/{key}_max', np.max(self.episode_cumulative_avg[key]), frame)
            self.new_finished_episodes = False
        # 直接写标量信息
        for k, v in self.direct_info.items():
            self.writer.add_scalar(f'{k}', v, epoch_num)


class MultiObserver(AlgoObserver):
    """Meta-observer that allows the user to add several observers."""

    def __init__(self, observers_):
        super().__init__()
        self.observers = observers_

    def _call_multi(self, method, *args_, **kwargs_):
        for o in self.observers:
            getattr(o, method)(*args_, **kwargs_)

    def before_init(self, base_name, config, experiment_name):
        self._call_multi('before_init', base_name, config, experiment_name)

    def after_init(self, algo):
        self._call_multi('after_init', algo)

    def process_infos(self, infos, done_indices):
        self._call_multi('process_infos', infos, done_indices)

    def after_steps(self):
        self._call_multi('after_steps')

    def after_clear_stats(self):
        self._call_multi('after_clear_stats')

    def after_print_stats(self, frame, epoch_num, total_time):
        self._call_multi('after_print_stats', frame, epoch_num, total_time)


"""
这段代码定义了一个强化学习环境包装类 RLGPUEnv，用于在 GPU 加速的强化学习框架中管理单个或多个环境实例
它继承自 vecenv.IVecEnv，表明它是一个“向量化环境”（VecEnv）接口的实现类，可以被上层并行环境管理器统一调度
"""
class RLGPUEnv(vecenv.IVecEnv):
    
    # config_name：环境配置名称，默认是 rlgpu
    # num_actors：并行执行的环境数量（在本类中未直接使用，可能用于兼容接口）
    # **kwargs：其他关键字参数，传递给环境创建函数
    # self.env 是是一个 Isaac Gym 矢量化任务环境对象（如 Ant）
    def __init__(self, config_name, num_actors, **kwargs):
        self.env = env_configurations.configurations[config_name]['env_creator'](**kwargs)

    # actions：环境动作
    # self.env.step(actions) 调用对应的类（例如 Ant）的 step(actions) 方法
    def step(self, actions):
        return  self.env.step(actions)

    # self.env.reset() 调用对应的类（例如 Ant）的 reset() 方法
    def reset(self):
        return self.env.reset()
    
    # 重置完成的环境
    # self.env.reset_done() 调用对应的类（例如 Ant）的 reset_done() 方法
    def reset_done(self):
        return self.env.reset_done()
    
    # 获取环境中智能体的数量
    # 实际上找不到该函数，怀疑不存在
    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    # 构建一个字典 info，用于收集环境的基本信息（空间结构等）
    def get_env_info(self):
        info = {}
        
        # self.env.action_space 获取动作空间，实际上是调用 action_space() 函数，但是因为该函数被 @property 装饰器修饰，所以可以当作属性调用，在 Env 类里定义，Ant 任务的动作空间是 Box(-1, 1, (8,), float32)
        info['action_space'] = self.env.action_space
        
        # self.env.observation_space 获取观测空间，在 Env 类里定义，Ant 任务的观测空间是 Box(-inf, inf, (60,), float32)
        info['observation_space'] = self.env.observation_space
        
        # 如果环境支持 AMP（Adversarial Motion Priors），则保存其专用的观测空间，当前任务中用不到
        if hasattr(self.env, "amp_observation_space"):
            info['amp_observation_space'] = self.env.amp_observation_space

        # 对于 Ant 任务来说，num_states 是 0，因此不会进入该分支
        if self.env.num_states > 0:
            info['state_space'] = self.env.state_space
            print(info['action_space'], info['observation_space'], info['state_space'])
        else:
            print(info['action_space'], info['observation_space'])

        # 对于 Ant 任务，info 只包含 action_space 和 observation_space 两个键值对
        return info

    # 向环境传递训练过程信息
    
    def set_train_info(self, env_frames, *args_, **kwargs_):
        """
        Send the information in the direction algo->environment.
        Most common use case: tell the environment how far along we are in the training process. This is useful
        for implementing curriculums and things such as that.
        """
        if hasattr(self.env, 'set_train_info'):
            # 保存当前训练总帧数到 total_train_env_frames 变量中
            self.env.set_train_info(env_frames, *args_, **kwargs_)

    def get_env_state(self):
        """
        Return serializable environment state to be saved to checkpoint.
        Can be used for stateful training sessions, i.e. with adaptive curriculums.
        """
        # Ant 和 Shadow_Hand 任务虽然有该函数，但是默认返回 None
        if hasattr(self.env, 'get_env_state'):
            return self.env.get_env_state()
        else:
            return None

    def set_env_state(self, env_state):
        # Ant 和 Shadow_Hand 任务虽然有该函数，但是默认不做任何操作
        if hasattr(self.env, 'set_env_state'):
            self.env.set_env_state(env_state)

class ComplexObsRLGPUEnv(vecenv.IVecEnv):
    
    def __init__(
        self,
        config_name,
        num_actors,
        obs_spec: Dict[str, Dict],
        **kwargs,
    ):
        """RLGPU wrapper for Isaac Gym tasks.

        Args:
            config_name: Name of rl games env_configurations configuration to use.
            obs_spec: Dictinoary listing out specification for observations to use.
                eg.
                {
                 'obs': {'names': ['obs_1', 'obs_2'], 'concat': True, space_name: 'observation_space'},},
                 'states': {'names': ['state_1', 'state_2'], 'concat': False, space_name: 'state_space'},}
                }
                Within each, if 'concat' is set, concatenates all the given observaitons into a single tensor of dim (num_envs, sum(num_obs)).
                    Assumes that each indivdual observation is single dimensional (ie (num_envs, k), so image observation isn't supported).
                    Currently applies to student and teacher both.
                "space_name" is given into the env info which RL Games reads to find the space shape
        """
        self.env = env_configurations.configurations[config_name]['env_creator'](**kwargs)

        self.obs_spec = obs_spec

    def _generate_obs(
        self, env_obs: Dict[str, torch.Tensor]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Generate the RL Games observations given the observations from the environment.

        Args:
            env_obs: environment observations
        Returns:
            Dict which contains keys with values corresponding to observations.
        """
        # rl games expects a dictionary with 'obs' and 'states'
        # corresponding to the policy observations and possible asymmetric
        # observations respectively

        rlgames_obs = {k: self.gen_obs_dict(env_obs, v['names'], v['concat']) for k, v in self.obs_spec.items()}

        return rlgames_obs

    def step(
        self, action: torch.Tensor
    ) -> Tuple[
        Dict[str, Dict[str, torch.Tensor]], torch.Tensor, torch.Tensor, Dict[str, Any]
    ]:
        """Step the Isaac Gym task.

        Args:
            action: Enivronment action.
        Returns:
            observations, rewards, dones, infos
            Returned obeservations are a dict which contains key 'obs' corresponding to a dictionary of observations,
            and possible 'states' key corresponding to dictionary of privileged observations.
        """
        env_obs, rewards, dones, infos = self.env.step(action)
        rlgames_obs = self._generate_obs(env_obs)
        return rlgames_obs, rewards, dones, infos

    def reset(self) -> Dict[str, Dict[str, torch.Tensor]]:
        env_obs = self.env.reset()
        return self._generate_obs(env_obs)

    def get_number_of_agents(self) -> int:
        return self.env.get_number_of_agents()

    def get_env_info(self) -> Dict[str, gym.spaces.Space]:
        """Gets information on the environment's observation, action, and privileged observation (states) spaces."""
        info = {}
        info["action_space"] = self.env.action_space

        for k, v in self.obs_spec.items():
            info[v['space_name']] = self.gen_obs_space(v['names'], v['concat'])

        return info
    
    def gen_obs_dict(self, obs_dict, obs_names, concat):
        """Generate the RL Games observations given the observations from the environment."""
        if concat:
            return torch.cat([obs_dict[name] for name in obs_names], dim=1)
        else:
            return {k: obs_dict[k] for k in obs_names}
            

    def gen_obs_space(self, obs_names, concat):
        """Generate the RL Games observation space given the observations from the environment."""
        if concat:
            return gym.spaces.Box(
                low=-np.Inf,
                high=np.Inf,
                shape=(sum([self.env.observation_space[s].shape[0] for s in obs_names]),),
                dtype=np.float32,
            )
        else:        
            return gym.spaces.Dict(
                    {k: self.env.observation_space[k] for k in obs_names}
                )

    def set_train_info(self, env_frames, *args_, **kwargs_):
        """
        Send the information in the direction algo->environment.
        Most common use case: tell the environment how far along we are in the training process. This is useful
        for implementing curriculums and things such as that.
        """
        if hasattr(self.env, 'set_train_info'):
            self.env.set_train_info(env_frames, *args_, **kwargs_)

    def get_env_state(self):
        """
        Return serializable environment state to be saved to checkpoint.
        Can be used for stateful training sessions, i.e. with adaptive curriculums.
        """
        if hasattr(self.env, 'get_env_state'):
            return self.env.get_env_state()
        else:
            return None

    def set_env_state(self, env_state):
        if hasattr(self.env, 'set_env_state'):
            self.env.set_env_state(env_state)                