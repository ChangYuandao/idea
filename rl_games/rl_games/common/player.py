import os
import shutil
import threading
import time
import gym
import numpy as np
import torch
import copy
from os.path import basename
from typing import Optional
from rl_games.common import vecenv
from rl_games.common import env_configurations
from rl_games.algos_torch import model_builder


"""
在 Python 中，object 是所有类的最顶层基类（root base class）；也就是说，Python 里所有类最终都继承自 object
Baseplayer 继承自 object，表示它是一个基础类，可以被其他类继承和扩展
"""
class BasePlayer(object):

    def __init__(self, params):
        
        # 从传入的参数字典中获取配置，并赋值给 self.config 和 config 变量
        self.config = config = params['config']
        
        # 调用 load_networks 方法加载模型，Ant 任务的模型是 ModelA2CContinuousLogStd
        self.load_networks(params)
        
        # 从配置中获取环境名称，默认是 rlgpu
        self.env_name = self.config['env_name']
        
        # 获取玩家配置，默认为空字典
        self.player_config = self.config.get('player', {})
        
        # 获取环境配置，默认为空字典
        self.env_config = self.config.get('env_config', {})
        
        # 优先使用玩家配置中的环境配置（如果存在），否则使用默认的环境配置
        self.env_config = self.player_config.get('env_config', self.env_config)
        
        # 获取环境信息（如果有的话）
        self.env_info = self.config.get('env_info')
        
        # 获取配置中的 clip_actions 是否启用（默认启用）
        self.clip_actions = config.get('clip_actions', True)
        
        # 获取种子（如果有的话），并从 env_config 中移除它
        self.seed = self.env_config.pop('seed', None)
        
        # 判断是否平衡环境奖励（通常用于多环境的情景），如果 balance_env_rewards 不存在则默认设置为 False
        self.balance_env_rewards = self.player_config.get('balance_env_rewards', False)

        # 如果环境信息为空，则创建环境实例
        if self.env_info is None:
            # 判断是否使用多环境（vecenv），Ant 任务默认是没有使用的
            use_vecenv = self.player_config.get('use_vecenv', False)
            if use_vecenv:
                print('[BasePlayer] Creating vecenv: ', self.env_name)
                
                # 创建一个 RLGPUEnv 类，RLGPUEnv 类的 self.env 是 Ant 等任务环境实例
                self.env = vecenv.create_vec_env(
                    self.env_name, self.config['num_actors'], **self.env_config)
                
                # 获取环境信息，对于 Ant 任务，info 只包含 action_space 和 observation_space 信息，维度分别是 (8,) 和 (60,)
                self.env_info = self.env.get_env_info()
            else:
                print('[BasePlayer] Creating regular env: ', self.env_name)
                
                # 创建环境实例，Ant 任务的环境实例是 Ant
                self.env = self.create_env()
                self.env_info = env_configurations.get_env_info(self.env)
        else:
            self.env = config.get('vec_env')
            
        # 获取环境中的代理数量，默认为1
        self.num_agents = self.env_info.get('agents', 1)
        
        # 获取环境中的值大小（通常用于多智能体的价值函数），默认为1
        self.value_size = self.env_info.get('value_size', 1)
        
        # 获取动作空间，Ant 任务的动作空间是 Box(-1.0, 1.0, (8,), float32)
        self.action_space = self.env_info['action_space']

        # 获取观测空间，Ant 任务的观测空间是 Box(-inf, inf, (60,), float32)
        self.observation_space = self.env_info['observation_space']
        
        # Isaac Gym 任务的 observation_space 大多默认不是 Dict 类型
        if isinstance(self.observation_space, gym.spaces.Dict):
            self.obs_shape = {}
            for k, v in self.observation_space.spaces.items():
                self.obs_shape[k] = v.shape
                print(f'obs key: {k} shape: {v.shape}')
        else:
            # 对于 Ant 任务，观测空间的 shape 是 (60,)
            self.obs_shape = self.observation_space.shape
            
        # 标记是否是张量类型的观测数据
        self.is_tensor_obses = False

        # 状态初始化为 None
        self.states = None
        
        # 重新获取玩家配置
        self.player_config = self.config.get('player', {})
        
        # 默认使用 CUDA 设备，GPU 加速
        self.use_cuda = True
        
        # 批量大小初始化为 1
        self.batch_size = 1
        
        # 是否具有批处理维度的标志
        self.has_batch_dimension = False
        
        # central_value_config 用于判断是否使用集中式价值函数，常出现在多智能体强化学习中
        self.has_central_value = self.config.get(
            'central_value_config') is not None
        
        # 获取设备名称，默认是 'cuda'，Ant 的训练配置里没有 device_name 字段
        self.device_name = self.config.get('device_name', 'cuda')
        
        # 渲染环境的标志，默认不渲染
        self.render_env = self.player_config.get('render', False)
        
        # 游戏总数，默认为十亿，足够大
        self.games_num = self.player_config.get('games_num', 1000000000)

        # 判断是否为确定性策略，默认 deterministic 不在 self.player_config 中，因此 is_deterministic 被设置为 True
        if 'deterministic' in self.player_config:
            self.is_deterministic = self.player_config['deterministic']
        else:
            self.is_deterministic = self.player_config.get(
                'deterministic', True)

        # 获取每个游戏生命周期的游戏数
        self.n_game_life = self.player_config.get('n_game_life', 1)
        
        # 是否打印统计信息，默认打印
        self.print_stats = self.player_config.get('print_stats', True)
        
        # 渲染时的暂停时间
        self.render_sleep = self.player_config.get('render_sleep', 0.002)
        
        # 最大步骤数，默认是 108000 // 4 = 27000 步
        self.max_steps = 108000 // 4
        
        # 创建一个 torch 设备对象（默认 'cuda'）
        self.device = torch.device(self.device_name)

        # 是否为评估模式，默认不是
        self.evaluation = self.player_config.get("evaluation", False)
        
        # 更新检查点的频率，默认是 100
        self.update_checkpoint_freq = self.player_config.get("update_checkpoint_freq", 100)
        
        # 如果是评估模式，获取检查点路径
        self.dir_to_monitor = self.player_config.get("dir_to_monitor")
        
        # 用于存储最新的检查点路径
        self.checkpoint_to_load: Optional[str] = None

        # 如果是评估模式并且有监控目录，默认没有
        if self.evaluation and self.dir_to_monitor is not None:
            
            # 为了防止多线程同时加载检查点，使用锁机制
            self.checkpoint_mutex = threading.Lock()
            
            # 评估检查点目录
            self.eval_checkpoint_dir = os.path.join(self.dir_to_monitor, "eval_checkpoints")
            
            # 创建检查点目录
            os.makedirs(self.eval_checkpoint_dir, exist_ok=True)

         # 设置文件观察模式，监听 "*.pth" 后缀的文件
            patterns = ["*.pth"]
            from watchdog.observers import Observer
            from watchdog.events import PatternMatchingEventHandler
            
            # 创建一个文件事件处理器，它会监听 .pth 文件的变化。patterns 参数设置了要匹配的文件类型
            self.file_events = PatternMatchingEventHandler(patterns)
            # 设置文件事件处理器的回调函数
            self.file_events.on_created = self.on_file_created
            self.file_events.on_modified = self.on_file_modified
            # 创建文件观察器，并开始监听
            self.file_observer = Observer()
            self.file_observer.schedule(self.file_events, self.dir_to_monitor, recursive=False)
            self.file_observer.start()

    """
    它的作用是在评估模式下，持续等待新检查点文件的到来
    """
    def wait_for_checkpoint(self):
        # 如果没有指定检查点监控目录，则直接返回，不进行任何操作
        if self.dir_to_monitor is None:
            return
        # 初始化尝试计数器，记录等待了多少次循环
        attempt = 0
        while True:
            
            # 每次循环尝试次数增加 1
            attempt += 1
            
            # 使用互斥锁（checkpoint_mutex）来确保线程安全，避免多个线程同时访问 checkpoin_to_load
            with self.checkpoint_mutex:
                
                 # 如果检查点路径不为空，说明已经有了新的检查点
                if self.checkpoint_to_load is not None:
                    
                    # 每10次尝试打印一次提示信息，告知用户正在等待新检查点
                    if attempt % 10 == 0:
                        print(f"Evaluation: waiting for new checkpoint in {self.dir_to_monitor}...")
                    # 跳出 while 循环，表示已经有了检查点
                    break
                
            # 等待 1 秒钟后再进行下一次循环，避免频繁占用 CPU 资源
            time.sleep(1.0)

        print(f"Checkpoint {self.checkpoint_to_load} is available!")

    
    """
    这段代码的作用是尝试加载一个新的检查点文件
    如果成功加载，它会使用 restore 方法恢复模型状态，如果失败，它会处理错误并打印出相关信息
    """
    def maybe_load_new_checkpoint(self):
        
        # 在加载新检查点时，使用互斥锁（checkpoint_mutex）来保证线程安全，避免多个线程同时操作检查点
        with self.checkpoint_mutex:
            
            # 如果有待加载的检查点文件（self.checkpoint_to_load 不为 None），则尝试加载
            if self.checkpoint_to_load is not None:
                print(f"Evaluation: loading new checkpoint {self.checkpoint_to_load}...")

                # 用于标记加载过程中是否发生了错误
                load_error = False
                try:
                    # 尝试加载检查点文件，这里只是简单验证文件是否有效
                    # 如果文件损坏，torch.load 会抛出异常
                    torch.load(self.checkpoint_to_load)
                except (OSError, IOError, torch.TorchError) as e:
                    # 如果加载过程中出现异常（如文件损坏），打印错误信息并将 load_error 设置为 True
                    print(f"Evaluation: checkpoint file is likely corrupted {self.checkpoint_to_load}: {e}")
                    load_error = True
                    
                # 如果加载没有出错（load_error 为 False），尝试恢复模型的状态
                if not load_error:
                    try:
                        # 使用 self.restore 方法恢复模型状态
                        self.restore(self.checkpoint_to_load)
                    except Exception as e:
                        # 如果恢复状态时发生异常，打印错误信息
                        print(f"Evaluation: failed to load new checkpoint {self.checkpoint_to_load}: {e}")

                # 无论是否成功加载检查点，都清空待加载检查点的路径
                # 这一步是防止后续继续尝试加载相同的检查点
                self.checkpoint_to_load = None

    def process_new_eval_checkpoint(self, path):
        
        # 使用线程锁 (checkpoint_mutex) 来确保在处理评估 checkpoints 文件时不会出现竞争条件
        with self.checkpoint_mutex:
            try:
                # 构建 checkpoints 文件的目标路径，将原始文件路径的文件名提取并添加到 checkpoints 目录中
                eval_checkpoint_path = os.path.join(self.eval_checkpoint_dir, basename(path))
                # 使用 shutil.copyfile() 方法将文件从源路径（path）复制到目标路径（eval_checkpoint_path）
                shutil.copyfile(path, eval_checkpoint_path)
                print(f"Successfully copied {path} to {eval_checkpoint_path}")
            except Exception as e:
                print(f"Failed to copy {path} to {eval_checkpoint_path}: {e}")
                return
            
            # 将新复制的检查点文件路径保存到 checkpoint_to_load 属性
            # 该属性用于后续加载模型检查点文件
            self.checkpoint_to_load = eval_checkpoint_path

    # 这个方法会在指定目录下有新文件创建时触发
    def on_file_created(self, event):
        self.process_new_eval_checkpoint(event.src_path)

    # 这个方法会在指定目录下的文件被修改时触发
    def on_file_modified(self, event):
        self.process_new_eval_checkpoint(event.src_path)


    def load_networks(self, params):
        # 生成一个网络实例
        builder = model_builder.ModelBuilder()
        
        # 根据参数 params 加载网络配置，并赋值给 self.config['network']
        self.config['network'] = builder.load(params)

    """
    对输入的观测批次（obs_batch）进行预处理，特别是针对像素值为 uint8 类型的情况，将其归一化到 [0, 1] 的范围内
    还处理了观测批次为字典的情况，每个字典条目会根据其数据类型分别处理
    """
    def _preproc_obs(self, obs_batch):
        
        # 如果 obs_batch 是字典类型（通常在处理多个观测时使用字典来存储不同的观测项）
        if type(obs_batch) is dict:
            
            # 创建 obs_batch 的副本，避免修改原始数据
            obs_batch = copy.copy(obs_batch)
            
            # 遍历字典中的每一项 (k 是键，v 是值)
            for k, v in obs_batch.items():
                # 如果观测数据的 dtype 是 uint8，表示每个像素的值在 [0, 255] 范围内
                if v.dtype == torch.uint8:
                    # 将像素值除以 255.0，将其转换到 [0, 1] 范围，并保持为浮点数
                    obs_batch[k] = v.float() / 255.0
                else:
                    # 如果 dtype 不是 uint8，则保持原始数据
                    obs_batch[k] = v
        else:
            # 如果 obs_batch 不是字典类型（假设是张量），并且它的数据类型是 uint8
            if obs_batch.dtype == torch.uint8:
                # 将像素值除以 255.0，归一化到 [0, 1] 范围
                obs_batch = obs_batch.float() / 255.0
        # 返回处理后的观测数据
        return obs_batch

    """
    执行一个环境步骤（env.step），并根据需要对观测数据（obs）、奖励（rewards）、结束标志（dones）等信息进行处理和转换
    还处理了数据格式的转换，确保能够适应不同类型的观测数据（如张量或 NumPy 数组）以及环境返回的奖励信息
    """
    def env_step(self, env, actions):
        
        # 如果观察到的不是张量格式（即 self.is_tensor_obses 为 False）
        if not self.is_tensor_obses:
            # 将 actions 从张量转换为 numpy 数组
            actions = actions.cpu().numpy()
            
        # 执行环境步骤，传入动作，返回观测（obs），奖励（rewards），结束标志（dones），以及额外信息（infos）
        obs, rewards, dones, infos = env.step(actions)
        
        # 如果观测数据的类型是 np.float64，将其转换为 np.float32（避免在训练中出现数据类型不一致的错误）
        if hasattr(obs, 'dtype') and obs.dtype == np.float64:
            obs = np.float32(obs)
        
        # 如果 value_size > 1，表示存在多个值函数的输出，将奖励值简化为第一个值，_init 里默认为1
        if self.value_size > 1:
            rewards = rewards[0]
        
        # 如果观测数据是张量格式（即 self.is_tensor_obses 为 True）
        # 则返回转换后的张量观测数据、奖励、结束标志和信息
        if self.is_tensor_obses:
            return self.obs_to_torch(obs), rewards.cpu(), dones.cpu(), infos
        else:
            # 如果 `dones` 是标量（单一的结束标志），则扩展其维度为 (1,)
            if np.isscalar(dones):
                rewards = np.expand_dims(np.asarray(rewards), 0)
                dones = np.expand_dims(np.asarray(dones), 0)
                
            # 返回转换后的张量观测数据、奖励、结束标志和信息
            # 将奖励和结束标志从 numpy 数组转换为 PyTorch 张量
            return self.obs_to_torch(obs), torch.from_numpy(rewards), torch.from_numpy(dones), infos

    """
    处理观测数据的方法，目标是将观测数据转换为 PyTorch 张量（tensor）
    如果观测数据是字典类型，它会递归地转换字典中的每个观测值
    如果观测数据本身就是一个数组或其他类型的数据，它将直接通过 cast_obs 方法进行转换
    """
    def obs_to_torch(self, obs):
        
        # 如果观测值是一个字典类型（即可能包含多个不同的观测信息）
        if isinstance(obs, dict):
            # 如果字典中包含键 'obs'，则提取该键的值作为观测
            if 'obs' in obs:
                obs = obs['obs']
                
            # 如果观测仍然是字典类型
            if isinstance(obs, dict):
                # 初始化一个空字典用于存储转换后的观测
                upd_obs = {}
                
                # 遍历观测字典中的每一个键值对，递归地将每个值转换为 PyTorch 张量
                for key, value in obs.items():
                    upd_obs[key] = self._obs_to_tensors_internal(value, False)
            else:
                # 如果观测不再是字典类型，直接转换为 PyTorch 张量
                upd_obs = self.cast_obs(obs)
        else:
            # 如果观测不是字典类型，直接调用 cast_obs 函数进行转换
            upd_obs = self.cast_obs(obs)
        
        # 返回转换后的观测
        return upd_obs

    """
    作用是递归地将观测 obs 转换为 PyTorch 张量
    """
    def _obs_to_tensors_internal(self, obs, cast_to_dict=True):
        
        # 如果观测是字典类型（即包含多个观测项）
        if isinstance(obs, dict):
            
            # 初始化一个空字典用于存储转换后的观测
            upd_obs = {}
            
            # 遍历字典中的每个键值对，递归地将每个值转换为张量
            for key, value in obs.items():
                
                # 递归地调用转换方法，逐项转换观测值
                upd_obs[key] = self._obs_to_tensors_internal(value, False)
        else:
            # 如果观测不是字典类型，直接通过 cast_obs 方法转换
            upd_obs = self.cast_obs(obs)
        
         # 返回转换后的观测（如果是字典，返回转换后的字典；如果是其他类型，返回张量）
        return upd_obs

    """
    将不同类型的观测数据（NumPy 数组、标量、PyTorch 张量等）统一转换为 PyTorch 张量，并确保数据被移到指定的设备（如 GPU）
    """
    def cast_obs(self, obs):
        
        # 如果输入的观测（obs）是一个 PyTorch Tensor
        if isinstance(obs, torch.Tensor):
            # 设置标记，表示当前的观测数据是一个张量
            self.is_tensor_obses = True
            
        # 如果观测是一个 NumPy 数组
        elif isinstance(obs, np.ndarray):
            # 确保 NumPy 数组的元素类型不是 np.int8，因为它不适用于转换为张量
            assert (obs.dtype != np.int8)
            
            # 如果 NumPy 数组的类型是 uint8（无符号8位整数），将其转换为 ByteTensor
            if obs.dtype == np.uint8:
                # 将其转换为 ByteTensor，并转移到指定的设备（如GPU）
                obs = torch.ByteTensor(obs).to(self.device)
            
            # 如果 NumPy 数组的类型是其他类型，将其转换为 FloatTensor
            else:
                # 将其转换为 FloatTensor，并转移到指定的设备（如GPU）
                obs = torch.FloatTensor(obs).to(self.device)
                
        # 如果观测是一个标量值（单一数值）
        elif np.isscalar(obs):
            # 将标量值包装成一个列表并转换为 FloatTensor
            obs = torch.FloatTensor([obs]).to(self.device)
            
        # 返回转换后的观测（现在是 PyTorch 张量
        return obs

    """
    该方法的主要作用是根据观测数据是否是张量（self.is_tensor_obses）来处理动作数据 actions
        - 如果当前的观测数据是一个张量（self.is_tensor_obses 为 True），那么它直接返回动作数据
        - 如果当前观测数据不是张量（self.is_tensor_obses 为 False），则将动作数据从 PyTorch 张量转换为 NumPy 数组
    """
    def preprocess_actions(self, actions):
        
        # 如果观测数据不是张量（即不是tensor类型）
        if not self.is_tensor_obses:
            # 将动作数据从 PyTorch 张量转换为 NumPy 数组，并移回 CPU
            actions = actions.cpu().numpy()
            
        # 返回处理后的动作数据（此时可能是 NumPy 数组，或者仍然是 PyTorch 张量）
        return actions

    """
    该方法的作用是重置环境，并将环境返回的初始观测数据转换为 PyTorch 张量格式
    这样可以保证在强化学习过程中，所有的观测数据都能作为 PyTorch 张量进行处理，便于后续的神经网络输入和梯度计算
    """
    def env_reset(self, env):
        # 调用环境的 reset 方法重置环境，返回环境的初始观测（通常是观测空间中的一项或多项数据）
        obs = env.reset()
        # 将观测数据转换为 PyTorch 张量并返回
        return self.obs_to_torch(obs)

    """
    restore 方法的作用通常是恢复某种状态或从文件中加载信息
    """
    def restore(self, fn):
        # 该方法没有实现，抛出一个 NotImplementedError 异常，表明该方法需要在子类中实现
        raise NotImplementedError('restore')

    """
    get_weights 方法的作用是提取当前模型的权重并返回
    """
    def get_weights(self):
        
        # 创建一个空字典 `weights`，用来存储模型的权重
        weights = {}
        
        # 获取当前模型（self.model）的所有参数（权重）字典
        # `self.model.state_dict()` 会返回一个包含模型所有可学习参数（如权重和偏置）的字典
        weights['model'] = self.model.state_dict()
        
        print('Getting weights from the model')
        print(weights['model'].keys())
        # 返回包含模型权重的字典
        return weights

    """
    set_weights 方法用于将外部提供的权重和归一化信息加载到模型中
    """
    def set_weights(self, weights):
        
        # 使用传入的 `weights` 字典加载模型的权重
        self.model.load_state_dict(weights['model'])
        
        # 如果需要对输入进行归一化，并且 `weights` 字典中包含 'running_mean_std'，则加载其对应的状态字典
        # Ant 任务默认 normalize_input 是 True
        if self.normalize_input and 'running_mean_std' in weights:
            self.model.running_mean_std.load_state_dict(
                weights['running_mean_std'])

    """
    create_env 方法用于创建强化学习环境实例
    """
    def create_env(self):
        # 从 env_configurations.configurations 中获取当前环境名称为 rlgpu 对应的配置
        # 配置中包含一个 'env_creator' 函数，用于创建环境实例，默认创建类似于 Ant 任务的环境
        return env_configurations.configurations[self.env_name]['env_creator'](**self.env_config)

    """
    get_action 方法用于根据当前的观测数据获取动作
    """
    def get_action(self, obs, is_deterministic=False):
        raise NotImplementedError('step')

    """
    get_masked_action 方法用于根据当前的观测数据和动作掩码获取动作
    """
    def get_masked_action(self, obs, mask, is_deterministic=False):
        raise NotImplementedError('step')

    """
    reset 方法用于重置玩家的状态
    """
    def reset(self):
        raise NotImplementedError('raise')

    """
    init_rnn 方法用于初始化模型中使用的 RNN 隐藏状态
    RNN 需要一个隐藏状态来存储和更新当前的状态信息，以便在处理序列数据时保持上下文信息
    """
    def init_rnn(self):
        # 检查模型是否启用了 RNN 结构
        if self.is_rnn:
            # 获取模型的默认 RNN 状态（通常是初始化的隐状态）
            rnn_states = self.model.get_default_rnn_state()
            # 为每个 RNN 状态创建一个零初始化的张量，并将其存储到 `self.states` 列表中
            self.states = [torch.zeros((s.size()[0], self.batch_size, s.size()[2]), dtype=torch.float32).to(self.device) for s in rnn_states]

    """
    run 方法让智能体在环境中执行多局游戏，记录奖励、步数、胜率等信息，并在评估模式下动态加载新模型检查点
    """
    def run(self):
        
        # 是否渲染环境，默认 self.render_env 是 False
        render = self.render_env
        
        # 每局游戏可以重复的次数，默认是 1
        n_game_life = self.n_game_life
        
        # 计算总的游戏次数 = 单局生命数 × 需要玩的游戏数量
        n_games = self.games_num * n_game_life
        
        # 是否使用确定性策略（True 时智能体总是选择同样的动作），默认是 True
        is_deterministic = self.is_deterministic
        
        # 累积奖励
        sum_rewards = 0
        
        # 累积步数
        sum_steps = 0
        
        # 累积游戏结果（如胜率）
        sum_game_res = 0
        
        # 已完成的游戏数量
        games_played = 0
        
        # 是否存在动作掩码（mask）机制（部分动作不可选）
        has_masks = False
        
        # 检查环境是否支持动作掩码（mask），如果存在 has_action_mask 则返回为 True
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None

        # 检查环境是否支持自对弈（self-play）智能体
        op_agent = getattr(self.env, "create_agent", None)
        
        if op_agent:
            agent_inited = True
            # print('setting agent weights for selfplay')
            # self.env.create_agent(self.env.config)
            # self.env.set_weights(range(8),self.get_weights())

        # 如果环境支持动作掩码，调用以确认当前是否启用掩码机制
        if has_masks_func:
            has_masks = self.env.has_action_mask()

        # 在评估模式下，持续等待新检查点文件的到来
        self.wait_for_checkpoint()

        # 是否需要初始化 RNN 状态（针对循环神经网络模型）
        need_init_rnn = self.is_rnn
        
        # 主循环：遍历要玩的所有游戏，默认是 1 亿局
        for _ in range(n_games):
            
            # 如果已完成指定数量的游戏则退出循环
            if games_played >= n_games:
                break
            
            # 环境重置，获得初始观察值（obs），但是返回的是张量格式
            obses = self.env_reset(self.env)
            
            # 默认批次大小为 1（单环境）
            batch_size = 1
            
            # 根据观测确定批次大小，默认是 train 文件里的并行环境数量
            batch_size = self.get_batch_size(obses, batch_size)

            # 如果是 RNN 模型，初始化其隐藏状态
            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            # 当前回合的累计奖励与步数（每个环境一个值），环境数量是多少就创建多少维度的张量
            cr = torch.zeros(batch_size, dtype=torch.float32)
            steps = torch.zeros(batch_size, dtype=torch.float32)

            # 如果启用了 reward 平衡机制（多环境评估平均化）
            if self.balance_env_rewards:
                
                # 每个环境累计奖励
                per_env_rewards = torch.zeros(batch_size, dtype=torch.float32)
                
                # 每个环境步数
                per_env_steps = torch.zeros(batch_size, dtype=torch.float32)
                
                # 每个环境胜负结果
                per_env_game_res = torch.zeros(batch_size, dtype=torch.float32)
                
                # 每个环境完成的游戏数量
                per_env_games_played = torch.zeros(batch_size, dtype=torch.float32)
            
            # 标记是否打印胜负结果
            print_game_res = False

            # 每一局游戏内部的时间步循环
            for n in range(self.max_steps):
                
                # 如果是评估模式，则定期尝试加载新的模型检查点
                if self.evaluation and n % self.update_checkpoint_freq == 0:
                    
                    # 尝试加载一个新的检查点文件，如果成功则会用 restore 恢复状态
                    self.maybe_load_new_checkpoint()

                # 根据是否有动作掩码选择不同的动作计算方式
                if has_masks:
                    masks = self.env.get_action_mask()
                    
                    # 获取动作掩码（哪些动作可选）
                    action = self.get_masked_action(
                        obses, masks, is_deterministic)
                else:
                    action = self.get_action(obses, is_deterministic)

                # 执行动作，与环境交互，获得新状态、奖励、完成标志和信息，但是格式是张量
                obses, r, done, info = self.env_step(self.env, action)
                
                # 累积当前回合的奖励
                cr += r
                
                # 累积步数
                steps += 1

                # 如果开启渲染，则显示环境状态
                if render:
                    self.env.render(mode='human')
                    # 控制渲染速度为 0.002 秒
                    time.sleep(self.render_sleep)

                # 返回张量 done 中所有为 True 的元素的索引（二维张量形式 [i, j] 表示坐标），用于标识哪些环境的 episode 已结束
                all_done_indices = done.nonzero(as_tuple=False)
                
                # 在多智能体环境中，每隔 num_agents 个 done 取一个索引，从而得到每个环境的完成标志
                done_indices = all_done_indices[::self.num_agents]
                
                # 完成的环境数量
                done_count = len(done_indices)
                
                # 更新已完成游戏数量
                games_played += done_count
                
                # 当某些环境完成一局游戏时
                if done_count > 0:
                    
                     # 如果模型是 RNN，则重置已完成环境的隐藏状态
                    if self.is_rnn:
                        for s in self.states:
                            s[:, all_done_indices, :] = s[:,
                                                        all_done_indices, :] * 0.0
                    # 当前局的胜负结果
                    game_res = 0.0
                    
                    # 如果 info 中包含比赛结果或得分信息，则提取
                    if isinstance(info, dict):
                        if 'battle_won' in info:
                            print_game_res = True
                            # 默认 0.5 代表平局
                            game_res = info.get('battle_won', 0.5)
                        if 'scores' in info:
                            print_game_res = True
                            game_res = info.get('scores', 0.5)

                    # 如果启用了奖励平衡机制（针对多环境）
                    if self.balance_env_rewards:
                        # 更新每个环境的累计数据
                        per_env_rewards[done_indices] += cr[done_indices]
                        per_env_steps[done_indices] += steps[done_indices]
                        per_env_games_played[done_indices] += 1
                        if print_game_res:
                            per_env_game_res[done_indices] += game_res

                         # 重置已完成环境的累计奖励和步数
                        cr[done_indices] = 0
                        steps[done_indices] = 0
                    else:
                        
                        # 如果未启用奖励平衡机制（常见于单环境或同步环境），就直接把本次完成的奖励与步数统计到全局变量中
                        # 已完成的所有环境的累计奖励之和
                        cur_rewards = cr[done_indices].sum().item()
                        # 已完成环境的步数之和
                        cur_steps = steps[done_indices].sum().item()

                        # 将所有环境的奖励、步数乘上 (1 - done)，对 done=True 的环境置 0，保留未完成的环境，相当于一种向量化的重置操作
                        cr = cr * (1.0 - done.float())
                        
                        # 更新步数
                        steps = steps * (1.0 - done.float())
                        
                        # 总奖励、总步数、总游戏结果累加
                        sum_rewards += cur_rewards
                        sum_steps += cur_steps
                        sum_game_res += game_res

                        # 如果启用了打印统计信息
                        if self.print_stats:
                            
                            # 平均每局奖励和步数
                            cur_rewards_done = cur_rewards / done_count
                            cur_steps_done = cur_steps / done_count
                            if print_game_res:
                                print(
                                    f'reward: {cur_rewards_done:.2f} steps: {cur_steps_done:.1f} w: {game_res}')
                            else:
                                print(
                                    f'reward: {cur_rewards_done:.2f} steps: {cur_steps_done:.1f}')
                    
                    # 如果所有环境都完成了游戏，或达到游戏上限，则退出
                    if batch_size // self.num_agents == 1 or games_played >= n_games:
                        break
        
        # 游戏结束后计算总体统计结果，默认为 False
        # 若为 True，表示每个并行环境的奖励、步数等独立统计并平均
        # 若为 False，所有环境的数据统一累计后平均
        if self.balance_env_rewards:
            # 找出参与过游戏的环境（过滤掉没参与的）
            valid_envs = per_env_games_played > 0
            per_env_avg_rewards = torch.zeros(batch_size, dtype=torch.float32)
            per_env_avg_steps = torch.zeros(batch_size, dtype=torch.float32)
            per_env_avg_game_res = torch.zeros(batch_size, dtype=torch.float32)

            # 计算每个环境的平均奖励与步数
            per_env_avg_rewards[valid_envs] = (
                per_env_rewards[valid_envs] / per_env_games_played[valid_envs])
            per_env_avg_steps[valid_envs] = (
                per_env_steps[valid_envs] / per_env_games_played[valid_envs])

            # 计算总体平均奖励与步数
            overall_avg_reward = per_env_avg_rewards[valid_envs].mean().item()
            overall_avg_steps = per_env_avg_steps[valid_envs].mean().item()

            # 如果存在胜率数据，计算平均胜率
            if print_game_res:
                per_env_avg_game_res[valid_envs] = (
                    per_env_game_res[valid_envs] / per_env_games_played[valid_envs])
                overall_winrate = per_env_avg_game_res[valid_envs].mean().item()
                print('av reward:', overall_avg_reward * n_game_life, 'av steps:', overall_avg_steps *
                    n_game_life, 'winrate:', overall_winrate * n_game_life)
            else:
                print('av reward:', overall_avg_reward * n_game_life,
                    'av steps:', overall_avg_steps * n_game_life)
        else:
            
            # 打印非平衡模式下的总体结果
            print(sum_rewards)
            if print_game_res:
                print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps /
                    games_played * n_game_life, 'winrate:', sum_game_res / games_played * n_game_life)
            else:
                print('av reward:', sum_rewards / games_played * n_game_life,
                    'av steps:', sum_steps / games_played * n_game_life)

    """
    get_batch_size 方法根据输入的观察数据 obses 计算并返回批次大小 batch_size
    如果观察数据包含批次维度，则从数据中提取批次大小；否则，使用传入的默认批次大小
    还会根据 obses 的形状来判断是否有批次维度，并更新类实例中的 self.batch_size 和 self.has_batch_dimension
    """
    def get_batch_size(self, obses, batch_size):
        
        # 获取观察空间的形状信息，存储在 self.obs_shape 中
        # Ant 任务的 obs_shape 是(60,)
        obs_shape = self.obs_shape
        
        # 如果观察空间是一个字典类型，默认情况下不是
        if type(self.obs_shape) is dict:
            # 如果输入的 obses 中包含 'obs' 键，则使用对应的 obs 数据
            if 'obs' in obses:
                obses = obses['obs']
            
            # 获取 self.obs_shape 字典中的所有键
            keys_view = self.obs_shape.keys()
            
            # 生成一个迭代器，用于遍历 obs_shape 字典的键
            keys_iterator = iter(keys_view)
            
            # 判断输入的 obs 中是否包含 'observation' 键
            if 'observation' in obses:
                first_key = 'observation'
            else:
                # 如果没有 'observation' 键，则使用字典中的第一个键
                first_key = next(keys_iterator)
            
            print('First obs key:', first_key)
            # 根据第一键获取该键对应的观察空间的形状
            obs_shape = self.obs_shape[first_key]
            
            # 根据第一键从输入的 obses 获取实际的观察数据
            obses = obses[first_key]

        print(f"self.obs_shape: {obs_shape}, obses.size(): {obses.size()}")
        # 如果 obs 的维度数大于观察空间的维度数，说明 obs 数据包含批次维度
        if len(obses.size()) > len(obs_shape):
            
            # 设置批次大小为 obs 的第一个维度大小（即批次维度）
            batch_size = obses.size()[0]
            print(f'Inferred batch_size: {batch_size}')
            # 标记模型已包含批次维度
            self.has_batch_dimension = True
            
        # 将计算得到的 batch_size 存储到类的实例中
        self.batch_size = batch_size

        return batch_size
