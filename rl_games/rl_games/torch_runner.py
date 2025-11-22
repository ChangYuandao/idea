import os
import time
import numpy as np
import random
from copy import deepcopy
import torch
import torch._dynamo
torch._dynamo.config.cache_size_limit = 64

from rl_games.common import object_factory
from rl_games.common import tr_helpers

from rl_games.algos_torch import a2c_continuous
from rl_games.algos_torch import a2c_discrete
from rl_games.algos_torch import players
from rl_games.common.algo_observer import DefaultAlgoObserver
from rl_games.algos_torch import sac_agent


# Limit tensor printouts to 3 decimal places globally
torch.set_printoptions(precision=3, sci_mode=False)


def _restore(agent, args):
    # 只有在 args 中存在有效的 checkpoint 路径时，才进行恢复
    if 'checkpoint' in args and args['checkpoint'] is not None and args['checkpoint'] !='':
        # 当前处于训练模式 (args['train'] 为 True)
        # 并且用户要求只加载 Critic 网络 (load_critic_only 为 True)
        if args['train'] and args.get('load_critic_only', False):
            # 检查 agent 是否有属性 has_central_value
            # 如果没有，说明该 agent 不支持只加载 Critic
            if not getattr(agent, 'has_central_value', False):
                raise ValueError('Loading critic only works only for asymmetric actor critic')
            agent.restore_central_value_function(args['checkpoint'])
            return
        agent.restore(args['checkpoint'])

def _override_sigma(agent, args):
    # 只有当 args 中存在有效的 sigma 参数时，才尝试修改网络中的 sigma
    if 'sigma' in args and args['sigma'] is not None:
        net = agent.model.a2c_network
        if hasattr(net, 'sigma') and hasattr(net, 'fixed_sigma'):
            if net.fixed_sigma:
                with torch.no_grad():
                    net.sigma.fill_(float(args['sigma']))
            else:
                print('Cannot set new sigma because fixed_sigma is False')


# 根据配置创建 Agent（算法实例）或 Player（推理实例）
# 管理训练循环与日志
# 处理多 GPU 配置和随机种
# 统一接口：run_train()、run_play()、run()
class Runner:
    """Runs training/inference (playing) procedures as per a given configuration.

    The Runner class provides a high-level API for instantiating agents for either training or playing
    with an RL algorithm. It further logs training metrics.

    """

    def __init__(self, algo_observer=None):
        """Initialise the runner instance with algorithms and observers.

        Initialises runners and players for all algorithms available in the library using `rl_games.common.object_factory.ObjectFactory`

        Args:
            algo_observer (:obj:`rl_games.common.algo_observer.AlgoObserver`, optional): Algorithm observer that logs training metrics.
                Defaults to `rl_games.common.algo_observer.DefaultAlgoObserver`

        """

        self.algo_factory = object_factory.ObjectFactory()
        self.algo_factory.register_builder('a2c_continuous', lambda **kwargs : a2c_continuous.A2CAgent(**kwargs))
        self.algo_factory.register_builder('a2c_discrete', lambda **kwargs : a2c_discrete.DiscreteA2CAgent(**kwargs)) 
        self.algo_factory.register_builder('sac', lambda **kwargs: sac_agent.SACAgent(**kwargs))


        self.player_factory = object_factory.ObjectFactory()
        self.player_factory.register_builder('a2c_continuous', lambda **kwargs : players.PpoPlayerContinuous(**kwargs))
        self.player_factory.register_builder('a2c_discrete', lambda **kwargs : players.PpoPlayerDiscrete(**kwargs))
        self.player_factory.register_builder('sac', lambda **kwargs : players.SACPlayer(**kwargs))
        
        # 任务收集轨迹
        self.player_factory.register_builder('ant_trajectory_collector', lambda **kwargs : players.AntTrajectoryCollector(**kwargs))
        self.player_factory.register_builder('shadowhand_trajectory_collector', lambda **kwargs : players.ShadowHandTrajectoryCollector(**kwargs))

        self.algo_observer = algo_observer if algo_observer else DefaultAlgoObserver()

        # 启用 TensorFloat32 (TF32) 提高矩阵乘法性能
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # cudnn.benchmark = True 让 cuDNN 自动选择最优卷积算法
        torch.backends.cudnn.benchmark = True

    def reset(self):
        pass
    
    # 主要作用是从传入的字典 params 初始化 Runner 的内部状态， params 是训练文件的配置
    def load_config(self, params):
        """Loads passed config params.

        Args:
            params (:obj:`dict`): Parameters passed in as a dict obtained from a yaml file or some other config format.

        """

        self.seed = params.get('seed', None)
        if self.seed is None:
            self.seed = int(time.time())

        self.local_rank = 0
        self.global_rank = 0
        self.world_size = 1
        # 尝试获取 key 'multi_gpu' 对应的值，不存在则为 False
        if params["config"].get('multi_gpu', False):
            # local rank of the GPU in a node
            self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
            # global rank of the GPU
            self.global_rank = int(os.getenv("RANK", "0"))
            # total number of GPUs across all nodes
            self.world_size = int(os.getenv("WORLD_SIZE", "1"))

            # set different random seed for each GPU
            self.seed += self.global_rank

            print(f"global_rank = {self.global_rank} local_rank = {self.local_rank} world_size = {self.world_size}")

        print(f"self.seed = {self.seed}")

        self.algo_params = params['algo']
        self.algo_name = self.algo_params['name']
        self.exp_config = None

        if self.seed:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)

            # 检查 env_config 字段是否则 config 下
            if 'env_config' in params['config']:
                # 如果环境配置没有指定 seed，就使用全局 seed
                if not 'seed' in params['config']['env_config']:
                    params['config']['env_config']['seed'] = self.seed
                # 多 GPU 情况下微调 seed
                else:
                    if params["config"].get('multi_gpu', False):
                        params['config']['env_config']['seed'] += self.seed

        config = params['config']
        config['reward_shaper'] = tr_helpers.DefaultRewardsShaper(**config['reward_shaper'])
        if 'features' not in config:
            config['features'] = {}
        config['features']['observer'] = self.algo_observer
        self.params = params

    def load(self, yaml_config):
        # 深拷贝 YAML 配置，防止修改原始字典
        config = deepcopy(yaml_config)
        # 保存默认参数 self.default_config
        self.default_config = deepcopy(config['params'])
        # 调用 load_config 初始化 Runner 内部状态
        self.load_config(params=self.default_config)

    def run_train(self, args):
        """Run the training procedure from the algorithm passed in.

        Args:
            args (:obj:`dict`): Train specific args passed in as a dict obtained from a yaml file or some other config format.

        """
        print('Started to train')
        agent = self.algo_factory.create(self.algo_name, base_name='run', params=self.params)

        # Restore weights (if any) BEFORE compiling the model.  Compiling first
        # wraps the model in an `OptimizedModule`, which changes parameter
        # names (adds the `_orig_mod.` prefix) and breaks `load_state_dict`
        # when loading checkpoints that were saved from an *un‑compiled*
        # model.

        _restore(agent, args)
        _override_sigma(agent, args)

        # Now compile the (already restored) model. Doing it after the restore
        # keeps parameter names consistent with the checkpoint.

        # mode="max-autotune" would be faster at runtime, but it has a much
        # longer compilation time. "reduce-overhead" gives a good trade‑off.
        agent.model = torch.compile(agent.model, mode="reduce-overhead")

        agent.train()

    def run_play(self, args):
        """Run the inference procedure from the algorithm passed in.

        Args:
            args (:obj:`dict`): Playing specific args passed in as a dict obtained from a yaml file or some other config format.

        """
        print('Started to play')
        player = self.create_player()
        _restore(player, args)
        _override_sigma(player, args)
        player.run()
    
    def run_collect(self, args):
        """根据任务名称运行对应的轨迹收集程序
        
        Args:
            args (:obj:`dict`): 包含collect字段的参数字典，collect应为任务名称(如'Ant', 'Humanoid')
        """
        task_name = args.get('collect', '')
        
        if not task_name:
            raise ValueError("collect参数不能为空，需要指定任务名称")
        
        # 将任务名转换为小写并添加_trajectory_collector后缀
        collector_name = f"{task_name.lower()}_trajectory_collector"
        
        print(f'Started to collect trajectories for task: {task_name}')
        print(f'Using collector: {collector_name}')
        
        try:
            collector = self.player_factory.create(collector_name, params=self.params)
        except KeyError:
            raise ValueError(f"未找到任务 {task_name} 对应的轨迹收集器: {collector_name}")
        
        _restore(collector, args)
        _override_sigma(collector, args)
        collector.run()

    def create_player(self):
        return self.player_factory.create(self.algo_name, params=self.params)

    def reset(self):
        pass

    def run(self, args):
        """Run either train/play depending on the args.

        Args:
            args (:obj:`dict`):  Args passed in as a dict obtained from a yaml file or some other config format.

        """
        if args['train']:
            self.run_train(args)
        elif args['play']:
            self.run_play(args)
        elif args.get('collect'):
            self.run_collect(args)
        else:
            self.run_train(args)
