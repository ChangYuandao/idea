# train.py
# Script to train policies in Isaac Gym
#
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
import datetime

# isaacgym 需要在 torch 之前引入，否则会报错
import isaacgym

import hydra
from hydra.utils import to_absolute_path
from isaacgymenvs.tasks import isaacgym_task_map
from omegaconf import DictConfig, OmegaConf
import gym
import shutil
from pathlib import Path

from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
from isaacgymenvs.utils.utils import set_np_formatting, set_seed


ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

def preprocess_train_config(cfg, config_dict):
    """
    Adding common configuration parameters to the rl_games train config.
    An alternative to this is inferring them in task-specific .yaml files, but that requires repeating the same
    variable interpolations in each config.
    """

    train_cfg = config_dict['params']['config']
    # 如果不存在 full_experiment_name 字段，则不会报错，返回 None
    train_cfg['full_experiment_name'] = cfg.get('full_experiment_name')

    try:
        model_size_multiplier = config_dict['params']['network']['mlp']['model_size_multiplier']
        if model_size_multiplier != 1:
            units = config_dict['params']['network']['mlp']['units']
            for i, u in enumerate(units):
                units[i] = u * model_size_multiplier
            print(f'Modified MLP units by x{model_size_multiplier} to {config_dict["params"]["network"]["mlp"]["units"]}')
    except KeyError:
        pass

    return config_dict

# 配置文件在项目根目录的 ./cfg 文件加，默认加载文件是 config.yaml 配置文件， hydra 运行的时候会将配置内容打包为 DictConfig 传入函数
@hydra.main(config_name="config", config_path="./cfg")
def launch_rlg_hydra(cfg: DictConfig):

    from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver, MultiObserver, ComplexObsRLGPUEnv
    from isaacgymenvs.utils.wandb_utils import WandbAlgoObserver
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner
    from rl_games.algos_torch import model_builder
    from isaacgymenvs.learning import amp_continuous
    from isaacgymenvs.learning import amp_players
    from isaacgymenvs.learning import amp_models
    from isaacgymenvs.learning import amp_network_builder
    import isaacgymenvs

    # 获取当前时间并将其格式化为字符串
    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # cfg.wandb_name 在配置文件中定义，表示 wandb 项目的名称，默认为任务名例如 Ant
    run_name = f"{cfg.wandb_name}_{time_str}"

    if cfg.checkpoint:
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)

    # 将配置转换为普通字典以便打印
    cfg_dict = omegaconf_to_dict(cfg)

    # 设置 numpy 的打印格式
    set_np_formatting()
    
    # LOCAL_RANK 表示当前进程在本机 GPU 的编号，不存在的时候默认是 0
    rank = int(os.getenv("LOCAL_RANK", "0"))
    cfg.seed += rank
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)
    cfg.train.params.config.multi_gpu = cfg.multi_gpu


    # **kwargs 表示函数可以接受 任意数量的关键字参数（即形如 key=value 的参数）
    def create_isaacgym_env(**kwargs):
        envs = isaacgymenvs.make(
            cfg.seed, 
            cfg.task_name, 
            cfg.task.env.numEnvs, 
            cfg.sim_device,
            cfg.rl_device,
            cfg.graphics_device_id,
            cfg.headless,
            cfg.multi_gpu,
            cfg.capture_video,
            cfg.force_render,
            cfg,
            **kwargs,
        )
        if cfg.capture_video:
            envs.is_vector_env = True
            # 从第 0 step 就可能开始录制
            if cfg.test:
                envs = gym.wrappers.RecordVideo(
                    envs,
                    f"videos/{run_name}",
                    step_trigger=lambda step: (step % cfg.capture_video_freq == 0),
                    video_length=cfg.capture_video_len,
                )
            # 跳过第 0 step，从第 1 step 开始录制，避免录制环境刚 reset 的“空白帧”
            else:
                envs = gym.wrappers.RecordVideo(
                    envs,
                    f"videos/{run_name}",
                    step_trigger=lambda step: (step % cfg.capture_video_freq == 0) and (step > 0),
                    video_length=cfg.capture_video_len,
                )
        return envs

    # 注册环境
    env_configurations.register('rlgpu', {
        'vecenv_type': 'RLGPU',
        'env_creator': lambda **kwargs: create_isaacgym_env(**kwargs),
    })
    

    try:
        output_file = f"{ROOT_DIR}/tasks/{cfg.task.env.env_name.lower()}.py"
        # copy 到当前的工作输出环境
        shutil.copy(output_file, f"env.py")
    except:
        import re
        def camel_to_snake(name):
            s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
            return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
        output_file = f"{ROOT_DIR}/tasks/{camel_to_snake(cfg.task.name)}.py"

        shutil.copy(output_file, f"env.py")

    vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))

    rlg_config_dict = omegaconf_to_dict(cfg.train)
    # 修改 train 的配置
    rlg_config_dict = preprocess_train_config(cfg, rlg_config_dict)


    def build_runner(algo_observer):
        runner = Runner(algo_observer)
        runner.algo_factory.register_builder('amp_continuous', lambda **kwargs : amp_continuous.AMPAgent(**kwargs))
        runner.player_factory.register_builder('amp_continuous', lambda **kwargs : amp_players.AMPPlayerContinuous(**kwargs))
        model_builder.register_model('continuous_amp', lambda network, **kwargs : amp_models.ModelAMPContinuous(network))
        model_builder.register_network('amp', lambda **kwargs : amp_network_builder.AMPBuilder())

        return runner

    # 实例化一个类，放在列表里
    observers = [RLGPUAlgoObserver()]

    if cfg.wandb_activate and rank ==0 :
        import wandb
        wandb_observer = WandbAlgoObserver(cfg)
        observers.append(wandb_observer)

  
    exp_date = cfg.train.params.config.name + '-{date:%Y-%m-%d_%H-%M-%S}'.format(date=datetime.datetime.now())
    # 'runs' 是根目录，exp_date 是子目录名
    experiment_dir = os.path.join('runs', exp_date)
    print("Network Directory:", Path.cwd() / experiment_dir / "nn")
    print("Tensorboard Directory:", Path.cwd() / experiment_dir / "summaries")
    # 递归创建目录，如果上级目录不存在也会创建；如果目录已存在，不会报错
    os.makedirs(experiment_dir, exist_ok=True)
    # 将 Hydra 配置对象 cfg 转为 YAML 字符串写入到路径下的 config.yaml 文件中
    with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))
    rlg_config_dict['params']['config']['full_experiment_name'] = exp_date

  
    runner = build_runner(MultiObserver(observers))
    runner.load(rlg_config_dict)
    runner.reset()

    statistics = runner.run({
        'train': not cfg.test,
        'play': cfg.test and not cfg.collect,
        'collect': cfg.test and cfg.collect,
        'checkpoint' : cfg.checkpoint,
        'sigma': cfg.sigma if cfg.sigma != '' else None
    })

    if cfg.wandb_activate and rank == 0:
        wandb.finish()
        
if __name__ == "__main__":
    launch_rlg_hydra()
