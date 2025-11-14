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

import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.gymtorch import *

from isaacgymenvs.utils.torch_jit_utils import *
from isaacgymenvs.tasks.base.vec_task import VecTask


class Ant(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        # 初始化函数，创建强化学习环境的基础组件
        # cfg 是配置字典，包含环境、任务、仿真参数等
        self.cfg = cfg

        # ====== 环境超参数 ======
        # 最大回合长度
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        
        # ====== 任务与随机化相关参数 ======
        # 读取随机化参数
        self.randomization_params = self.cfg["task"]["randomization_params"]
        # 是否启用随机化
        self.randomize = self.cfg["task"]["randomize"]
        # 关节速度缩放因子
        self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
        # 接触力缩放因子
        self.contact_force_scale = self.cfg["env"]["contactForceScale"]
        # 功率缩放因子
        self.power_scale = self.cfg["env"]["powerScale"]
        # 朝向奖励的权重
        self.heading_weight = self.cfg["env"]["headingWeight"]
        # 保持身体直立奖励的权重
        self.up_weight = self.cfg["env"]["upWeight"]
        # 动作代价系数
        self.actions_cost_scale = self.cfg["env"]["actionsCost"]
        # 能量消耗惩罚系数
        self.energy_cost_scale = self.cfg["env"]["energyCost"]
        # 关节达到极限惩罚系数
        self.joints_at_limit_cost_scale = self.cfg["env"]["jointsAtLimitCost"]
        # 摔倒等死亡惩罚系数
        self.death_cost = self.cfg["env"]["deathCost"]
        # 终止条件：身体高度低于该值则episode结束
        self.termination_height = self.cfg["env"]["terminationHeight"]
        
        # ====== 调试与地面属性 ======
        # 是否启用调试可视化
        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        # 地面静态摩擦系数
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        # 地面动态摩擦系数
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        # 地面恢复系数
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        # ====== 定义观察与动作维度 ======
        # 每个环境的观测维度
        self.cfg["env"]["numObservations"] = 60
        # 每个环境的动作维度
        self.cfg["env"]["numActions"] = 8
        
        # ====== 调用父类初始化（创建物理仿真与环境） ======
        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # ====== 如果有可视化窗口，则设置相机视角 ======
        if self.viewer != None:
            # 相机位置
            cam_pos = gymapi.Vec3(50.0, 25.0, 2.4)
            # 相机目标位置
            cam_target = gymapi.Vec3(45.0, 25.0, 0.0)
            # 设置相机位置与目标
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # ====== 获取 GPU 上的仿真状态张量 ======
        # 获取角色根状态张量（位置、旋转、速度等）
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        # 获取所有关节状态张量（角度、速度）
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        # 获取力传感器张量（接触力等）
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)

        # 每个环境中的传感器数量
        sensors_per_env = 4
        
        # 将原始传感器数据转为 torch 张量，并调整维度（每个传感器6个通道：3个力 + 3个力矩）
        self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, sensors_per_env * 6)

        # 刷新状态张量（确保 GPU 端数据与方针同步）
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        # ====== 保存根状态张量 ======
        # 当前根状态
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        # 保存初始状态用于重置
        self.initial_root_states = self.root_states.clone()
        # 将初始线速度和角速度设为0（防止环境初始化时带速度）
        self.initial_root_states[:, 7:13] = 0  

        # ====== 拆分关节状态张量 ======
        # 原始 DOF 状态（角度+速度）
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        # 提取关节角度
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        # 提取关节速度
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        
        # 创建初始关节角度与速度的张量
        self.initial_dof_pos = torch.zeros_like(self.dof_pos, device=self.device, dtype=torch.float)
        zero_tensor = torch.tensor([0.0], device=self.device)
        
        # 如果关节限制下界 > 0，则取下界；如果上界 < 0，则取上界；否则取0，确保初始姿态合法
        self.initial_dof_pos = torch.where(self.dof_limits_lower > zero_tensor, self.dof_limits_lower,
                                           torch.where(self.dof_limits_upper < zero_tensor, self.dof_limits_upper, self.initial_dof_pos))
        # 初始关节速度设为0
        self.initial_dof_vel = torch.zeros_like(self.dof_vel, device=self.device, dtype=torch.float)

        # ====== 初始化用于奖励计算或坐标变换的辅助向量 ======
        # 上方向向量
        self.up_vec = to_torch(get_axis_params(1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        # 前进方向向量
        self.heading_vec = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        # 初始旋转的共轭四元数（用于坐标变换）
        self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))

        # ====== 建立基向量（用于方向奖励） ======
        self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()

        # ====== 目标点与方向 ======
        # 每个环境的远处目标点（x=1000, y=0, z=0）
        self.targets = to_torch([1000, 0, 0], device=self.device).repeat((self.num_envs, 1))
        # 目标方向
        self.target_dirs = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        
        # ====== 仿真时间步与势能初始化 ======
        # 仿真时间步长
        self.dt = self.cfg["sim"]["dt"]
        # 当前势能
        self.potentials = to_torch([-1000./self.dt], device=self.device).repeat(self.num_envs)
        # 上一步势能  
        self.prev_potentials = self.potentials.clone()

    def create_sim(self):
            
        # 设置“上方向”的轴索引（up axis）
        # 在Isaac Gym中：X=0, Y=1, Z=2
        # 这里设为2，表示Z轴是“竖直方向”
        # 上方向轴索引：Y=1, Z=2
        self.up_axis_idx = 2 
        

        # 调用父类的 create_sim() 方法创建仿真对象
        # 参数包括：设备ID、图形设备ID、物理引擎类型和仿真参数
        # 返回的 self.sim 是整个物理仿真的核心对象（simulation handle）
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)


        # 调用自定义函数创建地面（ground plane）
        # 地面是所有仿真环境的基础支撑面
        self._create_ground_plane()
        
        # 输出调试信息：显示创建的环境数量和环境间距
        # envSpacing 决定多个环境实例之间的物理位置间隔，防止相互碰撞
        print(f'num envs {self.num_envs} env spacing {self.cfg["env"]["envSpacing"]}')
        
        # 创建多个仿真环境实例（例如多个Ant机器人）
        # 参数：
        #   num_envs: 环境数量
        #   envSpacing: 每个环境之间的间距
        #   int(np.sqrt(self.num_envs)): 每行放多少个环境（用于二维排列）
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # 如果启用了随机化（domain randomization），在仿真开始前立即应用
        # 这会对环境参数（如摩擦系数、质量、关节阻尼等）进行随机扰动
        # 目的是提高智能体对真实世界变化的泛化能力
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _create_ground_plane(self):
        
        # 创建仿真地面的函数
        # 该函数负责定义地面物理属性（法线方向、摩擦系数等），并将其添加到物理仿真中
        
        # 创建一个 PlaneParams 对象，用于存储地面平面的参数
        # 这是 Isaac Gym 提供的结构体，包含法线、摩擦、弹性系数等属性
        plane_params = gymapi.PlaneParams()
        
        # 设置地面法线向量 (x=0, y=0, z=1)
        # 意思是地面法线沿 z 轴正方向，说明地面在 XY 平面上（水平地面）
        # 若改成 (1, 0, 0)，则地面会变为垂直墙壁
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
            
        # 设置地面的静摩擦系数（阻止物体在未滑动前产生运动）
        # 从配置文件中读取 self.plane_static_friction，通常为 0.8~1.0
        plane_params.static_friction = self.plane_static_friction
        
        # 设置地面的动摩擦系数（当物体滑动时的摩擦）
        # 从配置文件中读取 self.plane_dynamic_friction，通常略小于静摩擦
        plane_params.dynamic_friction = self.plane_dynamic_friction
        
        # 调用 Isaac Gym API，将定义好的地面添加到当前仿真环境 (self.sim) 中
        # 这样所有 agent、机器人、刚体都能与地面交互（碰撞、摩擦、支撑等）
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        """
           创建多个强化学习环境实例（例如同时并行训练多个Ant机器人）

        Args:
            num_envs (_type_): 环境数量
            spacing (_type_): 环境之间的间距
            num_per_row (_type_): 每行环境的数量，用于排列环境网格
        """
        
        # 定义每个环境的边界范围，用于在物理空间中摆放多个环境（避免碰撞）
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # 定义默认的资源路径和模型文件，这里是一个 MuJoCo 格式的 Ant 机器人模型
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        asset_file = "mjcf/nv_ant.xml"

        # 若配置文件中指定了模型路径，则使用配置文件中的模型，否则用默认Ant模型
        if "asset" in self.cfg["env"]:
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        # 拼接完整路径后，分离出模型根目录与文件名，供 Isaac Gym 加载使用
        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        # 创建资源加载选项（AssetOptions）
        # 这个对象控制加载模型的方式，如关节模式、可视化、碰撞信息等
        asset_options = gymapi.AssetOptions()

        # 注意：DOF 模式在 MJCF 文件中定义，此处只覆盖默认值
        # 不主动控制关节
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        # 角速度阻尼设为0，使运动更自然（不人为减速）
        asset_options.angular_damping = 0.0

        # 从指定路径加载 Ant 机器人模型资源（含骨架、关节、传感器等）
        # 返回的是 Isaac Gym 内部的 asset 句柄
        ant_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        
        # 获取机器人关节数量（Degree of Freedom，自由度）
        self.num_dof = self.gym.get_asset_dof_count(ant_asset)
        # 获取机器人刚体数量（通常包括躯干、腿、脚等）
        self.num_bodies = self.gym.get_asset_rigid_body_count(ant_asset)

        # 从模型中读取执行器属性（电机信息）
        actuator_props = self.gym.get_asset_actuator_properties(ant_asset)
        motor_efforts = [prop.motor_effort for prop in actuator_props]
        # 提取每个执行器的最大力矩（电机功率），转换为 Torch 张量，用于后续控制计算
        self.joint_gears = to_torch(motor_efforts, device=self.device)

        start_pose = gymapi.Transform()
        # 定义初始位姿 Transform：设置机器人在空间中的起始高度（0.44 米离地）
        # up_axis_idx 通常为 2，对应 z 轴方向
        start_pose.p = gymapi.Vec3(*get_axis_params(0.44, self.up_axis_idx))

        # 将初始旋转（四元数）转为 Torch 张量，用于计算旋转方向或坐标变换
        self.start_rotation = torch.tensor([start_pose.r.x, start_pose.r.y, start_pose.r.z, start_pose.r.w], device=self.device)

        self.torso_index = 0
        # 躯干在刚体列表中的索引设为0（Ant模型的第一个刚体）
        self.num_bodies = self.gym.get_asset_rigid_body_count(ant_asset)
        
        # 获取所有刚体的名字（例如 ["torso", "front_left_leg", "front_left_foot", ...]）
        body_names = [self.gym.get_asset_rigid_body_name(ant_asset, i) for i in range(self.num_bodies)]
        
        # 提取包含 “foot” 的刚体名，即四个脚部刚体的名字
        extremity_names = [s for s in body_names if "foot" in s]
        
        # 为脚部刚体创建一个索引张量（稍后会填充真实索引）
        self.extremities_index = torch.zeros(len(extremity_names), dtype=torch.long, device=self.device)

        # ====== 为脚部创建力传感器 ======
        extremity_indices = [self.gym.find_asset_rigid_body_index(ant_asset, name) for name in extremity_names]
        # 找到每个脚部刚体在 asset 中的索引
        sensor_pose = gymapi.Transform()
        # 在每个脚上创建一个力传感器（用于检测接触力，辅助奖励或终止条件）
        for body_idx in extremity_indices:
            self.gym.create_asset_force_sensor(ant_asset, body_idx, sensor_pose)
            
        # ====== 初始化环境容器 ======
        # 存储每个环境中 Ant actor 的句柄
        self.ant_handles = []
        # 存储每个环境的 gym 环境指针
        self.envs = []
        # 记录关节角度下限
        self.dof_limits_lower = []
        # 记录关节角度上限
        self.dof_limits_upper = []

        # ====== 创建多个环境实例 ======
        for i in range(self.num_envs):
            # 创建一个独立的仿真子环境（物理空间的一部分）
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            
            # 在该环境中实例化一个 Ant 机器人（actor）
            # 参数含义：
            #   env_ptr: 环境
            #   ant_asset: 机器人模型
            #   start_pose: 初始位置
            #   "ant": 名称
            #   i: 环境索引
            #   1, 0: 分别是分组 ID 与碰撞过滤参数
            ant_handle = self.gym.create_actor(env_ptr, ant_asset, start_pose, "ant", i, 1, 0)

            # 给机器人的每个刚体设置颜色（橙色系），方便在可视化时区分身体部件
            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(
                    # gymapi.Vec3(0.97, 0.38, 0.06) 是橙色，可以自行修改颜色
                    env_ptr, ant_handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.97, 0.38, 0.06))

            self.envs.append(env_ptr)
            # 保存环境和机器人句柄，方便后续控制和状态访问
            self.ant_handles.append(ant_handle)

        # ====== 获取并保存关节角度限制 ======
        # 从最后一个环境中的 actor 获取关节属性（包含上下限、阻尼、刚度等）
        dof_prop = self.gym.get_actor_dof_properties(env_ptr, ant_handle)
        
        for j in range(self.num_dof):
            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                # 若上下限颠倒，交换顺序（防止MJCF文件异常）
                self.dof_limits_lower.append(dof_prop['upper'][j])
                self.dof_limits_upper.append(dof_prop['lower'][j])
            else:
                self.dof_limits_lower.append(dof_prop['lower'][j])
                self.dof_limits_upper.append(dof_prop['upper'][j])
                
        # 转换为 Torch 张量，后续用于动作裁剪、重置初始姿态等
        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

        # ====== 记录脚部刚体在 actor 中的索引 ======
        for i in range(len(extremity_names)):
            # 在第一个环境中查找每个“foot”刚体的句柄索引，便于后续读取传感器或接触力
            self.extremities_index[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.ant_handles[0], extremity_names[i])

    def compute_reward(self, actions):
        # 调用 JIT 加速函数 compute_ant_reward 来计算奖励和环境重置标志
        # compute_ant_reward 是一个 PyTorch JIT 脚本函数，可在 GPU 上高效执行
        # 返回值：
        #   - rew_buf: 每个环境当前步的奖励
        #   - reset_buf: 每个环境是否需要重置（如摔倒或达到最大步数）
        self.rew_buf[:], self.reset_buf[:] = compute_ant_reward(
            self.obs_buf, # 当前观测值张量
            self.reset_buf,# 当前环境的重置标志
            self.progress_buf,# 当前环境的步数计数
            self.actions,# 当前智能体动作张量
            self.up_weight,# 奖励中“保持向上”方向的权重
            self.heading_weight,# 奖励中“朝向目标方向”权重
            self.potentials,# 当前势能，用于奖励计算
            self.prev_potentials,# 上一步势能，用于计算前进进度奖励
            self.actions_cost_scale,# 动作平方惩罚权重
            self.energy_cost_scale,# 能量消耗惩罚权重
            self.joints_at_limit_cost_scale,# 关节到达极限惩罚权重
            self.termination_height,# 智能体低于该高度则视为倒地
            self.death_cost,# 倒地奖励惩罚值
            self.max_episode_length# 最大步数，用于判断是否重置
        )

    def compute_observations(self):
        
        # 刷新关节状态张量，从 GPU 中获取最新的 DOF（自由度）状态
        self.gym.refresh_dof_state_tensor(self.sim)
        
        # 刷新智能体根刚体状态张量，从 GPU 中获取最新的根部位置、旋转、速度等信息
        self.gym.refresh_actor_root_state_tensor(self.sim)
        
        # 刷新力传感器张量，从 GPU 获取脚部接触力等传感器信息
        self.gym.refresh_force_sensor_tensor(self.sim)


        # 调用 GPU JIT 加速函数 compute_ant_observations 计算观测值和势能信息
        # 返回值依次为：
        #   - obs_buf: 每个环境的观测张量
        #   - potentials: 当前势能（用于奖励计算）
        #   - prev_potentials: 上一步势能
        #   - up_vec: 智能体上方向向量
        #   - heading_vec: 智能体朝向向量
        self.obs_buf[:], self.potentials[:], self.prev_potentials[:], self.up_vec[:], self.heading_vec[:] = compute_ant_observations(
            self.obs_buf, # 输入观测张量（会被更新）
            self.root_states, # 根刚体状态，包括位置、旋转、速度
            self.targets, # 目标位置张量
            self.potentials,# 当前势能张量
            self.inv_start_rot, # 初始旋转的共轭四元数
            self.dof_pos, # 关节位置张量
            self.dof_vel,# 关节速度张量
            self.dof_limits_lower, # 关节位置下限张量
            self.dof_limits_upper, # 关节位置上限张量
            self.dof_vel_scale,# 关节速度缩放因子
            self.vec_sensor_tensor, # 力传感器张量
            self.actions, # 当前动作张量
            self.dt, # 仿真时间步长
            self.contact_force_scale,# 接触力缩放因子
            self.basis_vec0, # 智能体基准向量0（通常是朝向向量）
            self.basis_vec1, # 智能体基准向量1（通常是右侧向量）
            self.up_axis_idx # 上方向轴索引
        )

    # 该函数用于计算“真实目标值（true objective）”，常用于 PBT（Population-Based Training）
    # 在PBT中，每个智能体（或策略）会根据这个指标来比较优劣并进行超参数进化。
    def compute_true_objective(self):
        
        # 从 root_states 中提取智能体的线速度信息
        # root_states 是一个张量，通常包含每个环境中智能体根部（躯干）的状态
        # 其结构一般为 [pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, rot_w, vel_x, vel_y, vel_z, ang_vel_x, ang_vel_y, ang_vel_z]
        # 因此，索引 7:10 对应的是线速度向量 (vel_x, vel_y, vel_z)
        velocity = self.root_states[:, 7:10]

        # 强化学习的目标是让智能体尽可能快速地向前（x轴方向）移动
        # 因此取出 x 轴方向的速度分量作为优化目标
        # velocity[:, 0] 代表所有环境中每个智能体的 x 方向速度
        # squeeze() 去掉多余的维度，使其成为一维向量
        self.extras['true_objective'] = velocity[:, 0].squeeze()

    def reset_idx(self, env_ids):
        # 随机化（domain randomization）只能在重置时发生，因为它可以重置 GPU 上的 actor 位置
        if self.randomize:
            # 对环境参数进行随机化（如质量、摩擦等）
            self.apply_randomizations(self.randomization_params)

        # 生成每个关节的随机初始位置，范围 [-0.2, 0.2]，张量形状为 [num_envs_to_reset, num_dof]
        positions = torch_rand_float(-0.2, 0.2, (len(env_ids), self.num_dof), device=self.device)
        # 生成每个关节的随机初始速度，范围 [-0.1, 0.1]，形状同上
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)

        # 将关节位置加上初始位置，并限制在关节上下限之间
        # tensor_clamp 会确保位置不会超出物理限制
        # 第一个参数是初始位置加上随机偏移，第二个和第三个参数是上下限
        self.dof_pos[env_ids] = tensor_clamp(self.initial_dof_pos[env_ids] + positions, self.dof_limits_lower, self.dof_limits_upper)
        # 设置关节速度
        self.dof_vel[env_ids] = velocities

        # 将 env_ids 转换为 int32 类型，用于 Isaac Gym API
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        # 重置指定环境的 actor 根刚体状态（位置、旋转、速度等）为初始状态
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.initial_root_states), # 初始根状态张量
            gymtorch.unwrap_tensor(env_ids_int32), # 要重置的环境索引
            len(env_ids_int32)# 索引数量
        )

        # 重置指定环境的关节状态（位置、速度）为当前 dof_state
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),# 当前关节状态张量
            gymtorch.unwrap_tensor(env_ids_int32), # 要重置的环境索引
            len(env_ids_int32)# 索引数量
        )

        # 计算从智能体初始位置到目标位置的向量
        to_target = self.targets[env_ids] - self.initial_root_states[env_ids, 0:3]
        
        # 将竖直方向（Z轴）忽略，只计算水平平面距离
        to_target[:, 2] = 0.0
        
        # 更新前一步势能（用于奖励计算），负值表示离目标越远惩罚越大
        self.prev_potentials[env_ids] = -torch.norm(to_target, p=2, dim=-1) / self.dt
        
        # 初始化当前势能为前一步势能
        self.potentials[env_ids] = self.prev_potentials[env_ids].clone()

        # 重置步数计数器
        self.progress_buf[env_ids] = 0
        
        # 重置环境重置标志
        self.reset_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        
        # 将输入动作克隆一份，并移动到仿真使用的设备（CPU/GPU）
        # clone() 确保原始动作张量不被修改
        self.actions = actions.clone().to(self.device)
        

        # 计算每个关节施加的力
        # 公式: 力 = 动作 * 关节齿轮比 * 功率缩放
        # joint_gears 表示每个关节的电机齿轮比（来自 MJCF actuator 属性）
        # power_scale 是环境中定义的功率缩放系数
        forces = self.actions * self.joint_gears * self.power_scale
        
        # 将 PyTorch 张量转换为 Isaac Gym 可识别的原始张量指针
        force_tensor = gymtorch.unwrap_tensor(forces)
        
        # 设置仿真中所有环境的关节作用力
        # Isaac Gym 在下一步物理仿真中会使用这些力来更新关节状态
        self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)

    def post_physics_step(self):
         # 所有环境步数计数加1
        self.progress_buf += 1
        
        # 所有环境随机化计数加1（用于 Domain Randomization）
        self.randomize_buf += 1

        # 获取需要重置的环境 ID（重置标志 reset_buf 不为0的环境）
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            
            # 对这些需要重置的环境执行重置
            self.reset_idx(env_ids)

        # 计算所有环境的观测值
        self.compute_observations()
        
        # 计算奖励值和环境是否需要重置
        self.compute_reward(self.actions)
        
        # 计算真实目标值（用于 PBT）
        self.compute_true_objective()

        # 调试可视化：显示智能体的朝向和上方向
        if self.viewer and self.debug_viz:
            
            # 清除上一帧的调试线条
            self.gym.clear_lines(self.viewer)
            
            # 刷新根刚体状态，确保可视化位置最新
            self.gym.refresh_actor_root_state_tensor(self.sim)

            # 初始化点和颜色列表，用于绘制方向和上方向线
            points = []
            colors = []
            
            # 遍历每个环境
            for i in range(self.num_envs):
                
                # 获取环境原点坐标
                origin = self.gym.get_env_origin(self.envs[i])
                
                # 获取智能体根位置
                pose = self.root_states[:, 0:3][i].cpu().numpy()
                
                # 将环境原点与智能体位置相加得到全局位置
                glob_pos = gymapi.Vec3(origin.x + pose[0], origin.y + pose[1], origin.z + pose[2])
                
                # 添加朝向向量线（红色）
                points.append([glob_pos.x, glob_pos.y, glob_pos.z, glob_pos.x + 4 * self.heading_vec[i, 0].cpu().numpy(),
                               glob_pos.y + 4 * self.heading_vec[i, 1].cpu().numpy(),
                               glob_pos.z + 4 * self.heading_vec[i, 2].cpu().numpy()])
                
                # 红色表示朝向
                colors.append([0.97, 0.1, 0.06])
                
                # 添加上方向向量线（绿色）
                points.append([glob_pos.x, glob_pos.y, glob_pos.z, glob_pos.x + 4 * self.up_vec[i, 0].cpu().numpy(), glob_pos.y + 4 * self.up_vec[i, 1].cpu().numpy(),
                               glob_pos.z + 4 * self.up_vec[i, 2].cpu().numpy()])
                
                # 绿色表示上方向
                colors.append([0.05, 0.99, 0.04])

            # 将所有点和颜色添加到可视化窗口
            # 每个环境绘制2条线（朝向 + 上方向）
            self.gym.add_lines(self.viewer, None, self.num_envs * 2, points, colors)

#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_ant_reward(
    obs_buf, # 智能体观测
    reset_buf,# 环境重置标志
    progress_buf,# 环境步数计数
    actions,# 智能体动作
    up_weight,# 奖励中“保持向上”方向的权重
    heading_weight,# 奖励中“朝向目标方向”权重
    potentials,# 当前势能，用于前进奖励
    prev_potentials,# 上一步势能
    actions_cost_scale,# 动作平方惩罚权重
    energy_cost_scale,# 能量消耗惩罚权重
    joints_at_limit_cost_scale,# 关节到达极限惩罚权重
    termination_height,# 智能体低于该高度则视为倒地
    death_cost,# 倒地奖励惩罚值
    max_episode_length# 最大步数，用于判断是否重置
):
    # type: (Tensor, Tensor, Tensor, Tensor, float, float, Tensor, Tensor, float, float, float, float, float, float) -> Tuple[Tensor, Tensor]

    # 朝向奖励：根据智能体面向目标的方向
    heading_weight_tensor = torch.ones_like(obs_buf[:, 11]) * heading_weight
    # 如果朝向指标 > 0.8，则给满奖励，否则按比例缩放
    heading_reward = torch.where(obs_buf[:, 11] > 0.8, heading_weight_tensor, heading_weight * obs_buf[:, 11] / 0.8)

    # 站立奖励：智能体竖直方向与环境上方向一致
    up_reward = torch.zeros_like(heading_reward)
    up_reward = torch.where(obs_buf[:, 10] > 0.93, up_reward + up_weight, up_reward)

    # 能量惩罚：限制智能体动作幅度和消耗
    # 动作平方和
    actions_cost = torch.sum(actions ** 2, dim=-1)
    # 动作功率消耗
    electricity_cost = torch.sum(torch.abs(actions * obs_buf[:, 20:28]), dim=-1)
    # 关节到达极限惩罚
    dof_at_limit_cost = torch.sum(obs_buf[:, 12:20] > 0.99, dim=-1)

    # 生存奖励：鼓励智能体存活
    alive_reward = torch.ones_like(potentials) * 0.5
    # 前进奖励：基于势能差（距离目标减少）
    progress_reward = potentials - prev_potentials

    # 总奖励计算    
    total_reward = progress_reward + alive_reward + up_reward + heading_reward - \
        actions_cost_scale * actions_cost - energy_cost_scale * electricity_cost - dof_at_limit_cost * joints_at_limit_cost_scale

    # 倒地处理：如果智能体高度低于阈值，给死亡惩罚
    total_reward = torch.where(obs_buf[:, 0] < termination_height, torch.ones_like(total_reward) * death_cost, total_reward)

    # 更新重置标志
    reset = torch.where(obs_buf[:, 0] < termination_height, torch.ones_like(reset_buf), reset_buf)
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)

    # 返回奖励和重置标志
    return total_reward, reset


@torch.jit.script
def compute_ant_observations(obs_buf, root_states, targets, potentials,
                             inv_start_rot, dof_pos, dof_vel,
                             dof_limits_lower, dof_limits_upper, dof_vel_scale,
                             sensor_force_torques, actions, dt, contact_force_scale,
                             basis_vec0, basis_vec1, up_axis_idx):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, float, float, Tensor, Tensor, int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]

    # 获取躯干位置、旋转、线速度和角速度
    # xyz位置
    torso_position = root_states[:, 0:3]
    # 四元数旋转
    torso_rotation = root_states[:, 3:7]
    # 线速度
    velocity = root_states[:, 7:10]
    # 角速度
    ang_velocity = root_states[:, 10:13]

    # 计算从躯干到目标的向量，并忽略竖直方向（Z轴）
    to_target = targets - torso_position
    to_target[:, 2] = 0.0

    # 更新势能值，基于到目标的距离
    prev_potentials_new = potentials.clone()
    # 计算当前势能 = 到目标的水平距离 / dt
    potentials = -torch.norm(to_target, p=2, dim=-1) / dt

    # 计算躯干朝向和竖直方向投影
    # torso_quat: 躯干旋转相对于初始旋转
    # up_proj: 躯干上方向与环境上方向对齐程度
    # heading_proj: 躯干朝向目标方向的对齐程度
    # up_vec: 当前上方向向量
    # heading_vec: 当前朝向向量
    torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
        torso_rotation, 
        inv_start_rot, 
        to_target, 
        basis_vec0, 
        basis_vec1, 
        2
    )

    # 将速度和角速度转换到躯干局部坐标系，并计算 roll, pitch, yaw, 角度到目标
    vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
        torso_quat, velocity, ang_velocity, targets, torso_position)

    # 将关节位置归一化到 [-1, 1] 范围
    dof_pos_scaled = unscale(dof_pos, dof_limits_lower, dof_limits_upper)

    # 拼接观测向量
    # 观测向量包含：
    # - 躯干高度
    # - 躯干局部线速度
    # - 躯干局部角速度
    # - yaw, roll
    # - 躯干到目标的角度
    # - 上方向投影和朝向投影
    # - 归一化的关节位置
    # - 关节速度
    # - 躯干脚传感器力
    # - 动作
    obs = torch.cat((torso_position[:, up_axis_idx].view(-1, 1), vel_loc, angvel_loc,
                     yaw.unsqueeze(-1), roll.unsqueeze(-1), angle_to_target.unsqueeze(-1),
                     up_proj.unsqueeze(-1), heading_proj.unsqueeze(-1), dof_pos_scaled,
                     dof_vel * dof_vel_scale, sensor_force_torques.view(-1, 24) * contact_force_scale,
                     actions), dim=-1)

    # 返回观测、势能、上一帧势能、上方向向量、朝向向量
    return obs, potentials, prev_potentials_new, up_vec, heading_vec