from typing import Tuple, Dict
import math
import torch
from torch import Tensor
@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dof_pos: torch.Tensor, dof_vel: torch.Tensor, actions: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract relevant information from root states
    torso_position = root_states[:, 0:3]
    torso_rotation = root_states[:, 3:7]
    velocity = root_states[:, 7:10]
    
    # Calculate forward velocity reward (project velocity onto forward direction)
    # Assuming x-axis is forward direction in world coordinates
    forward_velocity = velocity[:, 0]  # x-component of velocity
    forward_velocity_reward = forward_velocity
    
    # Penalize lateral movement to encourage straight-line running
    lateral_velocity_penalty = -(torch.abs(velocity[:, 1]) + torch.abs(velocity[:, 2]))
    
    # Penalize energy expenditure (action magnitude)
    action_penalty_temp = 0.1
    action_penalty = -torch.sum(actions ** 2, dim=-1) * action_penalty_temp
    
    # Penalize excessive joint velocities (smoothness)
    dof_vel_penalty_temp = 0.01
    dof_vel_penalty = -torch.sum(dof_vel ** 2, dim=-1) * dof_vel_penalty_temp
    
    # Penalize deviation from nominal joint positions to encourage natural gait
    dof_pos_nominal = torch.zeros_like(dof_pos)
    dof_pos_penalty_temp = 0.05
    dof_pos_penalty = -torch.sum((dof_pos - dof_pos_nominal) ** 2, dim=-1) * dof_pos_penalty_temp
    
    # Combine all reward components
    total_reward = ( forward_velocity_reward + lateral_velocity_penalty * 0.5 + action_penalty + dof_vel_penalty + dof_pos_penalty )
    
    # Create info dictionary for debugging
    reward_components = {
        "forward_velocity_reward": forward_velocity_reward,
        "lateral_velocity_penalty": lateral_velocity_penalty,
        "action_penalty": action_penalty,
        "dof_vel_penalty": dof_vel_penalty,
        "dof_pos_penalty": dof_pos_penalty
    }
    
    return total_reward, reward_components
