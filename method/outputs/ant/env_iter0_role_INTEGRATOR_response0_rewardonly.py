from typing import Tuple, Dict
import math
import torch
from torch import Tensor
@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, actions: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract relevant information
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]
    
    # Compute distance to target (potential progress reward)
    to_target = targets - torso_position
    to_target[:, 2] = 0.0  # Ignore height difference
    current_potential = -torch.norm(to_target, p=2.0, dim=-1) / dt
    potential_diff = current_potential - prev_potentials
    
    # Forward velocity reward (encourage movement in forward direction)
    forward_velocity = velocity[:, 0]  # Assuming x-axis is forward direction
    forward_velocity_reward = forward_velocity
    
    # Energy efficiency penalty (prevent excessive action usage)
    action_penalty_temp = 0.1
    action_penalty = -torch.sum(actions ** 2, dim=-1) * action_penalty_temp
    
    # Height stability reward (keep torso at reasonable height)
    target_height = 0.5
    height_deviation = torch.abs(torso_position[:, 2] - target_height)
    height_reward_temp = 1.0
    height_reward = -height_deviation * height_reward_temp
    
    # Total reward composition
    total_reward = potential_diff + forward_velocity_reward + action_penalty + height_reward
    
    # Return individual components for analysis
    reward_components = {
        "potential_diff": potential_diff,
        "forward_velocity": forward_velocity_reward,
        "action_penalty": action_penalty,
        "height_reward": height_reward
    }
    
    return total_reward, reward_components
