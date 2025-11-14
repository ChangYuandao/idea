from typing import Tuple, Dict
import math
import torch
from torch import Tensor
@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, actions: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract relevant information
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]
    
    # Calculate forward velocity reward (encourage movement in forward direction)
    # Assuming forward is along the x-axis (common convention)
    forward_velocity = velocity[:, 0]  # x-component of velocity
    forward_velocity_reward = forward_velocity
    
    # Progress reward based on potential change (distance covered towards target)
    progress_reward = (potentials - prev_potentials) / dt
    
    # Energy efficiency penalty (penalize large actions to encourage efficient movement)
    action_penalty_temp = 0.1
    action_penalty = torch.exp(-action_penalty_temp * torch.sum(actions**2, dim=-1)) - 1.0
    
    # Stability reward (keep the ant upright, penalize falling)
    # Assuming z-axis is up, encourage torso height to stay around nominal height (typically 0.75 for Ant)
    torso_height = torso_position[:, 2]
    target_height = 0.75
    height_deviation = torch.abs(torso_height - target_height)
    stability_temp = 2.0
    stability_reward = torch.exp(-stability_temp * height_deviation)
    
    # Combine rewards
    total_reward = forward_velocity_reward + progress_reward + action_penalty + stability_reward
    
    # Return individual components for analysis
    reward_components = {
        "forward_velocity_reward": forward_velocity_reward,
        "progress_reward": progress_reward,
        "action_penalty": action_penalty,
        "stability_reward": stability_reward
    }
    
    return total_reward, reward_components
