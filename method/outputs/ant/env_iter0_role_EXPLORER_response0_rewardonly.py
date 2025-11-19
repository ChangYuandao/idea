from typing import Tuple, Dict
import math
import torch
from torch import Tensor
@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract relevant information from root_states
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]
    
    # Calculate forward velocity (assuming x-axis is forward direction)
    forward_velocity = velocity[:, 0]
    
    # Calculate distance to target
    to_target = targets - torso_position
    to_target[:, 2] = 0.0  # Ignore z-component for horizontal distance
    distance_to_target = torch.norm(to_target, p=2, dim=-1)
    
    # Primary reward: forward velocity
    forward_vel_reward = forward_velocity
    
    # Progress reward: negative distance to target (encourage moving toward target)
    progress_reward = -distance_to_target / dt
    
    # Exploration bonus: exponential of forward velocity to encourage high speeds
    exploration_temp = 0.1
    exploration_bonus = torch.exp(forward_velocity * exploration_temp)
    
    # Stability penalty: discourage excessive angular velocity
    ang_velocity = root_states[:, 10:13]
    angular_vel_penalty = -torch.norm(ang_velocity, p=2, dim=-1) * 0.01
    
    # Combine rewards
    total_reward = forward_vel_reward + progress_reward + exploration_bonus + angular_vel_penalty
    
    # Normalize reward to a fixed range using tanh
    total_reward = torch.tanh(total_reward)
    
    # Return individual reward components for analysis
    reward_components = {
        "forward_velocity": forward_vel_reward,
        "progress": progress_reward,
        "exploration_bonus": exploration_bonus,
        "angular_velocity_penalty": angular_vel_penalty
    }
    
    return total_reward, reward_components
