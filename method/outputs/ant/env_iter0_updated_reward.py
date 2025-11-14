from typing import Tuple, Dict
import math
import torch
from torch import Tensor
@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract relevant information from root_states
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]
    
    # Calculate direction to target (forward direction)
    to_target = targets - torso_position
    to_target[:, 2] = 0.0  # Zero out z-component for horizontal movement
    
    # Normalize the direction to target
    to_target_normalized = torch.nn.functional.normalize(to_target, p=2.0, dim=-1)
    
    # Calculate forward velocity (dot product of velocity and direction to target)
    forward_velocity = torch.sum(velocity * to_target_normalized, dim=-1)
    
    # Reward for forward speed
    speed_reward = forward_velocity
    
    # Penalty for moving sideways or upwards
    vertical_velocity_penalty = torch.abs(velocity[:, 2])
    
    # Energy efficiency reward (penalize large actions)
    # Note: This would require action information which is not available in the current signature
    # For now, we'll omit this component
    
    # Combine rewards
    total_reward = 0.936711 * speed_reward - 0.943992 * vertical_velocity_penalty
    
    # Create reward components dictionary
    reward_components = {
        "speed_reward": speed_reward,
        "vertical_velocity_penalty": vertical_velocity_penalty
    }
    
    return total_reward, reward_components
