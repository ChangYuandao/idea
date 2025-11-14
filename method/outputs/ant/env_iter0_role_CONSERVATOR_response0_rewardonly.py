from typing import Tuple, Dict
import math
import torch
from torch import Tensor
@torch.jit.script
def compute_reward(potentials: torch.Tensor, prev_potentials: torch.Tensor, torso_position: torch.Tensor, velocity: torch.Tensor, up_vec: torch.Tensor, heading_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Progress reward based on potential difference (distance covered toward target)
    progress_reward = potentials - prev_potentials
    
    # Speed reward - encourage movement in the forward direction
    forward_speed = torch.sum(velocity * heading_vec, dim=-1)
    speed_reward = forward_speed
    
    # Height reward - encourage maintaining proper body height
    target_height = 0.6
    current_height = torso_position[:, 2]
    height_diff = torch.abs(current_height - target_height)
    height_reward = -height_diff
    
    # Upright reward - encourage maintaining upright posture
    up_reward = up_vec[:, 2] - 1.0  # Should be close to 1.0 when perfectly upright
    
    # Combine rewards
    total_reward = progress_reward + 0.5 * speed_reward + 1.0 * height_reward + 0.1 * up_reward
    
    reward_components = {
        "progress_reward": progress_reward,
        "speed_reward": speed_reward,
        "height_reward": height_reward,
        "up_reward": up_reward
    }
    
    return total_reward, reward_components
