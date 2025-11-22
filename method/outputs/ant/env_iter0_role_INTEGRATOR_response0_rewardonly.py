from typing import Tuple, Dict
import math
import torch
from torch import Tensor
@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, prev_potentials: torch.Tensor, potentials: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract current position and velocity
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]
    
    # Reward forward velocity (x-direction velocity)
    forward_velocity = velocity[:, 0]  # Assuming x is forward direction
    forward_reward = forward_velocity
    
    # Reward for making progress toward target (potential-based reward)
    progress_reward = (prev_potentials - potentials) / dt
    
    # Combine rewards
    total_reward = forward_reward + progress_reward
    
    # Create reward components dictionary
    reward_components = {
        "forward_velocity_reward": forward_reward,
        "progress_reward": progress_reward
    }
    
    return total_reward, reward_components
