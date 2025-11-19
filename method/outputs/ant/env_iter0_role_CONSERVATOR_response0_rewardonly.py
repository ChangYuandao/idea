from typing import Tuple, Dict
import math
import torch
from torch import Tensor
@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, actions: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract velocity information
    velocity = root_states[:, 7:10]  # Linear velocity
    
    # Calculate progress reward based on potential difference
    progress_reward = (potentials - prev_potentials) / dt
    
    # Bonus for forward velocity (assuming forward is along x-axis based on typical ant env)
    forward_velocity = velocity[:, 0]  # x-component of velocity
    forward_reward = forward_velocity
    
    # Energy penalty to encourage efficient movement
    action_penalty = -torch.sum(actions ** 2, dim=-1) * 0.05
    
    # Combine rewards
    total_reward = progress_reward + forward_reward + action_penalty
    
    # Create reward components dictionary
    reward_components = {
        "progress_reward": progress_reward,
        "forward_reward": forward_reward,
        "action_penalty": action_penalty
    }
    
    return total_reward, reward_components
