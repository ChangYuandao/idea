from typing import Tuple, Dict
import math
import torch
from torch import Tensor
@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract relevant information from root_states
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]
    
    # Calculate distance to target (forward progress)
    to_target = targets - torso_position
    to_target[:, 2] = 0.0  # Ignore z-axis for forward progress
    distance_to_target = torch.norm(to_target, p=2.0, dim=-1)
    
    # Reward for moving forward (negative change in distance to target)
    forward_reward = -(distance_to_target - torch.norm(targets - torso_position.detach(), p=2.0, dim=-1)) / dt
    
    # Reward for speed in the forward direction
    forward_velocity = velocity[:, 0]  # Assuming x-axis is forward direction
    speed_reward = forward_velocity
    
    # Penalty for lateral movement
    lateral_movement_penalty = -torch.abs(velocity[:, 1])  # Assuming y-axis is lateral
    
    # Penalty for vertical movement (to encourage staying on ground)
    vertical_movement_penalty = -torch.abs(velocity[:, 2])  # Assuming z-axis is vertical
    
    # Energy efficiency penalty (to encourage efficient movement)
    energy_penalty_temperature = 0.1
    energy_penalty = -torch.exp(energy_penalty_temperature * torch.norm(velocity, p=2.0, dim=-1))
    
    # Combine rewards and penalties
    total_reward = ( 2.719379 * forward_reward + 2.383918 * speed_reward + 2.784904 * lateral_movement_penalty + 2.872568 * vertical_movement_penalty + 2.641255 * energy_penalty )
    
    # Return individual reward components for analysis
    reward_components = {
        "forward_reward": forward_reward,
        "speed_reward": speed_reward,
        "lateral_penalty": lateral_movement_penalty,
        "vertical_penalty": vertical_movement_penalty,
        "energy_penalty": energy_penalty
    }
    
    return total_reward, reward_components
