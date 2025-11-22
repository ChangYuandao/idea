from typing import Tuple, Dict
import math
import torch
from torch import Tensor
@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor, fingertip_pos: torch.Tensor, object_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute orientation alignment reward using quaternion dot product
    # Quaternion dot product gives us cos(theta/2) where theta is the rotation angle
    quat_dot = torch.sum(object_rot * goal_rot, dim=-1)
    # Take absolute value to handle quaternion double cover (q and -q represent same rotation)
    quat_dot = torch.abs(quat_dot)
    
    # Clamp to valid range [-1, 1] for numerical stability
    quat_dot = torch.clamp(quat_dot, -1.0, 1.0)
    
    # Convert to alignment reward (higher when closer to 1)
    orientation_alignment_reward = quat_dot
    
    # Temperature parameter for orientation reward transformation
    orientation_temperature = 5.0
    oriented_reward = torch.exp(orientation_temperature * (orientation_alignment_reward - 1.0))
    
    # Compute fingertip distance to object center to encourage grasping
    # Expand object position to match fingertip dimensions
    object_pos_expanded = object_pos.unsqueeze(1).expand_as(fingertip_pos)
    fingertip_to_object_dist = torch.norm(fingertip_pos - object_pos_expanded, dim=-1)
    
    # Mean distance across all fingertips
    mean_fingertip_distance = torch.mean(fingertip_to_object_dist, dim=-1)
    
    # Reward for keeping fingertips close to object (encourages grasp)
    grasp_temperature = 2.0
    grasp_reward = torch.exp(-grasp_temperature * mean_fingertip_distance)
    
    # Combine rewards
    total_reward = oriented_reward + 0.1 * grasp_reward
    
    # Return individual components for monitoring
    reward_components = {
        "oriented_reward": oriented_reward,
        "grasp_reward": grasp_reward,
        "orientation_alignment": orientation_alignment_reward,
        "mean_fingertip_distance": mean_fingertip_distance
    }
    
    return total_reward, reward_components
