from typing import Tuple, Dict
import math
import torch
from torch import Tensor
@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor, object_angvel: torch.Tensor, fingertip_pos: torch.Tensor, object_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute orientation error using quaternion distance
    # Normalize quaternions first
    object_rot_norm = torch.nn.functional.normalize(object_rot, p=2.0, dim=-1)
    goal_rot_norm = torch.nn.functional.normalize(goal_rot, p=2.0, dim=-1)
    
    # Quaternion dot product (equivalent to cos(theta/2) where theta is rotation angle)
    quat_dot = torch.sum(object_rot_norm * goal_rot_norm, dim=-1)
    # Ensure we take the shorter rotation path
    quat_dot = torch.abs(quat_dot)
    
    # Orientation reward - closer to 1 means better alignment
    orientation_reward = quat_dot
    
    # Encourage spinning behavior by rewarding angular velocity magnitude
    angvel_magnitude = torch.norm(object_angvel, dim=-1)
    spinning_reward_temp = 2.0
    spinning_reward = torch.exp(spinning_reward_temp * (angvel_magnitude - 1.0))  # Offset to encourage minimum spinning
    
    # Reward for fingertip contact distribution to encourage proper grip
    # Calculate variance of fingertip positions to encourage spread
    fingertip_mean = torch.mean(fingertip_pos, dim=1, keepdim=True)
    fingertip_variance = torch.mean(torch.var(fingertip_pos, dim=1), dim=-1)
    grip_reward_temp = 1.0
    grip_reward = torch.exp(grip_reward_temp * fingertip_variance)
    
    # Bonus for maintaining object height (prevent dropping)
    height_target = 0.5  # Adjust based on desired height
    current_height = object_pos[:, 2]  # z-coordinate
    height_diff = torch.abs(current_height - height_target)
    height_reward_temp = 3.0
    height_reward = torch.exp(-height_reward_temp * height_diff)
    
    # Combine rewards with weights
    total_reward = ( 2.0 * orientation_reward + 0.5 * spinning_reward + 0.3 * grip_reward + 1.0 * height_reward )
    
    reward_dict = {
        "orientation_reward": orientation_reward,
        "spinning_reward": spinning_reward,
        "grip_reward": grip_reward,
        "height_reward": height_reward
    }
    
    return total_reward, reward_dict
