from typing import Tuple, Dict
import math
import torch
from torch import Tensor
@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute quaternion difference (orientation error)
    # Since we don't have quat_conjugate or quat_mul, we'll compute dot product directly
    # For unit quaternions, dot product magnitude indicates similarity (1 = identical, 0 = perpendicular, -1 = opposite)
    
    # Normalize quaternions to ensure they're unit quaternions
    object_rot_norm = torch.nn.functional.normalize(object_rot, p=2.0, dim=-1)
    goal_rot_norm = torch.nn.functional.normalize(goal_rot, p=2.0, dim=-1)
    
    # Compute dot product between quaternions
    # This gives us a measure of orientation similarity
    quat_dot = torch.sum(object_rot_norm * goal_rot_norm, dim=-1)
    
    # Take absolute value to handle double cover property of quaternions
    # (q and -q represent the same rotation)
    quat_similarity = torch.abs(quat_dot)
    
    # Main orientation reward - exponential to provide smooth gradients
    orientation_temperature = 2.0
    orientation_reward = torch.exp(orientation_temperature * (quat_similarity - 1.0))
    
    # Bonus reward for being close to target orientation
    bonus_threshold = 0.95  # When dot product > 0.95
    bonus_reward = torch.where(quat_similarity > bonus_threshold, 
                              torch.tensor(1.0, device=quat_similarity.device), 
                              torch.tensor(0.0, device=quat_similarity.device))
    
    # Combine rewards
    total_reward = orientation_reward + bonus_reward
    
    # Return individual components for debugging
    reward_components = {
        "orientation_reward": orientation_reward,
        "bonus_reward": bonus_reward,
        "quat_similarity": quat_similarity
    }
    
    return total_reward, reward_components
