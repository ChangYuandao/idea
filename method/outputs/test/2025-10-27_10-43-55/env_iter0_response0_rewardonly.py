@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor, object_angvel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Normalize quaternions
    object_rot_normalized = object_rot / torch.norm(object_rot, dim=-1, keepdim=True)
    goal_rot_normalized = goal_rot / torch.norm(goal_rot, dim=-1, keepdim=True)
    
    # Compute quaternion distance (dot product)
    # This gives us a value between -1 and 1, where 1 means perfect alignment
    quat_dot = torch.sum(object_rot_normalized * goal_rot_normalized, dim=-1)
    
    # Take absolute value to handle quaternion double cover (q and -q represent same rotation)
    quat_dot_abs = torch.abs(quat_dot)
    
    # Main orientation reward - exponential to make it more sensitive near the target
    orientation_temp = 5.0
    orientation_reward = torch.exp(orientation_temp * (quat_dot_abs - 1.0))
    
    # Bonus for angular velocity aligned with rotation error
    # This encourages spinning in the right direction
    angvel_norm = torch.norm(object_angvel, dim=-1)
    angvel_temp = 1.0
    angvel_bonus = torch.exp(-angvel_temp * angvel_norm) * quat_dot_abs
    
    # Combine rewards
    total_reward = orientation_reward + 0.1 * angvel_bonus
    
    # Create reward components dictionary
    reward_components = {
        "orientation_reward": orientation_reward,
        "angvel_bonus": angvel_bonus,
        "quat_alignment": quat_dot_abs
    }
    
    return total_reward, reward_components
