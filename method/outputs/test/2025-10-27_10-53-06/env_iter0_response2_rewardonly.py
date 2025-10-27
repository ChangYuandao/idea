@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the difference between current object orientation and goal orientation
    # Using quaternion difference to measure orientation error
    quat_diff = quat_mul(object_rot, quat_conjugate(goal_rot))
    
    # Extract the angle from the quaternion difference
    # The angle is encoded in the w component: w = cos(angle/2)
    angle_diff = 2.0 * torch.acos(torch.abs(quat_diff[:, 3]))
    
    # Normalize the angle difference to [0, 1] range
    normalized_angle_diff = angle_diff / torch.pi
    
    # Create reward components
    reward_components = {}
    
    # Main orientation reward - exponential penalty based on angular difference
    orientation_temperature = 5.0
    orientation_reward = torch.exp(-orientation_temperature * normalized_angle_diff)
    reward_components["orientation_reward"] = orientation_reward
    
    # Bonus reward for being very close to target orientation
    bonus_threshold = 0.1  # 18 degrees
    bonus_reward = torch.where(normalized_angle_diff < bonus_threshold, 
                              torch.ones_like(normalized_angle_diff), 
                              torch.zeros_like(normalized_angle_diff))
    reward_components["bonus_reward"] = bonus_reward
    
    # Total reward
    total_reward = orientation_reward + bonus_reward
    
    return total_reward, reward_components
