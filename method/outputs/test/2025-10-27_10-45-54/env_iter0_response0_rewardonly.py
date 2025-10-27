@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the difference between current object orientation and goal orientation
    # Using quaternion conjugate to compute relative rotation
    quat_diff = quat_mul(object_rot, quat_conjugate(goal_rot))
    
    # Extract the angle from the quaternion (w component)
    # The w component of a unit quaternion is cos(theta/2) where theta is the rotation angle
    cos_half_angle = torch.abs(quat_diff[:, 0])  # Take absolute value to handle double cover
    
    # Clamp to avoid numerical issues
    cos_half_angle = torch.clamp(cos_half_angle, -1.0, 1.0)
    
    # Convert to actual angle difference (in radians)
    angle_diff = 2.0 * torch.acos(cos_half_angle)
    
    # Normalize angle difference to [0, 1] range
    # Maximum possible angle is pi radians
    normalized_angle_diff = angle_diff / torch.pi
    
    # Create reward components
    reward_components = {}
    
    # Main orientation reward: exponential decay based on angular difference
    # Using temperature parameter for scaling
    orientation_temperature = 5.0
    orientation_reward = torch.exp(-orientation_temperature * normalized_angle_diff)
    reward_components["orientation_reward"] = orientation_reward
    
    # Bonus reward for being very close to target orientation
    close_threshold = 0.1  # 10% of max angle (18 degrees)
    bonus_reward = torch.where(normalized_angle_diff < close_threshold, 
                              torch.ones_like(normalized_angle_diff), 
                              torch.zeros_like(normalized_angle_diff))
    reward_components["bonus_reward"] = bonus_reward
    
    # Total reward
    total_reward = orientation_reward + bonus_reward
    
    return total_reward, reward_components
