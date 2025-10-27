@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the difference between current object rotation and goal rotation
    # Using quaternion difference to measure orientation error
    quat_diff = quat_mul(object_rot, quat_conjugate(goal_rot))
    
    # Extract the angle of rotation from the quaternion (w component)
    # The w component of a unit quaternion is cos(theta/2) where theta is the rotation angle
    cos_half_angle = torch.abs(quat_diff[:, 3])  # Take absolute value to handle double cover
    
    # Clamp to avoid numerical issues
    cos_half_angle = torch.clamp(cos_half_angle, -1.0 + 1e-6, 1.0 - 1e-6)
    
    # Convert to actual angle difference (in radians)
    angle_diff = 2.0 * torch.acos(cos_half_angle)
    
    # Normalize angle difference to [0, 1] range where 0 is perfect match
    # Maximum possible angle difference is pi radians
    normalized_angle_diff = angle_diff / torch.pi
    
    # Compute reward: higher reward for smaller angle difference
    # Using exponential to provide smooth gradients
    orientation_reward_temperature = 5.0
    orientation_reward = torch.exp(-orientation_reward_temperature * normalized_angle_diff)
    
    # Total reward
    reward = orientation_reward
    
    # Return reward components
    reward_components = {
        "orientation_reward": orientation_reward,
        "angle_diff": angle_diff,
        "normalized_angle_diff": normalized_angle_diff
    }
    
    return reward, reward_components
