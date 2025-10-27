@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the difference between current object orientation and goal orientation
    # Using quaternion conjugate to compute relative rotation
    quat_diff = quat_mul(object_rot, quat_conjugate(goal_rot))
    
    # Extract the angle from the quaternion (w component)
    # The w component of a unit quaternion is cos(angle/2)
    cos_angle = torch.abs(quat_diff[:, 0])  # Take absolute value to handle double cover
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
    
    # Convert to angle and then to a reward (0 to 1, where 1 is perfect alignment)
    angle_diff = 2.0 * torch.acos(cos_angle)
    orientation_reward = 1.0 - (angle_diff / torch.pi)  # Normalize to [0,1]
    
    # Apply exponential transformation to make the reward sharper near the target
    temperature = 5.0
    exp_orientation_reward = torch.exp(temperature * (orientation_reward - 1.0))
    
    # Total reward is the orientation reward
    reward = exp_orientation_reward
    
    # Return reward components for debugging
    reward_components = {
        "orientation_reward": orientation_reward,
        "exp_orientation_reward": exp_orientation_reward,
        "angle_diff": angle_diff
    }
    
    return reward, reward_components
