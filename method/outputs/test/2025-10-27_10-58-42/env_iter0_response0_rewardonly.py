@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor, object_angvel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Normalize quaternions
    object_rot_norm = object_rot / torch.norm(object_rot, dim=-1, keepdim=True)
    goal_rot_norm = goal_rot / torch.norm(goal_rot, dim=-1, keepdim=True)
    
    # Compute rotation error as the angle between current and goal orientations
    # Using quaternion dot product: cos(theta/2) = |q1 · q2|
    quat_dot = torch.abs(torch.sum(object_rot_norm * goal_rot_norm, dim=-1))
    # Clamp to avoid numerical issues
    quat_dot = torch.clamp(quat_dot, -1.0 + 1e-8, 1.0 - 1e-8)
    # Convert to angular error (radians)
    rot_error = 2.0 * torch.acos(quat_dot)
    
    # Normalize rotation error to [0, 1] range
    normalized_rot_error = rot_error / torch.pi
    
    # Orientation reward: exponential decay based on rotation error
    orientation_temperature = 5.0
    orientation_reward = torch.exp(-orientation_temperature * normalized_rot_error)
    
    # Angular velocity penalty to encourage stable grasping
    angvel_magnitude = torch.norm(object_angvel, dim=-1)
    angvel_temperature = 0.5
    angvel_penalty = torch.exp(-angvel_temperature * angvel_magnitude)
    
    # Combine rewards
    total_reward = orientation_reward * angvel_penalty
    
    reward_components = {
        "orientation_reward": orientation_reward,
        "angvel_penalty": angvel_penalty,
        "rot_error": rot_error,
        "normalized_rot_error": normalized_rot_error
    }
    
    return total_reward, reward_components
