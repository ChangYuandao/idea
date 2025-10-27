@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the rotation error as the angle difference between current and goal orientations
    # First, compute the relative rotation (error rotation) from object to goal
    quat_conjugate = torch.zeros_like(object_rot)
    quat_conjugate[..., 0] = object_rot[..., 0]
    quat_conjugate[..., 1] = -object_rot[..., 1]
    quat_conjugate[..., 2] = -object_rot[..., 2]
    quat_conjugate[..., 3] = -object_rot[..., 3]
    
    # Compute relative rotation: goal_rot * conjugate(object_rot)
    rel_rot_x = goal_rot[..., 0] * quat_conjugate[..., 0] - goal_rot[..., 1] * quat_conjugate[..., 1] - goal_rot[..., 2] * quat_conjugate[..., 2] - goal_rot[..., 3] * quat_conjugate[..., 3]
    rel_rot_y = goal_rot[..., 0] * quat_conjugate[..., 1] + goal_rot[..., 1] * quat_conjugate[..., 0] - goal_rot[..., 2] * quat_conjugate[..., 3] + goal_rot[..., 3] * quat_conjugate[..., 2]
    rel_rot_z = goal_rot[..., 0] * quat_conjugate[..., 2] + goal_rot[..., 1] * quat_conjugate[..., 3] + goal_rot[..., 2] * quat_conjugate[..., 0] - goal_rot[..., 3] * quat_conjugate[..., 1]
    rel_rot_w = goal_rot[..., 0] * quat_conjugate[..., 3] - goal_rot[..., 1] * quat_conjugate[..., 2] + goal_rot[..., 2] * quat_conjugate[..., 1] + goal_rot[..., 3] * quat_conjugate[..., 0]
    
    # Convert quaternion to angle-axis representation to get the rotation error
    # For unit quaternions, the angle can be computed from the w component: angle = 2 * arccos(|w|)
    # To avoid numerical issues, we clamp the w component
    rel_rot_w_clamped = torch.clamp(torch.abs(rel_rot_w), 0.0, 1.0)
    rotation_error = 2.0 * torch.acos(rel_rot_w_clamped)
    
    # Normalize the error to [0, 1] range where 0 is perfect alignment
    normalized_rotation_error = rotation_error / torch.pi
    
    # Compute reward: higher reward for lower error
    # Using exponential to provide smooth gradients and strong reward signal near the goal
    orientation_reward_temperature = 5.0
    orientation_reward = torch.exp(-orientation_reward_temperature * normalized_rotation_error)
    
    # Total reward is the orientation reward
    reward = orientation_reward
    
    # Return individual reward components for debugging
    reward_components = {
        "orientation_reward": orientation_reward,
        "rotation_error": normalized_rotation_error
    }
    
    return reward, reward_components
