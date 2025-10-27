@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the quaternion distance between current object rotation and goal rotation
    # First, compute the conjugate of the object rotation
    object_rot_conj = quat_conjugate(object_rot)
    
    # Compute the relative rotation (difference) between goal and current orientation
    rel_rot = quat_mul(goal_rot, object_rot_conj)
    
    # The dot product of the quaternion with itself gives us a measure of alignment
    # We want to maximize the scalar part (first element) of the relative rotation
    # which represents cos(theta/2) where theta is the rotation angle between quaternions
    rot_alignment = rel_rot[:, 0]  # scalar part of quaternion
    
    # Clamp to avoid numerical issues
    rot_alignment = torch.clamp(rot_alignment, -1.0, 1.0)
    
    # Convert to rotation angle and normalize (0 to 1, where 1 is perfect alignment)
    # Since rot_alignment = cos(theta/2), we have theta = 2 * arccos(rot_alignment)
    # Normalize by pi to get range [0, 1]
    normalized_rot_error = 1.0 - (torch.acos(torch.abs(rot_alignment)) / (torch.pi / 2.0))
    
    # Apply exponential transformation to make the reward sharper near the target
    orientation_temperature = 5.0
    orientation_reward = torch.exp(orientation_temperature * (normalized_rot_error - 1.0))
    
    # Total reward
    reward = orientation_reward
    
    # Return individual reward components
    reward_components = {
        "orientation_reward": orientation_reward,
        "rot_alignment": rot_alignment,
        "normalized_rot_error": normalized_rot_error
    }
    
    return reward, reward_components
