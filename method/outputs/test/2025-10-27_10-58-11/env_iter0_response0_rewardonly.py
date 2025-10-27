@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor, fingertip_pos: torch.Tensor, object_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute orientation alignment reward using quaternion distance
    # Normalize quaternions first
    object_rot_norm = torch.nn.functional.normalize(object_rot, p=2, dim=-1)
    goal_rot_norm = torch.nn.functional.normalize(goal_rot, p=2, dim=-1)
    
    # Compute dot product between quaternions (equivalent to cos(theta/2) where theta is rotation angle)
    quat_dot = torch.sum(object_rot_norm * goal_rot_norm, dim=-1)
    # Take absolute value to handle quaternion double cover (q and -q represent same rotation)
    quat_dot = torch.abs(quat_dot)
    
    # Orientation reward: higher when quaternions are aligned
    orientation_temp = 10.0
    orientation_reward = torch.exp(orientation_temp * (quat_dot - 1.0))
    
    # Compute fingertip convergence reward (encourage fingertips to stay close together around object)
    # This helps maintain a stable grasp
    if fingertip_pos.shape[1] > 1:  # Check if we have multiple fingertips
        # Compute centroid of fingertips
        fingertip_centroid = torch.mean(fingertip_pos, dim=1)  # Shape: [num_envs, 3]
        
        # Compute distances from each fingertip to centroid
        fingertip_deviation = torch.norm(fingertip_pos - fingertip_centroid.unsqueeze(1), dim=-1)
        avg_fingertip_deviation = torch.mean(fingertip_deviation, dim=-1)  # Average deviation per env
        
        # Reward smaller deviations (fingertips staying together)
        fingertip_temp = 5.0
        fingertip_reward = torch.exp(-fingertip_temp * avg_fingertip_deviation)
    else:
        fingertip_reward = torch.ones_like(orientation_reward)
    
    # Object position stability reward (encourage object to stay in place while rotating)
    # Compute variance of object position over time could be useful, but since we don't have history,
    # we can encourage the object to stay near its current position
    object_height = object_pos[:, 2]  # z-coordinate
    # Assume objects should stay above a certain height (e.g., table surface)
    height_target = 0.0  # Adjust based on environment setup
    height_error = torch.abs(object_height - height_target)
    height_temp = 5.0
    height_reward = torch.exp(-height_temp * height_error)
    
    # Combine rewards
    total_reward = orientation_reward * fingertip_reward * height_reward
    
    reward_components = {
        "orientation_reward": orientation_reward,
        "fingertip_reward": fingertip_reward,
        "height_reward": height_reward,
        "quat_alignment": quat_dot
    }
    
    return total_reward, reward_components
