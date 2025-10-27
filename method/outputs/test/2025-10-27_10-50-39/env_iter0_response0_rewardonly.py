@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the difference between current object orientation and goal orientation
    # Using quaternion distance: 1 - |q1 · q2|
    # where · is the dot product of quaternions
    
    # Normalize quaternions to ensure they're unit quaternions
    object_rot_normalized = object_rot / torch.norm(object_rot, dim=-1, keepdim=True)
    goal_rot_normalized = goal_rot / torch.norm(goal_rot, dim=-1, keepdim=True)
    
    # Compute dot product between quaternions
    dot_product = torch.sum(object_rot_normalized * goal_rot_normalized, dim=-1)
    
    # Take absolute value to handle quaternion double cover (q and -q represent same rotation)
    abs_dot_product = torch.abs(dot_product)
    
    # Orientation reward: closer to 1 means better alignment
    orientation_reward = abs_dot_product
    
    # Transform with exponential to make reward smoother and bounded between 0 and 1
    orientation_temperature = 5.0
    transformed_orientation_reward = torch.exp(orientation_temperature * (orientation_reward - 1.0))
    
    # Total reward is the orientation reward
    reward = transformed_orientation_reward
    
    # Create reward components dictionary
    reward_components = {
        "orientation_reward": orientation_reward,
        "transformed_orientation_reward": transformed_orientation_reward
    }
    
    return reward, reward_components
