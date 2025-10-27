@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the difference between current object rotation and goal rotation
    # Using quaternion distance: 1 - |q1 · q2|
    # where q1 · q2 is the dot product of the quaternions
    
    # Normalize quaternions to ensure they're unit quaternions
    object_rot_norm = torch.nn.functional.normalize(object_rot, p=2, dim=-1)
    goal_rot_norm = torch.nn.functional.normalize(goal_rot, p=2, dim=-1)
    
    # Compute dot product between quaternions
    dot_product = torch.sum(object_rot_norm * goal_rot_norm, dim=-1)
    
    # Take absolute value to handle double cover of SO(3)
    abs_dot_product = torch.abs(dot_product)
    
    # Orientation reward: higher when orientations match (dot product approaches 1)
    # Clamp to avoid numerical issues
    orientation_reward = torch.clamp(abs_dot_product, min=0.0, max=1.0)
    
    # Apply exponential transformation to make reward steeper near the goal
    # Temperature parameter controls how steep the reward function is
    orientation_temperature = 5.0
    exp_orientation_reward = torch.exp(orientation_temperature * (orientation_reward - 1.0))
    
    # Total reward is the orientation reward
    reward = exp_orientation_reward
    
    # Create info dictionary with reward components
    reward_info = {
        "orientation_reward": orientation_reward,
        "exp_orientation_reward": exp_orientation_reward
    }
    
    return reward, reward_info
