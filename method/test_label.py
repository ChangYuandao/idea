import numpy as np
import pickle
import os
import numpy as np

import numpy as np

def trajectory_evaluate(trajectory_a, trajectory_b):
    """
    Both trajectory_a and trajectory_b are lists, each containing multiple states,
    and each state is represented as a dictionary.

    The label_list is a list of indices that corresponds to the states in the subtrajectory
    where A is definitively better or worse than B.

    The label indicates the quality of the sub-trajectory:
      - if A is better, the label is 1 ("Former")
      - if B is better, the label is -1 ("Latter")
      - if no valid sub-trajectory can be found between the two trajectories, label_list will be an empty list ([]).
    """
    traj_a_length = len(trajectory_a)
    traj_b_length = len(trajectory_b)
    assert traj_a_length == traj_b_length

    label_list = []

    for state_index in range(traj_a_length):
        state_a_info = trajectory_a[state_index]
        state_b_info = trajectory_b[state_index]

        # Extract object and goal poses
        object_pos_a = state_a_info['object_pos'][:, :3]  # shape (1, 3)
        object_rot_a = state_a_info['object_rot'][:, :4]  # shape (1, 4) - quaternion
        goal_pos_a = state_a_info['goal_pos'][:, :3]      # shape (1, 3)
        goal_rot_a = state_a_info['goal_rot'][:, :4]      # shape (1, 4) - quaternion

        object_pos_b = state_b_info['object_pos'][:, :3]  # shape (1, 3)
        object_rot_b = state_b_info['object_rot'][:, :4]  # shape (1, 4) - quaternion
        goal_pos_b = state_b_info['goal_pos'][:, :3]      # shape (1, 3)
        goal_rot_b = state_b_info['goal_rot'][:, :4]      # shape (1, 4) - quaternion

        # Calculate distance from object to goal position
        pos_dist_a = np.linalg.norm(object_pos_a - goal_pos_a, axis=1)
        pos_dist_b = np.linalg.norm(object_pos_b - goal_pos_b, axis=1)

        # Calculate orientation difference (using quaternion dot product)
        # For quaternions q1 and q2, the dot product gives cos(theta/2) where theta is the rotation angle
        # We want to measure how close the object's orientation is to the target orientation
        quat_dot_a = np.abs(np.sum(object_rot_a * goal_rot_a, axis=1))
        quat_dot_b = np.abs(np.sum(object_rot_b * goal_rot_b, axis=1))
        
        # Convert to orientation error (1 - |dot_product|) - smaller is better
        orient_error_a = 1 - quat_dot_a
        orient_error_b = 1 - quat_dot_b

        # Calculate object angular velocity magnitude (for spinning efficiency)
        object_angvel_a = state_a_info['object_angvel'][:, :3]  # shape (1, 3)
        object_angvel_b = state_b_info['object_angvel'][:, :3]  # shape (1, 3)
        
        angvel_mag_a = np.linalg.norm(object_angvel_a, axis=1)
        angvel_mag_b = np.linalg.norm(object_angvel_b, axis=1)

        # For efficiency evaluation, we consider:
        # 1. How close the object orientation is to the target (orient_error - smaller is better)
        # 2. How efficiently the object is rotating toward the target (higher angular velocity might be better if oriented correctly)
        # 3. Position proximity (smaller pos_dist is better)

        # Compare metrics: smaller orientation error and position distance are better
        # Higher angular velocity might indicate more efficient spinning if orientation is improving
        if (orient_error_a < orient_error_b and pos_dist_a <= pos_dist_b) or \
           (orient_error_a <= orient_error_b and pos_dist_a < pos_dist_b):
            # A is better in orientation and/or position
            label_list.append(1)
        elif (orient_error_a > orient_error_b and pos_dist_a >= pos_dist_b) or \
             (orient_error_a >= orient_error_b and pos_dist_a > pos_dist_b):
            # B is better in orientation and/or position
            label_list.append(-1)
        else:
            # Neither is clearly better
            label_list.append(0)

    return label_list

def reconstruct_trajectory(traj_dict):
    """
    将保存的轨迹字典拆分为每个时间步的状态字典列表
    """
    length = traj_dict['length']
    traj_list = []
    for t in range(length):
        state = {k: (v[t] if np.ndim(v) > 0 and hasattr(v, '__getitem__') else v) 
                for k, v in traj_dict.items() 
                if k not in ['length', 'total_reward']}
        traj_list.append(state)
    return traj_list


def load_and_compare_trajectories_from_file(pkl_file, traj_idx_a=0, traj_idx_b=1):
    """
    从pkl文件中加载多条轨迹,选择其中两条进行比较
    
    Args:
        pkl_file: 包含多条轨迹的pkl文件路径
        traj_idx_a: 第一条轨迹的索引
        traj_idx_b: 第二条轨迹的索引
    
    Returns:
        label_list: 比较结果列表
        traj_a: 第一条轨迹数据
        traj_b: 第二条轨迹数据
    """
    # 加载pkl文件
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    # 提取trajectories
    trajectories = data['trajectories']
    
    print(f"文件: {pkl_file}")
    print(f"总轨迹数: {len(trajectories)}")
    
    # 检查索引是否有效
    if traj_idx_a >= len(trajectories) or traj_idx_b >= len(trajectories):
        raise ValueError(f"轨迹索引超出范围。总轨迹数: {len(trajectories)}")
    
    # 选择两条轨迹
    traj_a = trajectories[traj_idx_a]
    traj_b = trajectories[traj_idx_b]
    
    traj_a = reconstruct_trajectory(traj_a)
    traj_b = reconstruct_trajectory(traj_b)
    print(type(traj_a), type(traj_b))
    # 比较轨迹
    label_list = trajectory_evaluate(traj_a, traj_b)
    
    # 打印结果统计
    print(f"\n比较轨迹 {traj_idx_a} vs 轨迹 {traj_idx_b}")
    print(f"轨迹长度: {len(label_list)}")
    print(f"轨迹{traj_idx_a}优于轨迹{traj_idx_b}的时间步数: {label_list.count(1)}")
    print(f"轨迹{traj_idx_b}优于轨迹{traj_idx_a}的时间步数: {label_list.count(-1)}")
    print(f"相等的时间步数: {label_list.count(0)}")
    print(f"\n标签列表: {label_list}")
    
    return label_list, traj_a, traj_b


if __name__ == "__main__":
    # 示例使用
    pkl_file = "/home/changyuandao/changyuandao/paperProject/idea/method/outputs/shadow_hand/iter0_trajectories.pkl"
    
    # 检查文件是否存在
    if os.path.exists(pkl_file):
        # 比较第0条和第1条轨迹
        labels, traj_a, traj_b = load_and_compare_trajectories_from_file(pkl_file, traj_idx_a=0, traj_idx_b=2)
        
    else:
        print(f"错误: 文件不存在 - {pkl_file}")