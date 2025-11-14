import logging 
import os 
from utils.reward_parser import extract_reward_parameters
from utils.reward_updater import update_reward_function_with_params
from utils.preference_learning import IsaacGymPreferenceLearning
from utils.trajectory_collector import collect_trajectories_from_checkpoint
from pathlib import Path

workspace_dir = Path("/home/changyuandao/changyuandao/paperProject/idea/method/outputs/ant")
best_reward_file = workspace_dir / f"env_iter0_role_CONSERVATOR_response0_rewardonly.py"
EUREKA_ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
ISAAC_ROOT_DIR = Path(os.path.abspath(os.path.join(EUREKA_ROOT_DIR, "../IsaacGymEnvs/isaacgymenvs")))

task = "Ant"
suffix = "GPT"
if not best_reward_file.exists():
    logging.error(f"❌ Best reward file not found: {best_reward_file}")
else:
    logging.info(f"📄 Best reward function file: {best_reward_file}")
    
    # 2. 从最佳奖励函数中提取参数
    try:
        hp_ranges, initial_values = extract_reward_parameters(str(best_reward_file))
        
        logging.info(f"Extracted {len(hp_ranges)} parameters:")
        for param_name, (min_val, max_val) in hp_ranges.items():
            init_val = initial_values.get(param_name, (min_val + max_val) / 2)
            logging.info(f"  {param_name}: {init_val:.4f} (range: [{min_val}, {max_val}])")
        
        # 3. 初始化偏好学习器
        preference_learner = IsaacGymPreferenceLearning(
            hp_ranges=hp_ranges,
            initial_values=initial_values,
            reward_file_path=str(best_reward_file),
            beta=1.0
        )
        
        # 开始偏好学习
        logging.info(f"Starting Preference Learning for Iteration {iter}")
        
        # 4. 构建checkpoint路径
        checkpoint_path = Path("/home/changyuandao/changyuandao/paperProject/idea/method/outputs/ant/Iteration_0/CONSERVATOR_0/runs/antgpt/nn/AntGPT.pth")

        logging.info(f"Checkpoint path: {checkpoint_path}")

        if checkpoint_path.exists():
            try:
                # 5. 收集轨迹
                logging.info("Step 1: Collecting trajectories...")
                trajectory_file = collect_trajectories_from_checkpoint(
                    isaac_root_dir=ISAAC_ROOT_DIR,
                    checkpoint_path=str(checkpoint_path),
                    task_name=task,
                    num_trajectories=10,  # 可配置
                    output_dir=workspace_dir,
                    save_filename=f"iter0_trajectories.pkl"
                )
                
                # 6. 加载轨迹
                logging.info("Step 2: Loading trajectories...")
                trajectories = preference_learner.load_trajectories(trajectory_file)
                
                # 7. 生成偏好对
                logging.info("Step 3: Generating preference pairs...")
                num_pref_pairs = 5
                preferences = preference_learner.generate_random_preferences(
                    trajectories, 
                    n_pairs=num_pref_pairs
                )
                
                # 8. 更新奖励函数参数
                logging.info("Step 4: Updating reward function parameters...")
                updated_params = preference_learner.update_reward_parameters(
                    trajectories,
                    preferences,
                    visualize=True
                )
                
                # 9. 更新奖励函数代码
                logging.info("Step 5: Updating reward function code...")
                updated_code_path = update_reward_function_with_params(
                    original_code_path=str(best_reward_file),
                    updated_params=updated_params,
                    output_path=workspace_dir / f"env_iter0_updated_reward.py"
                )
                
                logging.info(f"✅ Preference learning completed!")
                logging.info(f"Updated reward function saved to: {updated_code_path}")
                
            except Exception as e:
                logging.error(f"❌ Preference learning failed: {e}")
                import traceback
                logging.error(traceback.format_exc())
        else:
            logging.warning(f"⚠️ Checkpoint not found: {checkpoint_path}")
            
    except Exception as e:
        logging.error(f"❌ Failed to extract parameters from reward function: {e}")
        import traceback
        logging.error(traceback.format_exc())