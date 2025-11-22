"""
测试偏好学习流程的独立脚本
用于测试从加载轨迹到更新奖励函数参数的完整流程
"""
import os
import sys
import logging
from pathlib import Path
import traceback

# 添加当前目录到 Python 路径
EUREKA_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EUREKA_ROOT_DIR)

from utils.preference_learning import IsaacGymPreferenceLearning
from utils.reward_parser import extract_reward_parameters
from utils.reward_updater import update_reward_function_with_params

# 配置日志：新增将日志写入文件 test_run.log
log_file = Path(EUREKA_ROOT_DIR) / "test_run.log"

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),            # 保留原来的终端输出
        logging.FileHandler(log_file, mode='w', encoding='utf-8')  # 新增：输出到文件
    ]
)

logging.info(f"日志将保存到: {log_file}")

def test_preference_learning(
    trajectory_file: str,
    reward_file: str,
    evaluate_dir: str,
    task_name: str = "Ant",
    min_consecutive_steps: int = 5,
    beta: float = 1.0
):
    """
    测试偏好学习流程
    
    Args:
        trajectory_file: 轨迹文件路径 (.pkl)
        reward_file: 奖励函数文件路径 (_rewardonly.py)
        evaluate_dir: 评估函数目录路径
        task_name: 任务名称
        min_consecutive_steps: 最小连续偏好步数
        beta: 理性系数
    """
    logging.info("="*80)
    logging.info("Starting Preference Learning Test")
    logging.info("="*80)
    
    # 1. 验证文件存在性
    logging.info("[Step 1] Validating input files...")
    
    trajectory_path = Path(trajectory_file)
    reward_path = Path(reward_file)
    evaluate_path = Path(evaluate_dir)
    
    if not trajectory_path.exists():
        logging.error(f"❌ Trajectory file not found: {trajectory_path}")
        return False
    logging.info(f"✅ Trajectory file found: {trajectory_path}")
    
    if not reward_path.exists():
        logging.error(f"❌ Reward file not found: {reward_path}")
        return False
    logging.info(f"✅ Reward file found: {reward_path}")
    
    if not evaluate_path.exists():
        logging.warning(f"⚠️ Evaluate directory not found: {evaluate_path}")
        logging.warning("Will use random preference generation")
    else:
        logging.info(f"✅ Evaluate directory found: {evaluate_path}")
    
    try:
        # 2. 提取奖励函数参数
        logging.info("[Step 2] Extracting reward function parameters...")
        hp_ranges, initial_values = extract_reward_parameters(str(reward_path))
        
        # 3. 初始化偏好学习器
        logging.info("[Step 3] Initializing preference learner...")
        preference_learner = IsaacGymPreferenceLearning(
            hp_ranges=hp_ranges,
            initial_values=initial_values,
            reward_file_path=str(reward_path),
            beta=beta,
            task_name=task_name
        )
        logging.info("✅ Preference learner initialized")
        
        # 4. 加载轨迹
        logging.info("[Step 4] Loading trajectories...")
        trajectories = preference_learner.load_trajectories(str(trajectory_path))
        logging.info(f"✅ Loaded {len(trajectories)} trajectories")
        
        # 打印第一条轨迹的信息
        if len(trajectories) > 0:
            logging.info(f"First trajectory keys: {list(trajectories[0].keys())}")
            for key, value in trajectories[0].items():
                if hasattr(value, 'shape'):
                    logging.info(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                else:
                    logging.info(f"  {key}: type={type(value)}")
        
        # 5. 生成偏好对
        logging.info("[Step 5] Generating preference pairs...")
        
        if evaluate_path.exists():
            # 使用评估函数生成偏好对
            logging.info("Using evaluation functions to generate preferences...")
            preferences = preference_learner.generate_preference_buffer(
                trajectories,
                str(evaluate_path),
                min_consecutive=min_consecutive_steps
            )

        
        if len(preferences) == 0:
            logging.warning("⚠️ No preferences generated!")
            return False
        
        logging.info(f"✅ Generated {len(preferences)} preference pairs")
        
        # 6. 更新奖励函数参数
        logging.info("[Step 6] Updating reward function parameters...")
        updated_params = preference_learner.update_reward_parameters(
            trajectories,
            preferences,
        )
        
        logging.info("✅ Parameters updated successfully!")
        logging.info("Parameter changes:")
        for param_name in hp_ranges.keys():
            old_val = initial_values.get(param_name, 0)
            new_val = updated_params.get(param_name, 0)
            change = new_val - old_val
            change_pct = (change / old_val * 100) if old_val != 0 else 0
            logging.info(f"  {param_name}: {old_val:.4f} -> {new_val:.4f} ({change:+.4f}, {change_pct:+.2f}%)")
        
        # 7. 更新奖励函数代码
        logging.info("[Step 7] Updating reward function code...")
        output_dir = Path(EUREKA_ROOT_DIR) / "test_outputs"
        output_dir.mkdir(exist_ok=True)
        
        updated_code_path = update_reward_function_with_params(
            original_code_path=str(reward_path),
            updated_params=updated_params,
            output_path=output_dir / "updated_reward_function.py"
        )
        
        logging.info(f"✅ Updated reward function saved to: {updated_code_path}")
        
        # 8. 测试完成
        logging.info("\n" + "="*80)
        logging.info("🎉 Preference Learning Test PASSED!")
        logging.info("="*80)
        
        return True
        
    except Exception as e:
        logging.error(f"\n❌ Test FAILED with error: {e}")
        logging.error(traceback.format_exc())
        return False


def main():


    # 运行测试
    test_preference_learning(
        trajectory_file="/home/changyuandao/changyuandao/paperProject/idea/method/outputs/shadow_hand/iter0_trajectories.pkl",
        reward_file="/home/changyuandao/changyuandao/paperProject/idea/method/outputs/shadow_hand/env_iter0_role_CONSERVATOR_response0_rewardonly.py",
        evaluate_dir="/home/changyuandao/changyuandao/paperProject/idea/method/utils/prompts/evaluate_function/ShadowHand",
        task_name="ShadowHand",
        min_consecutive_steps=5,
        beta=1.0
    )
    


if __name__ == "__main__":
    main()