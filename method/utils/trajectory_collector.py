import subprocess
import logging
from pathlib import Path
import time
import os


def collect_trajectories_from_checkpoint(
    isaac_root_dir: str,
    checkpoint_path: str,
    task_name: str,
    num_trajectories: int = 10,
    output_dir: Path = None,
    save_filename: str = "trajectories.pkl"
) -> str:
    """
    从checkpoint收集轨迹
    
    Args:
        isaac_root_dir: IsaacGym根目录
        checkpoint_path: checkpoint文件路径
        task_name: 任务名称
        num_trajectories: 收集的轨迹数量
        output_dir: 输出目录
        
    Returns:
        轨迹文件路径
    """
    if output_dir is None:
        output_dir = Path('./trajectories')
    
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Collecting {num_trajectories} trajectories from checkpoint: {checkpoint_path}")
    logging.info(f"Output directory: {output_dir}")
    
    logging.info(f"task: {task_name}")
    rollout_cmd = [
        'python', '-u', f'{isaac_root_dir}/rollout.py',
        f'task={task_name}',
        f'checkpoint={checkpoint_path}',
        f'save_dir={str(output_dir)}',  # 传递保存目录
        f'num_trajectories={num_trajectories}',
        f'save_filename={save_filename}',
    ]
    
    # 执行rollout命令
    log_file = output_dir / 'rollout.log'
    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            rollout_cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=isaac_root_dir
        )
    
    logging.info(f"Rollout process started (PID: {process.pid})")
    logging.info(f"Log file: {log_file}")
    
    # 等待收集完成
    timeout = 600  # 10分钟超时
    start_time = time.time()
    
    while True:
        ret_code = process.poll()
        
        if ret_code is not None:
            if ret_code == 0:
                logging.info("✅ Trajectory collection completed successfully")
                break
            else:
                logging.error(f"❌ Trajectory collection failed with return code: {ret_code}")
                with open(log_file, 'r') as f:
                    logging.error(f.read())
                raise RuntimeError("Trajectory collection failed")
        
        # 检查超时
        if time.time() - start_time > timeout:
            process.kill()
            logging.error("❌ Trajectory collection timeout")
            raise TimeoutError("Trajectory collection timeout")
        
        time.sleep(5)
    
    # 查找生成的轨迹文件
    trajectory_file = output_dir / save_filename
    
    if not trajectory_file.exists():
        # 调试：列出目录下所有文件
        all_files = list(output_dir.glob("*"))
        logging.error(f"Target file not found: {trajectory_file}")
        logging.error(f"All files in directory: {[f.name for f in all_files]}")
        raise FileNotFoundError(f"Trajectory file not found: {trajectory_file}")
    
    logging.info(f"✅ Trajectory file found: {trajectory_file}")
    logging.info(f"File size: {trajectory_file.stat().st_size / 1024:.2f} KB")
    
    return str(trajectory_file)


