import re
import logging
from pathlib import Path
from typing import Dict


def update_reward_function_with_params(
    original_code_path: str,
    updated_params: Dict[str, float],
    output_path: Path
) -> Path:
    """
    使用更新后的参数修改奖励函数代码
    
    Args:
        original_code_path: 原始奖励函数代码路径
        updated_params: 更新后的参数字典
        output_path: 输出路径
        
    Returns:
        更新后的代码文件路径
    """
    logging.info(f"Updating reward function with new parameters...")
    
    # 读取原始代码
    with open(original_code_path, 'r') as f:
        original_code = f.read()
    
    # 更新参数
    updated_code = original_code
    
    for param_name, param_value in updated_params.items():
        # 处理温度参数
        if param_name.endswith('_temp'):
            pattern = rf'{param_name}\s*=\s*[\d\.]+'
            if re.search(pattern, updated_code):
                updated_code = re.sub(
                    pattern, 
                    f'{param_name} = {param_value:.6f}',
                    updated_code
                )
                logging.info(f"  Updated {param_name}: {param_value:.6f}")
            else:
                logging.warning(f"  Parameter {param_name} not found in code")
        
        # 处理权重参数
        elif param_name.endswith('_weight'):
            # 从 param_name 中提取奖励项名称
            reward_name = param_name[:-7]  # 去掉 '_weight'
            
            # 在 total_reward = 行中查找并替换
            # 匹配 total_reward = ... reward_name ... 的模式
            # 支持已有权重或没有权重的情况
            pattern = rf'(total_reward\s*=\s*[^=\n]*?)((?:[\d\.\-]+\s*\*\s*)?{reward_name}\b)'
            
            def replace_weight(match):
                prefix = match.group(1)
                reward_expr = match.group(2)
                # 移除旧的权重系数(如果存在)
                reward_only = re.sub(rf'^[\d\.\-]+\s*\*\s*', '', reward_expr)
                # 添加新的权重系数
                return f'{prefix}{param_value:.6f} * {reward_only}'
            
            if re.search(pattern, updated_code):
                updated_code = re.sub(pattern, replace_weight, updated_code)
                logging.info(f"  Updated {param_name}: {param_value:.6f} * {reward_name}")
            else:
                logging.warning(f"  Weight parameter {param_name} (reward: {reward_name}) not found in total_reward line")
    
    
    # 写入新文件
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(updated_code)
    
    logging.info(f"Updated reward function saved to: {output_path}")
    
    return output_path