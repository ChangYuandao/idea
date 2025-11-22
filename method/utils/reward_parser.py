import re
import ast
from pathlib import Path
from typing import Dict, Tuple, Any, List
import logging


import re
import logging
from typing import Dict, Tuple





def extract_reward_parameters(reward_file_path: str) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, float]]:
    """
    从奖励函数文件中提取可调参数（更鲁棒版本）
    支持以下特性：
      - total_reward / reward / final_reward 多行赋值
      - 任意括号格式
      - 隐式权重（没写出系数）
      - 支持整数、浮点、科学计数法
      - 自动截断 return 之后的内容
    """

    with open(reward_file_path, 'r') as f:
        code = f.read()

    hp_ranges: Dict[str, Tuple[float, float]] = {}
    initial_values: Dict[str, float] = {}

    # --- 1️⃣ 提取所有 _temp 参数 ---
    temp_pattern = r'(\w+_temp|temperature)\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'
    for param_name, value in re.findall(temp_pattern, code):
        try:
            val = float(value)
            initial_values[param_name] = val
            hp_ranges[param_name] = (0.01, 0.5)
            logging.info(f"Found temp parameter: {param_name} = {val}")
        except ValueError:
            continue

    # --- 2️⃣ 截断到 return 之前 ---
    # 只保留 compute_reward 主体部分
    match_func = re.search(r'def\s+compute_reward[\s\S]+', code)
    if match_func:
        func_body = match_func.group(0)
        func_body = re.split(r'\breturn\b', func_body, 1)[0]  # 截断到 return 之前
    else:
        func_body = code

    # --- 3️⃣ 清理文本（去注释、空格、换行）---
    func_body = re.sub(r'#.*', '', func_body)
    func_body = re.sub(r'\s+', '', func_body)

    # --- 4️⃣ 匹配 total_reward/ reward / final_reward ---
    reward_names = ['total_reward', 'reward', 'final_reward', 'rew']
    reward_expr = None

    for name in reward_names:
        pattern = rf'{name}=([^\n]+?)(?:;|$)'
        match = re.search(pattern, func_body)
        if match:
            reward_expr = match.group(1)
            logging.info(f"Found reward expression: {reward_expr}")
            break

    if not reward_expr:
        logging.warning("No total_reward or equivalent expression found.")
        return hp_ranges, initial_values

    # --- 5️⃣ 提取 reward 分量和权重 ---
    found_terms = set()

    # （1）数字在前： 0.5*forward_reward
    for weight, term in re.findall(r'([+-]?\d*\.?\d+(?:[eE][-+]?\d+)?)\*(\w+(?:_reward|_penalty|_bonus|_cost|_term))', reward_expr):
        try:
            val = float(weight)
            pname = f"{term}_weight"
            initial_values[pname] = val
            hp_ranges[pname] = (0.0, 2.0)
            found_terms.add(term)
            logging.info(f"Found weighted term: {val} * {term}")
        except ValueError:
            continue

    # （2）数字在后： forward_reward*0.5
    for term, weight in re.findall(r'(\w+(?:_reward|_penalty|_bonus|_cost|_term))\*([+-]?\d*\.?\d+(?:[eE][-+]?\d+)?)', reward_expr):
        try:
            val = float(weight)
            pname = f"{term}_weight"
            initial_values[pname] = val
            hp_ranges[pname] = (0.0, 2.0)
            found_terms.add(term)
            logging.info(f"Found weighted term: {term} * {val}")
        except ValueError:
            continue

    # （3）隐式权重： +forward_reward 或 -penalty
    for sign, term in re.findall(r'([+-]?)(\w+(?:_reward|_penalty|_bonus|_cost|_term))', reward_expr):
        if term in found_terms:
            continue
        sign_val = -1.0 if sign == '-' else 1.0
        pname = f"{term}_weight"
        initial_values[pname] = sign_val
        hp_ranges[pname] = (0.0, 2.0)
        found_terms.add(term)
        logging.info(f"Found implicit term: {sign_val} * {term}")

    # --- 6️⃣ 汇总 ---
    logging.info(f"=== Extracted {len(hp_ranges)} parameters ===")
    for pname, (low, high) in hp_ranges.items():
        logging.info(f"  {pname}: {initial_values[pname]} (range: [{low}, {high}])")

    return hp_ranges, initial_values




def load_reward_function_code(reward_file_path: str) -> str:
    """
    加载奖励函数代码
    
    Args:
        reward_file_path: 奖励函数文件路径
    
    Returns:
        奖励函数代码字符串
    """
    with open(reward_file_path, 'r') as f:
        code = f.read()
    return code




def create_parameterized_reward_code(original_code: str, params: Dict[str, float]) -> str:
    """
    将参数替换到奖励函数代码中
    
    Args:
        original_code: 原始奖励函数代码
        params: 参数字典
    
    Returns:
        参数化后的代码
    """
    code = original_code
    
    # 替换温度参数
    for param_name, param_value in params.items():
        if param_name.endswith('_temp'):
            # 匹配变量赋值
            pattern = rf'{param_name}\s*=\s*[\d\.]+'
            code = re.sub(pattern, f'{param_name} = {param_value:.6f}', code)
    
    # 替换权重参数
    for param_name, param_value in params.items():
        if param_name.endswith('_weight'):
            # 从 param_name 中提取奖励项名称
            reward_name = param_name[:-7]  # 去掉 '_weight'
            
            # 在 reward = (...) 中查找并替换
            # 匹配: 数值 * reward_name
            pattern = rf'[\d\.]+\s*\*\s*{reward_name}\b'
            replacement = f'{param_value:.6f} * {reward_name}'
            code = re.sub(pattern, replacement, code)
    
    return code
