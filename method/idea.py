import hydra
import numpy as np 
import json
import logging 
import matplotlib.pyplot as plt
import os
from openai import OpenAI
import re
import subprocess
from pathlib import Path
import shutil
import time 

from utils.misc import * 
from utils.file_utils import load_tensorboard_logs
from utils.create_task import create_task
from utils.extract_task_code import *
from utils.preference_learning import IsaacGymPreferenceLearning
from utils.trajectory_collector import collect_trajectories_from_checkpoint
from utils.reward_updater import update_reward_function_with_params
from utils.reward_parser import extract_reward_parameters, load_reward_function_code

# 当前脚本所在目录
EUREKA_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# 回到 idea，再进入 isaacGymEnvs
ISAAC_ROOT_DIR = os.path.abspath(os.path.join(EUREKA_ROOT_DIR, "../IsaacGymEnvs/isaacgymenvs"))


@hydra.main(config_path="cfg", config_name="config", version_base="1.1")
def main(cfg):

    workspace_dir = Path.cwd()
    logging.info(f"Eureka Root Dir: {EUREKA_ROOT_DIR}")
    logging.info(f"IsaacGymEnvs Root Dir: {ISAAC_ROOT_DIR}")
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {EUREKA_ROOT_DIR}")

    # chatGPT = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))
    qwen = OpenAI(
		api_key=os.getenv("DASHSCOPE_API_KEY"),
		base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
	)
    task = cfg.env.task
    task_description = cfg.env.description
    suffix = cfg.suffix
    model = cfg.model
    
    logging.info(f"Using LLM: {model}")
    logging.info(f"Task: {task}")
    logging.info(f"Task Description: {task_description}")

    env_name = cfg.env.env_name.lower()
    
    env_parent = 'isaac' if f'{env_name}.py' in os.listdir(f'{EUREKA_ROOT_DIR}/envs/isaac') else 'dexterity'

    logging.info(f"Env Parent: {env_parent}")
    
    # 训练文件路径
    task_file = f'{EUREKA_ROOT_DIR}/envs/{env_parent}/{env_name}.py'
    
    # 观察文件路径
    task_obs_file = f'{EUREKA_ROOT_DIR}/envs/{env_parent}/{env_name}_obs.py'
    
    # 复制初始环境文件到当前工作目录，方便后续调用
    shutil.copy(task_obs_file, f"env_init_obs.py")
    
    # 训练文件的字符串格式
    task_code_string  = file_to_string(task_file)
    
    # 观察文件的字符串格式
    task_obs_code_string  = file_to_string(task_obs_file)
    
    # 放到了isaacgym文件夹下的tasks目录
    output_file = f"{ISAAC_ROOT_DIR}/tasks/{env_name}{suffix.lower()}.py"

    # 不同角色
    roles = ["EXPLORER", "CONSERVATOR", "INTEGRATOR"]

    # prompt 文件夹路径
    prompt_dir = f'{EUREKA_ROOT_DIR}/utils/prompts'
    
    # 系统的初始 prompt
    initial_system = file_to_string(f'{prompt_dir}/initial_system.txt')
    
    # 定义角色文件映射
    role_files = {
        "EXPLORER": f"{prompt_dir}/role_explorer.txt",
        "CONSERVATOR": f"{prompt_dir}/role_conservator.txt",
        "INTEGRATOR": f"{prompt_dir}/role_integrator.txt"
    }

    # 不同角色的 prompt
    role_explorer_prompt = file_to_string(role_files["EXPLORER"])   
    role_conservator_prompt = file_to_string(role_files["CONSERVATOR"])
    role_integrator_prompt = file_to_string(role_files["INTEGRATOR"])

    # 输出提示和反馈
    code_output_tip = file_to_string(f'{prompt_dir}/code_output_tip.txt')
    code_feedback = file_to_string(f'{prompt_dir}/code_feedback.txt')
    
    # 用户的初始 prompt 和反馈
    initial_user = file_to_string(f'{prompt_dir}/initial_user.txt')
    
    # 任务奖励函数的签名
    reward_signature = file_to_string(f'{prompt_dir}/reward_signature.txt')
    
    
    # 策略反馈和执行错误反馈
    policy_feedback = file_to_string(f'{prompt_dir}/policy_feedback.txt')
    execution_error_feedback = file_to_string(f'{prompt_dir}/execution_error_feedback.txt')
    
    # 构建不同角色的消息列表
    initial_explorer_system = initial_system.format( task_reward_signature_string=reward_signature, role_prompt=role_explorer_prompt ) + code_output_tip
    initial_conservator_system = initial_system.format( task_reward_signature_string=reward_signature, role_prompt=role_conservator_prompt ) + code_output_tip
    initial_integrator_system = initial_system.format( task_reward_signature_string=reward_signature, role_prompt=role_integrator_prompt ) + code_output_tip
    initial_user = initial_user.format(task_obs_code_string=task_obs_code_string, task_description=task_description)
    
    # 不同角色的消息列表
    explore_messages = [{"role": "system", "content": initial_explorer_system}, {"role": "user", "content": initial_user}]
    conservator_messages = [{"role": "system", "content": initial_conservator_system}, {"role": "user", "content": initial_user}]
    integrator_messages = [{"role": "system", "content": initial_integrator_system}, {"role": "user", "content": initial_user}]

    # 不同角色的消息映射
    role_messages = {
        "EXPLORER": explore_messages,
        "CONSERVATOR": conservator_messages,    
        "INTEGRATOR": integrator_messages,
    }
    
	# 将task的文件里的任务名字都加上GPT后缀
    task_code_string = task_code_string.replace(task, task+suffix)
    
    # 在isaacgymenvs的cfg文件夹下创建对应的task和train的yaml文件
    create_task(ISAAC_ROOT_DIR, cfg.env.task, cfg.env.env_name, suffix)

    DUMMY_FAILURE = -10000.

    # 角色信息存储
    role_information = {}
    
    # 初始化角色信息存储结构
    for role in roles:
        role_information[role] = {
            "code_runs": [],
            "rl_runs": [],
            "code_paths": [],
            "rl_paths": [],
            "contents": [],
            "successes": [],
            "reward_correlations": [],
            "responses": [],
        }
    

    # 全局最优结果存储
    global_best = {
        "role": None,
        "response_id": None,
        "success": -float("inf"),
        "reward_corr": DUMMY_FAILURE,
        "content": None,
        "code_path": None
    }
    

    
    # 迭代次数
    for iter in range(cfg.iteration):
        logging.info(f"=== Iteration {iter} ===")
        
        # 为每个角色生成样本
        for role in roles:
            logging.info(f"Role: {role}")
            role_information[role]["responses"] = []
            response_cur = None
            total_samples = 0
            total_token = 0
            total_completion_token = 0
            chunk_size = cfg.sample
            logging.info(f"Generating {cfg.sample} samples with {cfg.model}")

            while True:
                if total_samples >= cfg.sample:
                    break
                for attempt in range(1000):
                    try:
                        if cfg.model.startswith("gpt-"):
                            response_cur = chatGPT.chat.completion.create(
                            model=model,
                            messages=role_messages[role],
                            temperature=cfg.temperature,
                            n=chunk_size
                        )
                        elif cfg.model.startswith("qwen3-"):
                            response_cur = qwen.chat.completions.create(
                                model=model,
                                messages=role_messages[role],
                                temperature=cfg.temperature,
                                n=chunk_size
                            )
                        total_samples += chunk_size
                        # 跳出attempt的循环
                        break
                    except Exception as e:
                        if attempt >= 10:
                            chunk_size = max(int(chunk_size / 2), 1)
                            print("Current Chunk Size", chunk_size)
                        logging.info(f"Attempt {attempt+1} failed with error: {e}")
                        time.sleep(1)
                        
                if response_cur is None:
                    logging.info("Code terminated due to too many failed attempts!")
                    exit()

                role_information[role]["responses"].extend(response_cur.choices)
                prompt_tokens = response_cur.usage.prompt_tokens
                total_completion_token += response_cur.usage.completion_tokens
                total_token += response_cur.usage.total_tokens

            # Logging Token Information
            logging.info(f"Iteration {iter}: Prompt Tokens: {prompt_tokens}, Completion Tokens: {total_completion_token}, Total Tokens: {total_token}")
            

        for role in roles:
            role_information[role]["code_runs"] = []
            role_information[role]["rl_runs"] = []
            
            for response_id in range(cfg.sample):
                
                
                response_cur = role_information[role]["responses"][response_id].message.content
                logging.info(f"Iteration {iter}: role_{role} Processing Code Run {response_id}")

                # 提取生成的 python 文件中对应的函数代码字符串
                patterns = [
                    r'```python(.*?)```',
                    r'```(.*?)```',
                    r'"""(.*?)"""',
                    r'""(.*?)""',
                    r'"(.*?)"',
                ]
                for pattern in patterns:
                    code_string = re.search(pattern, response_cur, re.DOTALL)
                    if code_string is not None:
                        code_string = code_string.group(1).strip()
                        break
                code_string = response_cur if not code_string else code_string

                # 找到第一个以 def 开头的函数定义，丢弃前面的内容
                lines = code_string.split("\n")
                for i, line in enumerate(lines):
                    if line.strip().startswith("def "):
                        code_string = "\n".join(lines[i:])
                        
                # 将函数签名添加到环境代码中
                try:
                    gpt_reward_signature, input_lst = get_function_signature(code_string)
                except Exception as e:
                    logging.info(f"Iteration {iter}: Role {role} - Code Run {response_id} cannot parse function signature!")
                    continue

                role_information[role]["code_runs"].append(code_string)
                reward_signature = [
                    f"self.rew_buf[:], self.rew_dict = {gpt_reward_signature}",
                    f"self.extras['gpt_reward'] = self.rew_buf.mean()",
                    f"for rew_state in self.rew_dict: self.extras[rew_state] = self.rew_dict[rew_state].mean()",
                ]
                indent = " " * 8
                reward_signature = "\n".join([indent + line for line in reward_signature])

                # task_code_string 是加上了GPT后缀的相关任务的训练文件
                if "def compute_reward(self)" in task_code_string:
                    task_code_string_iter = task_code_string.replace("def compute_reward(self):", "def compute_reward(self):\n" + reward_signature)
                elif "def compute_reward(self, actions)" in task_code_string:
                    task_code_string_iter = task_code_string.replace("def compute_reward(self, actions):", "def compute_reward(self, actions):\n" + reward_signature)
                else:
                    raise NotImplementedError

                # 写入到isaacgymenvs下的task目录里
                with open(output_file, 'w') as file:
                    file.writelines(task_code_string_iter + '\n')
                    file.writelines("from typing import Tuple, Dict" + '\n')
                    file.writelines("import math" + '\n')
                    file.writelines("import torch" + '\n')
                    file.writelines("from torch import Tensor" + '\n')
                    if "@torch.jit.script" not in code_string:
                        code_string = "@torch.jit.script\n" + code_string
                    file.writelines(code_string + '\n')
                    
                # 写入到hydra的工作目录下，方便后续调用
                with open(f"env_iter{iter}_role_{role}_response{response_id}_rewardonly.py", 'w') as file:
                    file.writelines("from typing import Tuple, Dict" + '\n')
                    file.writelines("import math" + '\n')
                    file.writelines("import torch" + '\n')
                    file.writelines("from torch import Tensor" + '\n')
                    file.writelines(code_string + '\n')

                # 复制一份到当前目录，方便后续查看
                shutil.copy(output_file, f"env_iter{iter}_role_{role}_response{response_id}.py")

                
                # 执行代码运行训练
                rl_filepath = f"env_iter{iter}_role_{role}_response{response_id}.txt"
                role_output_dir = f"./Iteration_{iter}/{role}_{response_id}"
                with open(rl_filepath, 'w') as f:
                    process = subprocess.Popen([
                        'python', '-u', f'{ISAAC_ROOT_DIR}/train.py',
                        f'hydra.run.dir={role_output_dir}',
                        f'hydra.output_subdir=null',
                        f'task={task}{suffix}',
                        f'wandb_activate={cfg.use_wandb}',
                        f'wandb_entity={cfg.wandb_username}',
                        f'wandb_project={cfg.wandb_project}',
                        f'headless={not cfg.capture_video}',
                        f'capture_video={cfg.capture_video}',
                        'force_render=False',
                        f'max_iterations={cfg.max_iterations}'
                    ], stdout=f, stderr=f)
                block_until_training(rl_filepath, log_status=True, iter_num=iter, response_id=response_id)
                role_information[role]["rl_runs"].append(process)
        
        # 处理每个角色的 RL 结果
        for role in roles:
            logging.info(f"Role: {role} - Processing RL Results")
            role_information[role]["contents"] = []
            role_information[role]["successes"] = []
            role_information[role]["reward_correlations"] = []
            role_information[role]["code_paths"] = []

            for response_id, (code_run, rl_run) in enumerate(zip(role_information[role]["code_runs"], role_information[role]["rl_runs"])):
                # 等待 RL 训练完成
                rl_run.communicate()
                rl_filepath = f"env_iter{iter}_role_{role}_response{response_id}.txt"
                role_information[role]["code_paths"].append(f"env_iter{iter}_role_{role}_response{response_id}.py")
                try:
                    with open(rl_filepath, 'r') as f:
                        stdout_str = f.read() 
                except: 
                    content = execution_error_feedback.format(traceback_msg="Code Run cannot be executed due to function signature error! Please re-write an entirely new reward function!")
                    content += code_output_tip
                    role_information[role]["contents"].append(content) 
                    role_information[role]["successes"].append(DUMMY_FAILURE)
                    role_information[role]["reward_correlations"].append(DUMMY_FAILURE)
                    continue

                content = ''
                traceback_msg = filter_traceback(stdout_str)

                if traceback_msg == '':
    
                    lines = stdout_str.split('\n')
                    for i, line in enumerate(lines):
                        if line.startswith('Tensorboard Directory:'):
                            break 
                    tensorboard_logdir = line.split(':')[-1].strip() 
                    tensorboard_logs = load_tensorboard_logs(tensorboard_logdir)
                    max_iterations = np.array(tensorboard_logs['gt_reward']).shape[0]
                    epoch_freq = max(int(max_iterations // 10), 1)
                    
                    content += policy_feedback.format(epoch_freq=epoch_freq)
                    
                    # 记录成功次数和奖励相关性
                    if "gt_reward" in tensorboard_logs and "gpt_reward" in tensorboard_logs:
                        gt_reward = np.array(tensorboard_logs["gt_reward"])
                        gpt_reward = np.array(tensorboard_logs["gpt_reward"])
                        reward_correlation = np.corrcoef(gt_reward, gpt_reward)[0, 1]
                        role_information[role]["reward_correlations"].append(reward_correlation)

                    # 记录各项指标
                    for metric in tensorboard_logs:
                        if "/" not in metric:
                            metric_cur = ['{:.2f}'.format(x) for x in tensorboard_logs[metric][::epoch_freq]]
                            metric_cur_max = max(tensorboard_logs[metric])
                            metric_cur_mean = sum(tensorboard_logs[metric]) / len(tensorboard_logs[metric])
                            if "consecutive_successes" == metric:
                                role_information[role]["successes"].append(metric_cur_mean)
                            metric_cur_min = min(tensorboard_logs[metric])
                            if metric != "gt_reward" and metric != "gpt_reward":
                                if metric != "consecutive_successes":
                                    metric_name = metric 
                                else:
                                    metric_name = "task_score"
                                content += f"{metric_name}: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"                    
                            else:
                                # Provide ground-truth score when success rate not applicable
                                if "consecutive_successes" not in tensorboard_logs:
                                    content += f"ground-truth score: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"                    
                    content += code_feedback  
                else:
                    # Otherwise, provide execution traceback error feedback
                    role_information[role]["successes"].append(DUMMY_FAILURE)
                    role_information[role]["reward_correlations"].append(DUMMY_FAILURE)
                    content += execution_error_feedback.format(traceback_msg=traceback_msg)

                content += code_output_tip
                role_information[role]["contents"].append(content)

        logging.info("Selecting the best reward function across all roles...")
        
        # === 在所有角色中选择最优结果 ===
        for role, info in role_information.items():
            successes = np.array(info["successes"])
            if len(successes) == 0:
                continue

            best_idx = int(np.argmax(successes))
            best_success = successes[best_idx]
            best_corr = info["reward_correlations"][best_idx]
            best_content = info["contents"][best_idx]
            best_code_path = info["code_paths"][best_idx]
            best_response = info["responses"][best_idx]

            logging.info(f"[{role}] Best sample {best_idx}: Success={best_success}, Corr={best_corr}")

            if best_success > global_best["success"]:
                global_best.update({
                    "role": role,
                    "response_id": best_idx,
                    "success": best_success,
                    "reward_corr": best_corr,
                    "content": best_content,
                    "code_path": best_code_path,
                    "response": best_response
                })
        
        # === 输出全局最优结果 ===
        if global_best["role"] is not None:
            logging.info(
                f"Iteration {iter} - "
                f"🏆 Global Best Reward Function from Role [{global_best['role']}] - "
                f"Response ID: {global_best['response_id']}, "
            )
            
            # ============ 动态提取参数并初始化偏好学习器 ============
            
            # 1. 获取最佳奖励函数文件路径
            best_reward_file = workspace_dir / f"env_iter{iter}_role_{global_best['role']}_response{global_best['response_id']}_rewardonly.py"
            
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
                        beta=1.0,
                        task_name=task
                    )
                    
                    # 开始偏好学习
                    logging.info(f"Starting Preference Learning for Iteration {iter}")
                    
                    # 4. 构建checkpoint路径
                    checkpoint_path = Path(workspace_dir) / \
                                    f"Iteration_{iter}" / \
                                    f"{global_best['role']}_{global_best['response_id']}" / \
                                    "runs" / f"{task.lower()}{suffix.lower()}" / "nn" / f"{task}{suffix}.pth"

                    logging.info(f"Checkpoint path: {checkpoint_path}")

                    if checkpoint_path.exists():
                        try:
                            # 5. 收集轨迹
                            logging.info("Step 1: Collecting trajectories...")
                            trajectory_file = collect_trajectories_from_checkpoint(
                                isaac_root_dir=ISAAC_ROOT_DIR,
                                checkpoint_path=str(checkpoint_path),
                                task_name=task,
                                num_trajectories=cfg.get('num_trajectories', 10),
                                output_dir=workspace_dir,
                                save_filename=f"iter{iter}_trajectories.pkl"
                            )
                            
                            # 6. 加载轨迹
                            logging.info("Step 2: Loading trajectories...")
                            trajectories = preference_learner.load_trajectories(trajectory_file)
                            
                            # 7. 生成偏好对
                            logging.info("Step 3: Generating preference pairs...")
                            
                            # 构建评估函数目录
                            evaluate_dir = Path(EUREKA_ROOT_DIR) / "utils" / "prompts" / "evaluate_function" / task
                            
                            if evaluate_dir.exists():
                                # 使用评估函数生成偏好对
                                logging.info("Using evaluation functions to generate preferences...")
                                preferences = preference_learner.generate_preference_buffer(
                                    trajectories,
                                    str(evaluate_dir),
                                    min_consecutive=5
                                )
                            
                            if len(preferences) == 0:
                                logging.warning("⚠️ No preferences generated, skipping preference learning")
                                continue
                            
                            # 8. 更新奖励函数参数
                            logging.info("Step 4: Updating reward function parameters...")
                            updated_params = preference_learner.update_reward_parameters(
                                trajectories,
                                preferences,
                            )
                            
                            # 9. 更新奖励函数代码
                            logging.info("Step 5: Updating reward function code...")
                            updated_code_path = update_reward_function_with_params(
                                original_code_path=str(best_reward_file),
                                updated_params=updated_params,
                                output_path=workspace_dir / f"env_iter{iter}_updated_reward.py"
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
            
            logging.info(f"\n{'='*80}\n")



if __name__ == "__main__":
    main()