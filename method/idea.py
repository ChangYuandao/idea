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

# 当前脚本所在目录（稳！）
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
		# 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
		api_key=os.getenv("DASHSCOPE_API_KEY"),
		base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
	)
    task = cfg.env.task
    task_description = cfg.env.description
    suffix = cfg.suffix
    model = cfg.model
    
    logging.info(f"Using LLM: {model}")
    logging.info("Task: " + task)
    logging.info("Task description: " + task_description)

    env_name = cfg.env.env_name.lower()
    env_parent = 'isaac' if f'{env_name}.py' in os.listdir(f'{EUREKA_ROOT_DIR}/envs/isaac') else 'dexterity'
    
    logging.info("Env parent:" + env_parent)
    
    task_file = f'{EUREKA_ROOT_DIR}/envs/{env_parent}/{env_name}.py'
    task_obs_file = f'{EUREKA_ROOT_DIR}/envs/{env_parent}/{env_name}_obs.py'
    shutil.copy(task_obs_file, f"env_init_obs.py")
    
    task_code_string  = file_to_string(task_file)
    task_obs_code_string  = file_to_string(task_obs_file)
    # 放到了isaacgym文件夹下的tasks目录
    output_file = f"{ISAAC_ROOT_DIR}/tasks/{env_name}{suffix.lower()}.py"

    # 不同角色
    roles = ["EXPLORER", "CONSERVATOR", "INTEGRATOR"]
    # Loading all text prompts
    prompt_dir = f'{EUREKA_ROOT_DIR}/utils/prompts'
    initial_system = file_to_string(f'{prompt_dir}/initial_system.txt')
    # 定义角色文件映射
    role_files = {
        "EXPLORER": f"{prompt_dir}/role_explorer.txt",
        "CONSERVATOR": f"{prompt_dir}/role_conservator.txt",
        "INTEGRATOR": f"{prompt_dir}/role_integrator.txt"
    }

    role_explorer_prompt = file_to_string(role_files["EXPLORER"])   
    role_conservator_prompt = file_to_string(role_files["CONSERVATOR"])
    role_integrator_prompt = file_to_string(role_files["INTEGRATOR"])



    code_output_tip = file_to_string(f'{prompt_dir}/code_output_tip.txt')
    code_feedback = file_to_string(f'{prompt_dir}/code_feedback.txt')
    initial_user = file_to_string(f'{prompt_dir}/initial_user.txt')
    reward_signature = file_to_string(f'{prompt_dir}/reward_signature.txt')
    policy_feedback = file_to_string(f'{prompt_dir}/policy_feedback.txt')
    execution_error_feedback = file_to_string(f'{prompt_dir}/execution_error_feedback.txt')
    
    initial_explorer_system = initial_system.format( task_reward_signature_string=reward_signature, role_prompt=role_explorer_prompt ) + code_output_tip
    initial_conservator_system = initial_system.format( task_reward_signature_string=reward_signature, role_prompt=role_conservator_prompt ) + code_output_tip
    initial_integrator_system = initial_system.format( task_reward_signature_string=reward_signature, role_prompt=role_integrator_prompt ) + code_output_tip
    initial_user = initial_user.format(task_obs_code_string=task_obs_code_string, task_description=task_description)
    
    explore_messages = [{"role": "system", "content": initial_explorer_system}, {"role": "user", "content": initial_user}]
    conservator_messages = [{"role": "system", "content": initial_conservator_system}, {"role": "user", "content": initial_user}]
    integrator_messages = [{"role": "system", "content": initial_integrator_system}, {"role": "user", "content": initial_user}]

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
    max_successes = []
    max_successes_reward_correlation = []
    execute_rates = []
    best_code_paths = []
    max_success_overall = DUMMY_FAILURE
    max_success_reward_correlation_overall = DUMMY_FAILURE
    max_reward_code_path = None 
    
    role_information = {}
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
    global_best = {
        "role": None,
        "response_id": None,
        "success": -float("inf"),
        "reward_corr": DUMMY_FAILURE,
        "content": None,
        "code_path": None
    }

    for iter in range(cfg.iteration):
        logging.info(f"=== Iteration {iter} ===")
        
        for role in roles:
            logging.info(f"Role: {role}")
            role_information[role]["responses"] = []
            response_cur = None
            total_samples = 0
            total_token = 0
            total_completion_token = 0
            chunk_size = cfg.sample if "qwen3-coder-plus" in model else 4
    
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

            if cfg.sample == 1:
                logging.info(f"Iteration {iter}: GPT Output:\n " + role_information[role]["responses"][0].message.content + "\n")

            # Logging Token Information
            logging.info(f"Iteration {iter}: Prompt Tokens: {prompt_tokens}, Completion Tokens: {total_completion_token}, Total Tokens: {total_token}")
            

        for role in roles:
            role_information[role]["code_runs"] = []
            role_information[role]["rl_runs"] = []
            
            for response_id in range(cfg.sample):
                response_cur = role_information[role]["responses"][response_id].message.content
                logging.info(f"Iteration {iter}: role_{role} Processing Code Run {response_id}")

                # Regex patterns to extract python code enclosed in GPT response
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

                # Remove unnecessary imports
                # 找到第一个以 def 开头的函数定义，丢弃前面的内容1
                lines = code_string.split("\n")
                for i, line in enumerate(lines):
                    if line.strip().startswith("def "):
                        code_string = "\n".join(lines[i:])
                        
                # Add the Eureka Reward Signature to the environment code
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

                # Save the new environment code when the output contains valid code string!
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
                    file.writelines(code_string + '\n')

                # Copy the generated environment code to hydra output directory for bookkeeping
                shutil.copy(output_file, f"env_iter{iter}_role_{role}_response{response_id}.py")

                # Find the freest GPU to run GPU-accelerated RL
                # set_freest_gpus(2)
                
                # Execute the python file with flags
                rl_filepath = f"env_iter{iter}_role_{role}_response{response_id}.txt"
                with open(rl_filepath, 'w') as f:
                    process = subprocess.Popen(['python', '-u', f'{ISAAC_ROOT_DIR}/train.py',  
                                                'hydra/output=subprocess',
                                                f'task={task}{suffix}', f'wandb_activate={cfg.use_wandb}',
                                                f'wandb_entity={cfg.wandb_username}', f'wandb_project={cfg.wandb_project}',
                                                f'headless={not cfg.capture_video}', f'capture_video={cfg.capture_video}', 'force_render=False',
                                                f'max_iterations={cfg.max_iterations}'],
                                                stdout=f, stderr=f)
                block_until_training(rl_filepath, log_status=True, iter_num=iter, response_id=response_id)
                role_information[role]["rl_runs"].append(process)
        
        exec_success = False 
        for role in roles:
            logging.info(f"Role: {role} - Processing RL Results")
            role_information[role]["contents"] = []
            role_information[role]["successes"] = []
            role_information[role]["reward_correlations"] = []
            role_information[role]["code_paths"] = []

            for response_id, (code_run, rl_run) in enumerate(zip(role_information[role]["code_runs"], role_information[role]["rl_runs"])):
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
                    # If RL execution has no error, provide policy statistics feedback
                    exec_success = True
                    lines = stdout_str.split('\n')
                    for i, line in enumerate(lines):
                        if line.startswith('Tensorboard Directory:'):
                            break 
                    tensorboard_logdir = line.split(':')[-1].strip() 
                    tensorboard_logs = load_tensorboard_logs(tensorboard_logdir)
                    max_iterations = np.array(tensorboard_logs['gt_reward']).shape[0]
                    epoch_freq = max(int(max_iterations // 10), 1)
                    
                    content += policy_feedback.format(epoch_freq=epoch_freq)
                    
                    # Compute Correlation between Human-Engineered and GPT Rewards
                    if "gt_reward" in tensorboard_logs and "gpt_reward" in tensorboard_logs:
                        gt_reward = np.array(tensorboard_logs["gt_reward"])
                        gpt_reward = np.array(tensorboard_logs["gpt_reward"])
                        reward_correlation = np.corrcoef(gt_reward, gpt_reward)[0, 1]
                        role_information[role]["reward_correlations"].append(reward_correlation)

                    # Add reward components log to the feedback
                    for metric in tensorboard_logs:
                        if "/" not in metric:
                            metric_cur = ['{:.2f}'.format(x) for x in tensorboard_logs[metric][::epoch_freq]]
                            metric_cur_max = max(tensorboard_logs[metric])
                            metric_cur_mean = sum(tensorboard_logs[metric]) / len(tensorboard_logs[metric])
                            if "consecutive_successes" == metric:
                                role_information[role]["successes"].append(metric_cur_max)
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
                f"🏆 Global Best Reward Function from Role [{global_best['role']}] "
                f"(Response {global_best['response_id']}) — Success={global_best['success']:.3f}, Corr={global_best['reward_corr']:.3f}"
            )

            best_role = global_best["role"]
            best_idx = global_best["response_id"]

            best_response_content = global_best["response"].message.content
            best_user_feedback = global_best["content"]

            # 更新 messages
            for role, messages in role_messages.items():
                best_response_content = role_information[role]["responses"][best_idx].message.content
                best_user_feedback = role_information[role]["contents"][best_idx]

                if len(messages) == 2:
                    messages += [
                        {"role": "assistant", "content": best_response_content},
                        {"role": "user", "content": best_user_feedback}
                    ]
                else:
                    messages[-2] = {"role": "assistant", "content": best_response_content}
                    messages[-1] = {"role": "user", "content": best_user_feedback}

            with open("messages_global_best.json", "w") as f:
                json.dump(messages, f, indent=4)

        else:
            logging.warning("❌ No valid reward function found across all roles!")
            
    if global_best["code_path"] is None or global_best["success"] == DUMMY_FAILURE:
        logging.info("❌ All roles failed to produce a valid reward function, aborting evaluation...")
        logging.info("Please check the env_iter*_response*.txt logs for detailed errors.")
        exit()

    best_role = global_best["role"]
    best_code_path = global_best["code_path"]
    best_success = global_best["success"]
    best_corr = global_best["reward_corr"]

    logging.info(f"🏁 Task: {task}")
    logging.info(f"🏆 Global Best Role: {best_role}")
    logging.info(f"✅ Max Training Success: {best_success:.3f}")
    logging.info(f"📈 Reward Correlation: {best_corr:.3f}")
    logging.info(f"📂 Best Reward Code Path: {best_code_path}")
    logging.info(f"🔁 Evaluating the best reward function {cfg.num_eval} times...")
    # shutil.copy(max_reward_code_path, output_file)
    
    # eval_runs = []
    # for i in range(cfg.num_eval):
    #     set_freest_gpus(2)
        
    #     # Execute the python file with flags
    #     rl_filepath = f"reward_code_eval{i}.txt"
    #     with open(rl_filepath, 'w') as f:
    #         process = subprocess.Popen(['python', '-u', f'{ISAAC_ROOT_DIR}/train.py',  
    #                                     'hydra/output=subprocess',
    #                                     f'task={task}{suffix}', f'wandb_activate={cfg.use_wandb}',
    #                                     f'wandb_entity={cfg.wandb_username}', f'wandb_project={cfg.wandb_project}',
    #                                     f'headless={not cfg.capture_video}', f'capture_video={cfg.capture_video}', 'force_render=False', f'seed={i}', f'max_iterations={cfg.max_iterations}'
    #                                     ],
    #                                     stdout=f, stderr=f)

    #     block_until_training(rl_filepath)
    #     eval_runs.append(process)

    # reward_code_final_successes = []
    # reward_code_correlations_final = []
    # for i, rl_run in enumerate(eval_runs):
    #     rl_run.communicate()
    #     rl_filepath = f"reward_code_eval{i}.txt"
    #     with open(rl_filepath, 'r') as f:
    #         stdout_str = f.read() 
    #     lines = stdout_str.split('\n')
    #     for i, line in enumerate(lines):
    #         if line.startswith('Tensorboard Directory:'):
    #             break 
    #     tensorboard_logdir = line.split(':')[-1].strip() 
    #     tensorboard_logs = load_tensorboard_logs(tensorboard_logdir)
    #     max_success = max(tensorboard_logs['consecutive_successes'])
    #     reward_code_final_successes.append(max_success)

    #     if "gt_reward" in tensorboard_logs and "gpt_reward" in tensorboard_logs:
    #         gt_reward = np.array(tensorboard_logs["gt_reward"])
    #         gpt_reward = np.array(tensorboard_logs["gpt_reward"])
    #         reward_correlation = np.corrcoef(gt_reward, gpt_reward)[0, 1]
    #         reward_code_correlations_final.append(reward_correlation)

    # logging.info(f"Final Success Mean: {np.mean(reward_code_final_successes)}, Std: {np.std(reward_code_final_successes)}, Raw: {reward_code_final_successes}")
    # logging.info(f"Final Correlation Mean: {np.mean(reward_code_correlations_final)}, Std: {np.std(reward_code_correlations_final)}, Raw: {reward_code_correlations_final}")
    # np.savez('final_eval.npz', reward_code_final_successes=reward_code_final_successes, reward_code_correlations_final=reward_code_correlations_final)


if __name__ == "__main__":
    main()