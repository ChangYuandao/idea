
import numpy as np
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator




def load_tensorboard_logs(path):
    data = defaultdict(list)
    event_acc = EventAccumulator(path)
    event_acc.Reload()  # Load all data written so far

    for tag in event_acc.Tags()["scalars"]:
        events = event_acc.Scalars(tag)
        for event in events:
            data[tag].append(event.value)
    
    return data

content = ''
policy_feedback = ''
reward_correlations = []
successes = []


rl_filepath = '/home/rtx4090/hnu/changyuandao/idea/method/outputs/test/2025-10-23_10-47-20/env_iter0_response2.txt'


with open(rl_filepath, 'r') as f:
    stdout_str = f.read()


lines = stdout_str.split('\n')
for i, line in enumerate(lines):
    if line.startswith('Tensorboard Directory:'):
        break 
tensorboard_logdir = line.split(':')[-1].strip() 
tensorboard_logs = load_tensorboard_logs(tensorboard_logdir)
max_iterations = np.array(tensorboard_logs['gt_reward/iter']).shape[0]
print("max_iterations:", max_iterations)
epoch_freq = max(int(max_iterations // 10), 1)

content += policy_feedback.format(epoch_freq=epoch_freq)

# Compute Correlation between Human-Engineered and GPT Rewards
if "gt_reward" in tensorboard_logs and "gpt_reward" in tensorboard_logs:
    gt_reward = np.array(tensorboard_logs["gt_reward"])
    gpt_reward = np.array(tensorboard_logs["gpt_reward"])
    reward_correlation = np.corrcoef(gt_reward, gpt_reward)[0, 1]
    reward_correlations.append(reward_correlation)

# Add reward components log to the feedback
for metric in tensorboard_logs:
    print(f"metric: {metric}")
    if "/" not in metric:
        metric_cur = ['{:.2f}'.format(x) for x in tensorboard_logs[metric][::epoch_freq]]
        print(f"No / Metric: {metric}, Values: {metric_cur}")
        print(f"tensorboard_logs[metric]: {tensorboard_logs[metric]}")
        metric_cur_max = max(tensorboard_logs[metric])
        metric_cur_mean = sum(tensorboard_logs[metric]) / len(tensorboard_logs[metric])
        if "consecutive_successes" == metric:
            successes.append(metric_cur_max)
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