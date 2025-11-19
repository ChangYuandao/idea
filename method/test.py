from tensorboard.backend.event_processing import event_accumulator

# 指定事件文件路径
event_file = "/home/changyuandao/changyuandao/paperProject/idea/IsaacGymEnvs/isaacgymenvs/outputs/train/2025-11-19_19-32-15/runs/antgpt/summaries/events.out.tfevents.1763551937.changyuandao-020318"  # 替换成你的事件文件

# 加载事件文件
ea = event_accumulator.EventAccumulator(event_file)
ea.Reload()

# 获取所有 scalar keys
scalar_keys = ea.Tags().get('scalars', [])


# 统计每个 key 的数量和平均值
for key in scalar_keys:
    if key == "consecutive_successes":
        events = ea.Scalars(key)
        values = [e.value for e in events]
        count = len(values)
        max_value = max(values) if count > 0 else float('nan')
        mean_value = sum(values) / count if count > 0 else float('nan')
        print(f"Key: {key}, Count: {count}, Mean: {mean_value:.6f}, Max: {max_value:.6f}")
