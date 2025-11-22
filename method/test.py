from tensorboard.backend.event_processing import event_accumulator

# Before
event_file = "/home/changyuandao/changyuandao/paperProject/idea/method/outputs/shadow_hand/Iteration_0/CONSERVATOR_0/runs/shadowhandgpt/summaries/events.out.tfevents.1763715947.changyuandao-020318"  # 替换成
ea = event_accumulator.EventAccumulator(event_file)
ea.Reload()
scalar_keys = ea.Tags().get('scalars', [])
for key in scalar_keys:
    if key == "consecutive_successes":
        events = ea.Scalars(key)
        values = [e.value for e in events]
        count = len(values)
        max_value = max(values) if count > 0 else float('nan')
        mean_value = sum(values) / count if count > 0 else float('nan')
        print(f"Before updated Key: {key}, Count: {count}, Mean: {mean_value:.6f}, Max: {max_value:.6f}")

#After
event_file = "/home/changyuandao/changyuandao/paperProject/idea/IsaacGymEnvs/isaacgymenvs/outputs/train/2025-11-22_13-59-43/runs/shadowhandgpt/summaries/events.out.tfevents.1763791186.changyuandao-020318"  # 替换成
ea = event_accumulator.EventAccumulator(event_file)
ea.Reload()
for key in scalar_keys:
    if key == "consecutive_successes":
        events = ea.Scalars(key)
        values = [e.value for e in events]
        count = len(values)
        max_value = max(values) if count > 0 else float('nan')
        mean_value = sum(values) / count if count > 0 else float('nan')
        print(f"After updated Key: {key}, Count: {count}, Mean: {mean_value:.6f}, Max: {max_value:.6f}")