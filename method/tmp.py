from tensorboard.backend.event_processing import event_accumulator
import random
# 你的 event 文件路径
log_dir = "/home/rtx4090/hnu/changyuandao/idea/IsaacGymEnvs/isaacgymenvs/outputs/train/2025-10-24_16-31-36/runs/ShadowHandGPT-2025-10-24_16-31-37/summaries/events.out.tfevents.1761294703.rtx4090-server"

# 加载事件文件
ea = event_accumulator.EventAccumulator(log_dir)
ea.Reload()

# 获取所有 scalar 类型的 tags
tags = ea.Tags().get('scalars', [])

for tag in tags:
    # 取出该 tag 的所有事件
    events = ea.Scalars(tag)
    values = [e.value for e in events]
    num_entries = len(values)

    # 随机抽取 5 个样本（如果少于 5 条就全取）
    sample_values = random.sample(values, min(5, num_entries))

    print(f"Tag: {tag}, Number of entries: {num_entries}, Sample: {sample_values}")