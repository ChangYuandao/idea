import pickle
import csv
import os

def export_each_trajectory_to_csv(pkl_path, out_dir):
    # 创建输出目录
    os.makedirs(out_dir, exist_ok=True)

    # 读取 pkl
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    trajectories = data.get("trajectories", [])
    print(f"Loaded {len(trajectories)} trajectories.")

    # 逐条保存
    for i, traj in enumerate(trajectories):
        csv_path = os.path.join(out_dir, f"trajectory_{i}.csv")

        with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow(["key", "value"])

            # 把轨迹的所有字段都写进去
            for key, value in traj.items():
                writer.writerow([key, str(value)])

        print(f"Saved: {csv_path}")

    print("\nExport finished.")



# 示例使用
if __name__ == "__main__":
    export_each_trajectory_to_csv(
        "/home/changyuandao/changyuandao/paperProject/idea/method/outputs/ant/iter0_trajectories.pkl",
        "trajectories_export.csv"
    )
