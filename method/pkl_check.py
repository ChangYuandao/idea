import pickle
import json
import numpy as np

def to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, bytes):
        return obj.decode(errors="ignore")
    elif hasattr(obj, "__dict__"):
        return obj.__dict__
    else:
        return str(obj)

def pkl_to_json(pkl_path, json_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, default=to_serializable, ensure_ascii=False, indent=4)

# 使用方法
pkl_to_json("/home/changyuandao/changyuandao/paperProject/idea/method/outputs/shadow_hand/iter0_trajectories.pkl", "/home/changyuandao/changyuandao/paperProject/idea/shadowhand.json")