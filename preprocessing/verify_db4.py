import glob
import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

DATA_DIR = os.environ.get("EMGFLOW_DB4_TARGET_DIR", "DB4_npy")
META_PATH = os.path.join(DATA_DIR, "emg_db4_dataset", "meta.json")

DEFAULT_STANDARD_CONFIG_GLOBAL = {
    "E1": {"count": 12, "min": 1, "max": 12},
    "E2": {"count": 17, "min": 13, "max": 29},
    "E3": {"count": 23, "min": 30, "max": 52},
}

DEFAULT_STANDARD_CONFIG_LOCAL = {
    "E1": {"count": 12, "min": 1, "max": 12},
    "E2": {"count": 17, "min": 1, "max": 17},
    "E3": {"count": 23, "min": 1, "max": 23},
}


def load_standard_config():
    if not os.path.exists(META_PATH):
        return DEFAULT_STANDARD_CONFIG_GLOBAL, "global"

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    label_mode = meta.get("label_mode", "global")
    exercise_config = meta.get("exercise_config", {})
    if exercise_config:
        standard = {}
        for exercise_id, cfg in exercise_config.items():
            saved_min, saved_max = cfg["saved_label_range"]
            standard[exercise_id] = {
                "count": cfg["count"],
                "min": saved_min,
                "max": saved_max,
            }
        return standard, label_mode

    if label_mode == "local":
        return DEFAULT_STANDARD_CONFIG_LOCAL, label_mode
    return DEFAULT_STANDARD_CONFIG_GLOBAL, label_mode


def verify_final_standard():
    standard_config, label_mode = load_standard_config()

    pattern_flat = os.path.join(DATA_DIR, "*_300ms.npy")
    pattern_recursive = os.path.join(DATA_DIR, "**", "*.npy")
    files = sorted(glob.glob(pattern_flat))
    if not files:
        all_npy = glob.glob(pattern_recursive)
        files = sorted(
            f for f in all_npy
            if not os.path.basename(f).startswith(".cache")
            and os.path.basename(f) in ("E1.npy", "E2.npy", "E3.npy")
        )

    if not files:
        print(f"❌ 未找到任何 .npy 文件，请检查路径: {DATA_DIR}")
        return

    print("🔍 开始验证 DB4 数据集...")
    print(f"🧭 标签模式: {label_mode}")
    print("=" * 95)
    print(f"{'文件':<22} | {'完整性(6 Reps)':<15} | {'Label范围':<12} | {'去噪长度':<10} | {'最终判定'}")
    print("-" * 95)

    passed_count = 0

    for file_path in tqdm(files, desc="Verifying"):
        try:
            display_name = os.path.relpath(file_path, DATA_DIR)
        except ValueError:
            display_name = os.path.basename(file_path)
        if len(display_name) > 22:
            display_name = "..." + display_name[-19:]

        file_name = os.path.basename(file_path)

        try:
            exercise_id = None
            if "_E1_" in file_name or file_name == "E1.npy":
                exercise_id = "E1"
            elif "_E2_" in file_name or file_name == "E2.npy":
                exercise_id = "E2"
            elif "_E3_" in file_name or file_name == "E3.npy":
                exercise_id = "E3"

            if not exercise_id:
                print(f"{display_name:<22} | ⚠️ 无法识别 Exercise 类型，跳过")
                continue

            data = np.load(file_path, allow_pickle=True).item()
            labels = data["label"]
            reps = data.get("rep", data.get("repetition"))
            emgs = data["emg"]
            if reps is None:
                raise KeyError("数据中既无 'rep' 也无 'repetition'")

            df = pd.DataFrame({"Label": labels, "Rep": reps})
            check_rep = df.groupby("Label")["Rep"].nunique()
            is_integrity_ok = (check_rep == 6).all()

            unique_labels = sorted(df["Label"].unique())
            actual_count = len(unique_labels)
            actual_min = min(unique_labels)
            actual_max = max(unique_labels)

            target = standard_config[exercise_id]
            is_range_ok = (
                actual_count == target["count"] and actual_min == target["min"] and actual_max == target["max"]
            )

            label_str = f"{actual_min}-{actual_max} ({actual_count})"

            avg_len = np.mean([e.shape[0] for e in emgs])
            is_trimmed_ok = 100 < avg_len < 13000

            status_integrity = "✅" if is_integrity_ok else "❌"
            status_range = "✅" if is_range_ok else f"❌(应为{target['min']}-{target['max']})"
            status_trim = f"{int(avg_len)}" if is_trimmed_ok else "❌异常"

            if is_integrity_ok and is_range_ok and is_trimmed_ok:
                final_verdict = "✅ 通过 (Normal)"
                passed_count += 1
            else:
                final_verdict = "❌ 失败"

            print(f"{display_name:<22} | {status_integrity:<15} | {label_str:<12} | {status_trim:<10} | {final_verdict}")

        except Exception as e:
            print(f"❌ {display_name} 读取出错: {e}")

    print("=" * 95)
    print(f"📊 验证结果: {passed_count}/{len(files)} 文件符合标准。")
    if passed_count == len(files):
        print("🎉 完美！所有文件的 Trial数、类别数、Label范围均完全正确。")


if __name__ == "__main__":
    verify_final_standard()
