import glob
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

# ================= ⚙️ 数据集标准定义 =================
DATA_DIR = os.environ.get("EMGFLOW_DB7_TARGET_DIR", "DB7_npy")

# DB7 只有 E1 和 E2（不像 DB2 有 E1/E2/E3）
STANDARD_CONFIG = {
    "E1": {
        "count": 17,
        "min": 1,
        "max": 17
    },
    "E2": {
        "count": 23,
        "min": 18,
        "max": 40
    }
}
# ========================================================


def verify_final_standard():
    # 找到所有生成的 .npy 文件（支持两种布局）
    # 布局A: DATA_DIR/S10_E2_300ms.npy
    # 布局B: DATA_DIR/Subject01/E1.npy, E2.npy
    pattern_flat = os.path.join(DATA_DIR, "*_300ms.npy")
    pattern_recursive = os.path.join(DATA_DIR, "**", "*.npy")
    files = sorted(glob.glob(pattern_flat))
    if not files:
        all_npy = glob.glob(pattern_recursive)
        # 排除 .cache 开头的文件，只保留 E1.npy / E2.npy / E3.npy
        files = sorted(
            f for f in all_npy
            if not os.path.basename(f).startswith(".cache")
            and os.path.basename(f) in ("E1.npy", "E2.npy", "E3.npy")
        )

    if not files:
        print(f"❌ 未找到任何 .npy 文件，请检查路径: {DATA_DIR}")
        return

    print(f"🔍 开始验证 DB7 数据集...")
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
            # 解析文件名中的 Exercise ID (E1/E2)
            exercise_id = None
            if "_E1_" in file_name or file_name == "E1.npy":
                exercise_id = "E1"
            elif "_E2_" in file_name or file_name == "E2.npy":
                exercise_id = "E2"

            if not exercise_id:
                print(f"{display_name:<22} | ⚠️ 无法识别 Exercise 类型，跳过")
                continue

            # 加载数据（兼容 'rep' 与 'repetition' 两种键名）
            data = np.load(file_path, allow_pickle=True).item()
            labels = data['label']
            reps = data.get('rep', data.get('repetition'))
            emgs = data['emg']
            if reps is None:
                raise KeyError("数据中既无 'rep' 也无 'repetition'")

            # --- A. 完整性检查 (Integrity) ---
            df = pd.DataFrame({'Label': labels, 'Rep': reps})
            check_rep = df.groupby('Label')['Rep'].nunique()
            is_integrity_ok = (check_rep == 6).all()

            # --- B. 标签范围匹配 (Standard Match) ---
            unique_labels = sorted(df['Label'].unique())
            actual_count = len(unique_labels)
            actual_min = min(unique_labels)
            actual_max = max(unique_labels)

            target = STANDARD_CONFIG[exercise_id]
            is_range_ok = (
                actual_count == target["count"] and actual_min == target["min"] and actual_max == target["max"]
            )

            label_str = f"{actual_min}-{actual_max} ({actual_count})"

            # --- C. 去噪检查 ---
            avg_len = np.mean([e.shape[0] for e in emgs])
            is_trimmed_ok = (100 < avg_len < 13000)

            # --- D. 综合判定 ---
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
