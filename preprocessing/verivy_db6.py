import os
from collections import Counter
from pathlib import Path

import numpy as np

# ================= 配置 =================
DATA_ROOT = Path("/bigdata/emgdata_public/DB_raw/DB6_npy")
# =======================================


def check_dataset_integrity():
    files = sorted(list(DATA_ROOT.glob("**/*.npy")))

    print(f"🔍 [完整性检查] 扫描 {len(files)} 个文件...\n")

    total_trials = 0
    error_count = 0
    warning_count = 0

    # 打印表头
    header = f"{'FILE':<35} | {'STATUS':<6} | {'TRIALS':<6} | {'SHAPE (Time, Ch)':<18} | {'LABELS':<10} | {'REPS':<8}"
    print(header)
    print("-" * len(header))

    for file_path in files:
        relative_name = f"{file_path.parent.name}/{file_path.name}"

        try:
            # 1. 加载
            data_dict = np.load(file_path, allow_pickle=True).item()

            # 检查键是否存在
            if 'repetition' not in data_dict:
                print(f"{relative_name:<35} | ❌FAIL  | Missing 'repetition' key!")
                error_count += 1
                continue

            emg_list = data_dict['emg']
            label_list = data_dict['label']
            rep_list = data_dict['repetition']  # <--- 新增
            meta = data_dict['meta']

            # 2. 长度一致性检查 (这是最重要的)
            if not (len(emg_list) == len(label_list) == len(rep_list)):
                print(
                    f"{relative_name:<35} | ❌FAIL  | Length Mismatch: E={len(emg_list)}, L={len(label_list)}, R={len(rep_list)}"
                )
                error_count += 1
                continue

            if len(emg_list) == 0:
                print(f"{relative_name:<35} | ⚠️WARN  | No Trials Found")
                warning_count += 1
                continue

            # 3. 数值检查
            # 检查 Repetition 是否正常 (比如应该是 1-10)
            min_rep, max_rep = rep_list.min(), rep_list.max()

            # 检查 Label 是否含有 0
            if 0 in label_list:
                print(f"{relative_name:<35} | ⚠️WARN  | Contains Label 0 (Rest)")
                warning_count += 1

            # 4. 统计信息
            total_trials += len(label_list)

            # 时长范围
            lengths = [x.shape[0] for x in emg_list]
            channels = emg_list[0].shape[1]
            len_info = f"T:{min(lengths)}-{max(lengths)} C:{channels}"

            # Label 和 Rep 范围
            lbl_info = f"{label_list.min()}-{label_list.max()}"
            rep_info = f"{min_rep}-{max_rep}"

            print(
                f"{relative_name:<35} | ✅ OK   | {len(emg_list):<6} | {len_info:<18} | {lbl_info:<10} | {rep_info:<8}"
            )

        except Exception as e:
            print(f"{relative_name:<35} | ❌ERR   | {str(e)}")
            error_count += 1

    print("-" * len(header))
    print(f"\n📊 审计总结:")
    print(f"   - 文件: {len(files)} | 总 Trial: {total_trials}")
    print(f"   - 错误: {error_count} | 警告: {warning_count}")
    if error_count == 0:
        print("   ✅ 数据集结构完整，字段对齐。")


if __name__ == "__main__":
    check_dataset_integrity()
