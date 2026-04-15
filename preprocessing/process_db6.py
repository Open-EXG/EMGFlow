import os
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.io

# ================= 配置区域 =================
# 原始数据路径 (存放所有 .mat 文件)
SOURCE_DIR = Path(os.environ.get("EMGFLOW_DB6_SOURCE_DIR", "NinaPro_DB6"))

# 输出路径
TARGET_DIR = Path(os.environ.get("EMGFLOW_DB6_TARGET_DIR", "DB6_npy"))

# DB6 采样率
FS = 2000

# 当不得不使用原始 stimulus 时，补偿的反应时间 (ms)
RAW_LABEL_DELAY_MS = 300

# 最小有效动作时长 (ms)，小于这个时长的片段会被丢弃 (防止误触噪声)
MIN_TRIAL_DURATION_MS = 150
# ===========================================


def get_trials_with_repetition(mat_data, fs):
    """
    智能切片函数：
    1. 判定 Label 源 (restimulus vs stimulus) 并计算 Delay。
    2. 切分 Trial。
    3. [NEW] 计算 Repetition ID (这是该动作在当前 session 的第几次重复)。
    """

    # --- 1. 智能判断标签源和延迟 ---
    if 'restimulus' in mat_data and mat_data['restimulus'].size > 0:
        stim = mat_data['restimulus'].flatten()
        delay_ms = 0
        source_type = 'restimulus'
    elif 'stimulus' in mat_data:
        stim = mat_data['stimulus'].flatten()
        delay_ms = RAW_LABEL_DELAY_MS
        source_type = 'stimulus (fallback)'
    else:
        return [], [], [], {}, "No Label Found"

    delay_samples = int((delay_ms / 1000) * fs)
    min_samples = int((MIN_TRIAL_DURATION_MS / 1000) * fs)

    # --- 2. 寻找动作区间 ---
    emg_signal = mat_data['emg']
    max_len = emg_signal.shape[0]

    # 找到所有非休息状态
    active_indices = np.where(stim != 0)[0]

    if len(active_indices) == 0:
        return [], [], [], {}, f"Empty {source_type}"

    # 利用 diff 找到断点
    splits = np.where(np.diff(active_indices) > 1)[0]

    trial_ranges = []
    start_idx = 0
    for split in splits:
        end_idx = split
        trial_ranges.append((active_indices[start_idx], active_indices[end_idx]))
        start_idx = split + 1
    trial_ranges.append((active_indices[start_idx], active_indices[-1]))

    # --- 3. 提取数据 + 计算 Repetition ---
    sliced_emg = []
    sliced_labels = []
    sliced_reps = []  # 存储重复次数

    # 计数器：记录当前 Session 内，每个 Label 已经出现了几次
    # 格式: { Label_ID : Count }
    rep_counter = defaultdict(int)

    for start, end in trial_ranges:
        # === A. 确定 Label 和 Repetition ===
        # 先获取这段动作的 Label (取中位数)
        # 注意：要在应用延迟前获取 Label，因为 Label 是定义在原始时间轴上的
        segment_label = int(np.median(stim[start:end + 1]))

        # 即使 Label 是 0 (极少数情况切片错误)，也跳过
        if segment_label == 0:
            continue

        # 计数 +1
        rep_counter[segment_label] += 1
        current_rep_id = rep_counter[segment_label]

        # === B. 应用延迟 (Time Shift) ===
        real_start = start + delay_samples
        real_end = end + delay_samples

        # 边界保护
        if real_start >= max_len:
            continue
        if real_end > max_len:
            real_end = max_len

        # === C. 提取 EMG ===
        segment_emg = emg_signal[real_start:real_end + 1, :]

        # 长度过滤
        if segment_emg.shape[0] > min_samples:
            # 转 float32 省空间
            sliced_emg.append(segment_emg.astype(np.float32))
            sliced_labels.append(segment_label)
            sliced_reps.append(current_rep_id)  # 保存 Rep ID

    # 构建元数据
    meta_info = {
        "label_source": source_type,
        "applied_delay_ms": delay_ms,
        "fs": fs,
        "total_trials": len(sliced_labels)
    }

    return sliced_emg, sliced_labels, sliced_reps, meta_info, "OK"


def process_db6_smart():
    if not TARGET_DIR.exists():
        TARGET_DIR.mkdir(parents=True)

    files = list(SOURCE_DIR.glob("**/*.mat"))
    files.sort()  # 排序保证处理顺序一致

    print(f"🚀 开始处理 {len(files)} 个文件 | 目标结构: Subject/Day/Session.npy")
    print("🔧 策略: 优先 Restimulus, 自动计算 Repetition ID\n")

    success_count = 0

    for file_path in files:
        filename = file_path.name

        # --- 解析文件名 (S1_D1_T1.mat) ---
        try:
            # DB6 文件名格式: S{subject}_D{day}_T{session}
            parts = filename.replace('.mat', '').split('_')
            subject_id = int(parts[0].replace('S', ''))
            day_num = int(parts[1].replace('D', ''))
            session_num = int(parts[2].replace('T', ''))
        except:
            print(f"[SKIP] 文件名解析失败: {filename}")
            continue

        # --- 核心处理 ---
        try:
            mat_data = scipy.io.loadmat(str(file_path))

            # 调用带 Repetition 逻辑的函数
            emgs, lbls, reps, meta, status = get_trials_with_repetition(mat_data, FS)

            if not emgs:
                print(f"  [WARN] {filename}: 数据为空 ({status})")
                continue

            # --- 保存 ---
            save_data = {
                "emg": np.array(emgs, dtype=object),  # (N_trials, ) of (Time, Ch)
                "label": np.array(lbls, dtype=np.int64),  # (N_trials, )
                "repetition": np.array(reps, dtype=np.int64),  # (N_trials, ) <--- NEW
                "meta": {
                    "subject": subject_id,
                    "day": day_num,
                    "session_type": "AM" if session_num == 1 else "PM",
                    "original_file": filename,
                    **meta
                }
            }

            # 路径: Subject01/day1/session1.npy
            sub_dir = TARGET_DIR / f"Subject{subject_id:02d}" / f"day{day_num}"
            sub_dir.mkdir(parents=True, exist_ok=True)

            save_path = sub_dir / f"session{session_num}.npy"
            np.save(save_path, save_data)

            print(
                f"✅ {filename} -> S{subject_id}/D{day_num}/s{session_num} | Trials: {len(lbls)} | Source: {meta['label_source']}"
            )
            success_count += 1

        except Exception as e:
            print(f"❌ {filename}: 处理出错 -> {e}")

    print(f"\n🎉 全部完成! 成功处理 {success_count} 个文件。")


if __name__ == "__main__":
    process_db6_smart()
