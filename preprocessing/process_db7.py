import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import scipy.io
from datasets import Dataset, Features, Sequence, Value

# ================= 配置区域 =================
# 1. 原始压缩包所在路径
ZIP_SOURCE_DIR = Path("/bigdata/emgdata_public/DB_raw/NinaPro_DB7")

# 2. 最终 Hugging Face Datasets 输出路径；总表保存为 TARGET_DIR / MASTER_DATASET_NAME
TARGET_DIR = Path("/bigdata/emgdata_public/DB_raw/DB7_npy")
MASTER_DATASET_NAME = "emg_db7_dataset"

# 3. 参数设置
FS = 2000  # DB7 采样率也是 2000Hz
RAW_LABEL_DELAY_MS = 0  # DB7 使用 restimulus（已对齐），延迟为 0
MIN_TRIAL_DURATION_MS = 150
# ===========================================
#
# 输出目录结构（TARGET_DIR / MASTER_DATASET_NAME）：
#   - *.arrow           Hugging Face 生成的 Arrow 数据分片（实际数据）
#   - dataset_info.json  Hugging Face 生成的表结构/描述，load_from_disk 依赖
#   - state.json         Hugging Face 生成的加载状态（指纹等），load_from_disk 依赖
#   - meta.json          本脚本写入的元信息：n_channels, fs, total_trials, sources（读表时取 n_channels 用）
# ===========================================

# DB7 标签配置（与 DB2 保持一致）
EXERCISE_CONFIG = {
    1: {
        "name": "E1",
        "count": 17,
        "min": 1,
        "max": 17
    },
    2: {
        "name": "E2",
        "count": 23,
        "min": 18,
        "max": 40
    }
}


def get_trials_db7_explicit(mat_data, fs):
    """
    DB7 切片核心逻辑:
    使用 restimulus + rerepetition（已对齐，不需延迟）
    """
    # --- 1. 变量选择 ---
    if (
        'restimulus' in mat_data and mat_data['restimulus'].size > 0 and np.any(mat_data['restimulus'] != 0)
        and 'rerepetition' in mat_data and mat_data['rerepetition'].size > 0
    ):
        stim = mat_data['restimulus'].flatten()
        rep_signal = mat_data['rerepetition'].flatten()
        delay_ms = 0
        source_type = 'restimulus'
    elif 'stimulus' in mat_data and 'repetition' in mat_data:
        stim = mat_data['stimulus'].flatten()
        rep_signal = mat_data['repetition'].flatten()
        delay_ms = RAW_LABEL_DELAY_MS
        source_type = 'stimulus (fallback)'
    else:
        return [], [], [], {}, "Missing Variables"

    # --- 2. 准备切片 ---
    delay_samples = int((delay_ms / 1000) * fs)
    min_samples = int((MIN_TRIAL_DURATION_MS / 1000) * fs)
    emg_signal = mat_data['emg']
    max_len = emg_signal.shape[0]

    active_indices = np.where(stim != 0)[0]
    if len(active_indices) == 0:
        return [], [], [], {}, f"Empty {source_type}"

    # 寻找断点（标签或重复改变）
    splits = np.where(np.diff(active_indices) > 1)[0]
    trial_ranges = []
    start_idx = 0
    for split in splits:
        end_idx = split
        trial_ranges.append((active_indices[start_idx], active_indices[end_idx]))
        start_idx = split + 1
    trial_ranges.append((active_indices[start_idx], active_indices[-1]))

    # --- 3. 提取数据 ---
    sliced_emg = []
    sliced_labels = []
    sliced_reps = []

    for start, end in trial_ranges:
        segment_label = int(np.median(stim[start:end + 1]))
        if segment_label == 0:
            continue

        segment_rep_id = int(np.median(rep_signal[start:end + 1]))
        if segment_rep_id == 0:
            continue

        real_start = start + delay_samples
        real_end = end + delay_samples

        if real_start >= max_len:
            continue
        if real_end > max_len:
            real_end = max_len

        segment_emg = emg_signal[real_start:real_end + 1, :]

        if segment_emg.shape[0] > min_samples:
            sliced_emg.append(segment_emg.astype(np.float32))
            sliced_labels.append(segment_label)
            sliced_reps.append(segment_rep_id)

    meta_info = {
        "label_source": source_type,
        "applied_delay_ms": delay_ms,
        "fs": fs,
        "total_trials": len(sliced_labels)
    }

    return sliced_emg, sliced_labels, sliced_reps, meta_info, "OK"


def pipeline_db7_auto_cleanup():
    """
    DB7 处理流程：
    1. 遍历每个 Subject_X.zip
    2. 解压出 SX_Ey_A1.mat 文件
    3. 按 restimulus/rerepetition 切分成 trials
    4. 分别保存为：
       - 按 Subject/Exercise 的 npy 文件
       - 单一的 Hugging Face Dataset
    """
    zip_files = sorted(list(ZIP_SOURCE_DIR.glob("Subject_*.zip")))
    if not zip_files:
        print(f"❌ 未找到压缩包: {ZIP_SOURCE_DIR}")
        return

    # 总表：所有 trial 的列式数据
    data_dict = {
        "subject": [],   # 如 "Subject01"
        "exercise": [],  # 如 "E1", "E2"
        "label": [],
        "repetition": [],
        "n_samples": [],  # 该行 emg 的时间点数量
        "emg": [],        # 每段 EMG 形状 (n_channels, n_samples)
    }
    n_channels = None
    global_meta = {"fs": FS, "sources": []}

    print(f"🚀 开始处理 {len(zip_files)} 个压缩包\n")

    for zip_file in zip_files:
        subject_name = zip_file.stem  # "Subject_1"
        # 提取数字：Subject_1 -> Subject01
        subject_num = int(subject_name.split('_')[1])
        subject_id_str = f"Subject{subject_num:02d}"

        print(f"📦 正在处理: {zip_file.name} ({subject_id_str})...")

        # 为该被试创建目录
        subject_dir = TARGET_DIR / subject_id_str
        subject_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(temp_path)

            mat_files = list(temp_path.glob("*.mat"))

            for mat_file in mat_files:
                filename = mat_file.name
                try:
                    # 解析文件名：S1_E1_A1.mat -> subject=1, exercise=1, activity=1
                    parts = filename.replace(".mat", "").split("_")
                    subject_from_file = int(parts[0][1:])
                    exercise_num = int(parts[1][1:])
                    # activity = int(parts[2][1:])  # DB7 目前只有 A1
                except Exception:
                    print(f"  [WARN] 无法解析文件名: {filename}")
                    continue

                if exercise_num not in EXERCISE_CONFIG:
                    print(f"  [WARN] 未知的 Exercise: {exercise_num}")
                    continue

                exercise_str = EXERCISE_CONFIG[exercise_num]["name"]

                try:
                    mat_data = scipy.io.loadmat(str(mat_file))
                    emgs, lbls, reps, meta, status = get_trials_db7_explicit(mat_data, FS)

                    if not emgs:
                        print(f"  [WARN] {filename}: 空数据 ({status})")
                        continue

                    if n_channels is None:
                        n_channels = int(emgs[0].shape[1])

                    n_trials = len(lbls)
                    data_dict["subject"].extend([subject_id_str] * n_trials)
                    data_dict["exercise"].extend([exercise_str] * n_trials)
                    data_dict["label"].extend(lbls)
                    data_dict["repetition"].extend(reps)
                    data_dict["n_samples"].extend([int(arr.shape[0]) for arr in emgs])
                    # 转置为 (n_channels, n_samples)
                    transposed_emgs = [arr.astype(np.float32).T for arr in emgs]
                    data_dict["emg"].extend(transposed_emgs)
                    global_meta["sources"].append(f"{subject_id_str}/{exercise_str} from {filename}")

                    # --- 保存为 npy 文件（按 Subject/Exercise） ---
                    npy_data = {
                        "emg": emgs,  # 保存原始形状 (n_samples, n_channels)，方便直接使用
                        "label": np.array(lbls, dtype=np.int64),
                        "repetition": np.array(reps, dtype=np.int64),
                        "meta": {
                            "exercise": exercise_str,
                            "n_trials": n_trials,
                            "fs": FS,
                            "n_channels": n_channels,
                            "label_source": meta["label_source"],
                        }
                    }
                    npy_file = subject_dir / f"{exercise_str}.npy"
                    np.save(str(npy_file), npy_data)

                    # 标签范围检查
                    expected_min = EXERCISE_CONFIG[exercise_num]["min"]
                    expected_max = EXERCISE_CONFIG[exercise_num]["max"]
                    actual_min, actual_max = min(lbls), max(lbls)

                    label_check = "✅" if (actual_min == expected_min and actual_max == expected_max) else "⚠️"

                    print(
                        f"  {label_check} {subject_id_str}/{exercise_str} | "
                        f"Trials: {n_trials} | Labels: {actual_min}-{actual_max} | "
                        f"Reps: {min(reps)}-{max(reps)}"
                    )

                except Exception as e:
                    print(f"  ❌ 错误 {filename}: {e}")

        print(f"  ✨ {subject_id_str} 处理完毕，已保存 npy 文件。\n")

    if not data_dict["label"]:
        print("❌ 未得到任何数据，未写入。")
        return

    # 构建总表
    features = Features({
        "subject": Value("string"),
        "exercise": Value("string"),
        "label": Value("int64"),
        "repetition": Value("int64"),
        "n_samples": Value("int64"),
        "emg": Sequence(Sequence(Value("float32"))),
    })
    hf_dataset = Dataset.from_dict(data_dict, features=features)

    out_path = TARGET_DIR / MASTER_DATASET_NAME
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    hf_dataset.save_to_disk(str(out_path))

    meta_path = out_path / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_channels": n_channels,
                "fs": global_meta["fs"],
                "total_trials": len(data_dict["label"]),
                "sources": global_meta["sources"][:10]
                + (["..."] if len(global_meta["sources"]) > 10 else []),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"🎉 全部完成！总表已写入: {out_path}")
    print(
        f"   总行数: {len(data_dict['label'])} | 列: subject, exercise, label, repetition, n_samples, emg"
    )
    print(f"   通道数: {n_channels} | 采样率: {FS} Hz")


if __name__ == "__main__":
    pipeline_db7_auto_cleanup()
