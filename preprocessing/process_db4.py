import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import scipy.io
from datasets import Dataset, Features, Sequence, Value

# ================= 配置区域 =================
# 1. 原始压缩包所在路径
ZIP_SOURCE_DIR = Path("/bigdata/emgdata_public/DB_raw/Ninapro_DB4")

# 2. 最终 Hugging Face Datasets 输出路径；总表保存为 TARGET_DIR / MASTER_DATASET_NAME
TARGET_DIR = Path("/bigdata/emgdata_public/DB_raw/DB4_npy")
MASTER_DATASET_NAME = "emg_db4_dataset"

# 3. 参数设置
FS = 2000
RAW_LABEL_DELAY_MS = 300
MIN_TRIAL_DURATION_MS = 150

# DB4 原始标签是 exercise 内局部编号：
#   E1: 1-12
#   E2: 1-17
#   E3: 1-23
# 为了和 DB2 / DB7 一样在整表里保持 label 全局不重号，这里默认 remap 为：
#   E1: 1-12
#   E2: 13-29
#   E3: 30-52
REMAP_TO_GLOBAL_LABELS = True
# ===========================================


EXERCISE_CONFIG = {
    1: {
        "name": "E1",
        "count": 12,
        "local_min": 1,
        "local_max": 12,
        "global_offset": 0,
    },
    2: {
        "name": "E2",
        "count": 17,
        "local_min": 1,
        "local_max": 17,
        "global_offset": 12,
    },
    3: {
        "name": "E3",
        "count": 23,
        "local_min": 1,
        "local_max": 23,
        "global_offset": 29,
    },
}


def to_global_label(exercise_num, local_label):
    config = EXERCISE_CONFIG[exercise_num]
    if not (config["local_min"] <= local_label <= config["local_max"]):
        raise ValueError(
            f"E{exercise_num} 的标签超出范围: {local_label} "
            f"(应在 {config['local_min']}-{config['local_max']})"
        )

    if not REMAP_TO_GLOBAL_LABELS:
        return local_label

    return local_label + config["global_offset"]


def expected_label_range(exercise_num):
    config = EXERCISE_CONFIG[exercise_num]
    if REMAP_TO_GLOBAL_LABELS:
        return (
            config["local_min"] + config["global_offset"],
            config["local_max"] + config["global_offset"],
        )
    return config["local_min"], config["local_max"]


def get_trials_db4_explicit(mat_data, fs, exercise_num):
    """
    DB4 切片核心逻辑：
    优先 restimulus + rerepetition（已对齐，delay=0）
    回退 stimulus + repetition（delay=300ms）
    """
    if (
        "restimulus" in mat_data
        and mat_data["restimulus"].size > 0
        and np.any(mat_data["restimulus"] != 0)
        and "rerepetition" in mat_data
        and mat_data["rerepetition"].size > 0
    ):
        stim = mat_data["restimulus"].flatten()
        rep_signal = mat_data["rerepetition"].flatten()
        delay_ms = 0
        source_type = "restimulus (refined)"
    elif "stimulus" in mat_data and "repetition" in mat_data:
        stim = mat_data["stimulus"].flatten()
        rep_signal = mat_data["repetition"].flatten()
        delay_ms = RAW_LABEL_DELAY_MS
        source_type = "stimulus (raw fallback)"
    else:
        return [], [], [], {}, "Missing Variables"

    delay_samples = int((delay_ms / 1000) * fs)
    min_samples = int((MIN_TRIAL_DURATION_MS / 1000) * fs)
    emg_signal = mat_data["emg"]
    max_len = emg_signal.shape[0]

    active_indices = np.where(stim != 0)[0]
    if len(active_indices) == 0:
        return [], [], [], {}, f"Empty {source_type}"

    splits = np.where(np.diff(active_indices) > 1)[0]
    trial_ranges = []
    start_idx = 0
    for split in splits:
        end_idx = split
        trial_ranges.append((active_indices[start_idx], active_indices[end_idx]))
        start_idx = split + 1
    trial_ranges.append((active_indices[start_idx], active_indices[-1]))

    sliced_emg = []
    sliced_labels = []
    sliced_reps = []

    for start, end in trial_ranges:
        local_label = int(np.median(stim[start:end + 1]))
        if local_label == 0:
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
            sliced_labels.append(to_global_label(exercise_num, local_label))
            sliced_reps.append(segment_rep_id)

    meta_info = {
        "label_source": source_type,
        "applied_delay_ms": delay_ms,
        "fs": fs,
        "total_trials": len(sliced_labels),
        "label_mode": "global" if REMAP_TO_GLOBAL_LABELS else "local",
    }

    return sliced_emg, sliced_labels, sliced_reps, meta_info, "OK"


def pipeline_db4_auto_cleanup():
    zip_files = sorted(list(ZIP_SOURCE_DIR.glob("s*.zip")))
    if not zip_files:
        print(f"❌ 未找到压缩包: {ZIP_SOURCE_DIR}")
        return

    data_dict = {
        "subject": [],
        "exercise": [],
        "label": [],
        "repetition": [],
        "n_samples": [],
        "emg": [],
    }
    n_channels = None
    global_meta = {
        "fs": FS,
        "sources": [],
        "label_mode": "global" if REMAP_TO_GLOBAL_LABELS else "local",
        "exercise_config": {},
    }

    for exercise_num, config in EXERCISE_CONFIG.items():
        expected_min, expected_max = expected_label_range(exercise_num)
        global_meta["exercise_config"][config["name"]] = {
            "count": config["count"],
            "local_label_range": [config["local_min"], config["local_max"]],
            "saved_label_range": [expected_min, expected_max],
        }

    print(f"🚀 开始处理 {len(zip_files)} 个压缩包\n")

    for zip_file in zip_files:
        subject_name = zip_file.stem  # "s1"
        subject_num = int(subject_name.replace("s", ""))
        subject_id_str = f"Subject{subject_num:02d}"

        print(f"📦 正在处理: {zip_file.name} ({subject_id_str})...")

        subject_dir = TARGET_DIR / subject_id_str
        subject_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(temp_path)

            mat_files = sorted(temp_path.glob("**/*.mat"))

            for mat_file in mat_files:
                filename = mat_file.name
                try:
                    parts = filename.replace(".mat", "").split("_")
                    subject_from_file = int(parts[0][1:])
                    exercise_num = int(parts[1][1:])
                except Exception:
                    print(f"  [WARN] 无法解析文件名: {filename}")
                    continue

                if subject_from_file != subject_num:
                    print(
                        f"  [WARN] 文件名被试号不匹配: zip={subject_num}, file={subject_from_file} ({filename})"
                    )
                    continue

                if exercise_num not in EXERCISE_CONFIG:
                    print(f"  [WARN] 未知的 Exercise: {exercise_num} ({filename})")
                    continue

                exercise_str = EXERCISE_CONFIG[exercise_num]["name"]

                try:
                    mat_data = scipy.io.loadmat(str(mat_file))

                    if "frequency" in mat_data:
                        file_fs = int(np.asarray(mat_data["frequency"]).reshape(-1)[0])
                        if file_fs != FS:
                            raise ValueError(f"采样率不一致: 期望 {FS}, 实际 {file_fs}")

                    emgs, lbls, reps, meta, status = get_trials_db4_explicit(mat_data, FS, exercise_num)

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
                    data_dict["emg"].extend([arr.astype(np.float32).T for arr in emgs])
                    global_meta["sources"].append(f"{subject_id_str}/{exercise_str} from {filename}")

                    npy_data = {
                        "emg": emgs,
                        "label": np.array(lbls, dtype=np.int64),
                        "repetition": np.array(reps, dtype=np.int64),
                        "meta": {
                            "exercise": exercise_str,
                            "n_trials": n_trials,
                            "fs": FS,
                            "n_channels": n_channels,
                            "label_source": meta["label_source"],
                            "label_mode": meta["label_mode"],
                        },
                    }
                    npy_file = subject_dir / f"{exercise_str}.npy"
                    np.save(str(npy_file), npy_data)

                    expected_min, expected_max = expected_label_range(exercise_num)
                    actual_min, actual_max = min(lbls), max(lbls)
                    label_check = "✅" if (actual_min == expected_min and actual_max == expected_max) else "⚠️"

                    print(
                        f"  {label_check} {subject_id_str}/{exercise_str} | "
                        f"Trials: {n_trials} | Labels: {actual_min}-{actual_max} | "
                        f"Reps: {min(reps)}-{max(reps)} | {meta['label_source']}"
                    )

                except Exception as e:
                    print(f"  ❌ 错误 {filename}: {e}")

        print(f"  ✨ {subject_id_str} 处理完毕，已保存 npy 文件。\n")

    if not data_dict["label"]:
        print("❌ 未得到任何数据，未写入。")
        return

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
                "label_mode": global_meta["label_mode"],
                "exercise_config": global_meta["exercise_config"],
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
    print(f"   通道数: {n_channels} | 采样率: {FS} Hz | 标签模式: {global_meta['label_mode']}")


if __name__ == "__main__":
    pipeline_db4_auto_cleanup()
