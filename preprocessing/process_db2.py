import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import scipy.io
from datasets import Dataset, Features, Sequence, Value

# ================= 配置区域 =================
# 1. 原始压缩包所在路径
ZIP_SOURCE_DIR = Path("/bigdata/emgdata_public/DB_raw/NinaPro_DB2")

# 2. 最终 Hugging Face Datasets 输出路径；总表保存为 TARGET_DIR / MASTER_DATASET_NAME
TARGET_DIR = Path("/bigdata/emgdata_public/DB_raw/DB2_npy")
MASTER_DATASET_NAME = "emg_db2_dataset"

# 3. 参数设置
FS = 2000
RAW_LABEL_DELAY_MS = 300
MIN_TRIAL_DURATION_MS = 150
# ===========================================
#
# 输出目录结构（TARGET_DIR / MASTER_DATASET_NAME）：
#   - *.arrow           Hugging Face 生成的 Arrow 数据分片（实际数据）
#   - dataset_info.json  Hugging Face 生成的表结构/描述，load_from_disk 依赖
#   - state.json         Hugging Face 生成的加载状态（指纹等），load_from_disk 依赖
#   - meta.json          本脚本写入的元信息：n_channels, fs, total_trials, sources（读表时取 n_channels 用）
# ===========================================


def get_trials_db2_explicit(mat_data, fs):
    """
    DB2 切片核心逻辑:
    优先 restimulus + rerepetition (Delay=0)
    回退 stimulus + repetition (Delay=300)
    """
    # --- 1. 变量选择 ---
    if (
        'restimulus' in mat_data and mat_data['restimulus'].size > 0 and np.any(mat_data['restimulus'] != 0)
        and 'rerepetition' in mat_data and mat_data['rerepetition'].size > 0
    ):

        stim = mat_data['restimulus'].flatten()
        rep_signal = mat_data['rerepetition'].flatten()
        delay_ms = 0
        source_type = 'restimulus (refined)'

    elif 'stimulus' in mat_data and 'repetition' in mat_data:
        stim = mat_data['stimulus'].flatten()
        rep_signal = mat_data['repetition'].flatten()
        delay_ms = RAW_LABEL_DELAY_MS
        source_type = 'stimulus (raw fallback)'
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

    # 寻找断点
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


def pipeline_db2_auto_cleanup():
    zip_files = sorted(list(ZIP_SOURCE_DIR.glob("*.zip")))
    if not zip_files:
        print(f"❌ 未找到压缩包: {ZIP_SOURCE_DIR}")
        return

    # 总表：所有 trial 的列式数据，便于后续按 subject / exercise / label / repetition 筛选
    # emg 每行存为 (n_channels, n_samples)：外层 list 为通道，内层为时间点
    data_dict = {
        "subject": [],   # 如 "Subject01"
        "exercise": [],  # 如 "E1", "E2", "E3"
        "label": [],
        "repetition": [],
        "n_samples": [], # 该行 emg 的时间点数量（段长度）
        "emg": [],      # 每段 EMG 形状 (n_channels, n_samples)，直接存 ndarray
    }
    n_channels = None
    global_meta = {"fs": FS, "sources": []}

    print(f"🚀 开始处理 {len(zip_files)} 个压缩包，写入单一总表\n")

    for zip_file in zip_files:
        subject_name = zip_file.stem  # "S1"
        print(f"📦 正在处理: {zip_file.name} ...")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(temp_path)

            mat_files = list(temp_path.glob("**/*.mat"))

            for mat_file in mat_files:
                filename = mat_file.name
                try:
                    parts = filename.replace(".mat", "").split("_")
                    subject_id_int = int(parts[0].replace("S", ""))
                    exercise_str = parts[1]
                except Exception:
                    continue

                subject_id_str = f"Subject{subject_id_int:02d}"

                try:
                    mat_data = scipy.io.loadmat(str(mat_file))
                    emgs, lbls, reps, meta, status = get_trials_db2_explicit(mat_data, FS)

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
                    data_dict["n_samples"].extend([int(arr.shape[0]) for arr in emgs])  # 每段时间点数量
                    data_dict["emg"].extend([arr.astype(np.float32).T for arr in emgs])
                    global_meta["sources"].append(f"{subject_id_str}/{exercise_str} from {filename}")

                    print(
                        f"  ✅ {subject_id_str}/{exercise_str} | "
                        f"Trials: {n_trials} | Reps: {min(reps)}-{max(reps)} | {meta['label_source']}"
                    )

                except Exception as e:
                    print(f"  ❌ 错误 {filename}: {e}")

        print(f"  ✨ {subject_name} 处理完毕，临时文件已清理。\n")

    if not data_dict["label"]:
        print("❌ 未得到任何数据，未写入。")
        return

    # 一次性构建总表并落盘。features 非必须（不传则 HF 从数据推断类型），但显式传入可保证
    # emg 被存为 Sequence(Sequence(float32))，形状 (n_channels, n_samples)，避免推断错误。
    # 这些类型信息会写入 dataset_info.json，load_from_disk 时据此解析列类型。
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
    # save_to_disk 会生成：Arrow 数据分片、dataset_info.json（HF 格式/描述）、state.json（HF 加载状态）
    # 下面再写入本脚本自定义的 meta.json（n_channels / fs 等），读表时用 meta.json 取 n_channels

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
        f"   总行数: {len(data_dict['label'])} | 列: subject, exercise, label, repetition, n_samples, emg (n_channels, n_samples)"
    )


# ---------------------------------------------------------------------------
# 读表与筛选示例（用于深度学习 pipeline）
# ---------------------------------------------------------------------------
# from datasets import load_from_disk
# import numpy as np
#
# ds = load_from_disk("/bigdata/emgdata_public/DB_raw/DB2_npy/emg_db2_dataset")
# with open(".../emg_db2_dataset/meta.json") as f:
#     meta = json.load(f)
# n_channels = meta["n_channels"]
#
# # 按 subject / exercise 筛选
# ds_e1 = ds.filter(lambda x: x["exercise"] == "E1")
# ds_sub01 = ds.filter(lambda x: x["subject"] == "Subject01")
#
# # 按 label / repetition 筛选
# ds_label_5 = ds.filter(lambda x: x["label"] == 5)
# ds_rep_1_3 = ds.filter(lambda x: x["repetition"] in (1, 2, 3))  # 需逐行，或先取列再掩码
#
# # 组合条件：某被试、某 exercise、某几个 label
# ds_slice = ds.filter(
#     lambda x: x["subject"] == "Subject01"
#     and x["exercise"] == "E2"
#     and x["label"] in (18, 19, 20)
# )
#
# # 取出一段 EMG，形状 (n_channels, n_samples)
# row = ds_slice[0]
# emg_segment = np.array(row["emg"], dtype=np.float32)  # (n_channels, n_samples)
# label, rep = row["label"], row["repetition"]
#
# # 或批量转成 numpy 供 DataLoader
# labels = np.array(ds_slice["label"])
# reps = np.array(ds_slice["repetition"])
# emgs = [np.array(e, dtype=np.float32) for e in ds_slice["emg"]]  # 每项 (n_channels, n_samples)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pipeline_db2_auto_cleanup()
