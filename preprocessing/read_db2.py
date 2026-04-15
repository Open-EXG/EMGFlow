import os

import numpy as np
from datasets import load_from_disk

# 使用内存映射：不一次性读入全部数据，按需从磁盘映射；只有访问某行/某列时才触及对应数据
DATASET_DIR = os.environ.get("EMGFLOW_DB2_DATASET_DIR", "DB2_npy/emg_db2_dataset")
ds = load_from_disk(DATASET_DIR)


def row_with_numpy(dataset, index):
    """读一行，返回的 dict 里 emg 已是 ndarray (n_channels, n_samples)，可直接用。"""
    row = dataset[int(index)]
    row = dict(row)
    row["emg"] = np.asarray(row["emg"], dtype=np.float32)
    return row


def slice_to_numpy(dataset):
    """把 Dataset 的 label / repetition / emg 整列转成 numpy 风格，emg 为 list of ndarray。"""
    labels = np.array(dataset["label"])
    reps = np.array(dataset["repetition"])
    emgs = [np.asarray(e, dtype=np.float32) for e in dataset["emg"]]
    return labels, reps, emgs


def filter_fast(dataset, **kwargs):
    """
    按列值筛选，只读轻量列（不碰 emg），比 ds.filter(lambda) 快很多。
    用法: filter_fast(ds, exercise="E1") 或 filter_fast(ds, subject="Subject01", label=5)
    """
    indices = None
    for col, val in kwargs.items():
        col_data = dataset[col]
        if indices is None:
            if isinstance(val, (list, tuple, set)):
                idx = [i for i, v in enumerate(col_data) if v in val]
            else:
                idx = [i for i, v in enumerate(col_data) if v == val]
            indices = set(idx)
        else:
            if isinstance(val, (list, tuple, set)):
                indices &= {i for i, v in enumerate(col_data) if v in val}
            else:
                indices &= {i for i, v in enumerate(col_data) if v == val}
    if not indices:
        return dataset.select([])
    return dataset.select(sorted(indices))


# with open(".../emg_db2_dataset/meta.json") as f:
#     meta = json.load(f)
# n_channels = meta["n_channels"]

# 按 subject / exercise 筛选（用 filter_fast 只读轻量列，不碰 emg，比下面 ds.filter 快很多）
ds_e1 = filter_fast(ds, exercise="E1")
ds_sub01 = filter_fast(ds, subject="Subject01")

# 按 label / repetition 筛选
ds_label_5 = filter_fast(ds, label=5)
ds_rep_1_3 = filter_fast(ds, repetition=[1, 2, 3])  # 多值用 list

# 组合条件：某被试、某 exercise、某几个 label
ds_slice = filter_fast(ds, subject="Subject01", exercise="E2", label=[18, 19, 20])

# 若仍用 ds.filter(lambda x: ...)：每行都会把整行（含大 emg）交给 Python，所以很慢
# ds_e1_slow = ds.filter(lambda x: x["exercise"] == "E1")

# 用辅助函数读：直接拿到 numpy，不用手写 asarray
row = row_with_numpy(ds_slice, 0)
emg_segment = row["emg"]  # 已是 ndarray (n_channels, n_samples)
label, rep = row["label"], row["repetition"]

# 整块转成 numpy 供 DataLoader
labels, reps, emgs = slice_to_numpy(ds_slice)  # emgs 为 list of ndarray

# ---------------------------------------------------------------------------
# Dataset (ds) 常见用法速查
# ---------------------------------------------------------------------------
# 索引与列
#   ds[0]                   第 0 行，得到 dict {"subject", "exercise", "label", "repetition", "n_samples", "emg"}
#   ds[1:10]                切片，得到新 Dataset（1~9 行）
#   ds["label"]             整列，返回该列序列（按需映射）
#   ds.column_names         列名列表
#
# 筛选与子集
#   ds.filter(fn)           按函数筛选行，fn(row_dict) -> bool
#   ds.select(indices)      按索引列表取行，indices 为 list[int]
#   ds.shuffle(seed=42)     打乱行顺序
#   ds.train_test_split(test_size=0.2, seed=42)  划分训练/测试
#
# 变换
#   ds.map(fn, batched=True)  对每行（或批量）做变换，可增删列
#   ds.remove_columns(names)  删除列
#   ds.rename_column(old, new) 重命名列
#   ds.flatten()             展平嵌套列（如 emg 的 list 结构）
#
# 迭代与导出
#   for row in ds: ...      逐行迭代
#   for batch in ds.iter(batch_size=32): ...  按批迭代
#   ds.to_pandas()          转成 pandas DataFrame（会按需加载）
#   ds.to_dict()            转成 dict of lists（整表进内存）
#
# 元信息
#   len(ds)                 行数
#   ds.shape                 (num_rows, num_columns)
#   ds.features             列类型（Features 对象）
#   ds.info                  DatasetInfo（描述等）
# ---------------------------------------------------------------------------
