# DB7 数据处理脚本说明

## 概述
根据 DB2 的处理方式，为 NinaPro DB7 数据集编写了对应的处理脚本，生成相同的 Hugging Face Dataset 格式。

## 数据特点对比

| Exercise | E1, E2, E3 | E1, E2 |
| E1 标签数 | 17 (label 1-17) | 17 (label 1-17) |
| E2 标签数 | 23 (label 18-40) | 23 (label 18-40) |
| E3 标签数 | 9 (label 41-49) | - |
| Total Trials | - | 4800 |
## 脚本文件


**运行方式**:
python script/process_db7.py
```

**输出**:
- 位置: `$EMGFLOW_DB7_TARGET_DIR/$EMGFLOW_DB7_DATASET_NAME`
- 包含:
  - `data-00000-of-00004.arrow` 等 Arrow 格式数据分片
  - `dataset_info.json` - HF Dataset 格式描述
  - `meta.json` - 元信息（通道数、采样率、总trial数等）
  - `state.json` - HF 加载状态

**处理逻辑**:
1. 遍历 `NinaPro_DB7/` 下的 20 个 subject zip 文件
2. 解压后读取 `SX_EY_A1.mat` 文件（每个subject有E1和E2两个exercise）
3. 使用 `restimulus` 和 `rerepetition` 按活动段切片
4. 提取>150ms 的有效 EMG 片段
5. 存储为单一的列式 Dataset（便于按subject/exercise/label筛选）

### 2. `read_db7.py` - 数据读取和筛选工具
**功能**: 提供便捷的 Dataset 读取、筛选、转换接口

**主要函数**:
- `row_with_numpy(dataset, index)` - 读取单行，emg转为numpy
- `slice_to_numpy(dataset)` - 将整个slice转为numpy arrays
- `filter_fast(dataset, **kwargs)` - 快速筛选（只读轻量列）

**使用示例**:
```python
from datasets import load_from_disk
import numpy as np

ds = load_from_disk("DB7_npy/emg_db7_dataset")

# 快速筛选（推荐，避免加载大的emg列）
ds_e1 = filter_fast(ds, exercise="E1")
ds_sub01 = filter_fast(ds, subject="Subject01")

# 按subject和exercise组合筛选
ds_slice = filter_fast(ds, subject="Subject01", exercise="E1", label=[1, 2, 3])

# 转为numpy供深度学习使用
labels, reps, emgs = slice_to_numpy(ds_slice)
# emgs 为 list of ndarray，每个shape为 (n_channels, n_samples)
```

### 3. `verify_db7.py` - 数据验证脚本
**功能**: 验证生成的数据是否符合标准

**验证项**:
- ✅ 每个标签的6个重复完整性
- ✅ 标签范围正确性（E1: 1-17, E2: 18-40）
- ✅ EMG 片段长度合理（100-13000 samples）

**运行方式**:
```bash
python script/verify_db7.py
```

## Dataset 结构

加载后的 Dataset 包含以下列：

```
{
  "subject": "Subject01",      # 被试编号 01-20
  "exercise": "E1",             # Exercise 类型：E1 或 E2
  "label": 5,                   # 动作标签（E1: 1-17, E2: 18-40）
  "repetition": 1,              # 重复编号：1-6
  "n_samples": 2048,            # 该条EMG的样本数
  "emg": [[...], [...], ...]    # EMG 数据：(n_channels=12, n_samples)
}
```

## 与 DB2 的兼容性

脚本遵循完全相同的方式处理数据和组织结构，保证 DB2 和 DB7 生成的 Dataset 可以使用完全相同的代码进行：
- 加载和筛选
- EMG 数据预处理
- 深度学习模型训练

这样可以轻松对比 DB2 和 DB7 的性能差异，或合并两个数据集进行训练。

## 处理统计

```
📊 DB7 处理结果:
   被试数: 20
   总 Trials: 4800
   - E1 Trials: 2040 (每被试 102 trials × 20被试)
   - E2 Trials: 2760 (每被试 138 trials × 20被试)
   通道数: 12
   采样率: 2000 Hz
   数据大小: ~1.8 GB (Arrow 格式)
```
