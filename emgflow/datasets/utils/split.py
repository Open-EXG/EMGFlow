from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import random
import numpy as np

# 统一的样本行格式：你也可以换成 dataclass
# (subject_id, npy_path, trial_id, repetition_id, start, end, label)
Sample = Tuple[int, str, int, int, int, int, int]

@dataclass(frozen=True)
class SplitSpec:
    group_by: str = "subject"         # "subject" | "repetition"
    seed: int = 42
    # if using repetition 
    train_trials: Optional[list[int]] = None
    val_trials: Optional[list[int]] = None
    test_trials: Optional[list[int]] = None
    # if using subject
    train_subjects: Optional[list[int]] = None
    val_subjects: Optional[list[int]] = None
    test_subjects: Optional[list[int]] = None

def split_index(index: List[Sample], spec: SplitSpec) -> Dict[str, List[Sample]]:
    train_samples = []
    val_samples = []
    test_samples = []

    if spec.group_by == "subject":
        # Pre-convert to sets for O(1) lookup
        train_s = set(spec.train_subjects or [])
        val_s   = set(spec.val_subjects or [])
        test_s  = set(spec.test_subjects or [])

        for s in index:
            # Sample: (data_path, subject_id, trial_id, repetition_id, start, end，label)
            sub_id = s[1]
            if sub_id in train_s:
                train_samples.append(s)
            elif sub_id in val_s:
                val_samples.append(s)
            elif sub_id in test_s:
                test_samples.append(s)
    
    elif spec.group_by == "repetition":
        train_r = set(spec.train_trials or [])
        val_r   = set(spec.val_trials or [])
        test_r  = set(spec.test_trials or [])

        for s in index:
            rep_id = s[3] # repetition_id
            if rep_id in train_r:
                train_samples.append(s)
            elif rep_id in val_r:
                val_samples.append(s)
            elif rep_id in test_r:
                test_samples.append(s)

    else:
        raise ValueError(f"Unknown group_by={spec.group_by}")

    return {"train": train_samples, "val": val_samples, "test": test_samples}
