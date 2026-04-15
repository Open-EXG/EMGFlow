"""DB6 dataset loader with session-based split support."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


Sample = Tuple[str, int, int, int, int, int, int]

DB6_RAW_LABELS: Tuple[int, ...] = (1, 3, 4, 6, 9, 10, 11)
DB6_RAW_TO_CONTIG: Dict[int, int] = {raw: idx + 1 for idx, raw in enumerate(DB6_RAW_LABELS)}
DB6_CONTIG_TO_RAW: Dict[int, int] = {mapped: raw for raw, mapped in DB6_RAW_TO_CONTIG.items()}


def map_db6_raw_label(raw_label: int) -> int:
    if raw_label == 0:
        return 0
    if raw_label not in DB6_RAW_TO_CONTIG:
        raise ValueError(
            f"Unexpected DB6 raw label {raw_label}. "
            f"Expected one of {list(DB6_RAW_LABELS)} or 0."
        )
    return DB6_RAW_TO_CONTIG[raw_label]


@dataclass(frozen=True)
class DB6SplitSpec:
    group_by: str = "session"
    seed: int = 42
    train_sessions: Optional[List[int]] = None
    val_sessions: Optional[List[int]] = None
    test_sessions: Optional[List[int]] = None
    train_subjects: Optional[List[int]] = None
    val_subjects: Optional[List[int]] = None
    test_subjects: Optional[List[int]] = None


class NinaproDB6Simple(Dataset):
    EMPTY_CHANNELS = [8, 9]

    def __init__(
        self,
        root,
        subjects,
        window_size: int = 400,
        stride: int = 100,
        remove_transient: bool = True,
        norm_mode: str = "per_channel",
        use_cache: bool = True,
        wavelet_level=None,
        norm_method: str = "zscore",
        samples=None,
        mean=None,
        std=None,
        denoise: bool = False,
        use_disk_cache: bool = True,
        split_spec: DB6SplitSpec | None = None,
    ):
        self.root = Path(root)
        self.subjects = subjects
        self.window_size = window_size
        self.stride = stride
        self.remove_transient = remove_transient
        self.norm_mode = norm_mode
        self.use_cache = use_cache
        self.wavelet_level = wavelet_level
        self.use_disk_cache = use_disk_cache
        self.denoise = denoise
        self.norm_method = norm_method
        self.mean = mean
        self.std = std
        self._cache = {}

        if samples is None:
            self.samples = []
            self._build_index()
        else:
            self.samples = samples

        if self.norm_method == "zscore" and self.mean is None and self.std is None:
            if split_spec is None:
                raise ValueError(
                    "DB6 zscore normalization needs split_spec so mean/std can be estimated from train sessions."
                )
            self.compute_mean_std(split_spec)

    def _iter_session_files(self, subj_dir: Path):
        session_id = 0
        for day_idx in range(1, 6):
            day_dir = subj_dir / f"day{day_idx}"
            if not day_dir.exists():
                continue
            for session_file in sorted(day_dir.glob("session*.npy")):
                session_id += 1
                yield session_file, day_idx, session_id

    def _build_index(self):
        for subj in self.subjects:
            subj_dir = self.root / f"Subject{subj:02d}"
            if not subj_dir.exists():
                continue

            for session_path, _day_id, global_session_id in self._iter_session_files(subj_dir):
                data = np.load(session_path, allow_pickle=True).item()
                emg_trials = data["emg"]
                labels = data["label"]
                meta = data.get("meta", {})
                total_trials = meta.get("total_trials", len(labels)) if isinstance(meta, dict) else len(labels)
                fs = int(meta.get("fs", 2000)) if isinstance(meta, dict) else 2000
                transient_margin = int(1.5 * fs) if self.remove_transient else 0

                for trial_id in range(total_trials):
                    y_raw = int(labels[trial_id])
                    if y_raw == 0:
                        continue
                    y = map_db6_raw_label(y_raw)

                    trial = emg_trials[trial_id].astype(np.float32)
                    t_total = trial.shape[0]
                    valid_start = transient_margin
                    valid_end = t_total - transient_margin
                    valid_len = valid_end - valid_start
                    if valid_len < self.window_size:
                        continue

                    for start in range(valid_start, valid_end - self.window_size + 1, self.stride):
                        end = start + self.window_size
                        self.samples.append(
                            (str(session_path), subj, global_session_id, trial_id, start, end, y)
                        )

        print(f"Total indexed real windows (before split): {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def _load_file(self, npy_path):
        if npy_path in self._cache:
            return self._cache[npy_path]

        path = Path(npy_path)
        cache_path = path.parent / f".cache_{path.stem}.npy"
        if self.use_disk_cache and cache_path.exists():
            try:
                data = np.load(cache_path, allow_pickle=True).item()
                if self.use_cache:
                    self._cache[npy_path] = data
                return data
            except Exception as exc:
                print(f"Warning: failed to load DB6 cache {cache_path}: {exc}")

        data = np.load(npy_path, allow_pickle=True).item()
        raw_emg = data["emg"]
        n_trials = len(raw_emg)
        processed_emg = np.empty(n_trials, dtype=object)
        valid_channels = [c for c in range(16) if c not in self.EMPTY_CHANNELS]

        for idx in range(n_trials):
            trial = raw_emg[idx].astype(np.float32)
            processed_emg[idx] = trial[:, valid_channels]

        data["emg"] = processed_emg

        if self.use_disk_cache:
            try:
                np.save(cache_path, data)
            except Exception as exc:
                print(f"Failed to save DB6 cache {cache_path}: {exc}")

        if self.use_cache:
            self._cache[npy_path] = data
        return data

    def __getitem__(self, idx):
        npy_path, subject_id, _session_id, trial_id, start, end, y = self.samples[idx]
        data = self._load_file(npy_path)
        emg_trial = data["emg"][trial_id]
        x = emg_trial[start:end]
        x = torch.from_numpy(x).float().transpose(0, 1)

        if self.norm_method == "zscore":
            if isinstance(self.mean, dict) and subject_id in self.mean:
                mean = self.mean[subject_id]
                std = self.std[subject_id]
                x = (x - mean.to(x.device)) / (std.to(x.device) + 1e-8)
            elif self.mean is not None and self.std is not None:
                x = (x - self.mean.to(x.device)) / (self.std.to(x.device) + 1e-8)
            else:
                raise ValueError("Mean and std must be provided for z-score normalization.")
        elif self.norm_method == "per_sample_zscore":
            if self.norm_mode == "per_channel":
                mean = x.mean(dim=1, keepdim=True)
                std = x.std(dim=1, keepdim=True)
                x = (x - mean) / (std + 1e-8)
            elif self.norm_mode == "global":
                mean = x.mean()
                std = x.std()
                x = (x - mean) / (std + 1e-8)

        return x, y

    def split(self, spec: DB6SplitSpec) -> Dict[str, List[Sample]]:
        train_samples = []
        val_samples = []
        test_samples = []

        if spec.group_by == "session":
            train_sessions = set(spec.train_sessions or [])
            val_sessions = set(spec.val_sessions or [])
            test_sessions = set(spec.test_sessions or [])

            for sample in self.samples:
                session_id = sample[2]
                if session_id in train_sessions:
                    train_samples.append(sample)
                elif session_id in val_sessions:
                    val_samples.append(sample)
                elif session_id in test_sessions:
                    test_samples.append(sample)
        elif spec.group_by == "subject":
            train_subjects = set(spec.train_subjects or [])
            val_subjects = set(spec.val_subjects or [])
            test_subjects = set(spec.test_subjects or [])

            for sample in self.samples:
                subject_id = sample[1]
                if subject_id in train_subjects:
                    train_samples.append(sample)
                elif subject_id in val_subjects:
                    val_samples.append(sample)
                elif subject_id in test_subjects:
                    test_samples.append(sample)
        else:
            raise ValueError(f"Unknown group_by={spec.group_by}")

        return {"train": train_samples, "val": val_samples, "test": test_samples}

    def subset(self, samples, **kwargs):
        params = {
            "root": self.root,
            "subjects": self.subjects,
            "window_size": self.window_size,
            "stride": self.stride,
            "remove_transient": self.remove_transient,
            "norm_mode": self.norm_mode,
            "use_cache": self.use_cache,
            "wavelet_level": self.wavelet_level,
            "norm_method": self.norm_method,
            "samples": samples,
            "mean": self.mean,
            "std": self.std,
            "denoise": self.denoise,
            "use_disk_cache": self.use_disk_cache,
            "split_spec": None,
        }
        params.update(kwargs)
        return NinaproDB6Simple(**params)

    def compute_mean_std(self, spec: DB6SplitSpec):
        print("Computing mean and std from training set (Per-Channel, Per-Subject)...")
        splits = self.split(spec)
        train_samples = splits["train"]

        self.mean = {}
        self.std = {}
        subjects_in_train = sorted(list(set(sample[1] for sample in train_samples)))

        for subj in subjects_in_train:
            print(f"  Processing Subject {subj}...")
            subj_samples = [sample for sample in train_samples if sample[1] == subj]
            temp_ds = self.subset(subj_samples, norm_method=None)
            loader = DataLoader(temp_ds, batch_size=256, shuffle=False, num_workers=4)

            count = 0
            sum_x = None
            sum_sq_x = None
            for x_batch, _ in loader:
                x = x_batch.to("cpu")
                batch_size, channels, length = x.shape
                flat_x = x.permute(1, 0, 2).reshape(channels, -1)

                if sum_x is None:
                    sum_x = torch.zeros(channels)
                    sum_sq_x = torch.zeros(channels)

                sum_x += flat_x.sum(dim=1)
                sum_sq_x += (flat_x ** 2).sum(dim=1)
                count += batch_size * length

            if count == 0:
                self.mean[subj] = torch.zeros(14, 1)
                self.std[subj] = torch.ones(14, 1)
            else:
                mean = sum_x / count
                std = torch.sqrt(torch.clamp(sum_sq_x / count - mean ** 2, min=1e-9))
                self.mean[subj] = mean.unsqueeze(1)
                self.std[subj] = std.unsqueeze(1)

        return self.mean, self.std
