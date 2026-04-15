import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from emgflow.datasets.utils.transforms import minmax_norm, wavelet_denoise_db7
from emgflow.datasets.utils.split import split_index, SplitSpec
from torch.utils.data import DataLoader
# ===== DB2 Dataset =====
#40 subjects sampling at 2kHz 49 kinds of gestures(rest not included) label range 1-49
#sample format：（data_path, subject_id, trial_id, repetition_id, start, end, label）
class NinaproDB2Simple(Dataset):
    """
    Sliding window dataset for your preprocessed DB2_npy with Disk Caching.

    Return:
        x: Tensor (12, window_size)
        y: int (gesture label, DB2 original index)
    """

    def __init__(
        self,
        root,
        subjects,
        window_size=2000,
        stride=2000,
        norm_mode="per_channel",
        use_cache=True,
        wavelet_level=None,
        norm_method="zscore",#or Minmax or None or per_sample_zscore
        samples = None,
        mean = None,
        std = None,
        denoise = True,
        use_disk_cache = True, # New Feature
        split_spec: SplitSpec = None
    ):

        self.root = Path(root)
        self.subjects = subjects
        self.window_size = window_size
        self.stride = stride
        self.norm_mode = norm_mode
        self.use_cache = use_cache
        self.wavelet_level = wavelet_level
        self.use_disk_cache = use_disk_cache
        self.denoise = denoise

        if samples is None:
            self.samples = []
            self._build_index()
        else:
            self.samples = samples
        self._cache = {}
        self.norm_method = norm_method

        self.mean = mean
        self.std = std
        
        # Auto-compute stats if needed
        if self.norm_method == "zscore":
            if self.mean is None and self.std is None:
                if split_spec is not None:
                    self.compute_mean_std(split_spec)
                else:
                    raise ValueError("For zscore normalization, 'split_spec' must be provided to compute statistics on training set (or provide 'mean' and 'std' manually).")




    # ---------------------------------------------------
    # 扫描所有 trial → 建立滑窗索引
    # ---------------------------------------------------

    def _build_index(self):

        for subj in self.subjects:

            subj_dir = self.root / f"Subject{subj:02d}"

            if not subj_dir.exists():
                continue

            for npy_file in sorted(subj_dir.glob("*.npy")):

                if npy_file.name.startswith("."):
                    continue

                data = np.load(npy_file, allow_pickle=True).item()

                emg_trials = data["emg"]
                labels = data["label"]
                repetition = data["repetition"]
                trials = data['meta']["total_trials"]

                for trial_id in range(trials):

                    y = int(labels[trial_id])

                    # DB2 中 0 是 rest
                    if y == 0:
                        continue

                    emg = emg_trials[trial_id]
                    T = emg.shape[0]

                    L = self.window_size
                    S = self.stride

                    if T < L:
                        continue
                    repetition_id = repetition[trial_id]
                    for start in range(0, T - L + 1, S):
                        end = start + L

                        self.samples.append(
                            (str(npy_file),subj, trial_id, repetition_id, start, end,y)
                        )

        print(f"Total indexed real windows (before split): {len(self.samples)}")

    # ---------------------------------------------------

    def __len__(self):
        return len(self.samples)

    # ---------------------------------------------------

    def _load_file(self, npy_path):
        if npy_path in self._cache:
            return self._cache[npy_path]

        # 1. Try Loading from Disk Cache (Processed)
        p = Path(npy_path)
        level_suffix = f"_lvl{self.wavelet_level}" if self.denoise and self.wavelet_level is not None else ""
        cache_name = f".cache_{p.stem}{level_suffix}.npy"
        cache_path = p.parent / cache_name

        if self.use_disk_cache and cache_path.exists():
            try:
                data = np.load(cache_path, allow_pickle=True).item()
                if self.use_cache:
                    self._cache[npy_path] = data
                return data
            except Exception as e:
                print(f"Warning: Failed to load cache {cache_path}: {e}")

        # 2. Slow Path: Load Original & Process
        data = np.load(npy_path, allow_pickle=True).item()
        
        # Handle jagged arrays (variable length trials)
        raw_emg = data["emg"] # object array (N_trials,)
        n_trials = len(raw_emg)
        
        # Prepare container for processed trials
        processed_emg = np.empty(n_trials, dtype=object)

        for i in range(n_trials):
            # Load trial and ensure float32
            trial_data = raw_emg[i].astype(np.float32)

            if self.denoise:
                T, C = trial_data.shape
                trial_dn = np.empty_like(trial_data)
                # apply wavelet denoising channel-wise
                for c in range(C):
                    trial_dn[:, c] = wavelet_denoise_db7(trial_data[:, c], level=self.wavelet_level)
                processed_emg[i] = trial_dn
            else:
                processed_emg[i] = trial_data
        
        data["emg"] = processed_emg
        
        # 3. Save to Disk Cache
        if self.use_disk_cache:
            try:
                np.save(cache_path, data)
            except Exception as e:
                print(f"Failed to save DB2 cache {cache_path}: {e}")

        if self.use_cache:
            self._cache[npy_path] = data

        return data

    # ---------------------------------------------------

    def __getitem__(self, idx):

        npy_path, subject_id, trial_id, repetition_id, start, end, y = self.samples[idx]

        data = self._load_file(npy_path)

        emg_trial = data["emg"][trial_id]        # (T,12)
        label = y

        x = emg_trial[start:end]                 # (L,12)

        # -> torch (C,L)
        x = torch.from_numpy(x).float().transpose(0, 1)
        if self.norm_method=="minmax":
            x = minmax_norm(x, mode=self.norm_mode)
        if self.norm_method=="zscore":
            if isinstance(self.mean, dict) and subject_id in self.mean:
                m = self.mean[subject_id]
                s = self.std[subject_id]
                x = (x - m.to(x.device)) / (s.to(x.device) + 1e-8)
            elif self.mean is not None and self.std is not None:
                x = (x - self.mean.to(x.device)) / (self.std.to(x.device) + 1e-8)
            elif self.mean is None or self.std is None:
                raise ValueError("Mean and std must be provided for z-score normalization.")
        if self.norm_method is None:
            pass
        if self.norm_method=="per_sample_zscore":
            if self.norm_mode=="per_channel":
                m = x.mean(dim=1, keepdim=True)
                s = x.std(dim=1, keepdim=True)
                x = (x - m) / (s + 1e-8)
            elif self.norm_mode=="global":   
                m = x.mean()
                s = x.std()
                x = (x - m) / (s + 1e-8)
        return x, label
    # ---------------------------------------------------  
    #split
    # ---------------------------------------------------
    def split(self, spec: SplitSpec):
        return split_index(self.samples, spec)
    
    def subset(self, samples, **kwargs):
        params = {
            "root": self.root,
            "subjects": self.subjects,
            "window_size": self.window_size,
            "stride": self.stride,
            "norm_mode": self.norm_mode,
            "use_cache": self.use_cache,
            "wavelet_level": self.wavelet_level,
            "norm_method": self.norm_method,
            "samples": samples,
            "mean": self.mean,
            "std": self.std,
            "denoise": self.denoise,
            "use_disk_cache": self.use_disk_cache,
            "split_spec": None
        }
        params.update(kwargs)
        return NinaproDB2Simple(**params)
    # ---------------------------------------------------
    #get mean and std from train set
    # ---------------------------------------------------
    def compute_mean_std(self, spec: SplitSpec):
        print("Computing mean and std from training set (Per-Subject)...")
        splits = self.split(spec)
        train_samples = splits["train"]
        
        self.mean = {}
        self.std = {}
        
        # Get unique subjects in training set
        subjects_in_train = sorted(list(set(s[1] for s in train_samples)))
        
        for subj in subjects_in_train:
            print(f"  Processing Subject {subj}...")
            subj_samples = [s for s in train_samples if s[1] == subj]
            
            # Use a temporary dataset with NO normalization to compute raw stats
            temp_ds = self.subset(subj_samples, norm_method=None)
            
            loader = DataLoader(temp_ds, batch_size=256, shuffle=False, num_workers=4)
            
            cnt = 0
            sum_x = None
            sum_sq_x = None
            device = torch.device('cpu')
            
            for x_batch, _ in loader:
                x = x_batch.to(device)
                B, C, L = x.shape
                flat_x = x.permute(1, 0, 2).reshape(C, -1)
                
                if sum_x is None:
                    sum_x = torch.zeros(C, device=device)
                    sum_sq_x = torch.zeros(C, device=device)
                    
                sum_x += flat_x.sum(dim=1)
                sum_sq_x += (flat_x ** 2).sum(dim=1)
                cnt += B * L
                
            if cnt == 0:
                self.mean[subj] = torch.zeros(12, 1)
                self.std[subj] = torch.ones(12, 1)
            else:
                m = sum_x / cnt
                s = torch.sqrt(torch.clamp(sum_sq_x / cnt - m ** 2, min=1e-9))
                self.mean[subj] = m.unsqueeze(1)
                self.std[subj] = s.unsqueeze(1)
                print(f"    Subject {subj} Mean (first 3): {self.mean[subj][:3].reshape(-1)}")
            
        return self.mean, self.std
