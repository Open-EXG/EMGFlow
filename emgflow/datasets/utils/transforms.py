from typing import Callable, List, Tuple
import torch
import numpy as np
import pywt

class Compose:
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, x, y):
        for t in self.transforms:
            x, y = t(x, y)
        return x, y


class ToFloat32:
    def __call__(self, x, y):
        if isinstance(x, torch.Tensor):
            return x.float(), y
        return torch.from_numpy(x).float(), y



class ZScoreNormalize:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean  # (C,1)
        self.std = std    # (C,1)

    def __call__(self, x, y):
        # x: (C,L)
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        x = (x - mean) / std
        return x, y


def wavelet_denoise_db7(x_1d: np.ndarray, level: int = None) -> np.ndarray:
    """
    Symlet-8 wavelet denoising for 1D signal.
    Uses universal threshold with MAD sigma estimation on the highest-frequency detail coeffs.
    """
    x_1d = np.asarray(x_1d, dtype=np.float32)
    w = pywt.Wavelet("db7")

    if level is None:
        # conservative: auto max level but cap to avoid over-smoothing
        max_level = pywt.dwt_max_level(len(x_1d), w.dec_len)
        level = min(max_level, 5) if max_level > 0 else 1

    coeffs = pywt.wavedec(x_1d, w, level=level)

    # Estimate noise sigma from the finest-scale detail coefficients (last entry)
    detail = coeffs[-1]
    if detail.size == 0:
        return x_1d

    sigma = np.median(np.abs(detail)) / 0.6745 if np.any(detail) else 0.0
    if sigma <= 0:
        return x_1d

    thr = sigma * np.sqrt(2.0 * np.log(len(x_1d) + 1.0))

    # Threshold detail coeffs (leave approximation coeffs untouched)
    coeffs_denoised = [coeffs[0]]
    for cD in coeffs[1:]:
        coeffs_denoised.append(pywt.threshold(cD, thr, mode="soft"))

    x_rec = pywt.waverec(coeffs_denoised, w)

    # waverec may return slightly longer due to padding
    x_rec = x_rec[: len(x_1d)].astype(np.float32, copy=False)
    return x_rec

def minmax_norm(x: torch.Tensor, mode: str = "per_channel", eps: float = 1e-8) -> torch.Tensor:
    """
    x: (C, L)
    mode:
      - "per_channel": each channel scaled to [0, 1]
      - "global": whole sample scaled to [0, 1]
    """
    if mode == "per_channel":
        x_min = x.amin(dim=1, keepdim=True)
        x_max = x.amax(dim=1, keepdim=True)
    elif mode == "global":
        x_min = x.amin()
        x_max = x.amax()
    else:
        raise ValueError(f"Unknown minmax mode: {mode}")

    denom = (x_max - x_min).clamp_min(eps)
    return (x - x_min) / denom


def z_score_norm(x: torch.Tensor, mode: str) -> torch.Tensor:
    """
    x: (C, L)
    mode:
      - "per_channel": each channel normalized to zero-mean, unit-std
      - "global": whole sample normalized to zero-mean, unit-std
    """
    if mode == "per_channel":
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
    elif mode == "global":
        mean = x.mean()
        std = x.std()
    else:
        raise ValueError(f"Unknown z-score mode: {mode}")

    std = std.clamp_min(1e-8)
    return (x - mean) / std