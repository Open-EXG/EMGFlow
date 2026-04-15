from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch

TensorLike = torch.Tensor | np.ndarray
NormalizeMode = Literal[None, "l2", "zscore"]


def _resolve_device(feat_real: TensorLike, feat_gen: TensorLike) -> torch.device:
    if torch.is_tensor(feat_real):
        return feat_real.device
    if torch.is_tensor(feat_gen):
        return feat_gen.device
    return torch.device("cpu")


def _as_2d_tensor(x: TensorLike, name: str, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(x):
        t = x.to(device=device)
    elif isinstance(x, np.ndarray):
        t = torch.from_numpy(x).to(device=device)
    else:
        raise TypeError(f"{name} must be a torch.Tensor or np.ndarray, got {type(x)}")
    if t.ndim != 2:
        raise ValueError(f"{name} must be 2D with shape (N, D), got shape={tuple(t.shape)}")
    return t


def _as_1d_label_tensor(x: TensorLike, name: str, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(x):
        t = x.to(device=device)
    elif isinstance(x, np.ndarray):
        t = torch.from_numpy(x).to(device=device)
    else:
        raise TypeError(f"{name} must be a torch.Tensor or np.ndarray, got {type(x)}")
    if t.ndim != 1:
        raise ValueError(f"{name} must be 1D with shape (N,), got shape={tuple(t.shape)}")
    return t.long()


def _normalize_features(
    feat_real: torch.Tensor,
    feat_gen: torch.Tensor,
    normalize_mode: NormalizeMode,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if normalize_mode is None:
        return feat_real, feat_gen

    mode = str(normalize_mode).lower()
    if mode == "l2":
        real_norm = feat_real.norm(p=2, dim=1, keepdim=True).clamp_min(eps)
        gen_norm = feat_gen.norm(p=2, dim=1, keepdim=True).clamp_min(eps)
        return feat_real / real_norm, feat_gen / gen_norm

    if mode == "zscore":
        mu = feat_real.mean(dim=0, keepdim=True)
        std = feat_real.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
        return (feat_real - mu) / std, (feat_gen - mu) / std

    raise ValueError(f"Unsupported normalize_mode={normalize_mode}, expected None|'l2'|'zscore'.")


def polynomial_kernel(
    x: torch.Tensor,
    y: torch.Tensor,
    degree: int = 3,
    gamma: float | None = None,
    coef0: float = 1.0,
) -> torch.Tensor:
    """
    Polynomial kernel: k(x, y) = (gamma * x^T y + coef0)^degree.

    Args:
        x: Tensor (N, D)
        y: Tensor (M, D)
        degree: Polynomial degree, default 3.
        gamma: If None, uses 1.0 / D.
        coef0: Additive constant, default 1.0.

    Returns:
        Kernel matrix with shape (N, M).
    """
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f"x and y must be 2D tensors, got x.ndim={x.ndim}, y.ndim={y.ndim}")
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"x and y must share feature dimension D, got {x.shape[1]} vs {y.shape[1]}")
    if degree <= 0:
        raise ValueError(f"degree must be positive, got {degree}")

    d = int(x.shape[1])
    gamma_value = (1.0 / float(d)) if gamma is None else float(gamma)
    return (gamma_value * (x @ y.t()) + float(coef0)) ** int(degree)


def mmd2_unbiased(k_xx: torch.Tensor, k_yy: torch.Tensor, k_xy: torch.Tensor) -> torch.Tensor:
    """
    Unbiased estimator of squared MMD:

      MMD^2_unbiased =
          sum_{i!=j} K_xx[i,j] / (n(n-1))
        + sum_{i!=j} K_yy[i,j] / (m(m-1))
        - 2 * sum(K_xy) / (nm)

    Notes:
        - The result can be slightly negative due to unbiased finite-sample estimation.
        - Do NOT clamp negative values to zero.
    """
    if k_xx.ndim != 2 or k_yy.ndim != 2 or k_xy.ndim != 2:
        raise ValueError("k_xx, k_yy, k_xy must all be 2D matrices.")
    if k_xx.shape[0] != k_xx.shape[1]:
        raise ValueError(f"k_xx must be square, got shape={tuple(k_xx.shape)}")
    if k_yy.shape[0] != k_yy.shape[1]:
        raise ValueError(f"k_yy must be square, got shape={tuple(k_yy.shape)}")
    n = int(k_xx.shape[0])
    m = int(k_yy.shape[0])
    if k_xy.shape != (n, m):
        raise ValueError(f"k_xy shape must be (n, m)=({n}, {m}), got {tuple(k_xy.shape)}")
    if n < 2 or m < 2:
        raise ValueError(f"Need n>=2 and m>=2 for unbiased MMD^2, got n={n}, m={m}")

    sum_xx_offdiag = (k_xx.sum() - torch.diagonal(k_xx).sum()) / (n * (n - 1))
    sum_yy_offdiag = (k_yy.sum() - torch.diagonal(k_yy).sum()) / (m * (m - 1))
    sum_xy = k_xy.sum() / (n * m)
    return sum_xx_offdiag + sum_yy_offdiag - 2.0 * sum_xy


def compute_kid(
    feat_real: TensorLike,
    feat_gen: TensorLike,
    subset_size: int = 1000,
    n_subsets: int = 100,
    degree: int = 3,
    gamma: float | None = None,
    coef0: float = 1.0,
    normalize_mode: NormalizeMode = None,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """
    Compute KID (Kernel Inception Distance) from generic feature tensors.

    This implementation is feature-extractor agnostic (no Inception dependency) and
    follows the standard KID practice:
      1) use polynomial kernel in feature space,
      2) estimate unbiased MMD^2 on repeated random subsets,
      3) return subset-level mean/std.

    Args:
        feat_real: Real features with shape (N, D).
        feat_gen: Generated features with shape (M, D).
        subset_size: Subset size per repeat. Effective size is min(subset_size, N, M).
        n_subsets: Number of repeated random subsets.
        degree: Polynomial kernel degree (default 3).
        gamma: Kernel gamma. If None, uses 1.0 / D.
        coef0: Polynomial kernel bias term (default 1.0).
        normalize_mode: None | "l2" | "zscore".
        eps: Numerical epsilon used by normalization.

    Returns:
        Dict with:
          - kid_mean: float
          - kid_std: float
          - kid_scores: Tensor[n_subsets] (float64, CPU)
    """
    if n_subsets < 1:
        raise ValueError(f"n_subsets must be >=1, got {n_subsets}")

    device = _resolve_device(feat_real, feat_gen)
    x_real = _as_2d_tensor(feat_real, "feat_real", device=device).to(dtype=torch.float64)
    x_gen = _as_2d_tensor(feat_gen, "feat_gen", device=device).to(dtype=torch.float64)

    if x_real.shape[1] != x_gen.shape[1]:
        raise ValueError(f"Feature dimensions must match, got {x_real.shape[1]} and {x_gen.shape[1]}")

    n = int(x_real.shape[0])
    m = int(x_gen.shape[0])
    if n < 2 or m < 2:
        raise ValueError(f"Need at least 2 samples in each set, got n={n}, m={m}")

    x_real, x_gen = _normalize_features(x_real, x_gen, normalize_mode=normalize_mode, eps=eps)

    eff_subset = int(min(int(subset_size), n, m))
    if eff_subset < 2:
        raise ValueError(
            f"Effective subset_size is {eff_subset}, need at least 2. "
            f"Got subset_size={subset_size}, n={n}, m={m}."
        )

    d = int(x_real.shape[1])
    gamma_value = (1.0 / float(d)) if gamma is None else float(gamma)

    kid_scores = torch.empty(int(n_subsets), dtype=torch.float64, device="cpu")
    for s in range(int(n_subsets)):
        idx_real = torch.randperm(n, device=device)[:eff_subset]
        idx_gen = torch.randperm(m, device=device)[:eff_subset]
        xr = x_real.index_select(dim=0, index=idx_real)
        xg = x_gen.index_select(dim=0, index=idx_gen)

        k_xx = polynomial_kernel(xr, xr, degree=degree, gamma=gamma_value, coef0=coef0)
        k_yy = polynomial_kernel(xg, xg, degree=degree, gamma=gamma_value, coef0=coef0)
        k_xy = polynomial_kernel(xr, xg, degree=degree, gamma=gamma_value, coef0=coef0)
        kid_scores[s] = mmd2_unbiased(k_xx, k_yy, k_xy).detach().cpu()

    return {
        "kid_mean": float(kid_scores.mean().item()),
        "kid_std": float(kid_scores.std(unbiased=False).item()),
        "kid_scores": kid_scores,
    }


def compute_classwise_kid(
    feat_real: TensorLike,
    label_real: TensorLike,
    feat_gen: TensorLike,
    label_gen: TensorLike,
    subset_size: int = 1000,
    n_subsets: int = 100,
    degree: int = 3,
    gamma: float | None = None,
    coef0: float = 1.0,
    normalize_mode: NormalizeMode = None,
    eps: float = 1e-12,
    classes: list[int] | None = None,
) -> dict[str, Any]:
    """
    Compute class-conditional KID and macro average.

    For each class c, compute KID(real[c], gen[c]) with repeated subsets.
    Classes with fewer than 2 real or generated samples are marked as insufficient.
    """
    device = _resolve_device(feat_real, feat_gen)
    x_real = _as_2d_tensor(feat_real, "feat_real", device=device)
    x_gen = _as_2d_tensor(feat_gen, "feat_gen", device=device)
    y_real = _as_1d_label_tensor(label_real, "label_real", device=device)
    y_gen = _as_1d_label_tensor(label_gen, "label_gen", device=device)

    if x_real.shape[0] != y_real.shape[0]:
        raise ValueError(f"feat_real and label_real size mismatch: {x_real.shape[0]} vs {y_real.shape[0]}")
    if x_gen.shape[0] != y_gen.shape[0]:
        raise ValueError(f"feat_gen and label_gen size mismatch: {x_gen.shape[0]} vs {y_gen.shape[0]}")

    if classes is None:
        classes_real = set(torch.unique(y_real).detach().cpu().tolist())
        classes_gen = set(torch.unique(y_gen).detach().cpu().tolist())
        class_list = sorted(int(c) for c in (classes_real & classes_gen))
    else:
        class_list = [int(c) for c in classes]

    if not class_list:
        raise ValueError("No classes available for class-wise KID computation.")

    per_class: dict[str, Any] = {}
    valid_means: list[float] = []
    valid_stds: list[float] = []
    for cls in class_list:
        mask_real = y_real == cls
        mask_gen = y_gen == cls
        n_real = int(mask_real.sum().item())
        n_gen = int(mask_gen.sum().item())

        class_key = str(cls)
        per_class[class_key] = {
            "num_real": n_real,
            "num_gen": n_gen,
            "subset_size_effective": int(min(subset_size, n_real, n_gen)),
            "kid_mean": float("nan"),
            "kid_std": float("nan"),
            "status": "ok",
        }

        if n_real < 2 or n_gen < 2:
            per_class[class_key]["status"] = "insufficient_samples"
            continue

        kid_res = compute_kid(
            feat_real=x_real[mask_real],
            feat_gen=x_gen[mask_gen],
            subset_size=subset_size,
            n_subsets=n_subsets,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            normalize_mode=normalize_mode,
            eps=eps,
        )
        per_class[class_key]["kid_mean"] = float(kid_res["kid_mean"])
        per_class[class_key]["kid_std"] = float(kid_res["kid_std"])
        valid_means.append(float(kid_res["kid_mean"]))
        valid_stds.append(float(kid_res["kid_std"]))

    if valid_means:
        means_tensor = torch.tensor(valid_means, dtype=torch.float64)
        stds_tensor = torch.tensor(valid_stds, dtype=torch.float64)
        macro_mean = float(means_tensor.mean().item())
        macro_std = float(means_tensor.std(unbiased=False).item())
        macro_within_std = float(stds_tensor.mean().item())
    else:
        macro_mean = float("nan")
        macro_std = float("nan")
        macro_within_std = float("nan")

    return {
        "classwise": per_class,
        "macro_kid_mean": macro_mean,
        "macro_kid_std": macro_std,
        "macro_kid_within_std_mean": macro_within_std,
        "valid_classes": int(len(valid_means)),
        "total_classes": int(len(class_list)),
        "classes": class_list,
    }


if __name__ == "__main__":
    # Minimal runnable example
    torch.manual_seed(7)
    n_real, n_gen, d, n_cls = 2048, 2048, 256, 8
    real_feat = torch.randn(n_real, d)
    gen_feat = real_feat[:n_gen] + 0.05 * torch.randn(n_gen, d)
    real_lbl = torch.randint(low=0, high=n_cls, size=(n_real,))
    gen_lbl = torch.randint(low=0, high=n_cls, size=(n_gen,))

    kid = compute_kid(
        feat_real=real_feat,
        feat_gen=gen_feat,
        subset_size=512,
        n_subsets=20,
        degree=3,
        gamma=None,  # defaults to 1.0 / D
        coef0=1.0,
        normalize_mode="l2",
    )
    print(f"KID mean={kid['kid_mean']:.6f}, std={kid['kid_std']:.6f}")

    cls_kid = compute_classwise_kid(
        feat_real=real_feat,
        label_real=real_lbl,
        feat_gen=gen_feat,
        label_gen=gen_lbl,
        subset_size=256,
        n_subsets=10,
        normalize_mode="zscore",
    )
    print(
        "Class-wise KID: "
        f"macro_mean={cls_kid['macro_kid_mean']:.6f}, "
        f"macro_std={cls_kid['macro_kid_std']:.6f}, "
        f"valid_classes={cls_kid['valid_classes']}/{cls_kid['total_classes']}"
    )
