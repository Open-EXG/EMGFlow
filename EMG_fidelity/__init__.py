from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

from .kid_metrics import compute_classwise_kid, compute_kid
from .metrics import (
    calculate_frechet_distance,
    calculate_inception_score,
    extract_activations,
    get_activations,
)
from .models import ResNet18_1D
from .robustmodel import Crossformer1D, EMGHandNet1D, SimpleConvNet

ClassifierBuilder = Callable[[int], nn.Module]
MetricFn = Callable[["EMG_Fidelity_Evaluator", dict[str, Any] | None, dict[str, Any] | None, dict[str, Any]], dict[str, Any]]

_CLASSIFIER_REGISTRY: dict[str, ClassifierBuilder] = {}
_METRIC_REGISTRY: dict[str, "MetricSpec"] = {}

_DATASET_NUM_CLASSES = {
    "NinaproDB2": 49,
    "NinaproDB7": 40,
    "NinaproDB4": 52,
    "NinaproDB6": 7,
}

DEFAULT_METRICS = ["fid", "is", "kid"]
DEFAULT_KID_CONFIG = {
    "subset_size": 5000,
    "n_subsets": 100,
    "degree": 3,
    "gamma": None,
    "coef0": 1.0,
    "normalize_mode": None,
    "eps": 1e-12,
}
DEFAULT_PRDC_CONFIG = {
    "nearest_k": 5,
}


@dataclass(frozen=True)
class MetricSpec:
    name: str
    fn: MetricFn
    need_real_features: bool = False
    need_fake_features: bool = False
    need_real_logits: bool = False
    need_fake_logits: bool = False
    need_real_probs: bool = False
    need_fake_probs: bool = False
    need_real_labels: bool = False
    need_fake_labels: bool = False


def _normalize_classifier_name(name: str) -> str:
    return str(name).strip().lower()


def _normalize_metric_name(name: str) -> str:
    key = str(name).strip().lower()
    alias_map = {
        "precision": "pr",
        "recall": "re",
        "improved_precision": "pr",
        "improved_recall": "re",
    }
    return alias_map.get(key, key)


def register_classifier(name: str, builder: ClassifierBuilder, overwrite: bool = False) -> None:
    key = _normalize_classifier_name(name)
    if not key:
        raise ValueError("Classifier name cannot be empty.")
    if (key in _CLASSIFIER_REGISTRY) and (not overwrite):
        raise ValueError(f"Classifier '{key}' already registered. Use overwrite=True to replace it.")
    _CLASSIFIER_REGISTRY[key] = builder


def available_classifiers() -> list[str]:
    return sorted(_CLASSIFIER_REGISTRY.keys())


def register_metric(
    name: str,
    fn: MetricFn,
    *,
    need_real_features: bool = False,
    need_fake_features: bool = False,
    need_real_logits: bool = False,
    need_fake_logits: bool = False,
    need_real_probs: bool = False,
    need_fake_probs: bool = False,
    need_real_labels: bool = False,
    need_fake_labels: bool = False,
    overwrite: bool = False,
) -> None:
    key = _normalize_metric_name(name)
    if not key:
        raise ValueError("Metric name cannot be empty.")
    if (key in _METRIC_REGISTRY) and (not overwrite):
        raise ValueError(f"Metric '{key}' already registered. Use overwrite=True to replace it.")
    _METRIC_REGISTRY[key] = MetricSpec(
        name=key,
        fn=fn,
        need_real_features=bool(need_real_features),
        need_fake_features=bool(need_fake_features),
        need_real_logits=bool(need_real_logits),
        need_fake_logits=bool(need_fake_logits),
        need_real_probs=bool(need_real_probs),
        need_fake_probs=bool(need_fake_probs),
        need_real_labels=bool(need_real_labels),
        need_fake_labels=bool(need_fake_labels),
    )


def available_metrics() -> list[str]:
    return sorted(_METRIC_REGISTRY.keys())


def _register_builtin_classifiers() -> None:
    register_classifier(
        "resnet18",
        lambda num_classes: ResNet18_1D(input_channels=12, num_classes=num_classes),
        overwrite=True,
    )
    register_classifier(
        "simpleconvnet",
        lambda num_classes: SimpleConvNet(num_classes=num_classes),
        overwrite=True,
    )
    register_classifier(
        "emghandnet",
        lambda num_classes: EMGHandNet1D(num_classes=num_classes),
        overwrite=True,
    )
    register_classifier(
        "crossformer",
        lambda num_classes: Crossformer1D(num_classes=num_classes, in_ch=12, in_len=400),
        overwrite=True,
    )


def _dataset_input_channels(dataset_name: str) -> int:
    return 14 if dataset_name == "NinaproDB6" else 12


class _ClassifierOutputAdapter(nn.Module):
    """Normalize model outputs to (features, logits) for metrics pipelines."""

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model

    def forward(self, x: torch.Tensor):
        out = self.base_model(x)
        if isinstance(out, tuple):
            if len(out) < 2:
                raise ValueError("Classifier forward tuple output must contain at least (features, logits).")
            feats, logits = out[0], out[1]
        else:
            feats, logits = out, out

        if not torch.is_tensor(logits):
            raise TypeError("Classifier logits must be a torch.Tensor.")
        if not torch.is_tensor(feats):
            feats = logits
        return feats, logits


def _build_classifier_model(classifier_name: str, num_classes: int, device: str, dataset_name: str) -> nn.Module:
    key = _normalize_classifier_name(classifier_name)
    if key not in _CLASSIFIER_REGISTRY:
        raise ValueError(
            f"Unknown classifier '{classifier_name}'. "
            f"Available classifiers: {available_classifiers()}."
        )
    in_channels = _dataset_input_channels(dataset_name)
    if key == "resnet18":
        model = ResNet18_1D(input_channels=in_channels, num_classes=num_classes)
    elif key == "simpleconvnet":
        model = SimpleConvNet(num_classes=num_classes, in_channels=in_channels)
    elif key == "emghandnet":
        model = EMGHandNet1D(num_classes=num_classes, in_ch=in_channels)
    elif key == "crossformer":
        model = Crossformer1D(num_classes=num_classes, in_ch=in_channels, in_len=400)
    else:
        model = _CLASSIFIER_REGISTRY[key](num_classes)
    return _ClassifierOutputAdapter(model).to(device)


def _extract_state_dict_blob(state_obj):
    """Normalize different checkpoint container formats to a plain state_dict."""
    if isinstance(state_obj, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in state_obj and isinstance(state_obj[key], dict):
                return state_obj[key]
    return state_obj


def _align_state_dict_keys(model: nn.Module, state_dict: dict) -> dict:
    """
    Align checkpoint keys with current model keys.
    Handles wrapper prefix migration between:
    - bare model keys:      cnn.0.weight
    - wrapped model keys:   base_model.cnn.0.weight
    """
    if not isinstance(state_dict, dict):
        return state_dict

    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    if not ckpt_keys:
        return state_dict
    if ckpt_keys == model_keys:
        return state_dict

    if (not any(k.startswith("base_model.") for k in ckpt_keys)) and any(k.startswith("base_model.") for k in model_keys):
        prefixed = {f"base_model.{k}": v for k, v in state_dict.items()}
        if set(prefixed.keys()) & model_keys:
            return prefixed

    if any(k.startswith("base_model.") for k in ckpt_keys) and (not any(k.startswith("base_model.") for k in model_keys)):
        stripped = {k[len("base_model.") :]: v for k, v in state_dict.items() if k.startswith("base_model.")}
        if set(stripped.keys()) & model_keys:
            return stripped

    return state_dict


def _resolve_classifier_ckpt(
    dataset_name: str,
    classifier_name: str,
    classifier_ckpt_path: str | None = None,
    subject_id: int | None = None,
    ckpt_path: str | None = None,
) -> str:
    # Backward compatibility with older callsites using `ckpt_path`.
    if classifier_ckpt_path is None and ckpt_path is not None:
        classifier_ckpt_path = ckpt_path
    if classifier_ckpt_path is not None:
        return str(classifier_ckpt_path)

    subject_id = 1 if subject_id is None else int(subject_id)
    classifier_name = _normalize_classifier_name(classifier_name)
    base = Path(__file__).resolve().parent / "checkpoints"

    default_path = base / dataset_name / f"subj{subject_id}" / f"{dataset_name}_{classifier_name}.pth"
    candidates = [default_path]

    if classifier_name == "emghandnet":
        candidates.append(base / dataset_name / f"subj{subject_id}" / f"{dataset_name}_EMGhandnet.pth")

    for p in candidates:
        if p.exists():
            return str(p)

    raise FileNotFoundError(
        f"No classifier checkpoint found for dataset={dataset_name}, subject={subject_id}, "
        f"classifier={classifier_name}. "
        f"Checked: {[str(p) for p in candidates]}. "
        f"Please run `python -m EMG_fidelity.train_classifier --dataset {dataset_name} "
        f"--subject {subject_id} --model {classifier_name}`."
    )


def _resolve_metric_list(metrics: list[str] | tuple[str, ...] | None) -> list[str]:
    raw = DEFAULT_METRICS if metrics is None else list(metrics)
    if not raw:
        raise ValueError("metrics cannot be empty")
    norm: list[str] = []
    seen: set[str] = set()
    for metric in raw:
        key = _normalize_metric_name(metric)
        if not key or key in seen:
            continue
        seen.add(key)
        norm.append(key)

    unknown = [m for m in norm if m not in _METRIC_REGISTRY]
    if unknown:
        raise ValueError(
            f"Unknown metrics: {unknown}. Available metrics: {available_metrics()}."
        )
    return norm


def _assert_2d_non_empty(name: str, arr: np.ndarray, min_n: int = 1) -> None:
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D with shape (N, D), got {tuple(arr.shape)}")
    if arr.shape[0] < int(min_n):
        raise ValueError(f"{name} needs at least {min_n} rows, got N={arr.shape[0]}")


def _pairwise_l2_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    y = np.asarray(y)
    _assert_2d_non_empty("x", x, min_n=1)
    _assert_2d_non_empty("y", y, min_n=1)
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"x and y must share the same feature dimension, got {x.shape[1]} vs {y.shape[1]}")

    dtype = np.float64 if (x.dtype == np.float64 or y.dtype == np.float64) else np.float32
    x = np.asarray(x, dtype=dtype, order="C")
    y = np.asarray(y, dtype=dtype, order="C")

    dist = np.sum(x * x, axis=1, keepdims=True)
    dist = dist + np.sum(y * y, axis=1, keepdims=True).T
    dist = dist - (2.0 * (x @ y.T))
    np.maximum(dist, 0.0, out=dist)
    np.sqrt(dist, out=dist)
    return dist


def _kth_nearest_radii(features: np.ndarray, nearest_k: int, name: str) -> np.ndarray:
    feats = np.asarray(features)
    _assert_2d_non_empty(name, feats, min_n=nearest_k + 1)

    dist = _pairwise_l2_distance(feats, feats)
    np.fill_diagonal(dist, np.inf)
    return np.partition(dist, kth=nearest_k - 1, axis=1)[:, nearest_k - 1]


def _compute_prdc_bundle(
    real_features: np.ndarray,
    fake_features: np.ndarray,
    *,
    nearest_k: int,
    need_recall: bool,
) -> dict[str, float]:
    real_feats = np.asarray(real_features)
    fake_feats = np.asarray(fake_features)

    _assert_2d_non_empty("real features", real_feats, min_n=nearest_k + 1)
    _assert_2d_non_empty("fake features", fake_feats, min_n=(nearest_k + 1 if need_recall else 1))

    real_radii = _kth_nearest_radii(real_feats, nearest_k=nearest_k, name="real features")
    dist_gr = _pairwise_l2_distance(fake_feats, real_feats)
    membership_gr = dist_gr <= real_radii[None, :]

    result: dict[str, float] = {
        "pr": float(np.mean(np.any(membership_gr, axis=1))),
        "density": float(np.sum(membership_gr, dtype=np.int64) / float(nearest_k * fake_feats.shape[0])),
        "coverage": float(np.mean(np.any(membership_gr, axis=0))),
    }

    if need_recall:
        fake_radii = _kth_nearest_radii(fake_feats, nearest_k=nearest_k, name="fake features")
        membership_rg = dist_gr.T <= fake_radii[None, :]
        result["re"] = float(np.mean(np.any(membership_rg, axis=1)))

    return result


def _get_prdc_bundle(
    real_pack: dict[str, Any] | None,
    fake_pack: dict[str, Any] | None,
    options: dict[str, Any],
    *,
    force_recall: bool = False,
) -> dict[str, float]:
    if real_pack is None or fake_pack is None:
        raise ValueError("PR / RE / Density / Coverage require both real and fake features.")

    cache = options.setdefault("_cache", {})
    cached = cache.get("prdc")

    requested_metrics = {_normalize_metric_name(name) for name in options.get("_requested_metrics", ())}
    need_recall = bool(force_recall or ("re" in requested_metrics))

    prdc_cfg = dict(DEFAULT_PRDC_CONFIG)
    prdc_cfg.update(options.get("prdc", {}))
    nearest_k = int(prdc_cfg["nearest_k"])
    if nearest_k < 1:
        raise ValueError(f"prdc.nearest_k must be >= 1, got {nearest_k}")

    if (
        cached is None
        or cached.get("nearest_k") != nearest_k
        or (need_recall and not cached.get("has_recall", False))
    ):
        bundle = _compute_prdc_bundle(
            real_features=np.asarray(real_pack["features"]),
            fake_features=np.asarray(fake_pack["features"]),
            nearest_k=nearest_k,
            need_recall=need_recall,
        )
        cached = {
            "nearest_k": nearest_k,
            "has_recall": bool("re" in bundle),
            "values": bundle,
        }
        cache["prdc"] = cached

    return dict(cached["values"])


def _metric_fid(
    evaluator: "EMG_Fidelity_Evaluator",
    real_pack: dict[str, Any] | None,
    fake_pack: dict[str, Any] | None,
    options: dict[str, Any],
) -> dict[str, Any]:
    if real_pack is None or fake_pack is None:
        raise ValueError("FID requires both real and fake activations.")

    real_feats = np.asarray(real_pack["features"])
    fake_feats = np.asarray(fake_pack["features"])
    _assert_2d_non_empty("real features", real_feats, min_n=2)
    _assert_2d_non_empty("fake features", fake_feats, min_n=2)

    mu1 = np.mean(real_feats, axis=0)
    sigma1 = np.cov(real_feats, rowvar=False)
    mu2 = np.mean(fake_feats, axis=0)
    sigma2 = np.cov(fake_feats, rowvar=False)

    return {"fid": float(calculate_frechet_distance(mu1, sigma1, mu2, sigma2))}


def _metric_is(
    evaluator: "EMG_Fidelity_Evaluator",
    real_pack: dict[str, Any] | None,
    fake_pack: dict[str, Any] | None,
    options: dict[str, Any],
) -> dict[str, Any]:
    if fake_pack is None:
        raise ValueError("IS requires fake activations.")

    probs = np.asarray(fake_pack["probs"])
    _assert_2d_non_empty("fake probabilities", probs, min_n=1)

    splits = int(options.get("is_splits", 10))
    is_mean, is_std = calculate_inception_score(probs, splits=splits)
    return {"is_mean": float(is_mean), "is_std": float(is_std)}


def _metric_kid(
    evaluator: "EMG_Fidelity_Evaluator",
    real_pack: dict[str, Any] | None,
    fake_pack: dict[str, Any] | None,
    options: dict[str, Any],
) -> dict[str, Any]:
    if real_pack is None or fake_pack is None:
        raise ValueError("KID requires both real and fake activations.")

    real_feats = np.asarray(real_pack["features"])
    fake_feats = np.asarray(fake_pack["features"])
    _assert_2d_non_empty("real features", real_feats, min_n=2)
    _assert_2d_non_empty("fake features", fake_feats, min_n=2)

    kid_cfg = dict(DEFAULT_KID_CONFIG)
    kid_cfg.update(options.get("kid", {}))

    kid_result = compute_kid(
        feat_real=real_feats,
        feat_gen=fake_feats,
        subset_size=int(kid_cfg["subset_size"]),
        n_subsets=int(kid_cfg["n_subsets"]),
        degree=int(kid_cfg["degree"]),
        gamma=kid_cfg.get("gamma"),
        coef0=float(kid_cfg["coef0"]),
        normalize_mode=kid_cfg.get("normalize_mode"),
        eps=float(kid_cfg["eps"]),
    )

    return {
        "kid_mean": float(kid_result["kid_mean"]),
        "kid_std": float(kid_result["kid_std"]),
    }


def _align_labels_for_logits(labels: np.ndarray, num_classes: int) -> np.ndarray:
    labels = np.asarray(labels).astype(np.int64)
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D, got shape={tuple(labels.shape)}")
    if labels.size < 1:
        raise ValueError("labels is empty")

    min_label = int(labels.min())
    max_label = int(labels.max())

    if 0 <= min_label and max_label < num_classes:
        return labels
    if 1 <= min_label and max_label <= num_classes:
        return labels - 1

    raise ValueError(
        f"Cannot align labels to logits classes. labels range=[{min_label}, {max_label}], "
        f"num_classes={num_classes}."
    )


def _metric_cas(
    evaluator: "EMG_Fidelity_Evaluator",
    real_pack: dict[str, Any] | None,
    fake_pack: dict[str, Any] | None,
    options: dict[str, Any],
) -> dict[str, Any]:
    if fake_pack is None:
        raise ValueError("CAS requires fake activations.")

    logits = np.asarray(fake_pack["logits"])
    labels = np.asarray(fake_pack["labels"])

    _assert_2d_non_empty("fake logits", logits, min_n=1)
    if logits.shape[0] != labels.shape[0]:
        raise ValueError(
            f"CAS needs equal sample count for logits and labels, got {logits.shape[0]} vs {labels.shape[0]}"
        )

    labels_aligned = _align_labels_for_logits(labels, num_classes=logits.shape[1])
    pred = np.argmax(logits, axis=1)
    acc = float(np.mean(pred == labels_aligned))
    return {"cas": acc}


def _metric_pr(
    evaluator: "EMG_Fidelity_Evaluator",
    real_pack: dict[str, Any] | None,
    fake_pack: dict[str, Any] | None,
    options: dict[str, Any],
) -> dict[str, Any]:
    return {"pr": float(_get_prdc_bundle(real_pack, fake_pack, options)["pr"])}


def _metric_re(
    evaluator: "EMG_Fidelity_Evaluator",
    real_pack: dict[str, Any] | None,
    fake_pack: dict[str, Any] | None,
    options: dict[str, Any],
) -> dict[str, Any]:
    return {"re": float(_get_prdc_bundle(real_pack, fake_pack, options, force_recall=True)["re"])}


def _metric_density(
    evaluator: "EMG_Fidelity_Evaluator",
    real_pack: dict[str, Any] | None,
    fake_pack: dict[str, Any] | None,
    options: dict[str, Any],
) -> dict[str, Any]:
    return {"density": float(_get_prdc_bundle(real_pack, fake_pack, options)["density"])}


def _metric_coverage(
    evaluator: "EMG_Fidelity_Evaluator",
    real_pack: dict[str, Any] | None,
    fake_pack: dict[str, Any] | None,
    options: dict[str, Any],
) -> dict[str, Any]:
    return {"coverage": float(_get_prdc_bundle(real_pack, fake_pack, options)["coverage"])}


def _register_builtin_metrics() -> None:
    register_metric(
        "fid",
        _metric_fid,
        need_real_features=True,
        need_fake_features=True,
        overwrite=True,
    )
    register_metric(
        "is",
        _metric_is,
        need_fake_probs=True,
        overwrite=True,
    )
    register_metric(
        "kid",
        _metric_kid,
        need_real_features=True,
        need_fake_features=True,
        overwrite=True,
    )
    register_metric(
        "cas",
        _metric_cas,
        need_fake_logits=True,
        need_fake_labels=True,
        overwrite=True,
    )
    register_metric(
        "pr",
        _metric_pr,
        need_real_features=True,
        need_fake_features=True,
        overwrite=True,
    )
    register_metric(
        "re",
        _metric_re,
        need_real_features=True,
        need_fake_features=True,
        overwrite=True,
    )
    register_metric(
        "density",
        _metric_density,
        need_real_features=True,
        need_fake_features=True,
        overwrite=True,
    )
    register_metric(
        "coverage",
        _metric_coverage,
        need_real_features=True,
        need_fake_features=True,
        overwrite=True,
    )


class EMG_Fidelity_Evaluator:
    def __init__(
        self,
        dataset_name,
        device,
        classifier_name: str = "emghandnet",
        classifier_ckpt_path: str | None = None,
        subject_id: int | None = None,
        ckpt_path: str | None = None,
    ):
        self.dataset_name = dataset_name
        self.device = device
        self.subject_id = 1 if subject_id is None else int(subject_id)
        self.classifier_name = _normalize_classifier_name(classifier_name)

        if dataset_name not in _DATASET_NUM_CLASSES:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        num_classes = _DATASET_NUM_CLASSES[dataset_name]
        self.model = _build_classifier_model(self.classifier_name, num_classes, device, dataset_name)
        self.model.eval()

        self.ckpt_path = _resolve_classifier_ckpt(
            dataset_name=dataset_name,
            classifier_name=self.classifier_name,
            classifier_ckpt_path=classifier_ckpt_path,
            subject_id=self.subject_id,
            ckpt_path=ckpt_path,
        )
        if not Path(self.ckpt_path).exists():
            raise FileNotFoundError(
                f"Checkpoint not found at {self.ckpt_path}. "
                f"Please run `python -m EMG_fidelity.train_classifier --dataset {dataset_name} "
                f"--subject {self.subject_id} --model {self.classifier_name}` first."
            )

        print(
            f"Loading fidelity classifier from {self.ckpt_path} "
            f"(dataset={dataset_name}, subject={self.subject_id}, classifier={self.classifier_name})..."
        )
        raw_state = torch.load(self.ckpt_path, map_location=device, weights_only=False)
        state = _extract_state_dict_blob(raw_state)
        state = _align_state_dict_keys(self.model, state)
        self.model.load_state_dict(state)

    def extract(
        self,
        dataloader,
        *,
        need_features: bool = True,
        need_logits: bool = False,
        need_probs: bool = False,
        need_labels: bool = False,
    ) -> dict[str, Any]:
        return extract_activations(
            model=self.model,
            dataloader=dataloader,
            device=self.device,
            need_features=need_features,
            need_logits=need_logits,
            need_probs=need_probs,
            need_labels=need_labels,
        )

    def evaluate(
        self,
        *,
        real_loader=None,
        fake_loader=None,
        metrics: list[str] | tuple[str, ...] | None = None,
        kid: dict[str, Any] | None = None,
        prdc: dict[str, Any] | None = None,
        is_splits: int = 10,
        metric_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        metric_list = _resolve_metric_list(metrics)
        opt = self._resolve_metric_options(kid=kid, prdc=prdc, is_splits=is_splits, metric_options=metric_options)
        opt["_requested_metrics"] = tuple(metric_list)
        req = self._collect_requirements(metric_list)

        need_real = any(
            req[k]
            for k in (
                "real_features",
                "real_logits",
                "real_probs",
                "real_labels",
            )
        )
        need_fake = any(
            req[k]
            for k in (
                "fake_features",
                "fake_logits",
                "fake_probs",
                "fake_labels",
            )
        )

        if need_real and real_loader is None:
            raise ValueError(f"Metrics {metric_list} require real_loader, but got None.")
        if need_fake and fake_loader is None:
            raise ValueError(f"Metrics {metric_list} require fake_loader, but got None.")

        real_pack = None
        fake_pack = None

        if need_real:
            real_pack = self.extract(
                real_loader,
                need_features=req["real_features"],
                need_logits=req["real_logits"],
                need_probs=req["real_probs"],
                need_labels=req["real_labels"],
            )
        if need_fake:
            fake_pack = self.extract(
                fake_loader,
                need_features=req["fake_features"],
                need_logits=req["fake_logits"],
                need_probs=req["fake_probs"],
                need_labels=req["fake_labels"],
            )

        return self.evaluate_from_activations(
            real_pack=real_pack,
            fake_pack=fake_pack,
            metrics=metric_list,
            kid=kid,
            prdc=prdc,
            is_splits=is_splits,
            metric_options=metric_options,
        )

    def evaluate_from_activations(
        self,
        *,
        real_pack: dict[str, Any] | None = None,
        fake_pack: dict[str, Any] | None = None,
        metrics: list[str] | tuple[str, ...] | None = None,
        kid: dict[str, Any] | None = None,
        prdc: dict[str, Any] | None = None,
        is_splits: int = 10,
        metric_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        metric_list = _resolve_metric_list(metrics)
        opt = self._resolve_metric_options(kid=kid, prdc=prdc, is_splits=is_splits, metric_options=metric_options)
        opt["_requested_metrics"] = tuple(metric_list)
        req = self._collect_requirements(metric_list)

        need_real = any(
            req[k]
            for k in (
                "real_features",
                "real_logits",
                "real_probs",
                "real_labels",
            )
        )
        need_fake = any(
            req[k]
            for k in (
                "fake_features",
                "fake_logits",
                "fake_probs",
                "fake_labels",
            )
        )

        if need_real and real_pack is None:
            raise ValueError(f"Metrics {metric_list} require real_pack, but got None.")
        if need_fake and fake_pack is None:
            raise ValueError(f"Metrics {metric_list} require fake_pack, but got None.")

        self._validate_pack(real_pack, req, prefix="real")
        self._validate_pack(fake_pack, req, prefix="fake")

        result: dict[str, Any] = {}
        for name in metric_list:
            spec = _METRIC_REGISTRY[name]
            metric_out = spec.fn(self, real_pack, fake_pack, opt)
            if not isinstance(metric_out, dict):
                raise TypeError(f"Metric '{name}' must return a dict, got {type(metric_out)}")
            result.update(metric_out)
        return result

    @staticmethod
    def _reduce_embedding(
        features: np.ndarray,
        *,
        method: str = "tsne",
        reducer_options: dict[str, Any] | None = None,
        random_seed: int = 42,
    ) -> np.ndarray:
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2:
            raise ValueError(f"features must be 2D, got shape {tuple(features.shape)}")
        n_samples = int(features.shape[0])
        if n_samples == 0:
            return np.empty((0, 2), dtype=np.float32)
        if n_samples == 1:
            return np.zeros((1, 2), dtype=np.float32)

        method = str(method).strip().lower()
        reducer_options = dict(reducer_options or {})

        if method == "tsne":
            from sklearn.manifold import TSNE

            reducer_options.setdefault("init", "pca")
            reducer_options.setdefault("learning_rate", "auto")
            reducer_options.setdefault("perplexity", max(1, min(30, n_samples - 1)))
            reducer = TSNE(n_components=2, random_state=int(random_seed), **reducer_options)
            embedding = reducer.fit_transform(features)
        elif method == "umap":
            try:
                import umap
            except ImportError as exc:
                raise ImportError(
                    "UMAP is not installed. Install `umap-learn` or use method='tsne'."
                ) from exc

            reducer = umap.UMAP(n_components=2, random_state=int(random_seed), **reducer_options)
            embedding = reducer.fit_transform(features)
        else:
            raise ValueError(f"Unsupported embedding reduction method '{method}'. Use one of: tsne, umap.")

        return np.asarray(embedding, dtype=np.float32)

    @staticmethod
    def _save_embedding_plot(
        *,
        real_embedding: np.ndarray,
        fake_embedding: np.ndarray,
        save_path: str | Path,
        method: str,
        real_label: str,
        fake_label: str,
        title: str | None,
    ) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(6, 5))
        if len(real_embedding) > 0:
            ax.scatter(
                real_embedding[:, 0],
                real_embedding[:, 1],
                s=14,
                alpha=0.75,
                label=real_label,
                c="#1f77b4",
            )
        if len(fake_embedding) > 0:
            ax.scatter(
                fake_embedding[:, 0],
                fake_embedding[:, 1],
                s=14,
                alpha=0.75,
                label=fake_label,
                c="#d62728",
            )

        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_title(title or f"{str(method).upper()} Feature Embedding")
        ax.grid(True, alpha=0.2)
        if len(real_embedding) > 0 or len(fake_embedding) > 0:
            ax.legend()
        fig.tight_layout()
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def visualize_embedding(
        self,
        *,
        real_loader=None,
        fake_loader=None,
        save_path: str | Path,
        method: str = "tsne",
        reducer_options: dict[str, Any] | None = None,
        real_label: str = "Real",
        fake_label: str = "Fake",
        title: str | None = None,
        random_seed: int = 42,
    ) -> None:
        if real_loader is None:
            raise ValueError("visualize_embedding requires real_loader, but got None.")
        if fake_loader is None:
            raise ValueError("visualize_embedding requires fake_loader, but got None.")

        real_pack = self.extract(
            real_loader,
            need_features=True,
            need_logits=False,
            need_probs=False,
            need_labels=False,
        )
        fake_pack = self.extract(
            fake_loader,
            need_features=True,
            need_logits=False,
            need_probs=False,
            need_labels=False,
        )
        return self.visualize_embedding_from_activations(
            real_pack=real_pack,
            fake_pack=fake_pack,
            save_path=save_path,
            method=method,
            reducer_options=reducer_options,
            real_label=real_label,
            fake_label=fake_label,
            title=title,
            random_seed=random_seed,
        )

    def visualize_embedding_from_activations(
        self,
        *,
        real_pack: dict[str, Any] | None = None,
        fake_pack: dict[str, Any] | None = None,
        save_path: str | Path,
        method: str = "tsne",
        reducer_options: dict[str, Any] | None = None,
        real_label: str = "Real",
        fake_label: str = "Fake",
        title: str | None = None,
        random_seed: int = 42,
    ) -> None:
        if real_pack is None:
            raise ValueError("visualize_embedding_from_activations requires real_pack, but got None.")
        if fake_pack is None:
            raise ValueError("visualize_embedding_from_activations requires fake_pack, but got None.")
        if "features" not in real_pack:
            raise ValueError("real_pack must contain 'features' for embedding visualization.")
        if "features" not in fake_pack:
            raise ValueError("fake_pack must contain 'features' for embedding visualization.")

        real_features = np.asarray(real_pack["features"], dtype=np.float32)
        fake_features = np.asarray(fake_pack["features"], dtype=np.float32)

        if real_features.ndim != 2 or fake_features.ndim != 2:
            raise ValueError(
                f"real/fake features must both be 2D, got {tuple(real_features.shape)} and {tuple(fake_features.shape)}"
            )
        if len(real_features) == 0 and len(fake_features) == 0:
            raise ValueError("At least one of real_features or fake_features must be non-empty.")

        combined = np.concatenate([real_features, fake_features], axis=0)
        embedding = self._reduce_embedding(
            combined,
            method=method,
            reducer_options=reducer_options,
            random_seed=random_seed,
        )

        n_real = int(len(real_features))
        real_embedding = embedding[:n_real]
        fake_embedding = embedding[n_real:]
        self._save_embedding_plot(
            real_embedding=real_embedding,
            fake_embedding=fake_embedding,
            save_path=save_path,
            method=method,
            real_label=real_label,
            fake_label=fake_label,
            title=title,
        )
        return None

    @staticmethod
    def _resolve_metric_options(
        *,
        kid: dict[str, Any] | None,
        prdc: dict[str, Any] | None,
        is_splits: int,
        metric_options: dict[str, Any] | None,
    ) -> dict[str, Any]:
        opt: dict[str, Any] = dict(metric_options or {})
        opt["is_splits"] = int(opt.get("is_splits", is_splits))

        kid_cfg = dict(DEFAULT_KID_CONFIG)
        kid_cfg.update(opt.get("kid", {}))
        if kid is not None:
            kid_cfg.update(kid)
        opt["kid"] = kid_cfg

        prdc_cfg = dict(DEFAULT_PRDC_CONFIG)
        prdc_cfg.update(opt.get("prdc", {}))
        if prdc is not None:
            prdc_cfg.update(prdc)
        if "k" in prdc_cfg:
            prdc_cfg["nearest_k"] = prdc_cfg["k"]
        prdc_cfg["nearest_k"] = int(prdc_cfg["nearest_k"])
        if prdc_cfg["nearest_k"] < 1:
            raise ValueError(f"prdc.nearest_k must be >= 1, got {prdc_cfg['nearest_k']}")
        opt["prdc"] = prdc_cfg
        return opt

    @staticmethod
    def _collect_requirements(metric_list: list[str]) -> dict[str, bool]:
        req = {
            "real_features": False,
            "fake_features": False,
            "real_logits": False,
            "fake_logits": False,
            "real_probs": False,
            "fake_probs": False,
            "real_labels": False,
            "fake_labels": False,
        }
        for name in metric_list:
            spec = _METRIC_REGISTRY[name]
            req["real_features"] = req["real_features"] or spec.need_real_features
            req["fake_features"] = req["fake_features"] or spec.need_fake_features
            req["real_logits"] = req["real_logits"] or spec.need_real_logits
            req["fake_logits"] = req["fake_logits"] or spec.need_fake_logits
            req["real_probs"] = req["real_probs"] or spec.need_real_probs
            req["fake_probs"] = req["fake_probs"] or spec.need_fake_probs
            req["real_labels"] = req["real_labels"] or spec.need_real_labels
            req["fake_labels"] = req["fake_labels"] or spec.need_fake_labels
        return req

    @staticmethod
    def _validate_pack(pack: dict[str, Any] | None, req: dict[str, bool], prefix: str) -> None:
        if pack is None:
            return
        key_map = {
            f"{prefix}_features": "features",
            f"{prefix}_logits": "logits",
            f"{prefix}_probs": "probs",
            f"{prefix}_labels": "labels",
        }
        for req_key, pack_key in key_map.items():
            if req.get(req_key, False) and (pack_key not in pack):
                raise ValueError(
                    f"Required key '{pack_key}' missing in {prefix}_pack for metrics requirements {req}."
                )

    def compute_fid(self, real_loader, fake_loader) -> float:
        return float(self.evaluate(real_loader=real_loader, fake_loader=fake_loader, metrics=["fid"])["fid"])

    def compute_is(self, fake_loader, splits: int = 10) -> tuple[float, float]:
        out = self.evaluate(fake_loader=fake_loader, metrics=["is"], is_splits=splits)
        return float(out["is_mean"]), float(out["is_std"])

    def compute_kid(self, real_loader, fake_loader, kid: dict[str, Any] | None = None) -> dict[str, float]:
        out = self.evaluate(real_loader=real_loader, fake_loader=fake_loader, metrics=["kid"], kid=kid)
        return {"kid_mean": float(out["kid_mean"]), "kid_std": float(out["kid_std"])}

    def compute_cas(self, fake_loader) -> float:
        out = self.evaluate(fake_loader=fake_loader, metrics=["cas"])
        return float(out["cas"])

    def compute_prdc(
        self,
        real_loader,
        fake_loader,
        prdc: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        out = self.evaluate(
            real_loader=real_loader,
            fake_loader=fake_loader,
            metrics=["pr", "re", "density", "coverage"],
            prdc=prdc,
        )
        return {
            "pr": float(out["pr"]),
            "re": float(out["re"]),
            "density": float(out["density"]),
            "coverage": float(out["coverage"]),
        }


# Backward compatibility alias.
FID_IS_Evaluator = EMG_Fidelity_Evaluator


_register_builtin_classifiers()
_register_builtin_metrics()

__all__ = [
    "EMG_Fidelity_Evaluator",
    "FID_IS_Evaluator",
    "DEFAULT_METRICS",
    "DEFAULT_KID_CONFIG",
    "DEFAULT_PRDC_CONFIG",
    "compute_kid",
    "compute_classwise_kid",
    "register_classifier",
    "available_classifiers",
    "register_metric",
    "available_metrics",
    "calculate_frechet_distance",
    "calculate_inception_score",
    "get_activations",
    "extract_activations",
]
