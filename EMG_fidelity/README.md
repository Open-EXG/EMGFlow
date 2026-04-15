# EMG_fidelity

Unified EMG generation quality evaluation package.
It supports configurable `FID / IS / KID / CAS / PR / RE / Density / Coverage` and avoids redundant forwards by extracting required tensors once per real/fake loader.

## 1. Scope

- Provide one evaluation entrypoint for experiments.
- Compute only the tensors required by selected metrics (`features/logits/probs/labels`).
- Default metrics are `fid,is,kid`.
- Metric names are case-insensitive. For example, `["PR", "RE", "Density", "Coverage"]` is valid.
- Built-in dataset class counts:
- `NinaproDB2 -> 49`
- `NinaproDB4 -> 52`

Out of scope:
- Generator training.
- Automatic classifier training (helper script only: `train_classifier.py`).

## 2. Quick Start

```python
from EMG_fidelity import EMG_Fidelity_Evaluator

evaluator = EMG_Fidelity_Evaluator(
    dataset_name="NinaproDB2",
    device="cuda:0",
    classifier_name="emghandnet",
    subject_id=1,
)

res = evaluator.evaluate(
    real_loader=real_loader,
    fake_loader=fake_loader,
    metrics=["fid", "is", "kid", "cas", "pr", "re", "density", "coverage"],
    prdc={"nearest_k": 5},
)
print(res)
```

Typical output keys by metric:
- `fid`
- `is_mean`, `is_std`
- `kid_mean`, `kid_std`
- `cas`
- `pr`
- `re`
- `density`
- `coverage`

## 3. Core API

Main class: `EMG_fidelity.EMG_Fidelity_Evaluator`

Methods:
- `extract(dataloader, need_features, need_logits, need_probs, need_labels)`
- `evaluate(real_loader, fake_loader, metrics, kid, is_splits, metric_options)`
- `evaluate_from_activations(real_pack, fake_pack, metrics, kid, prdc, is_splits, metric_options)`
- `compute_fid(...)`
- `compute_is(...)`
- `compute_kid(...)`
- `compute_cas(...)`
- `compute_prdc(...)`

Compatibility alias:
- `FID_IS_Evaluator = EMG_Fidelity_Evaluator`

## 4. Metric Dependency Matrix

- `fid`: needs `real.features` + `fake.features`
- `kid`: needs `real.features` + `fake.features`
- `pr`: needs `real.features` + `fake.features`
- `re`: needs `real.features` + `fake.features`
- `density`: needs `real.features` + `fake.features`
- `coverage`: needs `real.features` + `fake.features`
- `is`: needs `fake.probs`
- `cas`: needs `fake.logits` + `fake.labels`

Implications:
- If metrics are `["is","cas"]`, real forward is skipped.
- If metrics are `["fid","kid"]`, feature extraction is shared, not repeated.
- If metrics are `["pr","re","density","coverage"]`, feature extraction is shared and PRDC distance matrices are reused within the same evaluation call.

## 5. KID Defaults and Override

Default KID config (aligned with legacy `db2_ddpm_h64_full` settings):

```python
DEFAULT_KID_CONFIG = {
    "subset_size": 5000,
    "n_subsets": 100,
    "degree": 3,
    "gamma": None,
    "coef0": 1.0,
    "normalize_mode": None,
    "eps": 1e-12,
}
```

Override options:
- Pass `kid={...}` to `evaluate(...)`.
- Set `evaluation.kid` in YAML, then optionally override per stage (for example `baseline.kid`, `gen_test.kid`).

## 6. PRDC Defaults and Override

Default PRDC config:

```python
DEFAULT_PRDC_CONFIG = {
    "nearest_k": 5,
}
```

Supported PRDC metrics:
- `pr`: improved precision
- `re`: improved recall
- `density`
- `coverage`

Override options:
- Pass `prdc={"nearest_k": 3}` to `evaluate(...)`.
- Or pass `metric_options={"prdc": {"nearest_k": 10}}`.
- `prdc={"k": 5}` is also accepted as an alias for `nearest_k`.

## 7. Classifiers and Checkpoints

Built-in classifiers:
- `resnet18`
- `simpleconvnet`
- `emghandnet`
- `crossformer`

Add custom classifier:
- `register_classifier(name, builder)`

Checkpoint resolution priority:
- Explicit `classifier_ckpt_path`
- Backward-compatible `ckpt_path`
- Default path: `EMG_fidelity/checkpoints/{dataset}/subj{subject}/{dataset}_{classifier}.pth`
- Extra fallback for `emghandnet`: `{dataset}_EMGhandnet.pth`

## 8. Add New Metric

1. Implement:
- `fn(evaluator, real_pack, fake_pack, options) -> dict`

2. Register with dependencies:
- `register_metric(...)`
- Set required flags among:
- `need_real_features`, `need_fake_features`
- `need_real_logits`, `need_fake_logits`
- `need_real_probs`, `need_fake_probs`
- `need_real_labels`, `need_fake_labels`

3. Use it in `evaluate(..., metrics=[...])`.

Rules:
- Metric function must return a `dict`.
- Avoid key collisions with existing metrics.

## 9. Common Pitfalls

- Dataloader batch must be `tuple/list`; first item must be `x`. `CAS` needs labels in second item.
- `is_splits` should not exceed sample count too much, or no valid split will be produced.
- `CAS` accepts labels in `1..C` or `0..C-1`; out-of-range labels raise an error.
- `PR / RE / Density / Coverage` use Euclidean distances in feature space and require at least `nearest_k + 1` samples in the manifold side.
- `evaluate_from_activations` validates required keys only. Caller must ensure shape and semantic correctness.

## 10. Code Map

- Registry, evaluator, built-ins: `EMG_fidelity/__init__.py`
- FID/IS utilities and extraction: `EMG_fidelity/metrics.py`
- KID implementation: `EMG_fidelity/kid_metrics.py`
- Classifier definitions: `EMG_fidelity/models.py`, `EMG_fidelity/robustmodel.py`
- Classifier training script: `EMG_fidelity/train_classifier.py`
