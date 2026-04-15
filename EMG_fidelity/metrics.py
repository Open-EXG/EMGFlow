import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            f"adding {eps} to diagonal of cov estimates"
        )
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_inception_score(probs, splits=10):
    """Compute inception score from softmax probabilities (N, C)."""
    probs = np.asarray(probs)
    if probs.ndim != 2:
        raise ValueError(f"probs must be 2D with shape (N, C), got {tuple(probs.shape)}")

    n = probs.shape[0]
    if n < 1:
        raise ValueError("probs must contain at least one sample")

    split_scores = []
    for k in range(int(splits)):
        part = probs[k * (n // splits) : (k + 1) * (n // splits), :]
        if part.shape[0] == 0:
            continue
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(np.sum(pyx * (np.log(pyx + 1e-16) - np.log(py + 1e-16))))
        split_scores.append(np.exp(np.mean(scores)))

    if not split_scores:
        raise ValueError(
            f"No valid IS splits produced. N={n}, splits={splits}. "
            "Increase sample count or decrease splits."
        )
    return float(np.mean(split_scores)), float(np.std(split_scores))


def _split_batch(batch):
    if not isinstance(batch, (tuple, list)):
        raise TypeError(f"Expected dataloader batch as tuple/list, got {type(batch)}")
    if len(batch) < 1:
        raise ValueError("Batch must contain at least input tensor x")
    x = batch[0]
    y = batch[1] if len(batch) > 1 else None
    return x, y


def extract_activations(
    model,
    dataloader,
    device,
    *,
    need_features=True,
    need_logits=False,
    need_probs=False,
    need_labels=False,
):
    """Extract requested tensors in one forward pass over the dataloader."""
    if dataloader is None:
        raise ValueError("dataloader cannot be None")

    model.eval()
    features_list = []
    logits_list = []
    probs_list = []
    labels_list = []

    with torch.no_grad():
        for batch in dataloader:
            x, y = _split_batch(batch)
            x = x.to(device)
            feats, logits = model(x)

            if need_features:
                features_list.append(feats.detach().cpu())
            if need_logits:
                logits_list.append(logits.detach().cpu())
            if need_probs:
                probs_list.append(F.softmax(logits, dim=1).detach().cpu())
            if need_labels:
                if y is None:
                    raise ValueError("need_labels=True but dataloader batch has no labels")
                labels_list.append(y.detach().cpu().long())

    output = {}
    if need_features:
        output["features"] = (
            torch.cat(features_list, dim=0).numpy() if features_list else np.empty((0, 0), dtype=np.float32)
        )
    if need_logits:
        output["logits"] = torch.cat(logits_list, dim=0).numpy() if logits_list else np.empty((0, 0), dtype=np.float32)
    if need_probs:
        output["probs"] = torch.cat(probs_list, dim=0).numpy() if probs_list else np.empty((0, 0), dtype=np.float32)
    if need_labels:
        output["labels"] = (
            torch.cat(labels_list, dim=0).numpy().astype(np.int64)
            if labels_list
            else np.empty((0,), dtype=np.int64)
        )
    return output


def get_activations(model, dataloader, device):
    """Backward-compatible helper returning features and probabilities."""
    out = extract_activations(
        model=model,
        dataloader=dataloader,
        device=device,
        need_features=True,
        need_logits=False,
        need_probs=True,
        need_labels=False,
    )
    return out["features"], out["probs"]
