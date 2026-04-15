from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ProjectionCritic1D, ResBlock1D, count_parameters


class PureWGANGenerator1D(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        signal_length: int,
        num_classes: int,
        noise_dim: int,
        label_dim: int,
    ) -> None:
        super().__init__()
        if signal_length != 400:
            raise ValueError(f"PureWGANGenerator1D currently expects signal_length=400, got {signal_length}.")
        self.noise_dim = int(noise_dim)
        self.signal_length = int(signal_length)
        self.out_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.label_dim = int(label_dim)

        self.label_emb = nn.Embedding(num_classes, label_dim)
        self.fc = nn.Linear(noise_dim + label_dim, 256 * 50)
        self.block1 = ResBlock1D(256, 192, cond_dim=label_dim, upsample=True)
        self.block2 = ResBlock1D(192, 128, cond_dim=label_dim, upsample=True)
        self.block3 = ResBlock1D(128, 96, cond_dim=label_dim, upsample=True)
        self.out_norm = nn.GroupNorm(num_groups=min(8, 96), num_channels=96)
        self.out_conv = nn.Conv1d(96, in_channels, kernel_size=5, padding=2)

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2 or z.shape[1] != self.noise_dim:
            raise ValueError(f"z must have shape (B,{self.noise_dim}), got {tuple(z.shape)}")
        y_emb = self.label_emb(y)
        h = self.fc(torch.cat([z, y_emb], dim=1))
        h = h.view(h.size(0), 256, 50)
        h = self.block1(h, y_emb)
        h = self.block2(h, y_emb)
        h = self.block3(h, y_emb)
        h = F.silu(self.out_norm(h))
        return self.out_conv(h)


def build_pure_wgan_from_config(model_cfg: dict[str, Any]) -> tuple[PureWGANGenerator1D, ProjectionCritic1D]:
    pure_cfg = dict(model_cfg.get("pure_wgan_gp", {}))
    label_dim = int(pure_cfg.get("label_dim", 32))
    generator = PureWGANGenerator1D(
        in_channels=int(model_cfg["in_channels"]),
        signal_length=int(model_cfg["signal_length"]),
        num_classes=int(model_cfg["num_classes"]),
        noise_dim=int(model_cfg["noise_dim"]),
        label_dim=label_dim,
    )
    discriminator = ProjectionCritic1D(
        in_channels=int(model_cfg["in_channels"]),
        num_classes=int(model_cfg["num_classes"]),
    )
    return generator, discriminator


def compute_pure_wgan_parameter_budget(
    generator: PureWGANGenerator1D,
    discriminator: ProjectionCritic1D,
) -> dict[str, int]:
    g_count = count_parameters(generator)
    d_count = count_parameters(discriminator)
    return {
        "G": int(g_count),
        "D": int(d_count),
        "total": int(g_count + d_count),
    }
