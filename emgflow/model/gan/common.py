from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class FiLM1D(nn.Module):
    def __init__(self, num_channels: int, cond_dim: int) -> None:
        super().__init__()
        self.to_gamma = nn.Linear(cond_dim, num_channels)
        self.to_beta = nn.Linear(cond_dim, num_channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma = self.to_gamma(cond).unsqueeze(-1)
        beta = self.to_beta(cond).unsqueeze(-1)
        return x * (1.0 + gamma) + beta


class ResBlock1D(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        cond_dim: int | None = None,
        upsample: bool = False,
        downsample: bool = False,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        if upsample and downsample:
            raise ValueError("ResBlock1D cannot upsample and downsample at the same time.")

        padding = kernel_size // 2
        self.upsample = bool(upsample)
        self.downsample = bool(downsample)

        self.norm1 = nn.GroupNorm(num_groups=min(8, in_ch), num_channels=in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding)
        self.norm2 = nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding)

        self.film1 = FiLM1D(in_ch, cond_dim) if cond_dim is not None else None
        self.film2 = FiLM1D(out_ch, cond_dim) if cond_dim is not None else None
        self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def _resample(self, x: torch.Tensor) -> torch.Tensor:
        if self.upsample:
            return F.interpolate(x, scale_factor=2, mode="nearest")
        if self.downsample:
            return F.avg_pool1d(x, kernel_size=2, stride=2)
        return x

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        residual = self.skip(self._resample(x))

        h = self.norm1(x)
        if self.film1 is not None and cond is not None:
            h = self.film1(h, cond)
        h = F.silu(h)
        h = self.conv1(self._resample(h))

        h = self.norm2(h)
        if self.film2 is not None and cond is not None:
            h = self.film2(h, cond)
        h = F.silu(h)
        h = self.conv2(h)
        return h + residual


class ProjectionCritic1D(nn.Module):
    def __init__(self, *, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, 256)
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2)
        self.conv4 = nn.Conv1d(256, 256, kernel_size=5, stride=2, padding=2)
        self.fc = nn.Linear(256, 1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = self.act(self.conv1(x))
        h = self.act(self.conv2(h))
        h = self.act(self.conv3(h))
        h = self.act(self.conv4(h))
        h = torch.sum(h, dim=-1)
        out_uncond = self.fc(h).squeeze(-1)
        out_proj = torch.sum(h * self.label_emb(y), dim=1)
        return out_uncond + out_proj
