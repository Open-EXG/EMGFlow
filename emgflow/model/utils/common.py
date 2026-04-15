"""
Shared building blocks used across DDPM, Flow Matching, and MeanFlow models.

Includes:
    - EMA                           Exponential Moving Average helper
    - get_sinusoidal_time_embedding Sinusoidal positional / timestep embedding
    - TimeMLP                       Time-embedding projection MLP
    - EfficientAttention1D          Linear-complexity attention (softmax kernel)
    - ResBlock1D                    Conv-Norm-ELU residual block with selectable conditioning
"""

import math
import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# EMA (Exponential Moving Average)
# ---------------------------------------------------------------------------
class EMA:
    def __init__(self, model, decay=0.9999, start_step=0, ema_fp32=True):
        self.decay = decay
        self.start_step = start_step
        self.step = 0

        self.ema_model = copy.deepcopy(model)
        if ema_fp32:
            self.ema_model = self.ema_model.float()

        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.step += 1

        # warmup: 直接拷贝全量 state_dict（包含 parameters + buffers）
        if self.step < self.start_step:
            sd = model.state_dict()
            if next(self.ema_model.parameters()).dtype != torch.float32:
                self.ema_model.load_state_dict(sd, strict=True)
            else:
                # ema_model 是 fp32 时，state_dict 里的 float tensor 会自动 cast
                self.ema_model.load_state_dict(sd, strict=True)
            return

        # 1) EMA 更新 parameters
        model_params = dict(model.named_parameters())
        ema_params = dict(self.ema_model.named_parameters())

        for name, p in model_params.items():
            if name not in ema_params:
                continue
            src = p.detach()
            # 如果 ema_model 是 fp32，强制用 fp32 做更新更稳
            if ema_params[name].dtype == torch.float32 and src.dtype != torch.float32:
                src = src.float()
            ema_params[name].mul_(self.decay).add_(src, alpha=1.0 - self.decay)

        # 2) 同步 buffers（BN running_mean/var 等）
        model_bufs = dict(model.named_buffers())
        ema_bufs = dict(self.ema_model.named_buffers())

        for name, b in model_bufs.items():
            if name not in ema_bufs:
                continue
            src = b.detach()
            if ema_bufs[name].dtype == torch.float32 and src.dtype != torch.float32:
                src = src.float()
            ema_bufs[name].copy_(src)


# ---------------------------------------------------------------------------
# Sinusoidal time / timestep embedding
# ---------------------------------------------------------------------------
def get_sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    t: (B,) int64 or float32 (timesteps)
    return: (B, dim)
    """
    if t.dim() != 1:
        t = t.view(-1)
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(0, half, device=t.device, dtype=torch.float32) / max(half - 1, 1)
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # (B, 2*half)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


# ---------------------------------------------------------------------------
# TimeMLP – projects sinusoidal time embedding to conditioning dim
# ---------------------------------------------------------------------------
class TimeMLP(nn.Module):
    def __init__(self, time_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(time_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        return self.net(t_emb)


# ---------------------------------------------------------------------------
# Efficient (Linear) Attention 1D
# Paper eq: softmax2(Q) * softmax1(K)^T * V
# ---------------------------------------------------------------------------
class EfficientAttention1D(nn.Module):
    """
    Efficient Attention (as in your figure):
      Q' = softmax over feature dim (d_k)
      K' = softmax over length dim (L)
      Y  = Q' @ (K'^T @ V)
    """
    def __init__(self, channels: int, heads: int = 4, head_dim: int = 32):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        inner = heads * head_dim

        self.to_q = nn.Conv1d(channels, inner, kernel_size=1, bias=False)
        self.to_k = nn.Conv1d(channels, inner, kernel_size=1, bias=False)
        self.to_v = nn.Conv1d(channels, inner, kernel_size=1, bias=False)
        self.to_out = nn.Conv1d(inner, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        B, C, L = x.shape
        h, d = self.heads, self.head_dim

        # project: (B, h*d, L)
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # reshape to (B, h, L, d)
        q = q.view(B, h, d, L).permute(0, 1, 3, 2)  # (B,h,L,d)
        k = k.view(B, h, d, L).permute(0, 1, 3, 2)  # (B,h,L,d)
        v = v.view(B, h, d, L).permute(0, 1, 3, 2)  # (B,h,L,d_v) here d_v==d

        # softmax on different dims
        q = torch.softmax(q, dim=-1)  # over d: (B,h,L,d)
        k = torch.softmax(k, dim=-2)  # over L: (B,h,L,d)

        # context: (B,h,d,d) = (B,h,L,d)^T @ (B,h,L,d)
        context = torch.einsum("bhld,bhlm->bhdm", k, v)  # k^T v, sums over L

        # out: (B,h,L,d) = (B,h,L,d) @ (B,h,d,d)
        out = torch.einsum("bhld,bhdm->bhlm", q, context)

        # back to (B, h*d, L)
        out = out.permute(0, 1, 3, 2).contiguous().view(B, h * d, L)
        return self.to_out(out)


# ---------------------------------------------------------------------------
# ResNet-style Block (Conv-Norm-ELU) x2 + residual
# with selectable conditioning injection from (time + label) embedding
# ---------------------------------------------------------------------------
class ResBlock1D(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k1: int,
        k2: int,
        emb_dim: int,
        norm_type: str = "gn",
        cond_inject: str = "adagn",
    ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.norm_type = self._normalize_norm_type(norm_type)
        self.cond_inject = self._normalize_cond_inject(cond_inject)

        self.concat_fuse1: nn.Conv1d | None = None
        self.concat_fuse2: nn.Conv1d | None = None

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=k1, padding="same", bias=True)
        self.norm1 = self._build_norm(num_channels=out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=k2, padding="same", bias=True)
        self.norm2 = self._build_norm(num_channels=out_ch)

        self.act = nn.ELU(inplace=True)

        self.emb_proj1: nn.Linear | None = None
        self.emb_proj2: nn.Linear | None = None
        if emb_dim > 0:
            if self.cond_inject == "adagn":
                self.emb_proj1 = nn.Linear(emb_dim, 2 * out_ch)
                self.emb_proj2 = nn.Linear(emb_dim, 2 * out_ch)
            elif self.cond_inject == "add":
                self.emb_proj1 = nn.Linear(emb_dim, out_ch)
                self.emb_proj2 = nn.Linear(emb_dim, out_ch)
            elif self.cond_inject == "concat":
                self.emb_proj1 = nn.Linear(emb_dim, in_ch)
                self.emb_proj2 = nn.Linear(emb_dim, out_ch)
                self.concat_fuse1 = nn.Conv1d(2 * in_ch, in_ch, kernel_size=1, bias=True)
                self.concat_fuse2 = nn.Conv1d(2 * out_ch, out_ch, kernel_size=1, bias=True)

        self.skip = None
        if in_ch != out_ch:
            self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=True)

    @staticmethod
    def _normalize_norm_type(norm_type: str) -> str:
        key = str(norm_type).strip().lower()
        aliases = {
            "gn": "gn",
            "groupnorm": "gn",
            "group_norm": "gn",
            "bn": "bn",
            "batchnorm": "bn",
            "batch_norm": "bn",
        }
        if key not in aliases:
            raise ValueError(
                f"Unsupported norm_type='{norm_type}'. Use one of: GN, BN."
            )
        return aliases[key]

    @staticmethod
    def _normalize_cond_inject(cond_inject: str) -> str:
        key = str(cond_inject).strip().lower()
        aliases = {
            "adagn": "adagn",
            "ada_gn": "adagn",
            "ada-gn": "adagn",
            "add": "add",
            "concat": "concat",
            "cat": "concat",
            "concatenate": "concat",
        }
        if key not in aliases:
            raise ValueError(
                f"Unsupported cond_inject='{cond_inject}'. "
                "Use one of: adaGN, add, concat."
            )
        return aliases[key]

    def _build_norm(self, num_channels: int) -> nn.Module:
        if self.norm_type == "gn":
            return nn.GroupNorm(num_groups=8, num_channels=num_channels)
        if self.norm_type == "bn":
            return nn.BatchNorm1d(num_features=num_channels)
        raise RuntimeError(f"Unhandled norm_type={self.norm_type}")

    def _apply_AdaGN(self, h: torch.Tensor, emb: torch.Tensor, emb_proj: nn.Linear) -> torch.Tensor:
        scale, shift = emb_proj(emb).to(dtype=h.dtype).unsqueeze(-1).chunk(2, dim=1)
        return h * (1 + scale) + shift

    def _apply_add(self, h: torch.Tensor, emb: Optional[torch.Tensor], emb_proj: Optional[nn.Linear]) -> torch.Tensor:
        if emb is None or emb_proj is None:
            return h
        return h + emb_proj(emb).to(dtype=h.dtype).unsqueeze(-1)

    def _concat_condition(
        self,
        h: torch.Tensor,
        emb: Optional[torch.Tensor],
        emb_proj: Optional[nn.Linear],
        fuse: Optional[nn.Conv1d],
    ) -> torch.Tensor:
        if emb_proj is None or fuse is None:
            return h

        if emb is None:
            cond = torch.zeros(
                h.shape[0],
                emb_proj.out_features,
                h.shape[-1],
                device=h.device,
                dtype=h.dtype,
            )
        else:
            cond = emb_proj(emb).to(dtype=h.dtype).unsqueeze(-1).expand(-1, -1, h.shape[-1])
        return fuse(torch.cat([h, cond], dim=1))

    def forward(self, x: torch.Tensor, emb: Optional[torch.Tensor]) -> torch.Tensor:
        """
        x: (B, C, L)
        emb: (B, emb_dim) or None
        """
        residual = x if self.skip is None else self.skip(x)

        h_in = (
            self._concat_condition(x, emb, self.emb_proj1, self.concat_fuse1)
            if self.cond_inject == "concat"
            else x
        )
        h = self.conv1(h_in)
        h = self.norm1(h)

        if self.cond_inject == "adagn" and emb is not None and self.emb_proj1 is not None:
            h = self._apply_AdaGN(h, emb, self.emb_proj1)
        elif self.cond_inject == "add":
            h = self._apply_add(h, emb, self.emb_proj1)

        h = self.act(h)

        h_in = (
            self._concat_condition(h, emb, self.emb_proj2, self.concat_fuse2)
            if self.cond_inject == "concat"
            else h
        )
        h = self.conv2(h_in)
        h = self.norm2(h)
        if self.cond_inject == "adagn" and emb is not None and self.emb_proj2 is not None:
            h = self._apply_AdaGN(h, emb, self.emb_proj2)
        elif self.cond_inject == "add":
            h = self._apply_add(h, emb, self.emb_proj2)
        h = self.act(h)

        return h + residual
