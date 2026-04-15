from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from emgflow.model.utils.scheduler import linear_beta_schedule, get_alpha_terms
from emgflow.model.utils.patchEMG_extract import extract, random_two_consecutive_patches, ddim_timesteps
from emgflow.model.utils.common import (
    EMA,
    get_sinusoidal_time_embedding,
    TimeMLP,
    EfficientAttention1D,
    ResBlock1D,
)

#readme 
'''this file contains EMGFlow diffusion (DDPM/DDIM) model and utilities'''
#API :PatchEMGUNet1D, DiffusionPatchEMG


# ----------------------------
# PatchEMG-like 1D U-Net backbone for DDPM/DDIM
# ----------------------------
class PatchEMGUNet1D(nn.Module):
    """
    Generic DDPM/DDIM backbone.

    Input:
      x: (B, 12, 200)
      t: (B,) timesteps
      y: (B,) class labels or None

    Output:
      prediction: (B, 12, 200)
    """

    def __init__(
        self,
        in_ch: int = 12,
        base_ch: int = 128,
        bottleneck_ch: int = 256,
        num_classes: int = 53,      # include 0 as "null" if you want; you can also manage null externally
        time_dim: int = 128,
        emb_dim: int = 256,
        attn_heads: int = 8,
        attn_head_dim: int = 32,
        norm_type: str = "gn",
        cond_inject: str = "adagn",
    ):
        super().__init__()

        # time embedding
        self.time_dim = time_dim
        self.t_mlp = TimeMLP(time_dim=time_dim, out_dim=emb_dim)

        # label embedding (for conditional generation / CFG)
        self.num_classes = num_classes
        self.y_emb = nn.Embedding(num_classes, emb_dim)

        # --- Downsampling (Table I) ---

        self.down1 = ResBlock1D(
            in_ch,
            base_ch,
            k1=32,
            k2=3,
            emb_dim=emb_dim,
            norm_type=norm_type,
            cond_inject=cond_inject,
        )
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)  # 200 -> 100

        self.down2 = ResBlock1D(
            base_ch,
            bottleneck_ch,
            k1=24,
            k2=6,
            emb_dim=emb_dim,
            norm_type=norm_type,
            cond_inject=cond_inject,
        )
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)  # 100 -> 50

        # --- Bottleneck Attention ---
        self.attn = EfficientAttention1D(bottleneck_ch, heads=attn_heads, head_dim=attn_head_dim)
        # --- Upsampling (Table II) ---
        self.up1 = nn.ConvTranspose1d(bottleneck_ch, bottleneck_ch, kernel_size=4, stride=2, padding=1)

        self.up1_blk = ResBlock1D(
            bottleneck_ch,
            base_ch,
            k1=3,
            k2=3,
            emb_dim=emb_dim,
            norm_type=norm_type,
            cond_inject=cond_inject,
        )

        self.up2 = nn.ConvTranspose1d(base_ch, base_ch, kernel_size=4, stride=2, padding=1)


        self.up2_blk = ResBlock1D(
            base_ch,
            base_ch,
            k1=3,
            k2=3,
            emb_dim=emb_dim,
            norm_type=norm_type,
            cond_inject=cond_inject,
        )

        # final projection to noise prediction
        self.out = nn.Conv1d(base_ch, in_ch, kernel_size=3, padding="same")

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, 12, L)
        t: (B,)
        y: (B,) or None
        """
        # build conditioning embedding
        t_emb = get_sinusoidal_time_embedding(t, self.time_dim)  # (B, time_dim)
        emb = self.t_mlp(t_emb)  # (B, emb_dim)

        if y is not None:
            emb = emb + self.y_emb(y)

        # Down path
        d1 = self.down1(x, emb)          # (B, 128, 200)
        p1 = self.pool1(d1)              # (B, 128, 100)

        d2 = self.down2(p1, emb)       # (B, 256, 100)
        p2 = self.pool2(d2)             # (B, 256, 50)

        # Bottleneck attention + residual
        b = p2 + self.attn(p2)           # (B, 256, 50)

        # Up path
        u1 = self.up1(b)                 # (B, 128, 100)
        # optional skip: add d2a (128ch) rather than d2b (256ch) to match channels cleanly
        u1 = u1 + d2
        u1 = self.up1_blk(u1, emb)

        u2 = self.up2(u1)                # (B, 128, 200)
        u2 = u2 + d1
        u2 = self.up2_blk(u2, emb)

        return self.out(u2)

# ----------------------------
#DDPM wrapper class
# ----------------------------  


class DiffusionPatchEMG(nn.Module):
    """
    PatchEMG diffusion wrapper
    - DDPM training (epsilon prediction, Eq.(5))
    - DDIM / DDPM sampling
    - Classifier-Free Guidance (Eq.(17))
    - Patch-based training (Fig.2 description)
    """

    def __init__(
        self,
        backbone,
        T=1000,
        cfg_dropout=0.05,
        device="cuda",
        patch_strategy_enabled=True,
        use_ema=True,
        ema_decay=0.9999,
        ema_start_step=0,
        prediction_target="eps",
    ):
        super().__init__()
        self.backbone = backbone
        self.T = T
        self.cfg_dropout = cfg_dropout
        self.device = device
        self.prediction_target = self._normalize_prediction_target(prediction_target)

        # schedule (paper not specified → linear)
        betas = linear_beta_schedule(T).to(device)
        alphas, alphas_cumprod, alphas_cumprod_prev = get_alpha_terms(betas)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_mean_coef1 = (
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )
        self.patch_strategy_enabled = patch_strategy_enabled
        
        # EMA
        self.use_ema = use_ema
        if use_ema:
            self.ema = EMA(backbone, decay=ema_decay, start_step=ema_start_step)
        else:
            self.ema = None

    @staticmethod
    def _normalize_prediction_target(target: str) -> str:
        key = str(target).strip().lower()
        aliases = {
            "e": "eps",
            "eps": "eps",
            "epsilon": "eps",
            "noise": "eps",
            "x": "x0",
            "x0": "x0",
            "sample": "x0",
            "data": "x0",
        }
        if key not in aliases:
            raise ValueError(
                f"Unsupported DDPM prediction_target='{target}'. "
                "Use one of: e, eps, epsilon, x, x0."
            )
        return aliases[key]

    @staticmethod
    def _normalize_solver(solver: str | None) -> str:
        key = "ddim" if solver is None else str(solver).strip().lower()
        aliases = {
            "ddim": "ddim",
            "ddpm": "ddpm",
        }
        if key not in aliases:
            raise ValueError(
                f"Unsupported DDPM solver '{solver}'. "
                "Use one of: DDIM, DDPM."
            )
        return aliases[key]
            
    def update_ema(self):
        """Should be called after optimizer.step()"""
        if self.use_ema and self.ema is not None:
            self.ema.update(self.backbone)
    # ---------------------------------------------------
    # Forward diffusion q(x_t | x_0)
    # ---------------------------------------------------
    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        return (
            extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise
        )

    def _target_from_x0_noise(self, x0, noise):
        if self.prediction_target == "eps":
            return noise
        if self.prediction_target == "x0":
            return x0
        raise RuntimeError(f"Unhandled prediction_target={self.prediction_target}")

    def _predict_model_output(self, x_t, t, y=None, guidance_w=1.0, use_ema_model=True):
        model = self.ema.ema_model if (use_ema_model and self.ema is not None) else self.backbone

        if guidance_w == 1.0 or y is None:
            return model(x_t, t, y)

        pred_uncond = model(x_t, t, torch.zeros_like(y))
        pred_cond = model(x_t, t, y)
        return pred_uncond + guidance_w * (pred_cond - pred_uncond)

    def _x0_from_eps(self, x_t, t, eps):
        alpha_bar = extract(self.alphas_cumprod, t, x_t.shape)
        return (x_t - torch.sqrt(1 - alpha_bar) * eps) / torch.sqrt(alpha_bar)

    def _eps_from_x0(self, x_t, t, x0):
        alpha_bar = extract(self.alphas_cumprod, t, x_t.shape)
        return (x_t - torch.sqrt(alpha_bar) * x0) / torch.sqrt(1 - alpha_bar)

    # ---------------------------------------------------
    # Training step (Patch-based)
    # ---------------------------------------------------
    def training_step(self, x0, y):
        """
        x0: (B,C,L_full) already normalized
        y : (B,)
        """
        B = x0.shape[0]
        device = x0.device

        t = torch.randint(0, self.T, (B,), device=device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)

        # CFG label dropout
        if y is not None:
            drop = torch.rand(B, device=device) < self.cfg_dropout
            y = y.clone()
            y[drop] = 0  # assume 0 is null label (see dataloader)

        # Patch sampling (two consecutive patches)
        if self.patch_strategy_enabled:
            p1_xt, p2_xt, start, L1, L2 = random_two_consecutive_patches(x_t)
            p1_eps = noise[:, :, start : start + L1]
            p2_eps = noise[:, :, start + L1 : start + L1 + L2]
            p1_x0 = x0[:, :, start : start + L1]
            p2_x0 = x0[:, :, start + L1 : start + L1 + L2]
            target_1 = self._target_from_x0_noise(p1_x0, p1_eps)
            target_2 = self._target_from_x0_noise(p2_x0, p2_eps)

            pred_1 = self.backbone(p1_xt, t, y)
            pred_2 = self.backbone(p2_xt, t, y)

            # Loss (equal weights, paper not specified)
            loss1 = F.mse_loss(pred_1, target_1)
            loss2 = F.mse_loss(pred_2, target_2)
            weight = p1_xt.shape[-1]/(p1_xt.shape[-1]+p2_xt.shape[-1])

            loss = weight*loss1 + loss2*(1- weight)

            return loss, {
                "loss_patch1": loss1.item(),
                "loss_patch2": loss2.item(),
                "t_mean": t.float().mean().item(),
            }
        else:
            # Full sequence training
            target = self._target_from_x0_noise(x0, noise)
            pred = self.backbone(x_t, t, y)
            loss = F.mse_loss(pred, target)
            return loss, {
                "loss_full": loss.item(),
                "t_mean": t.float().mean().item(),
            }

    # ---------------------------------------------------
    # Epsilon prediction with CFG
    # ---------------------------------------------------
    def predict_eps(self, x_t, t, y=None, guidance_w=1.0, use_ema_model=True):
        pred = self._predict_model_output(x_t, t, y, guidance_w, use_ema_model)
        if self.prediction_target == "eps":
            return pred
        if self.prediction_target == "x0":
            return self._eps_from_x0(x_t, t, pred)
        raise RuntimeError(f"Unhandled prediction_target={self.prediction_target}")

    def predict_x0(self, x_t, t, y=None, guidance_w=1.0, use_ema_model=True):
        pred = self._predict_model_output(x_t, t, y, guidance_w, use_ema_model)
        if self.prediction_target == "x0":
            return pred
        if self.prediction_target == "eps":
            return self._x0_from_eps(x_t, t, pred)
        raise RuntimeError(f"Unhandled prediction_target={self.prediction_target}")

    # ---------------------------------------------------
    # Sampling
    # ---------------------------------------------------
    @torch.no_grad()
    def _sample_ddim(
        self,
        y,
        shape,
        steps=50,
        guidance_w=1.0,
        eta=0.0,
        use_ema=True,
    ):
        if steps <= 0:
            raise ValueError(f"DDIM steps must be > 0, got {steps}")

        B = shape[0]
        x = torch.randn(shape, device=self.device)
        tau = ddim_timesteps(self.T, steps).to(self.device)

        for i in reversed(range(steps)):
            t = tau[i].repeat(B)
            if i == 0:
                return self.predict_x0(x, t, y, guidance_w, use_ema_model=use_ema)

            eps = self.predict_eps(x, t, y, guidance_w, use_ema_model=use_ema)
            alpha_bar = extract(self.alphas_cumprod, t, x.shape)
            t_prev = tau[i - 1].repeat(B)
            alpha_bar_prev = extract(self.alphas_cumprod, t_prev, x.shape)

            x0_hat = (x - torch.sqrt(1 - alpha_bar) * eps) / torch.sqrt(alpha_bar)
            sigma = (
                eta
                * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
            )

            noise = torch.randn_like(x)
            x = (
                torch.sqrt(alpha_bar_prev) * x0_hat
                + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
                + sigma * noise
            )

        return x

    @torch.no_grad()
    def _sample_ddpm(
        self,
        y,
        shape,
        guidance_w=1.0,
        use_ema=True,
    ):
        B = shape[0]
        x = torch.randn(shape, device=self.device)

        for step in reversed(range(self.T)):
            t = torch.full((B,), step, device=self.device, dtype=torch.long)
            x0_hat = self.predict_x0(x, t, y, guidance_w, use_ema_model=use_ema)

            if step == 0:
                return x0_hat

            mean = (
                extract(self.posterior_mean_coef1, t, x.shape) * x0_hat
                + extract(self.posterior_mean_coef2, t, x.shape) * x
            )
            noise = torch.randn_like(x)
            var = extract(self.posterior_variance, t, x.shape).clamp_min(1e-20)
            x = mean + torch.sqrt(var) * noise

        return x

    @torch.no_grad()
    def sample(
        self,
        y,
        shape,
        solver="DDIM",
        steps=50,
        guidance_w=1.0,
        eta=0.0,
        use_ema=True
    ):
        """
        shape: (B,C,L_full)
        solver: "DDIM" (default) or "DDPM"
        use_ema: If True, use EMA model for sampling (if initialized)
        """
        B = shape[0]
        #aligning y 
        if y is not None:
            if not torch.is_tensor(y):
                y = torch.full((B,), int(y), device=self.device, dtype=torch.long)
            else:
                y = y.to(self.device)
                if y.ndim == 0:
                    y = y.repeat(B)
        solver_key = self._normalize_solver(solver)
        if solver_key == "ddpm":
            return self._sample_ddpm(
                y=y,
                shape=shape,
                guidance_w=guidance_w,
                use_ema=use_ema,
            )
        return self._sample_ddim(
            y=y,
            shape=shape,
            steps=steps,
            guidance_w=guidance_w,
            eta=eta,
            use_ema=use_ema,
        )
