import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Any, Optional, Tuple
from emgflow.model.utils.patchEMG_extract import random_two_consecutive_patches
from emgflow.model.utils.common import EMA

# Flow Matching Wrapper
# Implements Conditional Flow Matching (CFM) with Optimal Transport path
# Target velocity v_t(x) = x1 - x0
# Interpolation x_t = (1-t)x0 + t*x1 (t in [0,1])

class FlowMatchingPatchEMG(nn.Module):
    """
    PatchEMG Flow Matching wrapper.
    - Path: Optimal Transport (Linear interpolation)
    - x_t = (1 - t) * x0 + t * x1
    - Vector Field target: u_t(x_t|x0,x1) = x1 - x0
    - Training: Regression on vector field
    - Sampling: ODE solver (Euler or RK4)
    """

    def __init__(
        self,
        backbone,  # e.g., PatchEMGUNet1D or DiT1D
        sigma_min: float = 0.0,  # Min noise level (usually 0 or small)
        cfg_dropout: float = 0.05,
        device: str = "cuda",
        patch_strategy_enabled: bool = False,
        use_ema: bool = True,
        ema_decay: float = 0.9999,
        ema_start_step: int = 0,
        t_sampling: Optional[dict[str, Any]] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.backbone.to(device)
        self.sigma_min = sigma_min
        self.cfg_dropout = cfg_dropout
        self.device = device
        self.patch_strategy_enabled = patch_strategy_enabled

        # EMA Setup
        self.use_ema = use_ema
        if use_ema:
            self.ema = EMA(backbone, decay=ema_decay, start_step=ema_start_step)
        else:
            self.ema = None

        t_sampling = dict(t_sampling or {})
        self.t_sampling_mode = str(t_sampling.get("mode", "uniform")).strip().lower()
        if self.t_sampling_mode not in {"uniform", "logit_normal"}:
            raise ValueError(
                f"Unsupported t_sampling.mode='{self.t_sampling_mode}'. "
                "Use one of: uniform, logit_normal."
            )

        logit_normal_cfg = dict(t_sampling.get("logit_normal", {}))
        self.t_sampling_eps = float(t_sampling.get("eps", 1e-6))
        if not (0.0 <= self.t_sampling_eps < 0.5):
            raise ValueError(f"t_sampling.eps must be in [0, 0.5), got {self.t_sampling_eps}")
        self.t_logit_mu = float(logit_normal_cfg.get("mu", 0.0))
        self.t_logit_sigma = float(logit_normal_cfg.get("sigma", 1.0))
        if self.t_logit_sigma <= 0.0:
            raise ValueError(f"logit_normal sigma must be > 0, got {self.t_logit_sigma}")

    def update_ema(self):
        """Should be called after optimizer.step()"""
        if self.use_ema and self.ema is not None:
            self.ema.update(self.backbone)

    # ---------------------------------------------------
    # Core Flow Matching Logic
    # ---------------------------------------------------

    def _sample_t(self, batch: int, reference: torch.Tensor) -> torch.Tensor:
        if self.t_sampling_mode == "uniform":
            return torch.rand(batch, device=self.device, dtype=reference.dtype)

        z = torch.randn(batch, device=self.device, dtype=reference.dtype)
        t = torch.sigmoid(z * self.t_logit_sigma + self.t_logit_mu)
        return t.clamp(self.t_sampling_eps, 1.0 - self.t_sampling_eps)

    def compute_loss(self, x1, y, use_patch=False):
        """
        x1: Real data (B, C, L)
        y:  Labels (B,)
        """
        B, C, L = x1.shape
        
        # 1. Sample t in [0, 1]
        t = self._sample_t(B, x1)
        
        # 2. Sample x0 ~ N(0, I)
        x0 = torch.randn_like(x1)

        # 3. Compute x_t (Linear Interpolation)
        # Using broadcasting: t needs shape (B, 1, 1)
        t_expand = t.view(B, 1, 1)
        
        # Optimal Transport Path
        # x_t = (1 - (1 - sigma_min) * t) * x0 + t * x1  (If following some specific formulation)
        # Standard OT-CFM: x_t = (1 - t) * x0 + t * x1
        x_t = (1 - t_expand) * x0 + t_expand * x1
        
        # 4. Target Vector Field (Velocity)
        # u_t = x1 - x0
        target = x1 - x0
        
        # CFG label dropout
        if y is not None:
            drop = torch.rand(B, device=self.device) < self.cfg_dropout
            y_in = y.clone()
            y_in[drop] = 0  # 0 is null/uncond
        else:
            y_in = None

        # 5. Predict Vector Field
        # Let's scale t to [0, 1000] for embedding consistency with pre-trained models or existing architecture design.
        t_model = t * 1000.0 

        if use_patch and self.patch_strategy_enabled:
            # Patch-based Training
            # Crop patches from x_t and corresponding target
            p1_xt, p2_xt, start, L1, L2 = random_two_consecutive_patches(x_t)
            p1_target = target[:, :, start : start + L1]
            p2_target = target[:, :, start + L1 : start + L1 + L2]
            
            # Forward pass
            v_pred_1 = self.backbone(p1_xt, t_model, y_in)
            v_pred_2 = self.backbone(p2_xt, t_model, y_in)
            
            # Loss
            loss1 = F.mse_loss(v_pred_1, p1_target)
            loss2 = F.mse_loss(v_pred_2, p2_target)
            
            # Weighted loss
            weight = float(L1) / (L1 + L2)
            loss = weight * loss1 + (1 - weight) * loss2
            
            return loss, {
                "loss": loss.item(),
                "loss_p1": loss1.item(), 
                "loss_p2": loss2.item()
            }
        else:
            # Full Sequence Training
            v_pred = self.backbone(x_t, t_model, y_in)
            loss = F.mse_loss(v_pred, target)
            return loss, {"loss": loss.item()}

    def training_step(self, x1, y):
        return self.compute_loss(x1, y, use_patch=self.patch_strategy_enabled)

    # ---------------------------------------------------
    # Sampling (ODE Solver)
    # ---------------------------------------------------

    @torch.no_grad()
    def predict_v(self, x, t_float, y, guidance_w=1.0, use_ema_model=True):
        """
        x: (B, C, L) current state
        t_float: scalar time [0, 1]
        y: labels
        """
        B = x.shape[0]
        t_tensor = torch.full((B,), t_float, device=self.device) * 1000.0
        
        # Select model
        model = self.ema.ema_model if (use_ema_model and self.ema is not None) else self.backbone
        
        # CFG
        if guidance_w == 1.0 or y is None:
            return model(x, t_tensor, y)
        
        # v_uncond
        y_null = torch.zeros_like(y)
        v_uncond = model(x, t_tensor, y_null)
        v_cond = model(x, t_tensor, y)
        
        return v_uncond + guidance_w * (v_cond - v_uncond)

    @torch.no_grad()
    def sample(
        self, 
        y, 
        shape, 
        steps=50, 
        method='euler', 
        guidance_w=2, 
        use_ema=True
    ):
        """
        Generate samples using ODE solver.
        methods: 'euler', 'heun', 'midpoint', 'rk4'
        """
        method = str(method).lower()
        if method not in {'euler', 'heun', 'midpoint', 'rk4'}:
            raise ValueError(
                f"Unsupported FlowMatching solver '{method}'. "
                "Use one of: euler, heun, midpoint, rk4."
            )
        B = shape[0]
        # Align labels
        if y is not None:
            if not torch.is_tensor(y):
                y = torch.full((B,), int(y), device=self.device, dtype=torch.long)
            else:
                y = y.to(self.device)
                if y.ndim == 0:
                    y = y.repeat(B)

        # Initial noise x0
        x = torch.randn(shape, device=self.device)
        
        # Time steps [0 -> 1]
        # Note: Flow matching usually goes 0 -> 1 (Noise -> Data) or 1 -> 0 depending on definition.
        # Here we defined: x_t = (1-t)x0 + t*x1.
        # t=0 is x0 (Noise), t=1 is x1 (Data).
        # So we integrate from 0 to 1.
        ts = torch.linspace(0, 1, steps + 1, device=self.device)
        dt = 1.0 / steps

        for i in range(steps):
            t_curr = ts[i].item()
            
            if method == 'euler':
                v = self.predict_v(x, t_curr, y, guidance_w, use_ema_model=use_ema)
                x = x + v * dt
                
            elif method == 'heun':
                v1 = self.predict_v(x, t_curr, y, guidance_w, use_ema_model=use_ema)
                x_euler = x + v1 * dt
                v2 = self.predict_v(x_euler, t_curr + dt, y, guidance_w, use_ema_model=use_ema)
                x = x + 0.5 * dt * (v1 + v2)

            elif method == 'midpoint':
                # k1
                v1 = self.predict_v(x, t_curr, y, guidance_w, use_ema_model=use_ema)
                x_mid = x + 0.5 * v1 * dt
                # k2
                v2 = self.predict_v(x_mid, t_curr + 0.5 * dt, y, guidance_w, use_ema_model=use_ema)
                x = x + v2 * dt
                
            elif method == 'rk4':
                # k1
                v1 = self.predict_v(x, t_curr, y, guidance_w, use_ema_model=use_ema)
                # k2
                x2 = x + 0.5 * v1 * dt
                v2 = self.predict_v(x2, t_curr + 0.5 * dt, y, guidance_w, use_ema_model=use_ema)
                # k3
                x3 = x + 0.5 * v2 * dt
                v3 = self.predict_v(x3, t_curr + 0.5 * dt, y, guidance_w, use_ema_model=use_ema)
                # k4
                x4 = x + v3 * dt
                # Clamp t to 1.0 for last check? typically fine
                t_next = ts[i+1].item() 
                v4 = self.predict_v(x4, t_next, y, guidance_w, use_ema_model=use_ema)
                
                x = x + (dt / 6.0) * (v1 + 2*v2 + 2*v3 + v4)

        return x
