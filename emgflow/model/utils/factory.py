from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn

from emgflow.model.DDPM import DiffusionPatchEMG, PatchEMGUNet1D
from emgflow.model.flow_matching import FlowMatchingPatchEMG
from emgflow.model.gan.pure_wgan_gp_1d import PureWGANGenerator1D, build_pure_wgan_from_config


class BaseModelAdapter(ABC):
    def __init__(self, model, device: str):
        self.model = model
        self.device = device

    @abstractmethod
    def load_checkpoint(self, ckpt_path: str) -> None:
        pass

    @abstractmethod
    def sample(self, y, shape, **kwargs):
        pass

    def eval(self) -> None:
        self.model.eval()
        if hasattr(self.model, "ema") and self.model.ema:
            self.model.ema.ema_model.eval()

    @staticmethod
    def filter_supported_kwargs(fn, kwargs: dict[str, Any]) -> dict[str, Any]:
        sig = inspect.signature(fn)
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            return kwargs
        accepted = {
            name
            for name, p in sig.parameters.items()
            if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        return {k: v for k, v in kwargs.items() if k in accepted}


class DDPMAdapter(BaseModelAdapter):
    def load_checkpoint(self, ckpt_path: str) -> None:
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        self.model.backbone.load_state_dict(state_dict)
        if self.model.ema is not None and isinstance(ckpt, dict) and "ema" in ckpt:
            self.model.ema.ema_model.load_state_dict(ckpt["ema"])

    def sample(self, y, shape, **kwargs):
        return self.model.sample(
            y=y,
            shape=shape,
            solver=str(kwargs.get("solver", "ddim")).lower(),
            steps=kwargs.get("steps", 50),
            guidance_w=kwargs.get("guidance_w", 1.0),
            eta=kwargs.get("eta", 0.0),
            use_ema=kwargs.get("use_ema", True),
        )


class FlowMatchingAdapter(BaseModelAdapter):
    def load_checkpoint(self, ckpt_path: str) -> None:
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        self.model.backbone.load_state_dict(state_dict)
        if self.model.ema is not None and isinstance(ckpt, dict) and "ema" in ckpt:
            self.model.ema.ema_model.load_state_dict(ckpt["ema"])

    def sample(self, y, shape, **kwargs):
        return self.model.sample(
            y=y,
            shape=shape,
            steps=kwargs.get("steps", 50),
            method=str(kwargs.get("solver", kwargs.get("ode_solver", "heun"))).lower(),
            guidance_w=kwargs.get("guidance_w", 1.0),
            use_ema=kwargs.get("use_ema", True),
        )


class GANAdapter(BaseModelAdapter):
    def load_checkpoint(self, ckpt_path: str) -> None:
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        state_dict = ckpt
        if isinstance(ckpt, dict):
            for key in ("G_ema", "G", "model_state_dict", "model", "state_dict"):
                if key in ckpt:
                    state_dict = ckpt[key]
                    break
        self.model.load_state_dict(state_dict)

    def sample(self, y, shape, **kwargs):
        del kwargs
        bsz = int(shape[0])
        if not torch.is_tensor(y):
            y = torch.full((bsz,), int(y), device=self.device, dtype=torch.long)
        else:
            y = y.to(self.device, dtype=torch.long)
        z = torch.randn(bsz, int(self.model.noise_dim), device=self.device)
        return self.model(z, y)


class ModelFactory:
    @staticmethod
    def _build_unet(model_cfg: dict[str, Any], num_classes: int, device: str) -> nn.Module:
        return PatchEMGUNet1D(
            in_ch=int(model_cfg.get("in_ch", model_cfg.get("in_channels", 12))),
            base_ch=int(model_cfg.get("base_ch", 128)),
            bottleneck_ch=int(model_cfg.get("bottleneck_ch", 256)),
            num_classes=int(model_cfg.get("num_classes", num_classes)),
            time_dim=int(model_cfg.get("time_dim", 128)),
            emb_dim=int(model_cfg.get("emb_dim", 256)),
            attn_heads=int(model_cfg.get("attn_heads", 8)),
            attn_head_dim=int(model_cfg.get("attn_head_dim", 32)),
            norm_type=str(model_cfg.get("norm_type", model_cfg.get("norm", "gn"))),
            cond_inject=str(model_cfg.get("cond_inject", model_cfg.get("condition_inject", "adagn"))),
        ).to(device)

    @classmethod
    def create(cls, config: dict[str, Any], device: str, num_classes: int) -> BaseModelAdapter:
        model_cfg = dict(config["model"])
        model_type = str(model_cfg.get("type", "ddpm")).strip().lower()

        if model_type == "ddpm":
            backbone = cls._build_unet(model_cfg, num_classes, device)
            model = DiffusionPatchEMG(
                backbone=backbone,
                T=int(model_cfg.get("T", 1000)),
                cfg_dropout=float(model_cfg.get("cfg_dropout", 0.05)),
                device=device,
                patch_strategy_enabled=bool(model_cfg.get("patch_strategy_enabled", True)),
                use_ema=bool(model_cfg.get("use_ema", True)),
                ema_decay=float(model_cfg.get("ema_decay", 0.9999)),
                ema_start_step=int(model_cfg.get("ema_start_step", 0)),
                prediction_target=str(model_cfg.get("prediction_target", "eps")),
            )
            return DDPMAdapter(model, device)

        if model_type == "flowmatching":
            backbone = cls._build_unet(model_cfg, num_classes, device)
            model = FlowMatchingPatchEMG(
                backbone=backbone,
                sigma_min=float(model_cfg.get("sigma_min", 0.0)),
                cfg_dropout=float(model_cfg.get("cfg_dropout", 0.05)),
                device=device,
                patch_strategy_enabled=bool(model_cfg.get("patch_strategy_enabled", False)),
                use_ema=bool(model_cfg.get("use_ema", True)),
                ema_decay=float(model_cfg.get("ema_decay", 0.9999)),
                ema_start_step=int(model_cfg.get("ema_start_step", 0)),
                t_sampling=model_cfg.get("t_sampling"),
            )
            return FlowMatchingAdapter(model, device)

        if model_type == "pure_wgan_gp":
            gan_cfg = {
                "in_channels": int(model_cfg.get("in_channels", model_cfg.get("in_ch", 12))),
                "signal_length": int(model_cfg.get("signal_length", 400)),
                "num_classes": int(model_cfg.get("num_classes", num_classes)),
                "noise_dim": int(model_cfg.get("noise_dim", 64)),
                "pure_wgan_gp": dict(model_cfg.get("pure_wgan_gp", {})),
            }
            generator, _ = build_pure_wgan_from_config(gan_cfg)
            return GANAdapter(generator.to(device), device)

        raise ValueError(
            f"Unsupported model type '{model_type}'. "
            "Public release supports: ddpm, flowmatching, pure_wgan_gp."
        )
