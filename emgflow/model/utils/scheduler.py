# diffusion/schedules.py
import torch

def linear_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    """
    Paper does not specify schedule → use standard linear DDPM schedule.
    """
    return torch.linspace(beta_start, beta_end, T)

def get_alpha_terms(betas: torch.Tensor):
    """
    betas: (T,)
    returns:
      alphas, alphas_cumprod, alphas_cumprod_prev
    """
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat(
        [torch.ones(1, device=betas.device), alphas_cumprod[:-1]], dim=0
    )
    return alphas, alphas_cumprod, alphas_cumprod_prev
