# diffusion/utils.py
import torch
import random

def extract(a: torch.Tensor, t: torch.Tensor, x_shape):
    """
    a: (T,)
    t: (B,)
    return: (B, 1, 1) broadcastable to x_shape
    """
    out = a.gather(0, t)
    return out.view(-1, *([1] * (len(x_shape) - 1)))

def random_two_consecutive_patches(x, min_len=100, max_len=None):
    """
    PatchEMG patch strategy:
    - two consecutive patches
    - varying sizes
    - sampled from same signal
    """
    _, _, L = x.shape
    if max_len is None:
        max_len = L // 2

    L1 = random.randint(min_len, max_len)
    L2 = random.randint(min_len, max_len)
    
    total = L1 + L2
    if total >= L:
        total = L - 1
        L1 = total // 2
        L2 = total - L1
    L1 = (L1//4)*4  # make divisible by 4
    L2 = (L2//4)*4
    start = random.randint(0, L - total)

    p1 = x[:, :, start : start + L1]
    p2 = x[:, :, start + L1 : start + L1 + L2]
    return p1, p2, start, L1, L2

def ddim_timesteps(T, S):
    """
    Uniform DDIM subsequence τ
    """
    return torch.linspace(0, T - 1, S, dtype=torch.long)
