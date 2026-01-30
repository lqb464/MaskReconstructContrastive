import torch
import torch.nn as nn
from einops import rearrange
from typing import Optional

# -------------------------
# Utility
# -------------------------
def flip_lr(x: torch.Tensor) -> torch.Tensor:
    """Left-right flip on width dimension for NCHW tensor."""
    return torch.flip(x, dims=[-1])


def flip_lr_nhwc(x: torch.Tensor) -> torch.Tensor:
    """Left-right flip on width dimension for NHWC tensor."""
    return torch.flip(x, dims=[2])


def nchw_to_nhwc(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 2, 3, 1).contiguous()


def nhwc_to_nchw(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 3, 1, 2).contiguous()


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

# -------------------------
# Swin blocks (minimal)
# -------------------------
def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    B, H, W, C = x.shape
    return rearrange(x, "b (nh ws1) (nw ws2) c -> (b nh nw) ws1 ws2 c", ws1=window_size, ws2=window_size)


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int, B: int) -> torch.Tensor:
    return rearrange(
        windows,
        "(b nh nw) ws1 ws2 c -> b (nh ws1) (nw ws2) c",
        b=B,
        ws1=window_size,
        ws2=window_size,
        nh=H // window_size,
        nw=W // window_size,
    )


def compute_attn_mask(H: int, W: int, window_size: int, shift_size: int, device: torch.device) -> Optional[torch.Tensor]:
    if shift_size == 0:
        return None
    img_mask = torch.zeros((1, H, W, 1), device=device)
    cnt = 0
    h_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
    w_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1
    mask_windows = window_partition(img_mask, window_size).view(-1, window_size * window_size)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


_ATTN_MASK_CACHE: dict[tuple[int, int, int, int, str, int | None], torch.Tensor] = {}


def get_attn_mask_cached(
    H: int,
    W: int,
    window_size: int,
    shift_size: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if shift_size == 0:
        return None
    key = (H, W, window_size, shift_size, device.type, device.index)
    attn_mask = _ATTN_MASK_CACHE.get(key, None)
    if attn_mask is None or attn_mask.device != device:
        attn_mask = compute_attn_mask(H, W, window_size, shift_size, device)
        _ATTN_MASK_CACHE[key] = attn_mask
    return attn_mask
