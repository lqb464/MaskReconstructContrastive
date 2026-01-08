from __future__ import annotations

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


class UnsharpMask(nn.Module):
    def __init__(self, kernel_size: int = 5, sigma: float = 1.0, amount: float = 1.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.amount = amount
        kernel = self._make_gaussian_kernel(kernel_size, sigma)
        self.register_buffer("kernel", kernel)

    @staticmethod
    def _make_gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
        ax = torch.arange(kernel_size) - kernel_size // 2
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma * sigma))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, kernel_size, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [1,H,W] or [B,1,H,W]
        if x.dim() == 3:
            x_in = x.unsqueeze(0)
            squeeze_back = True
        else:
            x_in = x
            squeeze_back = False

        blur = F.conv2d(x_in, self.kernel, padding=self.kernel_size // 2)
        sharp = x_in + self.amount * (x_in - blur)
        sharp = torch.clamp(sharp, 0.0, 1.0)

        if squeeze_back:
            return sharp.squeeze(0)
        return sharp


def build_base_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ]
    )
