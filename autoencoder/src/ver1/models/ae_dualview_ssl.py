from __future__ import annotations

from .unet_dualview_ssl import UNetDualViewSSL

# AE backbone is a UNet with skip connections (recon-only dual-view).
AEDualViewSSL = UNetDualViewSSL

__all__ = ["AEDualViewSSL", "UNetDualViewSSL"]
