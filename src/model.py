"""U-Net for binary building segmentation (outputs logits)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Sequential):
    """(Conv 3x3 -> BatchNorm -> ReLU) x 2."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    """Symmetric encoder-decoder with skip connections (Ronneberger et al., 2015).

    Returns raw logits; apply ``torch.sigmoid`` for probabilities. Keeping the
    sigmoid out of the network lets the loss use the numerically stable
    ``BCEWithLogitsLoss``.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 32, depth: int = 4, dropout: float = 0.1):
        super().__init__()
        chs = [base_channels * 2 ** i for i in range(depth + 1)]
        self.encoders = nn.ModuleList()
        prev = in_channels
        for ch in chs[:-1]:
            self.encoders.append(ConvBlock(prev, ch))
            prev = ch
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(ConvBlock(chs[-2], chs[-1]), nn.Dropout2d(dropout))
        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for ch in reversed(chs[:-1]):
            self.upsamples.append(nn.ConvTranspose2d(ch * 2, ch, 2, stride=2))
            self.decoders.append(ConvBlock(ch * 2, ch))
        self.head = nn.Conv2d(chs[0], 1, 1)
        self.divisor = 2 ** depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        pad_h, pad_w = (-h) % self.divisor, (-w) % self.divisor
        if pad_h or pad_w:  # any input size works
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.upsamples, self.decoders, reversed(skips)):
            x = dec(torch.cat([up(x), skip], dim=1))
        return self.head(x)[..., :h, :w]


def build_model(cfg: dict) -> UNet:
    m = cfg["model"]
    return UNet(base_channels=m["base_channels"], depth=m["depth"], dropout=m["dropout"])
