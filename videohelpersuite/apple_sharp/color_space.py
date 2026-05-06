"""Contains color space utility functions.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

from typing import Literal

import torch

ColorSpace = Literal["sRGB", "linearRGB"]


def encode_color_space(color_space: ColorSpace) -> int:
    """Encode color space to integer."""
    return 0 if color_space == "sRGB" else 1


def decode_color_space(color_space_index: int) -> ColorSpace:
    """Decode color space index to color space."""
    return "sRGB" if color_space_index == 0 else "linearRGB"


def sRGB2linearRGB(sRGB: torch.Tensor) -> torch.Tensor:
    """sRGB to linearRGB (inference-only, no gradient workaround needed)."""
    THRESHOLD = 0.04045
    return torch.where(
        sRGB <= THRESHOLD,
        sRGB / 12.92,
        ((sRGB + 0.055) / 1.055) ** 2.4,
    )


def linearRGB2sRGB(linearRGB: torch.Tensor) -> torch.Tensor:
    """linearRGB to sRGB (inference-only, no gradient workaround needed)."""
    THRESHOLD = 0.0031308
    return torch.where(
        linearRGB <= THRESHOLD,
        linearRGB * 12.92,
        1.055 * (linearRGB.clamp(min=THRESHOLD) ** (1 / 2.4)) - 0.055,
    )
