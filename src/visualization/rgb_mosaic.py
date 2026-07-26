"""
Real-Data WCS Mosaic RGB Composite Synthesizer.

Merges reprojected multi-wavelength FITS mosaic bands into 3-channel RGB photographs
with non-linear AsinhStretch contrast scaling.
"""

from __future__ import annotations

import numpy as np
from src.preprocessing.sky_sub import subtract_sky_background
from src.visualization.super_res import apply_asinh_stretch, apply_log_stretch, unsharp_mask_filter


def synthesize_mosaic_rgb(
    red_mosaic: np.ndarray,
    green_mosaic: np.ndarray,
    blue_mosaic: np.ndarray,
    stretch: str = "asinh",
    asinh_a: float = 0.1,
    saturation: float = 1.4,
    gamma: float = 1.0,
    unsharp_amount: float = 0.6,
) -> np.ndarray:
    """Synthesize a 3-channel RGB photograph from reprojected FITS mosaic arrays.

    Parameters
    ----------
    red_mosaic, green_mosaic, blue_mosaic : np.ndarray
        2D spatial mosaic arrays for Red, Green, and Blue channels.
    stretch : str
        Intensity scaling ('asinh', 'log', 'linear').
    asinh_a : float
        Asinh softening parameter.
    saturation : float
        Color saturation multiplier.
    gamma : float
        Gamma correction exponent.
    unsharp_amount : float
        Filament sharpening strength factor.

    Returns
    -------
    np.ndarray
        RGB float array of shape (H, W, 3) in range [0, 1].
    """
    # 1. 2D Sky Background Subtraction
    r_sub = subtract_sky_background(red_mosaic)
    g_sub = subtract_sky_background(green_mosaic)
    b_sub = subtract_sky_background(blue_mosaic)

    # 2. Filament Unsharp Sharpening
    if unsharp_amount > 0:
        r_sub = unsharp_mask_filter(r_sub, amount=unsharp_amount)
        g_sub = unsharp_mask_filter(g_sub, amount=unsharp_amount)
        b_sub = unsharp_mask_filter(b_sub, amount=unsharp_amount)

    # 3. Non-Linear Contrast Stretch
    if stretch == "asinh":
        r_str = apply_asinh_stretch(r_sub, a=asinh_a)
        g_str = apply_asinh_stretch(g_sub, a=asinh_a)
        b_str = apply_asinh_stretch(b_sub, a=asinh_a)
    elif stretch == "log":
        r_str = apply_log_stretch(r_sub)
        g_str = apply_log_stretch(g_sub)
        b_str = apply_log_stretch(b_sub)
    else:
        def lin_norm(arr):
            v0, v1 = np.percentile(arr, (1, 99.5))
            return np.clip((arr - v0) / max(1e-12, v1 - v0), 0, 1) if v1 > v0 else np.zeros_like(arr)
        r_str = lin_norm(r_sub)
        g_str = lin_norm(g_sub)
        b_str = lin_norm(b_sub)

    # 4. Stack into RGB array
    rgb = np.dstack([r_str, g_str, b_str])

    # 5. Gamma Correction
    if gamma != 1.0 and gamma > 0:
        rgb = np.power(rgb, 1.0 / gamma)

    # 6. Saturation Enhancement
    if saturation != 1.0:
        gray = np.mean(rgb, axis=2, keepdims=True)
        rgb = gray + saturation * (rgb - gray)

    # Mask deep space background
    luminance = np.max(rgb, axis=2, keepdims=True)
    bg_mask = (luminance < 0.02)
    rgb[bg_mask.repeat(3, axis=2)] = 0.0

    return np.clip(rgb, 0.0, 1.0)
