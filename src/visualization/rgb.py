"""
3-Channel False-Color RGB Photo Synthesizer & Chemical Mapping Engine.

Blends multiple 2D monochromatic slices or continuum-subtracted emission line maps
into NASA press-release quality 3-channel RGB composite photographs.
"""

from __future__ import annotations

import numpy as np
from src.visualization.super_res import upsample_spatial_grid, apply_asinh_stretch, apply_log_stretch, unsharp_mask_filter


CHEMICAL_PRESETS = {
    "Crab Nebula NASA Signature (Orange/Cyan/Blue)": {
        "description": "Red: Warm Dust & S III (Outer Orange Filaments) | Green: Ne II & S IV (Cyan Web) | Blue: Ar III Synchrotron Core (Electric Blue)",
        "red": "emission_[S III]_18.71",
        "green": "emission_[Ne II]_12.81",
        "blue": "emission_[Ar III]_8.99",
    },
    "Star Formation & Dust Rings": {
        "description": "Red: PAH 11.3µm dust | Green: [Ne II] 12.81µm ionized gas | Blue: [Ar III] 8.99µm high-excitation",
        "red": "PAH_11.30",
        "green": "[Ne II]_12.81",
        "blue": "[Ar III]_8.99",
    },
    "Molecular vs Ionized Gas": {
        "description": "Red: Continuum dust | Green: H₂ 9.66µm molecular gas | Blue: [S IV] 10.51µm ionized gas",
        "red": "full_band_integrated",
        "green": "H₂_9.66",
        "blue": "[S IV]_10.51",
    },
    "High-Excitation Ionization Structure": {
        "description": "Red: [S III] 18.71µm | Green: [Ne II] 12.81µm | Blue: [S IV] 10.51µm",
        "red": "[S III]_18.71",
        "green": "[Ne II]_12.81",
        "blue": "[S IV]_10.51",
    },
}


def clean_channel_noise(img: np.ndarray, bg_percentile: float = 20.0) -> np.ndarray:
    """Subtract sky background noise floor to prevent random noise from becoming colorful blobs."""
    clean = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    bg_floor = np.percentile(clean, bg_percentile)
    subtracted = np.maximum(0.0, clean - bg_floor)
    
    # If peak signal is negligible (pure noise array), zero it out
    if np.max(subtracted) < 1e-10:
        return np.zeros_like(clean)
        
    return subtracted


def create_rgb_composite(
    red_img: np.ndarray,
    green_img: np.ndarray,
    blue_img: np.ndarray,
    stretch: str = "asinh",
    asinh_a: float = 0.1,
    saturation: float = 1.3,
    gamma: float = 1.0,
    super_res_factor: int = 10,
    unsharp_mask_amount: float = 0.0,
    bg_mask_threshold: float = 0.02,
) -> np.ndarray:
    """Synthesize a high-definition 3-channel RGB astronomical photograph.

    Parameters
    ----------
    red_img, green_img, blue_img : np.ndarray
        2D spatial arrays for Red, Green, and Blue channels.
    stretch : str
        Intensity scaling ('asinh', 'log', 'linear').
    asinh_a : float
        Asinh softening parameter.
    saturation : float
        Color saturation multiplier (1.0 = normal, >1.0 = vivid).
    gamma : float
        Gamma correction exponent.
    super_res_factor : int
        Spatial upsampling scale factor.
    unsharp_mask_amount : float
        Filament sharpening strength factor.
    bg_mask_threshold : float
        Background noise luminance cutoff below which pixels are masked to black.

    Returns
    -------
    np.ndarray
        RGB image array of shape (H, W, 3) with float values in [0.0, 1.0].
    """
    # 0. Clean Noise Floor
    r_clean = clean_channel_noise(red_img)
    g_clean = clean_channel_noise(green_img)
    b_clean = clean_channel_noise(blue_img)

    # 1. Spatial Super-Resolution Upsampling
    r_hd = upsample_spatial_grid(r_clean, factor=super_res_factor)
    g_hd = upsample_spatial_grid(g_clean, factor=super_res_factor)
    b_hd = upsample_spatial_grid(b_clean, factor=super_res_factor)

    # 1b. Unsharp Masking Filament Enhancement
    if unsharp_mask_amount > 0:
        r_hd = unsharp_mask_filter(r_hd, amount=unsharp_mask_amount)
        g_hd = unsharp_mask_filter(g_hd, amount=unsharp_mask_amount)
        b_hd = unsharp_mask_filter(b_hd, amount=unsharp_mask_amount)

    # 2. Non-Linear Dynamic Range Stretch
    if stretch == "asinh":
        r_str = apply_asinh_stretch(r_hd, a=asinh_a)
        g_str = apply_asinh_stretch(g_hd, a=asinh_a)
        b_str = apply_asinh_stretch(b_hd, a=asinh_a)
    elif stretch == "log":
        r_str = apply_log_stretch(r_hd)
        g_str = apply_log_stretch(g_hd)
        b_str = apply_log_stretch(b_hd)
    else:
        def lin_norm(arr):
            v0, v1 = np.percentile(arr, (1, 99.5))
            return np.clip((arr - v0) / max(1e-12, v1 - v0), 0, 1) if v1 > v0 else np.zeros_like(arr)
        r_str = lin_norm(r_hd)
        g_str = lin_norm(g_hd)
        b_str = lin_norm(b_hd)

    # 3. Stack into RGB array
    rgb = np.dstack([r_str, g_str, b_str])

    # 4. Gamma Correction
    if gamma != 1.0 and gamma > 0:
        rgb = np.power(rgb, 1.0 / gamma)

    # 5. Saturation Enhancement
    if saturation != 1.0:
        gray = np.mean(rgb, axis=2, keepdims=True)
        rgb = gray + saturation * (rgb - gray)

    # 6. Deep Cosmic Background Masking (mask low-luminance noise to pure black)
    luminance = np.max(rgb, axis=2, keepdims=True)
    bg_mask = (luminance < bg_mask_threshold)
    rgb[bg_mask.repeat(3, axis=2)] = 0.0

    return np.clip(rgb, 0.0, 1.0)
