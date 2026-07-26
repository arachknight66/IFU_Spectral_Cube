"""
Spatial Super-Resolution & Intensity Scaling Engine for Astronomical Imagery.

Provides sub-pixel spatial upsampling and non-linear dynamic range compression
(asinh / log stretches) used in STScI / NASA public release imagery.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom, gaussian_filter
try:
    from astropy.visualization import AsinhStretch, LogStretch, IntervalAsymmetricPercentile
    HAS_ASTROPY_VIS = True
except ImportError:
    HAS_ASTROPY_VIS = False


def unsharp_mask_filter(image: np.ndarray, sigma: float = 2.0, amount: float = 0.6) -> np.ndarray:
    """Enhance fine astronomical filaments and tendrils via unsharp masking.

    Parameters
    ----------
    image : np.ndarray
        2D spatial image array.
    sigma : float
        Gaussian blur kernel width.
    amount : float
        Filament sharpening strength factor.

    Returns
    -------
    np.ndarray
        Sharpened spatial image.
    """
    blurred = gaussian_filter(image, sigma=sigma)
    sharpened = image + amount * (image - blurred)
    return np.clip(sharpened, np.min(image), np.max(image))


def upsample_spatial_grid(
    image: np.ndarray,
    factor: int = 10,
    method: str = "bicubic",
) -> np.ndarray:
    """Upsample a coarse 2D IFU spatial grid into a high-definition smooth image.

    Parameters
    ----------
    image : np.ndarray
        Raw 2D spatial array (e.g. shape 30x30).
    factor : int
        Upsampling scale factor (e.g. 10 -> 300x300).
    method : str
        Interpolation method ('bicubic', 'bilinear', 'nearest').

    Returns
    -------
    np.ndarray
        HD spatial image.
    """
    if factor <= 1:
        return image.copy()

    # Fill NaNs with minimum valid value before interpolation
    valid_mask = np.isfinite(image)
    if not np.any(valid_mask):
        return np.zeros((image.shape[0] * factor, image.shape[1] * factor))

    clean_img = image.copy()
    fill_val = np.nanmin(clean_img[valid_mask])
    clean_img[~valid_mask] = fill_val

    order_map = {"nearest": 0, "bilinear": 1, "bicubic": 3}
    order = order_map.get(method, 3)

    # Perform bicubic zoom
    hd_img = zoom(clean_img, factor, order=order, mode="reflect")

    return hd_img


def apply_asinh_stretch(
    image: np.ndarray,
    a: float = 0.1,
    vmin_percentile: float = 1.0,
    vmax_percentile: float = 99.5,
) -> np.ndarray:
    """Apply NASA-standard non-linear `asinh` (arcsinh) intensity scaling.

    Reveals faint outer nebular gas without over-exposing bright central cores.

    Parameters
    ----------
    image : np.ndarray
        2D image array.
    a : float
        Softening parameter (smaller = more boost to faint structures).
    vmin_percentile : float
        Lower percentile threshold for black level.
    vmax_percentile : float
        Upper percentile threshold for white level.

    Returns
    -------
    np.ndarray
        Normalized image in range [0, 1].
    """
    clean = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

    if HAS_ASTROPY_VIS:
        try:
            interval = IntervalAsymmetricPercentile(vmin_percentile, vmax_percentile)
            vmin, vmax = interval.get_limits(clean)
        except Exception:
            vmin, vmax = np.min(clean), np.max(clean)
    else:
        vmin, vmax = np.percentile(clean, (vmin_percentile, vmax_percentile))

    if vmax <= vmin:
        return np.zeros_like(clean)

    # Clip into limits
    clipped = np.clip(clean, vmin, vmax)
    norm = (clipped - vmin) / (vmax - vmin)

    if HAS_ASTROPY_VIS:
        stretch = AsinhStretch(a=a)
        stretched = stretch(norm)
    else:
        # Pure numpy arcsinh stretch: asinh(x/a) / asinh(1/a)
        stretched = np.arcsinh(norm / max(1e-4, a)) / np.arcsinh(1.0 / max(1e-4, a))

    return np.clip(stretched, 0.0, 1.0)


def apply_log_stretch(
    image: np.ndarray,
    vmin_percentile: float = 1.0,
    vmax_percentile: float = 99.5,
) -> np.ndarray:
    """Apply logarithmic intensity scaling.

    Parameters
    ----------
    image : np.ndarray
        2D image array.
    vmin_percentile : float
        Lower percentile threshold.
    vmax_percentile : float
        Upper percentile threshold.

    Returns
    -------
    np.ndarray
        Normalized image in range [0, 1].
    """
    clean = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

    if HAS_ASTROPY_VIS:
        try:
            interval = IntervalAsymmetricPercentile(vmin_percentile, vmax_percentile)
            vmin, vmax = interval.get_limits(clean)
        except Exception:
            vmin, vmax = np.min(clean), np.max(clean)
    else:
        vmin, vmax = np.percentile(clean, (vmin_percentile, vmax_percentile))

    if vmax <= vmin:
        return np.zeros_like(clean)

    clipped = np.clip(clean, vmin, vmax)
    norm = (clipped - vmin) / (vmax - vmin)

    if HAS_ASTROPY_VIS:
        stretch = LogStretch()
        stretched = stretch(norm)
    else:
        # Pure numpy log stretch: log(1 + 1000*x) / log(1001)
        stretched = np.log1p(1000.0 * norm) / np.log1p(1000.0)

    return np.clip(stretched, 0.0, 1.0)
