"""
2D Background Sky Subtraction & Photometry Noise Floor Engine.

Performs robust 2D background sky estimation and noise floor subtraction
to ensure deep space is strictly black (0.0) without clipping faint nebular filaments.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter, gaussian_filter


def estimate_2d_background(
    image: np.ndarray,
    box_size: int = 32,
    filter_size: float = 3.0,
) -> np.ndarray:
    """Estimate 2D sky background brightness using spatial median filtering.

    Parameters
    ----------
    image : np.ndarray
        2D spatial image array.
    box_size : int
        Size of spatial background estimation box.
    filter_size : float
        Gaussian smoothing kernel size.

    Returns
    -------
    np.ndarray
        2D background sky model array.
    """
    clean = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    bg_low = median_filter(clean, size=box_size)
    bg_smooth = gaussian_filter(bg_low, sigma=filter_size)
    return bg_smooth


def subtract_sky_background(
    image: np.ndarray,
    percentile_cutoff: float = 15.0,
) -> np.ndarray:
    """Subtract sky background noise floor while preserving faint outer filaments.

    Parameters
    ----------
    image : np.ndarray
        2D spatial image array.
    percentile_cutoff : float
        Percentile cutoff for sky noise floor.

    Returns
    -------
    np.ndarray
        Sky-subtracted 2D image array.
    """
    clean = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

    # Estimate background noise floor
    bg_model = estimate_2d_background(clean)
    p_floor = np.percentile(clean, percentile_cutoff)

    # Subtract background
    subtracted = np.maximum(0.0, clean - (0.8 * bg_model + 0.2 * p_floor))

    # Mask noise
    noise_rms = np.std(clean[clean <= np.percentile(clean, 50.0)])
    if noise_rms > 0:
        subtracted[subtracted < 1.5 * noise_rms] = 0.0

    return subtracted
