"""
Unit tests for spatial super-resolution, asinh scaling, and RGB photo synthesis.
"""

from __future__ import annotations

import numpy as np
from src.visualization.super_res import upsample_spatial_grid, apply_asinh_stretch
from src.visualization.rgb import create_rgb_composite


def test_upsample_spatial_grid():
    """Verify upsampling increases spatial dimensions by scale factor."""
    img = np.random.rand(30, 30)
    hd_img = upsample_spatial_grid(img, factor=10)

    assert hd_img.shape == (300, 300)
    assert not np.any(np.isnan(hd_img))


def test_apply_asinh_stretch():
    """Verify asinh stretch normalizes array into [0, 1] range."""
    img = np.random.exponential(scale=10.0, size=(100, 100))
    str_img = apply_asinh_stretch(img, a=0.1)

    assert str_img.shape == (100, 100)
    assert np.min(str_img) >= 0.0
    assert np.max(str_img) <= 1.0


def test_create_rgb_composite():
    """Verify creating 3-channel RGB composite photograph array."""
    r = np.random.rand(20, 20)
    g = np.random.rand(20, 20)
    b = np.random.rand(20, 20)

    rgb = create_rgb_composite(r, g, b, stretch="asinh", super_res_factor=5)

    assert rgb.shape == (100, 100, 3)
    assert np.min(rgb) >= 0.0
    assert np.max(rgb) <= 1.0
