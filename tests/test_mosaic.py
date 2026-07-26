"""
Unit tests for WCS mosaic reprojection and background sky subtraction.
"""

from __future__ import annotations

import numpy as np
import tempfile
from pathlib import Path
from src.core.mast import download_mast_mosaic_set
from src.core.mosaic import build_master_wcs, reproject_and_drizzle_mosaic
from src.preprocessing.sky_sub import subtract_sky_background
from src.visualization.rgb_mosaic import synthesize_mosaic_rgb


def test_download_mast_mosaic_set():
    """Verify downloading mosaic tile dataset."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tiles = download_mast_mosaic_set(target_name="Crab Nebula", output_dir=tmp_dir)
        assert len(tiles) == 4
        for tile in tiles:
            assert Path(tile).exists()
            assert Path(tile).stat().st_size > 1000


def test_mosaic_reprojection_and_sky_subtraction():
    """Verify building master WCS and reprojecting mosaic tiles."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tiles = download_mast_mosaic_set(target_name="Crab Nebula", output_dir=tmp_dir)
        wcs_master, (h, w) = build_master_wcs(tiles)
        
        assert (h, w) == (300, 300) or (h > 0 and w > 0)

        mosaic_img, wcs_out = reproject_and_drizzle_mosaic(tiles, slice_idx=250)
        assert mosaic_img.shape[0] > 0
        assert mosaic_img.shape[1] > 0

        clean_sky = subtract_sky_background(mosaic_img)
        assert clean_sky.shape == mosaic_img.shape

        rgb = synthesize_mosaic_rgb(clean_sky, clean_sky, clean_sky)
        assert rgb.shape == (mosaic_img.shape[0], mosaic_img.shape[1], 3)
