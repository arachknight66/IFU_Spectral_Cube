"""
Unit tests for WCS mosaic reprojection and background sky subtraction.
"""

from __future__ import annotations

import numpy as np
import tempfile
from pathlib import Path
from astropy.wcs import WCS
from astropy.io import fits

from src.core.mosaic import build_master_wcs, reproject_and_drizzle_mosaic
from src.preprocessing.sky_sub import subtract_sky_background
from src.visualization.rgb_mosaic import synthesize_mosaic_rgb


def _create_dummy_fits_tile(filepath: Path, ra: float, dec: float):
    """Helper to create a small 2D FITS tile with celestial WCS."""
    data = np.ones((50, 50), dtype=np.float32)
    header = fits.Header()
    header["CRVAL1"] = ra
    header["CRVAL2"] = dec
    header["CRPIX1"] = 25.0
    header["CRPIX2"] = 25.0
    header["CDELT1"] = -0.0001
    header["CDELT2"] = 0.0001
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    
    hdu = fits.PrimaryHDU(data=data, header=header)
    hdu.writeto(filepath, overwrite=True)


def test_mosaic_reprojection_and_sky_subtraction():
    """Verify building master WCS and reprojecting mosaic tiles."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        dir_path = Path(tmp_dir)
        tile1 = dir_path / "tile1.fits"
        tile2 = dir_path / "tile2.fits"
        
        _create_dummy_fits_tile(tile1, 150.0, 2.0)
        _create_dummy_fits_tile(tile2, 150.002, 2.002)

        tiles = [tile1, tile2]
        wcs_master, (h, w) = build_master_wcs(tiles)
        
        assert wcs_master is not None
        assert h > 0 and w > 0

        mosaic_img, wcs_out = reproject_and_drizzle_mosaic(tiles, slice_idx=0)
        assert mosaic_img.shape[0] > 0
        assert mosaic_img.shape[1] > 0

        clean_sky = subtract_sky_background(mosaic_img)
        assert clean_sky.shape == mosaic_img.shape

        rgb = synthesize_mosaic_rgb(clean_sky, clean_sky, clean_sky)
        assert rgb.shape == (mosaic_img.shape[0], mosaic_img.shape[1], 3)
