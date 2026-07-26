"""
WCS Astrometric Sky Reprojection & Mosaic Stacking Engine.

Stitches and drizzles multiple overlapping FITS pointing tiles onto a single master
celestial coordinate grid (RA/Dec WCS) to construct true wide-field astronomical maps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
from astropy.io import fits

try:
    from astropy.wcs import WCS
    HAS_WCS = True
except ImportError:
    HAS_WCS = False


def build_master_wcs(
    fits_files: list[Path],
    pixel_scale_deg: float = 0.0001,
) -> tuple[WCS | None, tuple[int, int]]:
    """Calculate master celestial coordinate bounding box and create unified WCS grid.

    Parameters
    ----------
    fits_files : list of Path
        List of paths to individual FITS mosaic tiles.
    pixel_scale_deg : float
        Output pixel scale in degrees per pixel.

    Returns
    -------
    tuple of (WCS or None, (height, width))
        Master WCS object and total image dimensions.
    """
    if not fits_files or not HAS_WCS:
        return None, (300, 300)

    ra_min, ra_max = 1e9, -1e9
    dec_min, dec_max = 1e9, -1e9

    for fpath in fits_files:
        try:
            with fits.open(fpath) as hdul:
                for hdu in hdul:
                    if hdu.data is not None and len(hdu.data.shape) >= 2:
                        wcs_hdr = WCS(hdu.header, naxis=2)
                        h, w = hdu.data.shape[-2:]
                        corners = wcs_hdr.pixel_to_world_values(
                            [0, w, w, 0], [0, 0, h, h]
                        )
                        ras = corners[0]
                        decs = corners[1]
                        ra_min = min(ra_min, np.min(ras))
                        ra_max = max(ra_max, np.max(ras))
                        dec_min = min(dec_min, np.min(decs))
                        dec_max = max(dec_max, np.max(decs))
                        break
        except Exception:
            continue

    if ra_min >= ra_max or dec_min >= dec_max:
        return None, (300, 300)

    # Center RA & Dec
    center_ra = (ra_min + ra_max) / 2.0
    center_dec = (dec_min + dec_max) / 2.0

    width = int(np.ceil((ra_max - ra_min) / pixel_scale_deg)) + 40
    height = int(np.ceil((dec_max - dec_min) / pixel_scale_deg)) + 40

    master_wcs = WCS(naxis=2)
    master_wcs.wcs.crval = [center_ra, center_dec]
    master_wcs.wcs.crpix = [width / 2.0, height / 2.0]
    master_wcs.wcs.cdelt = [-pixel_scale_deg, pixel_scale_deg]
    master_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    return master_wcs, (height, width)


def reproject_and_drizzle_mosaic(
    fits_files: list[Path],
    slice_idx: int = 250,
) -> tuple[np.ndarray, WCS | None]:
    """Reproject and drizzle 2D spatial slices from multiple FITS tiles onto master WCS grid.

    Parameters
    ----------
    fits_files : list of Path
        List of paths to input FITS tile files.
    slice_idx : int
        Wavelength index to extract and mosaic.

    Returns
    -------
    tuple of (np.ndarray, WCS or None)
        Combined mosaic image array (height, width) and master WCS.
    """
    master_wcs, (out_h, out_w) = build_master_wcs(fits_files)
    canvas = np.zeros((out_h, out_w), dtype=np.float32)
    weights = np.zeros((out_h, out_w), dtype=np.float32)

    for fpath in fits_files:
        try:
            with fits.open(fpath) as hdul:
                data_hdu = None
                for hdu in hdul:
                    if hdu.data is not None and len(hdu.data.shape) >= 2:
                        data_hdu = hdu
                        break

                if data_hdu is None:
                    continue

                if len(data_hdu.data.shape) == 3:
                    s_idx = min(slice_idx, data_hdu.data.shape[0] - 1)
                    img_2d = data_hdu.data[s_idx]
                else:
                    img_2d = data_hdu.data

                wcs_tile = WCS(data_hdu.header, naxis=2) if HAS_WCS else None
                tile_h, tile_w = img_2d.shape

                if wcs_tile is not None and master_wcs is not None:
                    # Project tile pixel coordinates into master canvas pixel coordinates
                    grid_y, grid_x = np.mgrid[0:tile_h, 0:tile_w]
                    world_ras, world_decs = wcs_tile.pixel_to_world_values(grid_x.ravel(), grid_y.ravel())
                    canvas_x, canvas_y = master_wcs.world_to_pixel_values(world_ras, world_decs)

                    canvas_x = np.round(canvas_x).astype(int)
                    canvas_y = np.round(canvas_y).astype(int)

                    valid = (
                        (canvas_x >= 0) & (canvas_x < out_w) &
                        (canvas_y >= 0) & (canvas_y < out_h) &
                        np.isfinite(img_2d.ravel())
                    )

                    canvas[canvas_y[valid], canvas_x[valid]] += img_2d.ravel()[valid]
                    weights[canvas_y[valid], canvas_x[valid]] += 1.0
                else:
                    # Fallback tile placement
                    canvas[:tile_h, :tile_w] += img_2d
                    weights[:tile_h, :tile_w] += 1.0
        except Exception:
            continue

    weights[weights == 0] = 1.0
    mosaic_img = canvas / weights
    return mosaic_img, master_wcs
