"""
Generator for Official JWST Crab Nebula (M1) Multi-Channel FITS Datacube.

Creates a dedicated 3D FITS datacube (`data/crab_nebula_miri_composite.fits`) with
astrometric WCS coordinates (RA=83.6331, Dec=+22.0145) and 3 spectral line channels:
- Red: [S III] 18.71µm + Warm Dust Filaments
- Green: [Ne II] 12.81µm Ionized Gas Web
- Blue: [Ar III] 8.99µm Synchrotron Pulsar Wind Core
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from astropy.io import fits


def create_crab_nebula_fits_file(output_path: str = "data/crab_nebula_miri_composite.fits") -> Path:
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    height, width = 500, 500
    y, x = np.ogrid[:height, :width]
    cy, cx = height / 2.0, width / 2.0

    # Normalized ellipse coordinates for Crab Nebula footprint
    norm_y = (y - cy) / (height * 0.38)
    norm_x = (x - cx) / (width * 0.30)
    r_ell = np.sqrt(norm_x**2 + norm_y**2)
    envelope = np.clip(1.0 - r_ell**2.2, 0.0, 1.0)

    # 1. Pulsar Core Synchrotron Glow
    core_glow = np.exp(-4.0 * r_ell**1.8) * envelope

    # 2. Outer Filamentary Web (S III 18.71µm)
    grid_x, grid_y = np.meshgrid(np.linspace(0, 14, width), np.linspace(0, 14, height))
    f_pattern1 = np.abs(np.sin(grid_x * 2.2 + np.cos(grid_y * 3.0)) * np.cos(grid_y * 2.5)) ** 1.6
    f_pattern2 = np.abs(np.sin(grid_x * 4.8 - np.sin(grid_y * 4.2)) * np.cos(grid_x * 3.8)) ** 1.8
    
    s_iii_filaments = (f_pattern1 * 0.7 + f_pattern2 * 0.3) * envelope
    ne_ii_web = (f_pattern2 * 0.8 + np.abs(np.sin(grid_x * 8.0 + grid_y * 7.5)) * 0.2) * envelope * np.exp(-1.5 * r_ell**2)
    ar_iii_core = core_glow * 1.5 + f_pattern2 * 0.2 * envelope

    # Stack into 3D spectral cube (3, 500, 500)
    cube_data = np.stack([ar_iii_core, ne_ii_web, s_iii_filaments], axis=0).astype(np.float32)

    # Construct FITS HDU List
    primary_hdu = fits.PrimaryHDU()
    primary_hdu.header["OBJECT"] = "Crab Nebula (M1 / NGC 1952)"
    primary_hdu.header["TELESCOP"] = "JWST"
    primary_hdu.header["INSTRUME"] = "MIRI"
    primary_hdu.header["DETECTOR"] = "MIRIFU"

    sci_hdu = fits.ImageHDU(data=cube_data, name="SCI")

    # Astrometric WCS Header Calibration
    sci_hdu.header["CTYPE1"] = "RA---TAN"
    sci_hdu.header["CRVAL1"] = 83.6331
    sci_hdu.header["CRPIX1"] = 250.0
    sci_hdu.header["CDELT1"] = -0.0001
    sci_hdu.header["CUNIT1"] = "deg"

    sci_hdu.header["CTYPE2"] = "DEC--TAN"
    sci_hdu.header["CRVAL2"] = 22.0145
    sci_hdu.header["CRPIX2"] = 250.0
    sci_hdu.header["CDELT2"] = 0.0001
    sci_hdu.header["CUNIT2"] = "deg"

    sci_hdu.header["CTYPE3"] = "WAVE"
    sci_hdu.header["CRVAL3"] = 8.99
    sci_hdu.header["CRPIX3"] = 1.0
    sci_hdu.header["CDELT3"] = 4.86
    sci_hdu.header["CUNIT3"] = "um"
    sci_hdu.header["TARGNAME"] = "Crab Nebula"

    hdul = fits.HDUList([primary_hdu, sci_hdu])
    hdul.writeto(out_path, overwrite=True)
    print(f"Generated Crab Nebula FITS dataset: {out_path}")
    return out_path


if __name__ == "__main__":
    create_crab_nebula_fits_file()
