"""
Verification script for real FITS mosaic reprojection and drizzle stacking.
"""

import sys
sys.path.insert(0, ".")

from pathlib import Path
import matplotlib.pyplot as plt
from src.core.mast import download_mast_mosaic_set
from src.core.mosaic import reproject_and_drizzle_mosaic
from src.preprocessing.sky_sub import subtract_sky_background
from src.visualization.rgb_mosaic import synthesize_mosaic_rgb
from src.visualization.nasa_plot import render_nasa_photo


def test_real_mosaic_pipeline():
    print("1. Downloading 4-tile FITS mosaic set for Crab Nebula...")
    tile_paths = download_mast_mosaic_set(target_name="Crab Nebula", output_dir="data/mosaics")
    print(f"   Downloaded {len(tile_paths)} tiles: {[t.name for t in tile_paths]}")

    print("2. Reprojecting and drizzling mosaic tiles onto WCS celestial grid...")
    red_mosaic, wcs_out = reproject_and_drizzle_mosaic(tile_paths, slice_idx=100)
    green_mosaic, _ = reproject_and_drizzle_mosaic(tile_paths, slice_idx=250)
    blue_mosaic, _ = reproject_and_drizzle_mosaic(tile_paths, slice_idx=400)

    print(f"   Mosaic canvas shape: {red_mosaic.shape}")

    print("3. Performing 2D background sky estimation and noise floor subtraction...")
    red_sub = subtract_sky_background(red_mosaic)
    green_sub = subtract_sky_background(green_mosaic)
    blue_sub = subtract_sky_background(blue_mosaic)

    print("4. Synthesizing real FITS WCS RGB composite photograph...")
    rgb = synthesize_mosaic_rgb(red_sub, green_sub, blue_sub, stretch="asinh", unsharp_amount=0.8)

    print("5. Rendering celestial cartography photo...")
    fig = render_nasa_photo(
        rgb,
        title="JWST MIRI MRS — CRAB NEBULA WCS FITS MOSAIC",
        target_name="CRAB NEBULA (M1 / NGC 1952) 4-TILE FOOTPRINT",
        channel_labels={
            "red": "Slice λ=20.5µm",
            "green": "Slice λ=12.8µm",
            "blue": "Slice λ=6.5µm",
        },
        scale_bar_arcsec=30.0,
        show_compass=True,
    )

    out_dir = Path("output_demo")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "real_fits_mosaic_crab.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="black")
    plt.close(fig)

    print(f"Successfully generated Real FITS Mosaic Photograph: {out_path}")


if __name__ == "__main__":
    test_real_mosaic_pipeline()
