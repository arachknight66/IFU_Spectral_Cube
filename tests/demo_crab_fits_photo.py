"""
Verification script to generate and save NASA photograph from Crab Nebula FITS dataset.
"""

import sys
sys.path.insert(0, ".")

import matplotlib.pyplot as plt
from pathlib import Path
from src.core.loader import load_fits_cube
from src.visualization.rgb import create_rgb_composite
from src.visualization.nasa_plot import render_nasa_photo
from scripts.generate_crab_fits import create_crab_nebula_fits_file


def test_crab_fits_photo():
    fits_path = create_crab_nebula_fits_file("data/crab_nebula_miri_composite.fits")
    cube = load_fits_cube(str(fits_path))

    # Extract 3 spectral channels from FITS cube.data
    b_arr = cube.data[0] # [Ar III] 8.99µm
    g_arr = cube.data[1] # [Ne II] 12.81µm
    r_arr = cube.data[2] # [S III] 18.71µm

    rgb_hd = create_rgb_composite(
        r_arr, g_arr, b_arr,
        stretch="asinh",
        asinh_a=0.1,
        saturation=1.5,
        gamma=1.0,
        super_res_factor=2,
        unsharp_mask_amount=1.2,
    )

    fig = render_nasa_photo(
        rgb_hd,
        title="JWST MIRI MRS — CRAB NEBULA (M1 / NGC 1952)",
        target_name="CRAB NEBULA PULSAR & EJECTA FILAMENTS",
        channel_labels={
            "red": "S III 18.71µm Outer Dust",
            "green": "Ne II 12.81µm Ionized Web",
            "blue": "Ar III 8.99µm Pulsar Core",
        },
        scale_bar_arcsec=30.0,
        show_compass=True,
    )

    out_dir = Path("output_demo")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "crab_nebula_fits_nasa_photo.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="black")
    plt.close(fig)

    print(f"Successfully generated Crab Nebula FITS NASA Photograph: {out_path}")


if __name__ == "__main__":
    test_crab_fits_photo()
