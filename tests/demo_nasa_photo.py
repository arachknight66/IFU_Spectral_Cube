"""
Verification script to render and save a high-resolution NASA press-release composite photograph.
"""

from __future__ import annotations

import sys
sys.path.insert(0, ".")

import matplotlib.pyplot as plt
from pathlib import Path
from tests.synthetic import generate_synthetic_cube
from pipeline import SpectralPipeline, PipelineConfig
from src.visualization.rgb import create_rgb_composite, CHEMICAL_PRESETS
from src.visualization.nasa_plot import render_nasa_photo


def generate_nasa_demo_photo():
    cube = generate_synthetic_cube()
    pipeline = SpectralPipeline(PipelineConfig(peak_prominence=3.0))
    analysis = pipeline.analyze(cube, x=15, y=15)
    images = pipeline.generate_images(cube, analysis.line_matches)
    img_keys = list(images.keys())

    # Pick 3 channels for Red, Green, Blue
    r_img = images.get("emission_PAH_11.30", images[img_keys[0]])
    g_img = images.get("emission_[Ne II]_12.81", images[img_keys[1] if len(img_keys) > 1 else img_keys[0]])
    b_img = images.get("emission_[Ar III]_8.99", images[img_keys[2] if len(img_keys) > 2 else img_keys[0]])

    # Synthesize RGB
    rgb = create_rgb_composite(
        r_img, g_img, b_img,
        stretch="asinh",
        saturation=1.4,
        gamma=1.0,
        super_res_factor=10,
    )

    # Render NASA Cartography figure
    fig = render_nasa_photo(
        rgb,
        title="JWST MIRI IFU — Stephan's Quintet (NGC 7319)",
        target_name="NGC 7319 Core Shock",
        channel_labels={
            "red": "PAH 11.3µm Dust",
            "green": "[Ne II] 12.81µm Ionized Gas",
            "blue": "[Ar III] 8.99µm High-Ion",
        },
        scale_bar_arcsec=1.0,
        show_compass=True,
    )

    out_dir = Path("output_demo")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "nasa_composite_photo.png"

    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="black")
    plt.close(fig)

    print(f"Successfully generated NASA Photograph: {out_path}")


if __name__ == "__main__":
    generate_nasa_demo_photo()
