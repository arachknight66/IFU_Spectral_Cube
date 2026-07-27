"""
Comprehensive unit test suite for Phase 6: reproducible NASA-style false-colour rendering.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from astropy.wcs import WCS

from src.imaging.component_map import ComponentMap
from src.imaging.recipes import ImageRecipe
from src.visualization.false_color import (
    FalseColorResult,
    IncompatibleMapError,
    render_false_color,
)


def _make_dummy_component_map(
    shape: tuple[int, int] = (10, 10),
    val: float = 10.0,
    channel_name: str = "red",
    has_err: bool = True,
) -> ComponentMap:
    """Helper to construct a dummy ComponentMap for testing."""
    ny, nx = shape
    data = np.full((ny, nx), val, dtype=float)
    # Mask edge
    data[0, 0] = np.nan

    err = np.full((ny, nx), 1.0, dtype=float) if has_err else None
    snr = data / err if err is not None else None
    dq = np.zeros((ny, nx), dtype=np.int32)
    cov = np.ones((ny, nx), dtype=int)

    wcs_2d = WCS(naxis=2)
    wcs_2d.wcs.crval = [150.0, 2.0]
    wcs_2d.wcs.crpix = [nx / 2.0, ny / 2.0]
    wcs_2d.wcs.cdelt = [-0.0001, 0.0001]
    wcs_2d.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    provenance = {
        "source_filepath": f"dummy_{channel_name}.fits",
        "component_name": f"feature_{channel_name}",
    }

    return ComponentMap(
        data=data,
        uncertainty=err,
        snr=snr,
        dq=dq,
        coverage=cov,
        wcs=wcs_2d,
        unit="MJy / sr * um",
        component_name=f"feature_{channel_name}",
        recipe_channel=channel_name,
        on_band_flux=data.copy(),
        continuum_map=np.zeros_like(data),
        has_authoritative_uncertainty=has_err,
        provenance=provenance,
    )


@pytest.fixture
def dummy_recipe(tmp_path: Path) -> ImageRecipe:
    """Fixture providing a valid ImageRecipe."""
    json_path = Path("src/imaging/recipes/miri_emission_line_rgb.json")
    if json_path.exists():
        return ImageRecipe.load(json_path)

    # Fallback inline creation
    return ImageRecipe.from_dict({
        "name": "test_recipe",
        "target_name": "Test Target",
        "mapping_intent": "scientific",
        "mapping_note": "Test note",
        "expected_product": {"instrument": "MIRI", "product_type": "s3d", "observing_mode": "MRS IFU"},
        "channels": [
            {"feature_name": "[S III]", "central_wavelength_um": 18.71, "integration_width_um": 0.2, "continuum_subtraction": {"enabled": False, "method": None, "sideband_width_um": None, "gap_um": None}, "display_colour": "red"},
            {"feature_name": "[Ne II]", "central_wavelength_um": 12.81, "integration_width_um": 0.1, "continuum_subtraction": {"enabled": False, "method": None, "sideband_width_um": None, "gap_um": None}, "display_colour": "green"},
            {"feature_name": "[Ar III]", "central_wavelength_um": 8.99, "integration_width_um": 0.1, "continuum_subtraction": {"enabled": False, "method": None, "sideband_width_um": None, "gap_um": None}, "display_colour": "blue"},
        ],
        "rendering": {
            "stretch": "asinh",
            "percentile_clip": [1.0, 99.5],
            "background_subtraction": "none",
            "colour_balance": "preserve_flux_ratios",
            "output_resolution": [300, 300],
            "output_filename": "test.png",
        },
    })


def test_input_validation_channel_count_and_shape(dummy_recipe):
    """Requirement 2: Reject invalid channel count or spatial shape mismatch."""
    c_red = _make_dummy_component_map(shape=(10, 10), val=10.0, channel_name="red")
    c_green = _make_dummy_component_map(shape=(10, 10), val=15.0, channel_name="green")
    c_blue_wrong = _make_dummy_component_map(shape=(12, 12), val=20.0, channel_name="blue")

    # Mismatched shape throws IncompatibleMapError
    with pytest.raises(IncompatibleMapError):
        render_false_color({"red": c_red, "green": c_green, "blue": c_blue_wrong}, recipe=dummy_recipe)

    # Missing red channel throws IncompatibleMapError
    with pytest.raises(IncompatibleMapError):
        render_false_color({"green": c_green, "blue": c_blue_wrong}, recipe=dummy_recipe)


def test_display_stretches_linear_log_asinh(dummy_recipe):
    """Requirement 3: Test intensity stretches (linear, log, asinh)."""
    c_red = _make_dummy_component_map(shape=(10, 10), val=10.0, channel_name="red")
    c_green = _make_dummy_component_map(shape=(10, 10), val=20.0, channel_name="green")
    c_blue = _make_dummy_component_map(shape=(10, 10), val=30.0, channel_name="blue")

    maps = {"red": c_red, "green": c_green, "blue": c_blue}

    for str_mode in ("linear", "log", "asinh"):
        res = render_false_color(
            maps,
            recipe=dummy_recipe,
            rendering_mode="scientific",
            rendering_overrides={"stretch": str_mode},
        )
        assert isinstance(res, FalseColorResult)
        assert res.rgb_array.shape == (10, 10, 3)
        assert np.all(res.rgb_array >= 0.0) and np.all(res.rgb_array <= 1.0)
        assert res.provenance["stretch"] == str_mode


def test_scientific_vs_presentation_modes(dummy_recipe):
    """Requirement 4: Verify rendering mode tracking and gain overrides."""
    c_red = _make_dummy_component_map(val=10.0, channel_name="red")
    c_green = _make_dummy_component_map(val=15.0, channel_name="green")
    c_blue = _make_dummy_component_map(val=20.0, channel_name="blue")
    maps = {"red": c_red, "green": c_green, "blue": c_blue}

    # Scientific mode
    res_sci = render_false_color(maps, recipe=dummy_recipe, rendering_mode="scientific")
    assert res_sci.rendering_mode == "scientific"
    assert res_sci.provenance["rendering_mode"] == "scientific"

    # Presentation mode with gains
    res_pres = render_false_color(
        maps,
        recipe=dummy_recipe,
        rendering_mode="presentation",
        rendering_overrides={"gain_red": 1.5, "gain_blue": 0.8},
    )
    assert res_pres.rendering_mode == "presentation"
    assert res_pres.provenance["gains"]["red"] == 1.5


def test_snr_masking_and_background_subtraction(dummy_recipe):
    """Requirement 3: Test SNR masking and background subtraction."""
    c_red = _make_dummy_component_map(val=10.0, channel_name="red")
    c_green = _make_dummy_component_map(val=15.0, channel_name="green")
    c_blue = _make_dummy_component_map(val=20.0, channel_name="blue")

    # Manually drop SNR for one spaxel to test masking
    c_red.snr[2, 2] = 0.5  # Below threshold 3.0

    maps = {"red": c_red, "green": c_green, "blue": c_blue}

    res = render_false_color(
        maps,
        recipe=dummy_recipe,
        rendering_overrides={"snr_threshold": 3.0, "background_subtraction": "per_channel_median"},
    )
    # Red channel at (2,2) should be zeroed out
    assert res.rgb_array[2, 2, 0] == 0.0


def test_alpha_channel_and_transparency(dummy_recipe):
    """Requirement 3 & 6: Unobserved/NaN pixels rendered transparently in RGBA."""
    c_red = _make_dummy_component_map(val=10.0, channel_name="red")
    c_green = _make_dummy_component_map(val=15.0, channel_name="green")
    c_blue = _make_dummy_component_map(val=20.0, channel_name="blue")

    maps = {"red": c_red, "green": c_green, "blue": c_blue}
    res = render_false_color(maps, recipe=dummy_recipe)

    # Pixel (0,0) was NaN -> alpha should be 0.0
    assert res.rgba_array[0, 0, 3] == 0.0
    assert res.rgb_array[0, 0, 0] == 0.0


def test_png_tiff_and_manifest_export(dummy_recipe):
    """Requirement 5 & 6: Export 8-bit PNG, 16-bit TIFF, and sidecar JSON manifest."""
    c_red = _make_dummy_component_map(val=10.0, channel_name="red")
    c_green = _make_dummy_component_map(val=15.0, channel_name="green")
    c_blue = _make_dummy_component_map(val=20.0, channel_name="blue")
    maps = {"red": c_red, "green": c_green, "blue": c_blue}

    res = render_false_color(maps, recipe=dummy_recipe, rendering_mode="scientific")

    with tempfile.TemporaryDirectory() as tmp_dir:
        dir_p = Path(tmp_dir)

        # PNG Export
        png_path = dir_p / "output_render.png"
        saved_png = res.save_png(png_path, overwrite=True)
        assert saved_png.exists()

        png_manifest = saved_png.with_name(f"{saved_png.name}.manifest.json")
        assert png_manifest.exists()

        manifest_data = json.loads(png_manifest.read_text(encoding="utf-8"))
        assert manifest_data["recipe_name"] == dummy_recipe.name
        assert manifest_data["rendering_mode"] == "scientific"

        # TIFF Export
        tiff_path = dir_p / "output_render.tiff"
        saved_tiff = res.save_tiff(tiff_path, bit_depth=16, overwrite=True)
        assert saved_tiff.exists()
        tiff_manifest = saved_tiff.with_name(f"{saved_tiff.name}.manifest.json")
        assert tiff_manifest.exists()
