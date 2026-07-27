"""
Unit test suite for Phase 7: Streamlit end-to-end workflow logic and session state transitions.
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
from tests.synthetic import generate_synthetic_cube


def _make_dummy_component_map(channel_name: str) -> ComponentMap:
    """Helper to construct dummy ComponentMap."""
    ny, nx = 8, 8
    data = np.ones((ny, nx), dtype=float)
    err = np.full((ny, nx), 0.1, dtype=float)
    snr = data / err
    dq = np.zeros((ny, nx), dtype=np.int32)
    cov = np.ones((ny, nx), dtype=int)

    wcs_2d = WCS(naxis=2)
    wcs_2d.wcs.crval = [150.0, 2.0]
    wcs_2d.wcs.crpix = [4.0, 4.0]
    wcs_2d.wcs.cdelt = [-0.0001, 0.0001]
    wcs_2d.wcs.ctype = ["RA---TAN", "DEC--TAN"]

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
        has_authoritative_uncertainty=True,
        provenance={"source": "test"},
    )


def test_invalid_workflow_ordering_protection():
    """Requirement 9: Attempting false-colour rendering without 3 component maps raises an error."""
    recipe = ImageRecipe.load(Path("src/imaging/recipes/miri_emission_line_rgb.json"))
    c_red = _make_dummy_component_map("red")
    c_green = _make_dummy_component_map("green")

    # Incomplete channel dict (only 2 channels)
    with pytest.raises(IncompatibleMapError):
        render_false_color({"red": c_red, "green": c_green}, recipe=recipe)


def test_recipe_adjustment_persistence_and_provenance():
    """Requirement 9: User rendering overrides persist into FalseColorResult provenance."""
    recipe = ImageRecipe.load(Path("src/imaging/recipes/miri_emission_line_rgb.json"))
    cmaps = {
        "red": _make_dummy_component_map("red"),
        "green": _make_dummy_component_map("green"),
        "blue": _make_dummy_component_map("blue"),
    }

    overrides = {
        "stretch": "log",
        "percentile_clip": (2.0, 98.0),
        "background_subtraction": "per_channel_median",
        "gain_red": 1.4,
    }

    res = render_false_color(cmaps, recipe=recipe, rendering_mode="presentation", rendering_overrides=overrides)

    assert res.provenance["stretch"] == "log"
    assert res.provenance["percentile_clip"] == [2.0, 98.0]
    assert res.provenance["gains"]["red"] == 1.4
    assert res.rendering_mode == "presentation"


def test_export_manifest_completeness():
    """Requirement 9: Export manifest JSON contains full source provenance and disclaimer."""
    recipe = ImageRecipe.load(Path("src/imaging/recipes/miri_emission_line_rgb.json"))
    cmaps = {
        "red": _make_dummy_component_map("red"),
        "green": _make_dummy_component_map("green"),
        "blue": _make_dummy_component_map("blue"),
    }

    res = render_false_color(cmaps, recipe=recipe, rendering_mode="scientific")

    with tempfile.TemporaryDirectory() as tmp_dir:
        png_p = Path(tmp_dir) / "test_export.png"
        saved = res.save_png(png_p, overwrite=True)
        manifest_p = saved.with_name(f"{saved.name}.manifest.json")

        assert manifest_p.exists()
        manifest_data = json.loads(manifest_p.read_text(encoding="utf-8"))

        assert "recipe_name" in manifest_data
        assert "rendering_mode" in manifest_data
        assert "provenance" in manifest_data
        assert "warning" in manifest_data["provenance"]
        assert "False-colour" in manifest_data["provenance"]["warning"]
