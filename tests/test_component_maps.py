"""
Comprehensive unit test suite for Phase 5: scientifically traceable continuum-subtracted spectral component maps.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from src.core.cube import SpectralCube
from src.imaging.component_map import (
    ComponentMap,
    WavelengthCoverageError,
    compute_bin_widths,
    extract_component_map,
    extract_recipe_component_maps,
)
from src.imaging.recipes import ImageRecipe
from tests.synthetic import generate_synthetic_cube


def _make_fixture_cube(
    data: np.ndarray,
    wavelength: np.ndarray,
    err: np.ndarray | None = None,
    dq: np.ndarray | None = None,
    flux_unit: str = "MJy/sr",
) -> SpectralCube:
    """Helper to construct a test SpectralCube with 2D celestial WCS."""
    ny, nx = data.shape[1:]
    header = fits.Header()
    header["CRVAL1"] = 150.0
    header["CRVAL2"] = 2.0
    header["CRPIX1"] = float(nx // 2 + 1)
    header["CRPIX2"] = float(ny // 2 + 1)
    header["CDELT1"] = -0.0001
    header["CDELT2"] = 0.0001
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["BUNIT"] = flux_unit

    wcs_3d = WCS(naxis=3)
    wcs_3d.wcs.crval = [150.0, 2.0, wavelength[0]]
    wcs_3d.wcs.crpix = [nx // 2 + 1, ny // 2 + 1, 1]
    wcs_3d.wcs.cdelt = [-0.0001, 0.0001, (wavelength[-1] - wavelength[0]) / (len(wavelength) - 1)]
    wcs_3d.wcs.ctype = ["RA---TAN", "DEC--TAN", "WAVE"]

    return SpectralCube(
        data=data,
        wavelength=wavelength,
        header=header,
        err=err,
        dq=dq,
        wcs=wcs_3d,
        flux_unit=flux_unit,
        filepath="fixture_cube.fits",
    )


def test_uniform_and_non_uniform_bin_widths():
    """Requirement 2: Calculate bin widths Δλ correctly for uniform and non-uniform grids."""
    wl_linear = np.linspace(5.0, 10.0, 6)  # step = 1.0
    dwl_linear = compute_bin_widths(wl_linear)
    np.testing.assert_allclose(dwl_linear, 1.0)

    # Non-uniform grid
    wl_non_linear = np.array([5.0, 5.2, 5.6, 6.2, 7.0])
    dwl_non_linear = compute_bin_widths(wl_non_linear)
    assert len(dwl_non_linear) == 5
    assert dwl_non_linear[0] == pytest.approx(0.2)
    assert dwl_non_linear[-1] == pytest.approx(0.8)


def test_known_emission_line_flux_recovery():
    """Requirement 2 & 8: Verify recovery of injected Gaussian line flux."""
    n_wl = 100
    wl = np.linspace(10.0, 14.0, n_wl)
    dwl = (14.0 - 10.0) / (n_wl - 1)  # ~0.0404 um

    ny, nx = 5, 5
    data = np.zeros((n_wl, ny, nx), dtype=float)

    # Inject Gaussian line centered at 12.0 um with amplitude 10.0, sigma 0.1 um
    center = 12.0
    amp = 10.0
    sigma = 0.1
    gauss = amp * np.exp(-0.5 * ((wl - center) / sigma) ** 2)

    for iy in range(ny):
        for ix in range(nx):
            data[:, iy, ix] = gauss

    cube = _make_fixture_cube(data, wl)
    cmap = extract_component_map(
        cube,
        central_wavelength_um=12.0,
        integration_width_um=0.6,  # +/- 3 sigma spans full line
        feature_name="H2_S1",
        continuum_subtraction=False,
    )

    # Theoretical integrated line flux = amp * sigma * sqrt(2*pi)
    expected_flux = amp * sigma * np.sqrt(2.0 * np.pi)
    # Numerical integral over on-band
    np.testing.assert_allclose(cmap.data[2, 2], expected_flux, rtol=0.02)


def test_continuum_subtraction_with_sloped_continuum():
    """Requirement 3: Verify continuum subtraction under a linear continuum slope m*λ + b."""
    n_wl = 100
    wl = np.linspace(10.0, 14.0, n_wl)
    ny, nx = 5, 5

    # Linear continuum: I(λ) = 2.0 * λ + 5.0
    m_slope = 2.0
    b_int = 5.0
    cont_spectrum = m_slope * wl + b_int

    # Injected emission line at 12.0 um with height 8.0
    line_center = 12.0
    line_sigma = 0.05
    gauss = 8.0 * np.exp(-0.5 * ((wl - line_center) / line_sigma) ** 2)

    data = np.zeros((n_wl, ny, nx), dtype=float)
    for iy in range(ny):
        for ix in range(nx):
            data[:, iy, ix] = cont_spectrum + gauss

    cube = _make_fixture_cube(data, wl)

    cmap = extract_component_map(
        cube,
        central_wavelength_um=12.0,
        integration_width_um=0.3,
        feature_name="line_subtracted",
        continuum_subtraction={
            "enabled": True,
            "sideband_width_um": 0.3,
            "gap_um": 0.2,
            "method": "adjacent_sidebands",
        },
    )

    # Expected continuum intensity at 12.0 um is 2.0*12 + 5 = 29.0
    expected_cont_intensity = m_slope * line_center + b_int
    on_indices = np.where((wl >= 12.0 - 0.15) & (wl <= 12.0 + 0.15))[0]
    bin_w = compute_bin_widths(wl)
    on_width_tot = np.sum(bin_w[on_indices])

    # Scaled continuum map should match expected_cont_intensity * on_width_tot
    np.testing.assert_allclose(cmap.continuum_map[2, 2], expected_cont_intensity * on_width_tot, rtol=1e-2)

    # Net subtracted map should equal pure line flux
    pure_line_flux = np.sum(gauss[on_indices] * bin_w[on_indices])
    np.testing.assert_allclose(cmap.data[2, 2], pure_line_flux, rtol=1e-2)


def test_invalid_masked_and_dq_propagation():
    """Requirement 4: NaN input pixels remain NaN; DQ flags combine via bitwise OR."""
    n_wl = 20
    wl = np.linspace(10.0, 12.0, n_wl)
    ny, nx = 4, 4

    data = np.ones((n_wl, ny, nx), dtype=float)
    data[:, 0, 0] = np.nan  # Masked pixel at (0,0)

    dq = np.zeros((n_wl, ny, nx), dtype=np.int32)
    dq[9, 2, 2] = 2   # Flag 2 (index 9 is ~10.95 um, within 10.75-11.25 um window)
    dq[10, 2, 2] = 8  # Flag 8 (index 10 is ~11.05 um, within 10.75-11.25 um window)

    cube = _make_fixture_cube(data, wl, dq=dq)

    cmap = extract_component_map(
        cube,
        central_wavelength_um=11.0,
        integration_width_um=0.5,
        feature_name="dq_test",
        continuum_subtraction=False,
    )

    # Masked pixel remains NaN
    assert np.isnan(cmap.data[0, 0])
    # Bitwise OR: 2 | 8 = 10
    assert cmap.dq[2, 2] == 10


def test_uncertainty_and_snr_propagation():
    """Requirement 4: Propagate uncertainties and compute calibrated SNR map."""
    n_wl = 50
    wl = np.linspace(10.0, 12.0, n_wl)
    ny, nx = 4, 4

    data = np.full((n_wl, ny, nx), 10.0, dtype=float)
    err = np.full((n_wl, ny, nx), 1.0, dtype=float)

    cube = _make_fixture_cube(data, wl, err=err)

    cmap = extract_component_map(
        cube,
        central_wavelength_um=11.0,
        integration_width_um=0.4,
        feature_name="snr_test",
        continuum_subtraction=False,
    )

    assert cmap.has_authoritative_uncertainty is True
    assert cmap.uncertainty is not None
    assert cmap.snr is not None
    assert np.all(cmap.snr > 0)

    # SNR should equal data / uncertainty
    np.testing.assert_allclose(cmap.snr, cmap.data / cmap.uncertainty, rtol=1e-5)


def test_wavelength_coverage_rejection():
    """Requirement 3: Raise WavelengthCoverageError when feature or continuum is out of bounds."""
    cube = generate_synthetic_cube(wl_start=5.0, wl_end=10.0)

    # Feature at 12.0 um is outside 5-10 um range
    with pytest.raises(WavelengthCoverageError):
        extract_component_map(
            cube,
            central_wavelength_um=12.0,
            integration_width_um=0.5,
            feature_name="out_of_bounds",
        )

    # Continuum windows extend outside range
    with pytest.raises(WavelengthCoverageError):
        extract_component_map(
            cube,
            central_wavelength_um=5.2,
            integration_width_um=0.3,
            feature_name="edge_continuum_out",
            continuum_subtraction={
                "enabled": True,
                "sideband_width_um": 0.5,
                "gap_um": 0.3,
            },
            allow_one_sided_continuum=False,
        )


def test_broad_pah_vs_narrow_line_recipes():
    """Requirement 3: Support broad PAH features separately from narrow atomic lines."""
    n_wl = 100
    wl = np.linspace(10.0, 14.0, n_wl)
    data = np.ones((n_wl, 5, 5), dtype=float)
    cube = _make_fixture_cube(data, wl)

    # Broad PAH 11.3 um feature (width 0.4 um)
    cmap_pah = extract_component_map(
        cube,
        central_wavelength_um=11.3,
        integration_width_um=0.4,
        feature_name="PAH_11.3",
        continuum_subtraction=False,
    )
    assert cmap_pah.component_name == "PAH_11.3"
    assert cmap_pah.data.shape == (5, 5)

    # Narrow atomic line [Ne II] 12.814 um (width 0.06 um)
    cmap_line = extract_component_map(
        cube,
        central_wavelength_um=12.814,
        integration_width_um=0.06,
        feature_name="[Ne II]",
        continuum_subtraction=False,
    )
    assert cmap_line.component_name == "[Ne II]"


def test_fits_and_provenance_export_and_reload():
    """Requirement 5 & 6: Export component map to multi-extension FITS + JSON sidecar and verify."""
    n_wl = 30
    wl = np.linspace(10.0, 12.0, n_wl)
    data = np.ones((n_wl, 6, 6), dtype=float)
    err = np.full((n_wl, 6, 6), 0.5, dtype=float)
    dq = np.zeros((n_wl, 6, 6), dtype=np.int32)

    cube = _make_fixture_cube(data, wl, err=err, dq=dq)
    cmap = extract_component_map(
        cube,
        central_wavelength_um=11.0,
        integration_width_um=0.2,
        feature_name="export_test",
        continuum_subtraction=False,
        recipe_channel="green",
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        fits_path = Path(tmp_dir) / "test_map.fits"
        saved = cmap.save_fits(fits_path, export_sidecar=True)

        assert saved.exists()
        sidecar = saved.with_name(f"{saved.name}.provenance.json")
        assert sidecar.exists()

        # Inspect FITS extensions
        with fits.open(saved) as hdul:
            ext_names = [hdu.name for hdu in hdul]
            assert "SCI" in ext_names
            assert "ERR" in ext_names
            assert "SNR" in ext_names
            assert "DQ" in ext_names
            assert "COV" in ext_names
            assert hdul[0].header["COMPONENT"] == "export_test"
            assert hdul[0].header["CHANNEL"] == "green"

        # Inspect JSON sidecar
        import json
        prov_data = json.loads(sidecar.read_text(encoding="utf-8"))
        assert prov_data["component_name"] == "export_test"
        assert prov_data["recipe_channel"] == "green"
        assert prov_data["has_authoritative_uncertainty"] is True
