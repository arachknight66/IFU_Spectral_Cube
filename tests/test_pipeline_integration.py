"""End-to-end integration test: synthetic cube → full pipeline → validate outputs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from ifu_spectral_cube.config import PipelineConfig
from ifu_spectral_cube.pipeline import run_pipeline


def _create_realistic_synthetic_cube(path: str) -> dict[str, float]:
    """Create a synthetic MIRI cube with known emission lines.

    Returns the injected line parameters for validation.
    """
    rng = np.random.default_rng(42)
    n_lambda, ny, nx = 500, 15, 15
    wmin, wmax = 10.0, 20.0
    wl = np.linspace(wmin, wmax, n_lambda)

    # Flat continuum + noise
    data = np.full((n_lambda, ny, nx), 1.0, dtype=np.float32)
    data += rng.normal(0, 0.05, data.shape).astype(np.float32)

    # Inject known emission lines at center spaxel
    injected = {
        "[Ne II] 12.8136": {"mu": 12.8136, "amp": 5.0, "sig": 0.02},
        "[Ne III] 15.5551": {"mu": 15.5551, "amp": 4.0, "sig": 0.025},
        "H2 S(1) 17.0348": {"mu": 17.0348, "amp": 2.5, "sig": 0.015},
    }

    for params in injected.values():
        line_profile = params["amp"] * np.exp(
            -0.5 * ((wl - params["mu"]) / params["sig"]) ** 2
        )
        # Add to central 3×3 region
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                data[:, ny // 2 + dy, nx // 2 + dx] += line_profile

    # Build header
    header = fits.Header()
    header["NAXIS"] = 3
    header["NAXIS1"] = nx
    header["NAXIS2"] = ny
    header["NAXIS3"] = n_lambda
    header["CRVAL3"] = wmin
    header["CDELT3"] = (wmax - wmin) / (n_lambda - 1)
    header["CRPIX3"] = 1.0
    header["CUNIT3"] = "um"
    header["CTYPE3"] = "WAVE"
    header["BUNIT"] = "MJy/sr"
    header["INSTRUME"] = "MIRI"
    header["DETECTOR"] = "MIRIMAGE"
    header["TARGNAME"] = "SYNTHETIC_TEST"

    err = np.full_like(data, 0.05)

    hdul = fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(data=data, header=header, name="SCI"),
        fits.ImageHDU(data=err, name="ERR"),
    ])
    hdul.writeto(path, overwrite=True)
    return {name: params["mu"] for name, params in injected.items()}


class TestPipelineIntegration:
    """Full pipeline integration test with synthetic data."""

    def test_full_pipeline_circular(self, tmp_path) -> None:
        cube_path = str(tmp_path / "synthetic_cube.fits")
        out_dir = str(tmp_path / "outputs")
        injected = _create_realistic_synthetic_cube(cube_path)

        config = PipelineConfig(
            cube_path=cube_path,
            output_dir=out_dir,
            wmin=10.0,
            wmax=20.0,
        )
        # Center extraction on the source
        config.extraction.center_x = 7.0
        config.extraction.center_y = 7.0
        config.extraction.radius = 2.0
        # Lower threshold for synthetic data
        config.peaks.prominence_sigma = 3.0
        config.peaks.min_snr = 3.0
        config.preprocessing.continuum_window = 51

        # Run pipeline
        summary = run_pipeline(config)

        # Validate outputs exist
        out = Path(out_dir)
        assert (out / "summary.json").exists()
        assert (out / "detected_peaks.csv").exists()
        assert (out / "line_matches.csv").exists()
        assert (out / "gaussian_fits.csv").exists()
        assert (out / "intermediate_spectra.npz").exists()
        assert (out / "smoothing_comparison.png").exists()
        assert (out / "peaks_and_matches.png").exists()

        # Validate summary content
        assert summary["n_peaks_detected"] >= 2
        assert summary["n_lines_matched"] >= 1
        assert summary["n_spectral_channels"] == 500
        assert "timings_seconds" in summary

        # Validate at least one injected line was recovered
        matched_names = set()
        with open(out / "line_matches.csv", "r") as f:
            import csv
            reader = csv.DictReader(f)
            for row in reader:
                matched_names.add(row["line_name"])

        ne_ii_found = any("[Ne II]" in n for n in matched_names)
        ne_iii_found = any("[Ne III]" in n for n in matched_names)
        assert ne_ii_found or ne_iii_found, f"Expected to find Ne II or Ne III. Found: {matched_names}"

    def test_pipeline_percentile_mode(self, tmp_path) -> None:
        cube_path = str(tmp_path / "synthetic_cube.fits")
        out_dir = str(tmp_path / "outputs_pct")
        _create_realistic_synthetic_cube(cube_path)

        config = PipelineConfig(
            cube_path=cube_path,
            output_dir=out_dir,
        )
        config.extraction.mode = "percentile"
        config.extraction.percentile = 80.0

        summary = run_pipeline(config)
        assert summary["n_peaks_detected"] >= 1

    def test_pipeline_with_voigt(self, tmp_path) -> None:
        cube_path = str(tmp_path / "synthetic_cube.fits")
        out_dir = str(tmp_path / "outputs_voigt")
        _create_realistic_synthetic_cube(cube_path)

        config = PipelineConfig(
            cube_path=cube_path,
            output_dir=out_dir,
        )
        config.extraction.center_x = 7.0
        config.extraction.center_y = 7.0
        config.extraction.radius = 2.0
        config.fitting.try_voigt = True
        config.peaks.prominence_sigma = 3.0
        config.peaks.min_snr = 3.0
        config.preprocessing.continuum_window = 51

        summary = run_pipeline(config)
        assert summary["n_peaks_detected"] >= 1

    def test_summary_json_valid(self, tmp_path) -> None:
        cube_path = str(tmp_path / "synthetic_cube.fits")
        out_dir = str(tmp_path / "outputs_json")
        _create_realistic_synthetic_cube(cube_path)

        config = PipelineConfig(
            cube_path=cube_path,
            output_dir=out_dir,
        )
        config.extraction.center_x = 7.0
        config.extraction.center_y = 7.0

        run_pipeline(config)

        with open(Path(out_dir) / "summary.json", "r") as f:
            data = json.load(f)
        assert "estimated_redshift_initial" in data
        assert "estimated_redshift_fitted" in data
        assert "reliability_counts" in data
        assert isinstance(data["timings_seconds"], dict)
