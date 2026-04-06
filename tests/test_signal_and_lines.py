"""Tests for line database, matching, confidence scoring, and reliability."""
from __future__ import annotations

import numpy as np
import pytest

from ifu_spectral_cube.lines import (
    classify_detection_reliability,
    estimate_redshift_from_line_consensus,
    load_line_database,
    match_peaks_to_lines,
)
from ifu_spectral_cube.models import GaussianFit, LineMatch, Peak


class TestLineDatabase:
    def test_loads_bundled_db(self) -> None:
        db = load_line_database()
        assert len(db) >= 40  # Expanded database
        names = [r["line_name"] for r in db]
        assert any("[Ne II]" in n for n in names)

    def test_wavelength_filter(self) -> None:
        db = load_line_database(wmin_micron=10.0, wmax_micron=15.0)
        for row in db:
            assert 10.0 <= float(row["rest_wavelength_micron"]) <= 15.0

    def test_has_pah_features(self) -> None:
        db = load_line_database()
        species = [r.get("species_type", "") for r in db]
        assert any("pah" in s for s in species)

    def test_has_molecular_hydrogen(self) -> None:
        db = load_line_database()
        names = [r["line_name"] for r in db]
        h2_lines = [n for n in names if "H2" in n]
        assert len(h2_lines) >= 6


class TestRedshiftConsensus:
    def test_known_redshift(self) -> None:
        z_true = 0.012
        peaks = [
            Peak(index=10, wavelength_micron=12.8136 * (1 + z_true), flux=1.0,
                 prominence=1.0, width_samples=3.0, snr=9.0),
            Peak(index=20, wavelength_micron=15.5551 * (1 + z_true), flux=0.8,
                 prominence=0.8, width_samples=3.0, snr=8.0),
        ]
        db = [
            {"line_name": "[Ne II]", "rest_wavelength_micron": 12.8136},
            {"line_name": "[Ne III]", "rest_wavelength_micron": 15.5551},
        ]
        z_est = estimate_redshift_from_line_consensus(peaks, db, z_min=-0.01, z_max=0.05, dz=1e-3)
        assert abs(z_est - z_true) < 3e-3

    def test_no_peaks_returns_zero(self) -> None:
        assert estimate_redshift_from_line_consensus([], []) == 0.0


class TestLineMatching:
    def test_matches_known_lines(self) -> None:
        peaks = [
            Peak(index=100, wavelength_micron=12.8136, flux=2.0,
                 prominence=1.5, width_samples=5, snr=10.0),
        ]
        db = [
            {"line_name": "[Ne II] 12.8136", "rest_wavelength_micron": 12.8136,
             "ionization_class": "low", "species_type": "atomic_fine_structure", "notes": ""},
        ]
        matches = match_peaks_to_lines(peaks, db, redshift=0.0, min_confidence=0.1)
        assert len(matches) == 1
        assert matches[0].line_name == "[Ne II] 12.8136"

    def test_no_match_outside_tolerance(self) -> None:
        peaks = [Peak(index=50, wavelength_micron=20.0, flux=1.0,
                      prominence=1.0, width_samples=5, snr=5.0)]
        db = [{"line_name": "test", "rest_wavelength_micron": 10.0,
               "ionization_class": "low", "species_type": "", "notes": ""}]
        matches = match_peaks_to_lines(peaks, db, redshift=0.0)
        assert len(matches) == 0

    def test_ambiguous_flagged(self) -> None:
        # Two lines very close together
        peaks = [Peak(index=100, wavelength_micron=14.34, flux=1.0,
                      prominence=0.8, width_samples=5, snr=6.0)]
        db = [
            {"line_name": "[Ne V]", "rest_wavelength_micron": 14.3217,
             "ionization_class": "very_high", "species_type": "", "notes": ""},
            {"line_name": "[Cl II]", "rest_wavelength_micron": 14.3684,
             "ionization_class": "low", "species_type": "", "notes": ""},
        ]
        matches = match_peaks_to_lines(peaks, db, redshift=0.0, min_confidence=0.05,
                                        abs_tolerance_micron=0.05)
        if matches:
            assert matches[0].ambiguous


class TestReliability:
    def test_classification(self) -> None:
        matches = [
            LineMatch(peak_index=1, observed_wavelength_micron=12.8, line_name="[Ne II]",
                      rest_wavelength_micron=12.8136, ionization_class="low",
                      delta_micron=0.01, redshift=0.0, confidence=0.85, ambiguous=False),
            LineMatch(peak_index=2, observed_wavelength_micron=15.5, line_name="[Ne III]",
                      rest_wavelength_micron=15.5551, ionization_class="intermediate",
                      delta_micron=0.05, redshift=0.0, confidence=0.25, ambiguous=True),
        ]
        fits = [
            GaussianFit(peak_index=1, amplitude=2.0, center_micron=12.8, sigma_micron=0.01,
                        baseline_offset=0.0, baseline_slope=0.0,
                        center_uncertainty_micron=0.001, sigma_uncertainty_micron=0.001,
                        reduced_chi2=1.1),
        ]
        classified = classify_detection_reliability(matches, fits)
        assert classified[0].reliability == "secure"
        assert classified[1].reliability == "marginal"  # confidence 0.25 < 0.30 threshold
