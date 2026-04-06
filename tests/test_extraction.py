"""Tests for spectrum extraction: masks, background subtraction, optimal extraction."""
from __future__ import annotations

import numpy as np
import pytest

from ifu_spectral_cube.extraction import (
    annular_mask,
    circular_mask,
    extract_optimal_spectrum,
    extract_percentile_spectrum,
    extract_region_spectrum,
    extract_with_background,
    rectangular_mask,
)


class TestMasks:
    def test_circular_mask_center(self) -> None:
        mask = circular_mask((11, 11), center_xy=(5.0, 5.0), radius=2.0)
        assert mask[5, 5]  # center
        assert not mask[0, 0]  # corner

    def test_circular_mask_count(self) -> None:
        mask = circular_mask((21, 21), center_xy=(10.0, 10.0), radius=3.0)
        # Area ≈ π·r² ≈ 28.3; mask should be close
        assert 25 <= mask.sum() <= 32

    def test_rectangular_mask(self) -> None:
        mask = rectangular_mask((10, 10), x_min=2, x_max=5, y_min=3, y_max=7)
        assert mask[4, 3]
        assert not mask[0, 0]
        assert mask.sum() == (7 - 3) * (5 - 2)

    def test_annular_mask(self) -> None:
        mask = annular_mask((21, 21), center_xy=(10.0, 10.0), inner_radius=3.0, outer_radius=6.0)
        assert not mask[10, 10]  # center excluded
        assert mask.sum() > 0
        # Annulus area ≈ π(R² - r²) ≈ π(36-9) ≈ 84.8
        assert 70 <= mask.sum() <= 100


class TestExtraction:
    def _make_cube(self) -> np.ndarray:
        """Create a simple test cube: (50, 10, 10)."""
        rng = np.random.default_rng(42)
        return rng.uniform(1.0, 2.0, size=(50, 10, 10))

    def test_mean_extraction(self) -> None:
        cube = self._make_cube()
        mask = circular_mask((10, 10), (5.0, 5.0), 2.0)
        spec, n = extract_region_spectrum(cube, mask, statistic="mean")
        assert spec.shape == (50,)
        assert n > 0

    def test_sum_extraction(self) -> None:
        cube = np.ones((20, 5, 5))
        mask = np.ones((5, 5), dtype=bool)
        spec, n = extract_region_spectrum(cube, mask, statistic="sum")
        np.testing.assert_allclose(spec, 25.0)

    def test_empty_mask_raises(self) -> None:
        cube = self._make_cube()
        mask = np.zeros((10, 10), dtype=bool)
        with pytest.raises(ValueError, match="zero spaxels"):
            extract_region_spectrum(cube, mask)

    def test_background_subtraction(self) -> None:
        cube = np.ones((30, 15, 15)) * 5.0  # Uniform background
        # Add a strong source to the center spaxel
        cube[:, 7, 7] += 10.0
        src = circular_mask((15, 15), (7.0, 7.0), 1.0)
        bg = annular_mask((15, 15), (7.0, 7.0), 2.0, 5.0)
        net, bg_spec, n = extract_with_background(cube, src, bg, statistic="mean")
        # Background is ~5.0, so bg subtraction removes it.
        # Source region mean includes the bright center averaged with neighbours.
        # Net should be positive (source excess above background).
        assert np.mean(net) > 0.0

    def test_optimal_extraction(self) -> None:
        rng = np.random.default_rng(55)
        cube = rng.normal(10.0, 1.0, (30, 5, 5))
        err = np.full((30, 5, 5), 1.0)
        # Make one spaxel very noisy
        err[:, 0, 0] = 100.0
        mask = np.ones((5, 5), dtype=bool)
        spec, unc, n = extract_optimal_spectrum(cube, err, mask)
        assert spec.shape == (30,)
        assert unc.shape == (30,)
        assert n == 25

    def test_percentile_extraction(self) -> None:
        rng = np.random.default_rng(66)
        cube = rng.uniform(0, 1, (20, 10, 10))
        cube[:, 5, 5] = 100.0  # Bright spaxel
        spec, mask = extract_percentile_spectrum(cube, percentile=95.0)
        assert spec.shape == (20,)
        assert mask[5, 5]  # Bright spaxel should be selected
