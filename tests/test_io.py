"""Tests for FITS I/O with synthetic cubes."""
from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from ifu_spectral_cube.io import clip_cube_wavelength, load_miri_ifu_cube


def _create_synthetic_fits(
    path: str,
    n_lambda: int = 100,
    ny: int = 10,
    nx: int = 10,
    wmin: float = 10.0,
    wmax: float = 20.0,
    add_err: bool = False,
    add_dq: bool = False,
) -> None:
    """Create a minimal synthetic MIRI-like FITS cube."""
    rng = np.random.default_rng(42)
    data = rng.uniform(0.5, 1.5, size=(n_lambda, ny, nx)).astype(np.float32)

    # Add Gaussian emission line at center
    wl = np.linspace(wmin, wmax, n_lambda)
    line_profile = 5.0 * np.exp(-0.5 * ((wl - 15.0) / 0.02) ** 2)
    data[:, ny // 2, nx // 2] += line_profile

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
    header["TARGNAME"] = "SYNTHETIC"

    hdul = fits.HDUList()
    hdul.append(fits.PrimaryHDU())
    hdul.append(fits.ImageHDU(data=data, header=header, name="SCI"))

    if add_err:
        err = np.full_like(data, 0.1)
        hdul.append(fits.ImageHDU(data=err, name="ERR"))

    if add_dq:
        dq = np.zeros_like(data, dtype=np.int32)
        dq[50, 0, 0] = 1  # Flag one voxel
        hdul.append(fits.ImageHDU(data=dq, name="DQ"))

    hdul.writeto(path, overwrite=True)


class TestIOLoad:
    def test_load_synthetic_cube(self, tmp_path) -> None:
        path = str(tmp_path / "test_cube.fits")
        _create_synthetic_fits(path)
        cube = load_miri_ifu_cube(path)
        assert cube.data.ndim == 3
        assert cube.data.shape[0] == 100  # spectral axis first
        assert cube.wavelength_micron.shape[0] == 100
        assert cube.metadata["instrument"] == "MIRI"
        assert cube.flux_unit == "MJy/sr"

    def test_wavelength_range(self, tmp_path) -> None:
        path = str(tmp_path / "test_cube.fits")
        _create_synthetic_fits(path, wmin=5.0, wmax=28.0)
        cube = load_miri_ifu_cube(path)
        assert cube.wavelength_micron[0] == pytest.approx(5.0, abs=0.5)
        assert cube.wavelength_micron[-1] == pytest.approx(28.0, abs=0.5)

    def test_loads_err_extension(self, tmp_path) -> None:
        path = str(tmp_path / "test_cube.fits")
        _create_synthetic_fits(path, add_err=True)
        cube = load_miri_ifu_cube(path)
        assert cube.err is not None
        assert cube.err.shape == cube.data.shape

    def test_loads_dq_extension(self, tmp_path) -> None:
        path = str(tmp_path / "test_cube.fits")
        _create_synthetic_fits(path, add_dq=True)
        cube = load_miri_ifu_cube(path)
        assert cube.dq is not None


class TestClip:
    def test_clip_wavelength(self, tmp_path) -> None:
        path = str(tmp_path / "test_cube.fits")
        _create_synthetic_fits(path, wmin=5.0, wmax=25.0, n_lambda=200)
        cube = load_miri_ifu_cube(path)
        clipped = clip_cube_wavelength(cube, wmin=10.0, wmax=20.0)
        assert clipped.wavelength_micron[0] >= 10.0
        assert clipped.wavelength_micron[-1] <= 20.0
        assert clipped.data.shape[0] == clipped.wavelength_micron.shape[0]

    def test_clip_none_returns_original(self, tmp_path) -> None:
        path = str(tmp_path / "test_cube.fits")
        _create_synthetic_fits(path)
        cube = load_miri_ifu_cube(path)
        same = clip_cube_wavelength(cube, None, None)
        assert same.data.shape == cube.data.shape
