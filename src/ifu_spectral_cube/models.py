"""Data models for the JWST MIRI IFU spectral line identification pipeline.

All domain objects are plain dataclasses so they serialize trivially and
carry no hidden state.  The ``SpectralCube`` container enforces the
spectral-axis-first convention used throughout the package.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from astropy.io.fits import Header
from astropy.wcs import WCS


@dataclass(slots=True)
class SpectralCube:
    """Canonical IFU cube container with spectral axis first.

    Shape convention: ``(n_lambda, ny, nx)`` — the spectral dimension
    is always axis 0.  This is enforced at load time by ``io.py``.
    """

    data: np.ndarray
    wavelength_micron: np.ndarray
    header: Header
    wcs: WCS | None
    flux_unit: str | None = None
    err: np.ndarray | None = None
    dq: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Peak:
    """Detected 1D peak in an extracted spectrum.

    ``local_snr`` is computed in a per-peak window and may differ from the
    global ``snr`` that uses the whole-spectrum noise estimate.
    """

    index: int
    wavelength_micron: float
    flux: float
    prominence: float
    width_samples: float
    snr: float
    local_snr: float = float("nan")


@dataclass(slots=True)
class LineMatch:
    """Proposed assignment of a detected peak to a rest-frame transition.

    ``reliability`` is a human-readable classification of detection quality
    (``secure``, ``probable``, ``tentative``, ``marginal``).
    """

    peak_index: int
    observed_wavelength_micron: float
    line_name: str
    rest_wavelength_micron: float
    ionization_class: str
    delta_micron: float
    redshift: float
    confidence: float
    ambiguous: bool
    notes: str = ""
    reliability: str = "tentative"


@dataclass(slots=True)
class GaussianFit:
    """Gaussian + linear baseline fit around a detected spectral line.

    ``residual_normality_p`` is the Shapiro-Wilk p-value of the fit
    residuals (high → residuals look Gaussian → good fit).
    ``durbin_watson`` is the DW statistic for residual autocorrelation
    (≈2 → uncorrelated ≈ good; <1.5 → positive autocorrelation ≈ fringes).
    """

    peak_index: int
    amplitude: float
    center_micron: float
    sigma_micron: float
    baseline_offset: float
    baseline_slope: float
    center_uncertainty_micron: float
    sigma_uncertainty_micron: float
    reduced_chi2: float
    residual_normality_p: float = float("nan")
    durbin_watson: float = float("nan")


@dataclass(slots=True)
class VoigtFit:
    """Voigt + linear baseline fit.

    The Voigt profile is the convolution of Gaussian (thermal/instrumental
    broadening, sigma_G) and Lorentzian (natural/pressure broadening, gamma_L).
    ``bic`` is the Bayesian Information Criterion for model comparison.
    """

    peak_index: int
    amplitude: float
    center_micron: float
    sigma_gauss_micron: float
    gamma_lorentz_micron: float
    baseline_offset: float
    baseline_slope: float
    center_uncertainty_micron: float
    reduced_chi2: float
    bic: float = float("nan")
    residual_normality_p: float = float("nan")
    durbin_watson: float = float("nan")
