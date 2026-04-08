"""
Central Configuration for JWST MIRI IFU Spectral Pipeline.

All tunable parameters are defined here as a single dataclass.
No scientific processing parameter should be hardcoded anywhere else
in the codebase — everything flows from PipelineConfig.

Design rationale:
    Using a dataclass rather than a YAML/JSON config file keeps
    the configuration type-checked and IDE-navigable. The Streamlit
    frontend constructs a PipelineConfig from widget values, and the
    CLI constructs one from argparse, ensuring a single source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class PipelineConfig:
    """Master configuration for the spectral analysis pipeline.

    Attributes
    ----------
    savgol_window : int
        Window length for Savitzky-Golay filter (must be odd).
        Larger windows smooth more but risk washing out narrow features.
        Default of 11 channels corresponds to ~7 resolution elements
        for MIRI MRS Channel 1 (R~3500).
    savgol_polyorder : int
        Polynomial order for SG filter. Order 3 preserves up to the
        3rd moment of line profiles (position, width, skewness).
    continuum_poly_order : int
        Degree of polynomial used for continuum estimation.
        Order 3 handles gentle curvature; increase for complex continua
        (e.g., deeply embedded sources with silicate absorption).
    sigma_clip : float
        Number of sigma for iterative clipping during continuum fitting.
        Emission features above this threshold are excluded from the fit.
    peak_height : float or None
        Minimum absolute flux for a peak. None disables this filter.
        With continuum-subtracted data, use with caution.
    peak_prominence : float
        Detection threshold in units of noise RMS (σ). Set to 3.0 for
        the classical 3-sigma detection limit. Lower values find more
        (potentially spurious) peaks; higher values are more conservative.
    peak_distance : int
        Minimum separation between peaks in channels. Should be ≥ the
        instrument line-spread-function FWHM to avoid splitting single
        features.
    peak_width : tuple or None
        Expected range of peak widths in channels (min, max).
        Rejects cosmic-ray spikes (very narrow) and broad artifacts.
    tolerance_um : float
        Maximum |Δλ| (µm) for a line identification match.
        0.05 µm corresponds to Δv ≈ 400 km/s at 12 µm.
    redshift : float
        Source redshift for shifting the line database to the
        observed frame. z=0 for Galactic targets.
    default_band_width : float
        Default spectral width (µm) for band-integrated images.
    output_dir : str
        Directory for saving pipeline output products.
    """

    # — Preprocessing: Denoising —
    savgol_window: int = 11
    savgol_polyorder: int = 3

    # — Preprocessing: Continuum —
    continuum_poly_order: int = 3
    sigma_clip: float = 3.0

    # — Peak Detection —
    peak_height: Optional[float] = None
    peak_prominence: float = 3.0
    peak_distance: int = 5
    peak_width: Optional[tuple[int, int]] = None

    # — Line Identification —
    tolerance_um: float = 0.05
    redshift: float = 0.0

    # — Imaging —
    default_band_width: float = 0.5

    # — Output —
    output_dir: str = "output"

    def __post_init__(self) -> None:
        """Validate configuration upon creation."""
        if self.savgol_window % 2 == 0:
            raise ValueError(
                f"savgol_window must be odd, got {self.savgol_window}"
            )
        if self.savgol_polyorder >= self.savgol_window:
            raise ValueError(
                "savgol_polyorder must be < savgol_window: "
                f"{self.savgol_polyorder} >= {self.savgol_window}"
            )
        if self.peak_distance < 1:
            raise ValueError(
                f"peak_distance must be >= 1, got {self.peak_distance}"
            )
        # Ensure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
