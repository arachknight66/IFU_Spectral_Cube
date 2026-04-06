"""Pipeline configuration dataclasses and YAML loader.

All pipeline parameters are expressed as nested dataclasses so they
serialize trivially, support IDE autocompletion, and can be overridden
per-section from YAML or CLI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(slots=True)
class PreprocessConfig:
    """Continuum subtraction and noise reduction parameters."""
    continuum_window: int = 101
    iterative_continuum: bool = False
    continuum_sigma_clip: float = 3.0
    continuum_n_iter: int = 3
    savgol_window: int = 11
    savgol_polyorder: int = 3
    gaussian_sigma_channels: float = 1.5
    defringe: bool = False
    defringe_period_min: float = 0.01
    defringe_period_max: float = 0.1
    defringe_max_components: int = 3
    apply_dq_mask: bool = True
    bad_dq_flags: int = 1


@dataclass(slots=True)
class ExtractionConfig:
    """Spectrum extraction geometry and method."""
    mode: str = "circular"  # circular | percentile | optimal
    center_x: float | None = None
    center_y: float | None = None
    radius: float = 2.5
    statistic: str = "mean"
    percentile: float = 90.0
    background_inner_radius: float | None = None
    background_outer_radius: float | None = None


@dataclass(slots=True)
class PeakConfig:
    """Peak detection thresholds."""
    prominence_sigma: float = 4.5
    height_sigma: float = 0.0
    resolving_power: float = 2500.0
    min_snr: float = 4.5
    use_cwt: bool = False
    merge_cwt: bool = True
    cwt_min_snr: float = 4.0
    local_snr_window: int = 50


@dataclass(slots=True)
class IdentificationConfig:
    """Line matching and confidence parameters."""
    line_db_path: str | None = None
    z_min: float = -0.02
    z_max: float = 0.2
    dz: float = 5e-4
    abs_tolerance_micron: float = 0.008
    sigma_tolerance: float = 2.5
    min_confidence: float = 0.2


@dataclass(slots=True)
class FitConfig:
    """Gaussian/Voigt fitting parameters."""
    enabled: bool = True
    window_half_width_micron: float = 0.08
    try_voigt: bool = True
    deblend_radius_micron: float = 0.05


@dataclass(slots=True)
class VisualizationConfig:
    """Control which diagnostic plots are generated."""
    smoothing_comparison: bool = True
    peaks_and_matches: bool = True
    gaussian_fits: bool = True
    redshift_histogram: bool = True
    noise_profile: bool = True
    emission_maps: bool = True
    line_diagnostic_grid: bool = True
    interactive_html: bool = False


@dataclass(slots=True)
class PipelineConfig:
    """Top-level pipeline configuration."""
    cube_path: str
    output_dir: str
    wmin: float | None = None
    wmax: float | None = None
    preprocessing: PreprocessConfig = field(default_factory=PreprocessConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    peaks: PeakConfig = field(default_factory=PeakConfig)
    identification: IdentificationConfig = field(default_factory=IdentificationConfig)
    fitting: FitConfig = field(default_factory=FitConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    """Load pipeline configuration from a YAML file."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    return PipelineConfig(
        cube_path=raw["cube_path"],
        output_dir=raw["output_dir"],
        wmin=raw.get("wmin"),
        wmax=raw.get("wmax"),
        preprocessing=PreprocessConfig(**raw.get("preprocessing", {})),
        extraction=ExtractionConfig(**raw.get("extraction", {})),
        peaks=PeakConfig(**raw.get("peaks", {})),
        identification=IdentificationConfig(**raw.get("identification", {})),
        fitting=FitConfig(**raw.get("fitting", {})),
        visualization=VisualizationConfig(**raw.get("visualization", {})),
    )
