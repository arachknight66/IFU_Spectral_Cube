"""End-to-end pipeline orchestrator.

Executes the full spectral line identification workflow:

    Load FITS → DQ mask → Extract → Continuum subtract → Defringe →
    Smooth → Detect peaks → Refine SNR → Match lines → Fit profiles →
    Classify reliability → Visualize → Write outputs

Each stage is logged with timing information for performance profiling.
"""
from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import PipelineConfig
from .extraction import (
    annular_mask,
    circular_mask,
    extract_optimal_spectrum,
    extract_percentile_spectrum,
    extract_region_spectrum,
    extract_with_background,
)
from .fitting import estimate_redshift_from_matches, fit_all_peaks, fit_multiplet
from .io import clip_cube_wavelength, load_miri_ifu_cube
from .lines import (
    classify_detection_reliability,
    estimate_redshift_from_line_consensus,
    load_line_database,
    match_peaks_to_lines,
)
from .models import GaussianFit, LineMatch, Peak, VoigtFit
from .peaks import (
    detect_peaks_1d,
    detect_peaks_cwt,
    merge_peak_lists,
    refine_peak_local_snr,
    suggest_miri_peak_params,
)
from .preprocessing import (
    apply_dq_mask,
    apply_gaussian_smoothing,
    apply_savgol_smoothing,
    compare_smoothing_metrics,
    defringe_1d,
    subtract_continuum_1d,
)
from .visualization import (
    export_interactive_html,
    plot_gaussian_fits,
    plot_line_diagnostic_grid,
    plot_line_emission_map,
    plot_noise_profile,
    plot_peaks_and_matches,
    plot_redshift_histogram,
    plot_smoothing_comparison,
)

logger = logging.getLogger("ifu_spectral_cube")


# ---------------------------------------------------------------------------
# Timer context manager
# ---------------------------------------------------------------------------

class _Timer:
    """Simple context manager for stage timing."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.elapsed: float = 0.0

    def __enter__(self) -> "_Timer":
        self._start = time.perf_counter()
        logger.info("▶ %s …", self.name)
        return self

    def __exit__(self, *args: object) -> None:
        self.elapsed = time.perf_counter() - self._start
        logger.info("  ✓ %s completed in %.2f s", self.name, self.elapsed)


# ---------------------------------------------------------------------------
# CSV / JSON helpers
# ---------------------------------------------------------------------------

def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _serialize_peaks(peaks: list[Peak]) -> list[dict[str, object]]:
    return [
        {
            "index": p.index,
            "wavelength_micron": p.wavelength_micron,
            "flux": p.flux,
            "prominence": p.prominence,
            "width_samples": p.width_samples,
            "snr": p.snr,
            "local_snr": p.local_snr,
        }
        for p in peaks
    ]


def _serialize_matches(matches: list[LineMatch]) -> list[dict[str, object]]:
    return [asdict(m) for m in matches]


def _serialize_fits(fits: list[GaussianFit]) -> list[dict[str, object]]:
    return [asdict(f) for f in fits]


def _serialize_voigt_fits(fits: list[VoigtFit]) -> list[dict[str, object]]:
    return [asdict(f) for f in fits]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(config: PipelineConfig) -> dict[str, object]:
    """Execute the full spectral line identification pipeline.

    Returns a summary dict with all key results and timing information.
    """
    # Set up logging if not already configured
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    timings: dict[str, float] = {}
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Stage 1: Load FITS cube ──────────────────────────────────────────
    with _Timer("Loading FITS cube") as t:
        cube = load_miri_ifu_cube(config.cube_path)
        cube = clip_cube_wavelength(cube, config.wmin, config.wmax)
        logger.info("  Cube shape: %s, λ range: %.3f – %.3f µm",
                     cube.data.shape,
                     float(np.nanmin(cube.wavelength_micron)),
                     float(np.nanmax(cube.wavelength_micron)))
    timings["load"] = t.elapsed

    # ── Stage 2: DQ masking ──────────────────────────────────────────────
    if config.preprocessing.apply_dq_mask and cube.dq is not None:
        with _Timer("Applying DQ mask") as t:
            cube_data = apply_dq_mask(cube.data, cube.dq, config.preprocessing.bad_dq_flags)
        timings["dq_mask"] = t.elapsed
    else:
        cube_data = cube.data.astype(float)

    # ── Stage 3: Spectrum extraction ─────────────────────────────────────
    with _Timer("Extracting spectrum") as t:
        spatial_shape = (cube_data.shape[1], cube_data.shape[2])

        if config.extraction.mode == "circular":
            center_x = config.extraction.center_x
            center_y = config.extraction.center_y
            if center_x is None or center_y is None:
                center_x = (cube_data.shape[2] - 1) / 2.0
                center_y = (cube_data.shape[1] - 1) / 2.0

            source_mask = circular_mask(spatial_shape, (center_x, center_y), config.extraction.radius)

            # Background subtraction if annulus is configured
            if (config.extraction.background_inner_radius is not None and
                    config.extraction.background_outer_radius is not None):
                bg_mask = annular_mask(
                    spatial_shape, (center_x, center_y),
                    config.extraction.background_inner_radius,
                    config.extraction.background_outer_radius,
                )
                spectrum, bg_spectrum, n_spaxels = extract_with_background(
                    cube_data, source_mask, bg_mask, statistic=config.extraction.statistic)
                logger.info("  Background subtraction: %d bg spaxels", int(bg_mask.sum()))
            else:
                spectrum, n_spaxels = extract_region_spectrum(
                    cube_data, source_mask, statistic=config.extraction.statistic)

        elif config.extraction.mode == "optimal":
            if cube.err is None:
                raise ValueError("Optimal extraction requires ERR extension in FITS cube.")
            center_x = config.extraction.center_x
            center_y = config.extraction.center_y
            if center_x is None or center_y is None:
                center_x = (cube_data.shape[2] - 1) / 2.0
                center_y = (cube_data.shape[1] - 1) / 2.0

            source_mask = circular_mask(spatial_shape, (center_x, center_y), config.extraction.radius)
            spectrum, uncertainty, n_spaxels = extract_optimal_spectrum(cube_data, cube.err, source_mask)
            logger.info("  Optimal extraction: median uncertainty %.3e", float(np.nanmedian(uncertainty)))

        elif config.extraction.mode == "percentile":
            spectrum, pmask = extract_percentile_spectrum(
                cube_data=cube_data,
                percentile=config.extraction.percentile,
                statistic=config.extraction.statistic,
            )
            n_spaxels = int(pmask.sum())
        else:
            raise ValueError(f"Unsupported extraction mode: {config.extraction.mode}")

        logger.info("  %d spaxels used", n_spaxels)
    timings["extraction"] = t.elapsed

    # ── Stage 3b: Clean NaN channels ─────────────────────────────────
    nan_count = int(np.sum(~np.isfinite(spectrum)))
    if nan_count > 0:
        logger.info("  Cleaning %d NaN channels (%.1f%%)", nan_count, 100.0 * nan_count / spectrum.size)
        spectrum = np.where(np.isfinite(spectrum), spectrum, 0.0)

    # ── Stage 4: Continuum subtraction ───────────────────────────────────
    with _Timer("Continuum subtraction") as t:
        continuum_subtracted, continuum = subtract_continuum_1d(
            spectrum,
            window_length=config.preprocessing.continuum_window,
            iterative=config.preprocessing.iterative_continuum,
            sigma_clip=config.preprocessing.continuum_sigma_clip,
            n_iter=config.preprocessing.continuum_n_iter,
        )
    timings["continuum"] = t.elapsed

    # ── Stage 5: Defringing (optional) ───────────────────────────────────
    defringe_info: dict[str, object] = {}
    if config.preprocessing.defringe:
        with _Timer("Defringing") as t:
            continuum_subtracted, defringe_info = defringe_1d(
                continuum_subtracted,
                cube.wavelength_micron,
                period_range_micron=(
                    config.preprocessing.defringe_period_min,
                    config.preprocessing.defringe_period_max,
                ),
                max_components=config.preprocessing.defringe_max_components,
            )
            logger.info("  Removed %d fringe components", defringe_info.get("n_components_removed", 0))
        timings["defringe"] = t.elapsed

    # ── Stage 6: Noise reduction ─────────────────────────────────────────
    with _Timer("Noise reduction") as t:
        savgol = apply_savgol_smoothing(
            continuum_subtracted,
            window_length=config.preprocessing.savgol_window,
            polyorder=config.preprocessing.savgol_polyorder,
        )
        gaussian = apply_gaussian_smoothing(
            continuum_subtracted,
            sigma_channels=config.preprocessing.gaussian_sigma_channels,
        )
        smoothing_metrics = compare_smoothing_metrics(continuum_subtracted, savgol, gaussian)
    timings["smoothing"] = t.elapsed

    # ── Stage 7: Peak detection ──────────────────────────────────────────
    with _Timer("Peak detection") as t:
        default_peak = suggest_miri_peak_params(
            cube.wavelength_micron,
            resolving_power=config.peaks.resolving_power,
            min_snr=config.peaks.min_snr,
        )
        peaks, _, noise_sigma = detect_peaks_1d(
            cube.wavelength_micron,
            savgol,
            prominence_sigma=config.peaks.prominence_sigma or float(default_peak["prominence_sigma"]),
            height_sigma=config.peaks.height_sigma,
            width=default_peak["width"],
            distance=int(default_peak["distance"]),
        )
        logger.info("  Standard detection: %d peaks", len(peaks))

        # CWT detection (optional)
        if config.peaks.use_cwt:
            cwt_peaks, _ = detect_peaks_cwt(
                cube.wavelength_micron, savgol,
                min_snr=config.peaks.cwt_min_snr,
            )
            logger.info("  CWT detection: %d peaks", len(cwt_peaks))
            if config.peaks.merge_cwt and cwt_peaks:
                peaks = merge_peak_lists(peaks, cwt_peaks, merge_radius_channels=3)
                logger.info("  Merged: %d unique peaks", len(peaks))

        # Local SNR refinement
        if peaks:
            peaks = refine_peak_local_snr(
                cube.wavelength_micron, savgol, peaks,
                local_window_channels=config.peaks.local_snr_window,
            )
    timings["peak_detection"] = t.elapsed

    # ── Stage 8: Line identification ─────────────────────────────────────
    with _Timer("Line identification") as t:
        line_db = load_line_database(
            path=config.identification.line_db_path,
            wmin_micron=float(np.nanmin(cube.wavelength_micron)),
            wmax_micron=float(np.nanmax(cube.wavelength_micron)),
        )
        z_init = estimate_redshift_from_line_consensus(
            peaks=peaks,
            line_db=line_db,
            z_min=config.identification.z_min,
            z_max=config.identification.z_max,
            dz=config.identification.dz,
        )
        logger.info("  Initial redshift estimate: z = %.6f", z_init)

        matches = match_peaks_to_lines(
            peaks=peaks,
            line_db=line_db,
            redshift=z_init,
            resolving_power=config.peaks.resolving_power,
            abs_tolerance_micron=config.identification.abs_tolerance_micron,
            sigma_tolerance=config.identification.sigma_tolerance,
            min_confidence=config.identification.min_confidence,
        )
        logger.info("  %d lines matched", len(matches))
    timings["identification"] = t.elapsed

    # ── Stage 9: Profile fitting ─────────────────────────────────────────
    gauss_fits: list[GaussianFit] = []
    voigt_fits: list[VoigtFit] = []
    if config.fitting.enabled and peaks:
        with _Timer("Profile fitting") as t:
            gauss_fits, voigt_fits = fit_all_peaks(
                wavelength_micron=cube.wavelength_micron,
                flux=continuum_subtracted,
                peaks=peaks,
                window_half_width_micron=config.fitting.window_half_width_micron,
                try_voigt=config.fitting.try_voigt,
            )
            logger.info("  %d Gaussian fits, %d Voigt-preferred fits",
                         len(gauss_fits), len(voigt_fits))
        timings["fitting"] = t.elapsed

    # ── Stage 10: Reliability classification ─────────────────────────────
    with _Timer("Reliability classification") as t:
        # Re-match with fit quality information
        if gauss_fits:
            matches = match_peaks_to_lines(
                peaks=peaks,
                line_db=line_db,
                redshift=z_init,
                resolving_power=config.peaks.resolving_power,
                abs_tolerance_micron=config.identification.abs_tolerance_micron,
                sigma_tolerance=config.identification.sigma_tolerance,
                min_confidence=config.identification.min_confidence,
                gaussian_fits=gauss_fits,
            )
        matches = classify_detection_reliability(matches, gauss_fits)
        z_fit, z_scatter = estimate_redshift_from_matches(matches, gauss_fits)
        logger.info("  Refined redshift: z = %.6f ± %.6f", z_fit, z_scatter)

        # Count by reliability
        rel_counts = {}
        for m in matches:
            rel_counts[m.reliability] = rel_counts.get(m.reliability, 0) + 1
        for r, c in sorted(rel_counts.items()):
            logger.info("    %s: %d lines", r, c)
    timings["classification"] = t.elapsed

    # ── Stage 11: Write data outputs ─────────────────────────────────────
    with _Timer("Writing outputs") as t:
        _write_csv(out_dir / "detected_peaks.csv", _serialize_peaks(peaks))
        _write_csv(out_dir / "line_matches.csv", _serialize_matches(matches))
        _write_csv(out_dir / "gaussian_fits.csv", _serialize_fits(gauss_fits))
        if voigt_fits:
            _write_csv(out_dir / "voigt_fits.csv", _serialize_voigt_fits(voigt_fits))

        # Save intermediate spectra as numpy
        np.savez_compressed(
            out_dir / "intermediate_spectra.npz",
            wavelength_micron=cube.wavelength_micron,
            raw_spectrum=spectrum,
            continuum=continuum,
            continuum_subtracted=continuum_subtracted,
            savgol_smoothed=savgol,
            gaussian_smoothed=gaussian,
        )
    timings["write_outputs"] = t.elapsed

    # ── Stage 12: Visualization ──────────────────────────────────────────
    with _Timer("Generating plots") as t:
        viz = config.visualization

        if viz.smoothing_comparison:
            plot_smoothing_comparison(
                wavelength_micron=cube.wavelength_micron,
                original=spectrum,
                continuum=continuum,
                continuum_subtracted=continuum_subtracted,
                savgol=savgol,
                gaussian=gaussian,
                output_path=out_dir / "smoothing_comparison.png",
            )

        if viz.peaks_and_matches:
            plot_peaks_and_matches(
                wavelength_micron=cube.wavelength_micron,
                processed_spectrum=savgol,
                peaks=peaks,
                matches=matches,
                output_path=out_dir / "peaks_and_matches.png",
            )

        if viz.gaussian_fits and gauss_fits:
            plot_gaussian_fits(
                wavelength_micron=cube.wavelength_micron,
                flux=continuum_subtracted,
                peaks=peaks,
                fits=gauss_fits,
                matches=matches,
                output_path=out_dir / "gaussian_fit_panels.png",
                window_half_width_micron=config.fitting.window_half_width_micron,
            )

        if viz.redshift_histogram:
            plot_redshift_histogram(
                peaks=peaks,
                line_db=line_db,
                z_estimate=z_init,
                output_path=out_dir / "redshift_histogram.png",
                z_min=config.identification.z_min,
                z_max=config.identification.z_max,
                dz=config.identification.dz,
            )

        if viz.noise_profile:
            plot_noise_profile(
                wavelength_micron=cube.wavelength_micron,
                flux=continuum_subtracted,
                output_path=out_dir / "noise_profile.png",
            )

        if viz.emission_maps and matches:
            strongest = max(matches, key=lambda m: m.confidence)
            try:
                plot_line_emission_map(
                    cube=cube,
                    line_center_micron=strongest.observed_wavelength_micron,
                    half_width_micron=0.03,
                    output_path=out_dir / "strongest_line_map.png",
                    line_name=strongest.line_name,
                )
            except Exception as e:
                logger.warning("  Emission map failed: %s", e)

        if viz.line_diagnostic_grid and matches:
            try:
                plot_line_diagnostic_grid(
                    cube=cube,
                    matches=matches,
                    output_path=out_dir / "line_diagnostic_grid.png",
                )
            except Exception as e:
                logger.warning("  Diagnostic grid failed: %s", e)

        if viz.interactive_html:
            try:
                export_interactive_html(
                    wavelength_micron=cube.wavelength_micron,
                    processed_spectrum=savgol,
                    peaks=peaks,
                    matches=matches,
                    output_path=out_dir / "interactive_spectrum.html",
                )
            except Exception as e:
                logger.warning("  Interactive HTML export failed: %s", e)
    timings["visualization"] = t.elapsed

    # ── Summary ──────────────────────────────────────────────────────────
    summary: dict[str, object] = {
        "cube_path": config.cube_path,
        "output_dir": str(out_dir),
        "n_spectral_channels": int(cube.data.shape[0]),
        "spatial_shape": [int(cube.data.shape[1]), int(cube.data.shape[2])],
        "wavelength_range_micron": [
            float(np.nanmin(cube.wavelength_micron)),
            float(np.nanmax(cube.wavelength_micron)),
        ],
        "n_spaxels_used": int(n_spaxels),
        "noise_sigma": float(noise_sigma),
        "n_peaks_detected": len(peaks),
        "n_lines_matched": len(matches),
        "n_gaussian_fits": len(gauss_fits),
        "n_voigt_fits": len(voigt_fits),
        "reliability_counts": rel_counts,
        "estimated_redshift_initial": float(z_init),
        "estimated_redshift_fitted": float(z_fit),
        "redshift_scatter": float(z_scatter),
        "smoothing_metrics": smoothing_metrics,
        "defringe_info": defringe_info,
        "timings_seconds": timings,
        "metadata": cube.metadata,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    total_time = sum(timings.values())
    logger.info("━━ Pipeline complete in %.2f s ━━", total_time)
    return summary
