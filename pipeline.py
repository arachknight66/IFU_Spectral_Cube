"""
Pipeline Orchestrator — End-to-End Spectral Analysis Workflow.

The ``SpectralPipeline`` class coordinates all backend processing in a
defined sequence:

    LOAD → PREPROCESS → EXTRACT → DETECT → FIT → IDENTIFY → IMAGE → VISUALIZE

The Streamlit frontend and the CLI both instantiate and call this class,
ensuring that scientific computation is never duplicated in UI code.

All methods accept and return well-defined types (SpectralCube, PeakResult,
GaussianFit, LineMatch, etc.) so that behaviour is testable independently
of the interface layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any

import numpy as np

from src.core.loader import load_fits_cube
from src.core.cube import SpectralCube
from src.preprocessing.denoise import savgol_denoise
from src.preprocessing.baseline import subtract_continuum
from src.analysis.spectrum import extract_spectrum, extract_region_average
from src.analysis.peaks import detect_peaks, PeakResult
from src.analysis.fitting import fit_gaussian_peaks, GaussianFit
from src.analysis.line_id import identify_lines, LineMatch
from src.imaging.slice import wavelength_slice, wavelength_slice_at
from src.imaging.integrate import integrated_image
from src.imaging.emission import emission_line_map
from src.visualization.spectra_plot import plot_spectrum
from src.visualization.image_plot import plot_image, plot_image_grid
from src.utils.config import PipelineConfig
from src.utils.constants import get_line_database


# ===========================================================================
# Result containers
# ===========================================================================

@dataclass
class AnalysisResult:
    """Complete single-pixel analysis result.

    Bundles all outputs from extracting and analysing a single spaxel.
    """

    x: int
    y: int
    wavelength: np.ndarray
    raw_spectrum: np.ndarray
    denoised_spectrum: np.ndarray
    continuum: np.ndarray
    processed_spectrum: np.ndarray
    peaks: PeakResult
    gaussian_fits: list[GaussianFit]
    line_matches: list[LineMatch | None]
    raw_err: np.ndarray | None = None
    processed_err: np.ndarray | None = None


@dataclass
class PipelineResult:
    """Complete pipeline run result."""

    cube: SpectralCube
    analysis: AnalysisResult
    images: dict[str, np.ndarray]
    config: PipelineConfig


# ===========================================================================
# Pipeline
# ===========================================================================

class SpectralPipeline:
    """Orchestrates the JWST MIRI IFU spectral analysis pipeline.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration (all tunable parameters).

    Examples
    --------
    >>> pipeline = SpectralPipeline(PipelineConfig())
    >>> cube = pipeline.load("data/raw/my_cube.fits")
    >>> result = pipeline.analyze(cube, x=15, y=20)
    >>> pipeline.save_results(result, "output/")
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()

    # ------------------------------------------------------------------ #
    #  Stage 1: Load
    # ------------------------------------------------------------------ #

    def load(self, filepath: str | Path) -> SpectralCube:
        """Load a FITS spectral cube.

        Parameters
        ----------
        filepath : str or Path
            Path to the FITS file.

        Returns
        -------
        SpectralCube
        """
        return load_fits_cube(filepath)

    def load_multiple(self, filepaths: list[str | Path]) -> SpectralCube:
        """Load and stitch multiple FITS spectral cubes.

        Parameters
        ----------
        filepaths : list of str or Path
            Paths to the FITS files.

        Returns
        -------
        SpectralCube
            A single, combined (stitched) data cube.
        """
        if not filepaths:
            raise ValueError("No files provided for loading.")

        cubes = [self.load(f) for f in filepaths]

        if len(cubes) == 1:
            return cubes[0]

        from src.core.stitch import stitch_cubes
        return stitch_cubes(cubes, reference_index=0)

    # ------------------------------------------------------------------ #
    #  Stage 2: Preprocess
    # ------------------------------------------------------------------ #

    def preprocess(
        self,
        wavelength: np.ndarray,
        spectrum: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Denoise and subtract continuum from a 1D spectrum.

        Returns
        -------
        dict
            Keys: 'denoised', 'continuum', 'processed'
        """
        denoised = savgol_denoise(
            spectrum,
            window_length=self.config.savgol_window,
            polyorder=self.config.savgol_polyorder,
        )

        continuum_sub, continuum = subtract_continuum(
            wavelength, denoised,
            poly_order=self.config.continuum_poly_order,
            sigma_clip=self.config.sigma_clip,
        )

        return {
            "denoised": denoised,
            "continuum": continuum,
            "processed": continuum_sub,
        }

    # ------------------------------------------------------------------ #
    #  Stage 3: Detect and Identify
    # ------------------------------------------------------------------ #

    def detect_and_identify(
        self,
        wavelength: np.ndarray,
        spectrum: np.ndarray,
        err: np.ndarray | None = None,
    ) -> tuple[PeakResult, list[GaussianFit], list[LineMatch | None]]:
        """Run peak detection, Gaussian fitting, and line identification.

        Parameters
        ----------
        wavelength : np.ndarray
            Wavelength axis (µm).
        spectrum : np.ndarray
            Processed (continuum-subtracted) 1D spectrum.
        err : np.ndarray, optional
            1D standard error array.

        Returns
        -------
        tuple of (PeakResult, list[GaussianFit], list[LineMatch | None])
        """
        # Detect peaks
        peaks = detect_peaks(
            spectrum, wavelength,
            sigma_threshold=self.config.peak_prominence,
            min_distance=self.config.peak_distance,
            height=self.config.peak_height,
        )

        # Gaussian fitting with error propagation
        if peaks.n_peaks > 0:
            fits = fit_gaussian_peaks(
                wavelength, spectrum, peaks.indices,
                err=err,
            )
        else:
            fits = []

        # Line identification
        if peaks.n_peaks > 0:
            matches = identify_lines(
                peaks.wavelengths,
                tolerance_um=self.config.tolerance_um,
                redshift=self.config.redshift,
            )
        else:
            matches = []

        return peaks, fits, matches

    # ------------------------------------------------------------------ #
    #  Stage 4: Full single-pixel analysis
    # ------------------------------------------------------------------ #

    def analyze(
        self,
        cube: SpectralCube,
        x: int,
        y: int,
    ) -> AnalysisResult:
        """Run full analysis on a single spaxel.

        Parameters
        ----------
        cube : SpectralCube
        x, y : int
            Pixel coordinates.

        Returns
        -------
        AnalysisResult
        """
        # Extract & preprocess
        extraction = extract_spectrum(cube, x, y, config=self.config)

        wavelength = extraction["wavelength"]
        raw = extraction["raw"]
        raw_err = extraction.get("raw_err")
        denoised = extraction["denoised"]
        continuum = extraction["continuum"]
        processed = extraction["processed"]
        processed_err = extraction.get("processed_err")

        # Detect, fit, identify
        peaks, fits, matches = self.detect_and_identify(wavelength, processed, err=processed_err)

        return AnalysisResult(
            x=x, y=y,
            wavelength=wavelength,
            raw_spectrum=raw,
            denoised_spectrum=denoised,
            continuum=continuum,
            processed_spectrum=processed,
            peaks=peaks,
            gaussian_fits=fits,
            line_matches=matches,
            raw_err=raw_err,
            processed_err=processed_err,
        )

    # ------------------------------------------------------------------ #
    #  Stage 5: Generate images
    # ------------------------------------------------------------------ #

    def generate_images(
        self,
        cube: SpectralCube,
        line_matches: Optional[list[LineMatch | None]] = None,
    ) -> dict[str, np.ndarray]:
        """Generate standard image products from the cube.

        Generates:
        - A wavelength slice at the cube midpoint
        - A full-band integrated image
        - Emission line maps for each identified line (if provided)

        Parameters
        ----------
        cube : SpectralCube
        line_matches : list, optional
            Identified lines for emission map generation.

        Returns
        -------
        dict of str → np.ndarray
        """
        images: dict[str, np.ndarray] = {}

        # Mid-band slice
        mid_idx = cube.n_wavelengths // 2
        mid_wl = cube.get_wavelength_at(mid_idx)
        images[f"slice_{mid_wl:.2f}um"] = wavelength_slice(cube, mid_idx)

        # Full-band integration
        wl_min, wl_max = cube.wavelength_range
        images["full_band_integrated"] = integrated_image(cube, wl_min, wl_max, method="mean")

        # Emission maps for identified lines
        if line_matches:
            seen_species = set()
            for match in line_matches:
                if match is None:
                    continue
                key = f"{match.matched_species}_{match.rest_wavelength_um:.3f}"
                if key in seen_species:
                    continue
                seen_species.add(key)

                try:
                    em_map = emission_line_map(
                        cube,
                        line_center=match.peak_wavelength_um,
                        line_width=self.config.default_band_width * 0.3,
                        continuum_width=self.config.default_band_width,
                    )
                    images[f"emission_{match.matched_species}_{match.rest_wavelength_um:.2f}"] = em_map
                except (ValueError, IndexError):
                    pass

        return images

    # ------------------------------------------------------------------ #
    #  Stage 6: Full pipeline run
    # ------------------------------------------------------------------ #

    def run_full(
        self,
        filepath: str | Path,
        x: int,
        y: int,
    ) -> PipelineResult:
        """Execute the complete pipeline end-to-end.

        Parameters
        ----------
        filepath : str or Path
            FITS file to process.
        x, y : int
            Pixel coordinates for spectral analysis.

        Returns
        -------
        PipelineResult
        """
        cube = self.load(filepath)
        analysis = self.analyze(cube, x, y)
        images = self.generate_images(cube, analysis.line_matches)

        return PipelineResult(
            cube=cube,
            analysis=analysis,
            images=images,
            config=self.config,
        )

    # ------------------------------------------------------------------ #
    #  Output
    # ------------------------------------------------------------------ #

    def save_results(
        self,
        result: PipelineResult,
        output_dir: Optional[str] = None,
    ) -> Path:
        """Save all pipeline outputs to disk.

        Saves:
        - Spectrum plot (PNG)
        - Image plots (PNG)
        - Peak table (CSV)
        - Summary text file

        Parameters
        ----------
        result : PipelineResult
        output_dir : str, optional
            Override output directory. Defaults to config.output_dir.

        Returns
        -------
        Path
            Directory where outputs were saved.
        """
        out = Path(output_dir or self.config.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        analysis = result.analysis

        # Build peak labels
        peak_labels = []
        for match in analysis.line_matches:
            if match is not None:
                peak_labels.append(f"{match.matched_species}")
            else:
                peak_labels.append("")

        # Save spectrum plot
        plot_spectrum(
            wavelength=analysis.wavelength,
            raw=analysis.raw_spectrum,
            processed=analysis.processed_spectrum,
            continuum=analysis.continuum,
            peak_wavelengths=analysis.peaks.wavelengths if analysis.peaks.n_peaks > 0 else None,
            peak_labels=peak_labels if peak_labels else None,
            noise_rms=analysis.peaks.noise_rms,
            title=f"Spectrum at ({analysis.x}, {analysis.y})",
            save_path=str(out / "spectrum.png"),
        )

        # Save image plots
        if result.images:
            img_list = list(result.images.values())
            img_titles = list(result.images.keys())
            plot_image_grid(
                img_list, img_titles,
                save_path=str(out / "images.png"),
            )

            # Also save individual images
            import re
            for name, img in result.images.items():
                safe_name = re.sub(r'[^a-zA-Z0-9_\-.]', '', name.replace(" ", "_").replace("₂", "2"))
                plot_image(
                    img, title=name,
                    save_path=str(out / f"{safe_name}.png"),
                )

        # Save peak table as CSV
        if analysis.peaks.n_peaks > 0:
            _save_peak_csv(analysis, out / "peaks.csv")

        # Save summary
        _save_summary(result, out / "summary.txt")

        return out


def _save_peak_csv(analysis: AnalysisResult, filepath: Path) -> None:
    """Save detected peaks to a CSV file with full uncertainty propagation and kinematics."""
    import csv
    from src.analysis.advanced import compute_intrinsic_kinematics

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "peak_index", "wavelength_um", "flux", "prominence",
            "snr", "width_channels", "width_um",
            "gaussian_center", "gaussian_center_err",
            "gaussian_fwhm", "gaussian_fwhm_err",
            "gaussian_flux", "gaussian_flux_err",
            "reduced_chi2", "dof",
            "resolving_power", "fwhm_inst_um", "fwhm_intrinsic_um",
            "sigma_v_kms", "sigma_v_err_kms",
            "identified_species", "rest_wavelength", "confidence",
        ])

        for i in range(analysis.peaks.n_peaks):
            gfit = analysis.gaussian_fits[i] if i < len(analysis.gaussian_fits) else None
            match = analysis.line_matches[i] if i < len(analysis.line_matches) else None

            if gfit and gfit.fit_success:
                center_val = f"{gfit.center:.5f}"
                center_err = f"{gfit.center_err:.5f}"
                fwhm_val = f"{gfit.fwhm:.5f}"
                fwhm_err = f"{gfit.fwhm_err:.5f}"
                flux_val = f"{gfit.integrated_flux:.6e}"
                flux_err = f"{gfit.integrated_flux_err:.6e}"
                rchi2_val = f"{gfit.reduced_chi2:.3f}"
                dof_val = int(gfit.dof)

                kin = compute_intrinsic_kinematics(gfit.fwhm, gfit.fwhm_err, gfit.center)
                r_power = f"{kin['resolving_power']:.0f}"
                fwhm_inst = f"{kin['fwhm_inst_um']:.5f}"
                fwhm_intr = f"{kin['fwhm_intrinsic_um']:.5f}"
                sig_v = f"{kin['sigma_v_kms']:.2f}"
                sig_v_err = f"{kin['sigma_v_err_kms']:.2f}"
            else:
                center_val = center_err = fwhm_val = fwhm_err = flux_val = flux_err = rchi2_val = ""
                dof_val = ""
                r_power = fwhm_inst = fwhm_intr = sig_v = sig_v_err = ""

            writer.writerow([
                int(analysis.peaks.indices[i]),
                f"{analysis.peaks.wavelengths[i]:.5f}",
                f"{analysis.peaks.heights[i]:.6e}",
                f"{analysis.peaks.prominences[i]:.6e}",
                f"{analysis.peaks.snr[i]:.2f}",
                f"{analysis.peaks.widths[i]:.2f}",
                f"{analysis.peaks.widths_um[i]:.5f}",
                center_val, center_err,
                fwhm_val, fwhm_err,
                flux_val, flux_err,
                rchi2_val, dof_val,
                r_power, fwhm_inst, fwhm_intr,
                sig_v, sig_v_err,
                match.matched_species if match else "",
                f"{match.rest_wavelength_um:.5f}" if match else "",
                f"{match.confidence:.3f}" if match else "",
            ])


def _save_summary(result: PipelineResult, filepath: Path) -> None:
    """Save a human-readable analysis summary with formal uncertainties."""
    from src.analysis.advanced import compute_intrinsic_kinematics

    analysis = result.analysis
    cube = result.cube
    config = result.config

    lines: list[str] = []
    lines.append("=" * 75)
    lines.append("JWST MIRI IFU Pipeline — Rigorous Physical Analysis Summary")
    lines.append("=" * 75)
    lines.append("")
    lines.append(f"Source file:  {cube.filepath}")
    lines.append(f"Cube shape:   {cube.shape}")
    lines.append(f"Wavelength:   {cube.wavelength_range[0]:.4f} – {cube.wavelength_range[1]:.4f} µm")
    lines.append(f"Pixel:        ({analysis.x}, {analysis.y})")
    lines.append(f"Errors present: {cube.has_err}")
    lines.append(f"DQ present:    {cube.has_dq}")
    lines.append("")
    lines.append("--- Configuration ---")
    lines.append(f"SG window:    {config.savgol_window}")
    lines.append(f"SG polyorder: {config.savgol_polyorder}")
    lines.append(f"Continuum:    poly order {config.continuum_poly_order}, "
                 f"σ-clip={config.sigma_clip}")
    lines.append(f"Peak detect:  prominence={config.peak_prominence}, "
                 f"distance={config.peak_distance}")
    lines.append(f"Line ID:      tolerance={config.tolerance_um} µm, "
                 f"z={config.redshift}")
    lines.append("")
    lines.append("--- Results ---")
    lines.append(f"Noise RMS:       {analysis.peaks.noise_rms:.6e}")
    lines.append(f"Peaks detected:  {analysis.peaks.n_peaks}")

    n_identified = sum(1 for m in analysis.line_matches if m is not None)
    lines.append(f"Lines identified: {n_identified}")
    lines.append("")

    if analysis.peaks.n_peaks > 0:
        lines.append("--- Detected Peaks & Physical Line Parameters ---")
        lines.append(
            f"{'#':>2}  {'λ_obs (µm)':>10}  {'SNR':>6}  {'Species':>12}  "
            f"{'Rest λ':>9}  {'Flux (±err)':>22}  {'χ²_red':>7}  {'σ_v (km/s)':>14}"
        )
        lines.append("-" * 90)

        for i in range(analysis.peaks.n_peaks):
            match = analysis.line_matches[i] if i < len(analysis.line_matches) else None
            gfit = analysis.gaussian_fits[i] if i < len(analysis.gaussian_fits) else None

            if gfit and gfit.fit_success:
                flux_str = f"{gfit.integrated_flux:.3e}±{gfit.integrated_flux_err:.1e}"
                rchi2_str = f"{gfit.reduced_chi2:.2f}"
                kin = compute_intrinsic_kinematics(gfit.fwhm, gfit.fwhm_err, gfit.center)
                sig_v_str = f"{kin['sigma_v_kms']:.1f}±{kin['sigma_v_err_kms']:.1f}"
            else:
                flux_str = "—"
                rchi2_str = "—"
                sig_v_str = "—"

            lines.append(
                f"{i + 1:>2}  "
                f"{analysis.peaks.wavelengths[i]:>10.4f}  "
                f"{analysis.peaks.snr[i]:>6.1f}  "
                f"{(match.matched_species if match else '—'):>12}  "
                f"{(f'{match.rest_wavelength_um:.4f}' if match else '—'):>9}  "
                f"{flux_str:>22}  "
                f"{rchi2_str:>7}  "
                f"{sig_v_str:>14}"
            )

    lines.append("")
    lines.append(f"Images generated: {len(result.images)}")
    for name in result.images:
        lines.append(f"  • {name}")

    filepath.write_text("\n".join(lines), encoding="utf-8")
