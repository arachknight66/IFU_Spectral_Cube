"""
Publication-Quality Spectrum Plotting.

Generates richly annotated spectral plots showing:
    - Raw spectrum (gray) with processed overlay (color)
    - Detected peak markers with vertical dashed lines
    - Identified species labels at peak positions
    - Continuum model overlay
    - Noise-level shading (±1σ, ±3σ)

Matplotlib styling follows the Nature Astronomy house style:
    - Serif fonts (Computer Modern) for labels
    - Clean axis spines
    - High DPI for publication
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Use non-interactive backend when saving to file
matplotlib.rcParams.update({
    "font.size": 11,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "figure.dpi": 150,
})


def plot_spectrum(
    wavelength: np.ndarray,
    raw: np.ndarray,
    processed: Optional[np.ndarray] = None,
    continuum: Optional[np.ndarray] = None,
    peak_wavelengths: Optional[np.ndarray] = None,
    peak_labels: Optional[list[str]] = None,
    noise_rms: Optional[float] = None,
    title: str = "IFU Spectrum",
    xlabel: str = "Wavelength (µm)",
    ylabel: str = "Flux (MJy/sr)",
    save_path: Optional[str] = None,
    figsize: tuple[float, float] = (12, 5),
    return_figure: bool = False,
) -> Figure | None:
    """Create a publication-quality spectrum plot.

    Parameters
    ----------
    wavelength : np.ndarray
        Wavelength axis in µm.
    raw : np.ndarray
        Raw spectrum flux array.
    processed : np.ndarray, optional
        Processed (denoised + continuum-subtracted) spectrum.
    continuum : np.ndarray, optional
        Estimated continuum model.
    peak_wavelengths : np.ndarray, optional
        Wavelengths of detected peaks (µm).
    peak_labels : list of str, optional
        Species labels for each peak. Must match length of peak_wavelengths.
    noise_rms : float, optional
        Noise RMS for ±σ shading on the processed spectrum.
    title : str
        Plot title.
    xlabel, ylabel : str
        Axis labels.
    save_path : str, optional
        If provided, save figure to this path (PNG, PDF, etc.).
    figsize : tuple
        Figure size in inches.
    return_figure : bool
        If True, return the Figure object instead of showing it.

    Returns
    -------
    Figure or None
        The Matplotlib figure if return_figure is True.
    """
    fig, axes = plt.subplots(
        2 if processed is not None else 1, 1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]} if processed is not None else None,
    )

    if processed is None:
        ax_main = axes
        ax_proc = None
    else:
        ax_main = axes[0]
        ax_proc = axes[1]

    # ----- Main panel: raw spectrum + continuum -----
    ax_main.plot(
        wavelength, raw,
        color="#8E99A4", linewidth=0.6, alpha=0.7,
        label="Raw", zorder=1,
    )

    if continuum is not None:
        ax_main.plot(
            wavelength, continuum,
            color="#E74C3C", linewidth=1.2, linestyle="--",
            label="Continuum", alpha=0.8, zorder=2,
        )

    ax_main.set_ylabel(ylabel)
    ax_main.set_title(title, fontweight="bold", fontsize=13)
    ax_main.legend(loc="upper right", fontsize=9, framealpha=0.8)

    # ----- Bottom panel: processed spectrum with peaks -----
    if ax_proc is not None and processed is not None:
        ax_proc.plot(
            wavelength, processed,
            color="#2980B9", linewidth=0.9,
            label="Processed", zorder=3,
        )

        # Noise shading
        if noise_rms is not None and noise_rms > 0:
            ax_proc.axhspan(-noise_rms, noise_rms,
                            color="#BDC3C7", alpha=0.2, label="±1σ")
            ax_proc.axhspan(-3 * noise_rms, 3 * noise_rms,
                            color="#BDC3C7", alpha=0.08, label="±3σ")

        ax_proc.axhline(0, color="gray", linewidth=0.5, linestyle=":")

        # Peak markers
        if peak_wavelengths is not None and len(peak_wavelengths) > 0:
            for i, pwl in enumerate(peak_wavelengths):
                # Find nearest index for flux value
                idx = int(np.argmin(np.abs(wavelength - pwl)))
                flux_at_peak = processed[idx]

                ax_proc.axvline(
                    pwl, color="#E67E22", linewidth=0.7,
                    linestyle="--", alpha=0.6, zorder=4,
                )
                ax_proc.plot(
                    pwl, flux_at_peak, "v",
                    color="#E74C3C", markersize=6, zorder=5,
                )

                # Species label
                if peak_labels is not None and i < len(peak_labels) and peak_labels[i]:
                    ax_proc.annotate(
                        peak_labels[i],
                        (pwl, flux_at_peak),
                        textcoords="offset points",
                        xytext=(0, 12),
                        ha="center", va="bottom",
                        fontsize=7.5, fontweight="bold",
                        color="#2C3E50",
                        rotation=45,
                        bbox=dict(
                            boxstyle="round,pad=0.2",
                            facecolor="white", edgecolor="#BDC3C7",
                            alpha=0.8,
                        ),
                    )

        ax_proc.set_ylabel("Continuum-subtracted flux")
        ax_proc.set_xlabel(xlabel)
        ax_proc.legend(loc="upper right", fontsize=8, framealpha=0.8)
    else:
        ax_main.set_xlabel(xlabel)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if return_figure:
        return fig
    else:
        plt.close(fig)
        return None


def plot_filter_comparison(
    wavelength: np.ndarray,
    raw: np.ndarray,
    savgol: np.ndarray,
    gaussian: np.ndarray,
    title: str = "Filter Comparison: Savitzky-Golay vs Gaussian",
    save_path: Optional[str] = None,
    return_figure: bool = False,
) -> Figure | None:
    """Compare SG and Gaussian filter results side by side.

    Parameters
    ----------
    wavelength : np.ndarray
        Wavelength axis.
    raw, savgol, gaussian : np.ndarray
        Raw and filtered spectra.
    title : str
        Plot title.
    save_path : str, optional
        Save path for the figure.
    return_figure : bool
        If True, return Figure.

    Returns
    -------
    Figure or None
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # Top: spectra overlay
    ax1.plot(wavelength, raw, color="#BDC3C7", linewidth=0.5, alpha=0.7, label="Raw")
    ax1.plot(wavelength, savgol, color="#2980B9", linewidth=1.0, label="Savitzky-Golay")
    ax1.plot(wavelength, gaussian, color="#E74C3C", linewidth=1.0, label="Gaussian", linestyle="--")
    ax1.set_ylabel("Flux")
    ax1.set_title(title, fontweight="bold")
    ax1.legend(fontsize=9)

    # Bottom: residuals
    ax2.plot(wavelength, raw - savgol, color="#2980B9", linewidth=0.6, alpha=0.8, label="SG residual")
    ax2.plot(wavelength, raw - gaussian, color="#E74C3C", linewidth=0.6, alpha=0.8, label="Gaussian residual", linestyle="--")
    ax2.axhline(0, color="gray", linewidth=0.5)
    ax2.set_ylabel("Residual")
    ax2.set_xlabel("Wavelength (µm)")
    ax2.legend(fontsize=9)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if return_figure:
        return fig
    else:
        plt.close(fig)
        return None
