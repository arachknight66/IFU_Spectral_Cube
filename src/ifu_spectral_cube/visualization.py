"""Publication-quality visualization for spectral analysis results.

All functions accept output paths and produce self-contained figures
suitable for journal submission.

Design principles
-----------------
* Dark background style for astronomical data (high contrast).
* Color-coded line labels by ionization class.
* Confidence annotations on peak markers.
* Individual fit panels with data + model + residuals.
* Multi-panel spatial emission maps for matched lines.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np

from .models import GaussianFit, LineMatch, Peak, SpectralCube, VoigtFit

# ---------------------------------------------------------------------------
# Style configuration
# ---------------------------------------------------------------------------

_IONIZATION_COLORS = {
    "low": "#4FC3F7",           # Light blue
    "intermediate": "#81C784",  # Green
    "high": "#FFB74D",          # Orange
    "very_high": "#EF5350",     # Red
    "molecular": "#CE93D8",     # Purple
    "neutral": "#FFEE58",       # Yellow (PAH)
    "recombination": "#80DEEA", # Cyan
    "absorption": "#BDBDBD",    # Grey
}

_RELIABILITY_MARKERS = {
    "secure": ("*", 80),       # Star, large
    "probable": ("D", 40),     # Diamond, medium
    "tentative": ("o", 25),    # Circle, small
    "marginal": ("v", 20),     # Triangle, tiny
}


def _apply_science_style() -> None:
    """Apply a publication-quality matplotlib style."""
    plt.rcParams.update({
        "figure.facecolor": "#0D1117",
        "axes.facecolor": "#161B22",
        "axes.edgecolor": "#30363D",
        "axes.labelcolor": "#C9D1D9",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.color": "#8B949E",
        "ytick.color": "#8B949E",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "text.color": "#C9D1D9",
        "legend.facecolor": "#21262D",
        "legend.edgecolor": "#30363D",
        "legend.fontsize": 8,
        "grid.color": "#21262D",
        "grid.alpha": 0.5,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.facecolor": "#0D1117",
        "font.family": "sans-serif",
        "font.size": 10,
    })


def _get_ion_color(ionization_class: str) -> str:
    """Map ionization class to display color."""
    return _IONIZATION_COLORS.get(ionization_class, "#C9D1D9")


def _prepare_output_path(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


# ---------------------------------------------------------------------------
# 1. Smoothing comparison
# ---------------------------------------------------------------------------

def plot_smoothing_comparison(
    wavelength_micron: np.ndarray,
    original: np.ndarray,
    continuum: np.ndarray,
    continuum_subtracted: np.ndarray,
    savgol: np.ndarray,
    gaussian: np.ndarray,
    output_path: str | Path,
) -> None:
    """Two-panel plot: raw+continuum (top), smoothed spectra (bottom)."""
    _apply_science_style()
    out = _prepare_output_path(output_path)
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    # Top panel: extracted spectrum + continuum
    axes[0].plot(wavelength_micron, original, lw=0.8, color="#58A6FF", alpha=0.8, label="Extracted")
    axes[0].plot(wavelength_micron, continuum, lw=1.2, color="#F78166", label="Continuum Model")
    axes[0].set_ylabel("Flux")
    axes[0].set_title("Extracted Spectrum and Continuum Model", fontweight="bold")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    # Bottom panel: continuum-subtracted + smoothed
    axes[1].plot(wavelength_micron, continuum_subtracted, lw=0.6, alpha=0.5, color="#8B949E",
                 label="Continuum Subtracted")
    axes[1].plot(wavelength_micron, savgol, lw=1.2, color="#7EE787", label="Savitzky-Golay")
    axes[1].plot(wavelength_micron, gaussian, lw=1.2, color="#D2A8FF", label="Gaussian")
    axes[1].set_xlabel("Wavelength [µm]")
    axes[1].set_ylabel("Flux")
    axes[1].set_title("Noise-Reduced Spectra Comparison", fontweight="bold")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout(pad=1.5)
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Peaks and line matches
# ---------------------------------------------------------------------------

def plot_peaks_and_matches(
    wavelength_micron: np.ndarray,
    processed_spectrum: np.ndarray,
    peaks: list[Peak],
    matches: list[LineMatch],
    output_path: str | Path,
) -> None:
    """Spectrum with detected peaks and identified line labels, color-coded
    by ionization class with confidence annotations."""
    _apply_science_style()
    out = _prepare_output_path(output_path)
    fig, ax = plt.subplots(figsize=(14, 5.5))

    ax.plot(wavelength_micron, processed_spectrum, color="#58A6FF", lw=0.9, zorder=1)
    ax.set_xlabel("Wavelength [µm]")
    ax.set_ylabel("Flux")
    ax.set_title("Detected Peaks and Identified Lines", fontweight="bold")
    ax.grid(True, alpha=0.2)

    # Plot unmatched peaks
    matched_indices = {m.peak_index for m in matches}
    unmatched = [p for p in peaks if p.index not in matched_indices]
    if unmatched:
        ax.scatter(
            [p.wavelength_micron for p in unmatched],
            [p.flux for p in unmatched],
            color="#8B949E", s=18, marker="x", label="Unidentified peaks", zorder=3,
        )

    # Plot matched peaks with ionization-class colors
    top_matches = sorted(matches, key=lambda m: m.confidence, reverse=True)[:30]
    y_max = float(np.nanmax(processed_spectrum)) if processed_spectrum.size > 0 else 1.0

    for m in top_matches:
        color = _get_ion_color(m.ionization_class)
        marker, size = _RELIABILITY_MARKERS.get(m.reliability, ("o", 25))

        # Find the peak flux for this match
        peak_flux = None
        for p in peaks:
            if p.index == m.peak_index:
                peak_flux = p.flux
                break
        if peak_flux is None:
            peak_flux = y_max * 0.5

        ax.scatter(m.observed_wavelength_micron, peak_flux,
                   color=color, s=size, marker=marker, zorder=4, edgecolors="white", linewidths=0.3)
        ax.axvline(m.observed_wavelength_micron, color=color, lw=0.5, alpha=0.3, zorder=2)

        # Label with line name and confidence
        label_text = f"{m.line_name}\n({m.confidence:.2f})"
        ax.annotate(
            label_text,
            xy=(m.observed_wavelength_micron, peak_flux),
            xytext=(0, 12),
            textcoords="offset points",
            fontsize=6,
            color=color,
            ha="center",
            va="bottom",
            rotation=45,
        )

    # Build legend for ionization classes
    legend_handles = []
    ion_classes_present = {m.ionization_class for m in top_matches}
    for cls in sorted(ion_classes_present):
        color = _get_ion_color(cls)
        h = plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=color,
                        markersize=6, label=cls.replace("_", " ").title())
        legend_handles.append(h)
    if unmatched:
        h = plt.Line2D([0], [0], marker="x", color="#8B949E", linestyle="none",
                        markersize=6, label="Unidentified")
        legend_handles.append(h)
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", fontsize=7)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Gaussian fit panels
# ---------------------------------------------------------------------------

def plot_gaussian_fits(
    wavelength_micron: np.ndarray,
    flux: np.ndarray,
    peaks: list[Peak],
    fits: list[GaussianFit],
    matches: list[LineMatch],
    output_path: str | Path,
    window_half_width_micron: float = 0.08,
    max_panels: int = 12,
) -> None:
    """Grid of individual line-fit panels showing data, model, and residuals."""
    from .fitting import _gaussian_linear

    _apply_science_style()
    out = _prepare_output_path(output_path)

    if not fits:
        return

    # Select top fits by SNR
    peak_by_index = {p.index: p for p in peaks}
    match_by_index = {m.peak_index: m for m in matches}
    display_fits = sorted(fits, key=lambda f: peak_by_index.get(f.peak_index, Peak(0, 0, 0, 0, 0, 0)).snr, reverse=True)[:max_panels]

    n = len(display_fits)
    ncols = min(4, n)
    nrows = max(1, (n + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 4 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    wl = np.asarray(wavelength_micron, dtype=float)
    y = np.asarray(flux, dtype=float)

    for idx, gf in enumerate(display_fits):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        # Extract fit window
        win = np.abs(wl - gf.center_micron) <= window_half_width_micron
        xw = wl[win]
        yw = y[win]

        if xw.size < 3:
            ax.set_visible(False)
            continue

        model = _gaussian_linear(xw, gf.amplitude, gf.center_micron, gf.sigma_micron,
                                  gf.baseline_offset, gf.baseline_slope)
        resid = yw - model

        # Data + model
        ax.plot(xw, yw, "o", color="#58A6FF", ms=2.5, alpha=0.7)
        ax.plot(xw, model, "-", color="#F78166", lw=1.2)

        # Residuals (offset below)
        offset = float(np.nanmin(yw)) - 0.3 * (float(np.nanmax(yw)) - float(np.nanmin(yw)))
        ax.plot(xw, resid + offset, ".", color="#8B949E", ms=1.5, alpha=0.6)
        ax.axhline(offset, color="#30363D", lw=0.5, ls="--")

        # Title: line name if matched, else channel index
        m = match_by_index.get(gf.peak_index)
        title = m.line_name if m else f"ch {gf.peak_index}"
        color = _get_ion_color(m.ionization_class) if m else "#C9D1D9"
        ax.set_title(title, fontsize=8, color=color, fontweight="bold")

        # Annotation: χ², σ
        info = f"χ²={gf.reduced_chi2:.2f}\nσ={gf.sigma_micron * 1000:.1f}nm"
        ax.text(0.97, 0.97, info, transform=ax.transAxes, fontsize=6,
                va="top", ha="right", color="#8B949E")

        ax.set_xlabel("λ [µm]", fontsize=7)
        ax.tick_params(labelsize=7)

    # Hide unused panels
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle("Gaussian Line Fits", fontweight="bold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Redshift consensus histogram
# ---------------------------------------------------------------------------

def plot_redshift_histogram(
    peaks: list[Peak],
    line_db: list[dict[str, str | float]],
    z_estimate: float,
    output_path: str | Path,
    z_min: float = -0.02,
    z_max: float = 0.20,
    dz: float = 5e-4,
) -> None:
    """Visualize the SNR-weighted redshift consensus histogram."""
    _apply_science_style()
    out = _prepare_output_path(output_path)

    z_candidates: list[float] = []
    weights: list[float] = []
    for p in peaks:
        for line in line_db:
            lam0 = float(line["rest_wavelength_micron"])
            z = p.wavelength_micron / lam0 - 1.0
            if z_min <= z <= z_max:
                z_candidates.append(z)
                weights.append(max(0.0, p.snr))

    fig, ax = plt.subplots(figsize=(10, 4))
    if z_candidates:
        bins = np.arange(z_min, z_max + dz, dz)
        ax.hist(z_candidates, bins=bins, weights=weights, color="#58A6FF", alpha=0.7,
                edgecolor="#30363D", linewidth=0.5)
    ax.axvline(z_estimate, color="#F78166", lw=1.5, ls="--", label=f"z = {z_estimate:.5f}")
    ax.set_xlabel("Redshift (z)")
    ax.set_ylabel("SNR-weighted Count")
    ax.set_title("Redshift Consensus Histogram", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Noise profile
# ---------------------------------------------------------------------------

def plot_noise_profile(
    wavelength_micron: np.ndarray,
    flux: np.ndarray,
    output_path: str | Path,
    window_channels: int = 100,
) -> None:
    """Plot local noise estimate vs wavelength across the bandpass."""
    from .preprocessing import estimate_noise_sigma

    _apply_science_style()
    out = _prepare_output_path(output_path)

    wl = np.asarray(wavelength_micron, dtype=float)
    y = np.asarray(flux, dtype=float)
    n = y.size
    half = window_channels // 2

    centers: list[float] = []
    sigmas: list[float] = []
    for i in range(half, n - half, max(1, half // 2)):
        seg = y[i - half: i + half]
        s = estimate_noise_sigma(seg)
        if np.isfinite(s):
            centers.append(float(wl[i]))
            sigmas.append(s)

    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.fill_between(centers, 0, sigmas, color="#58A6FF", alpha=0.3)
    ax.plot(centers, sigmas, color="#58A6FF", lw=1.0)
    ax.set_xlabel("Wavelength [µm]")
    ax.set_ylabel("Local Noise σ")
    ax.set_title("Noise Profile Across Bandpass", fontweight="bold")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Spatial emission maps
# ---------------------------------------------------------------------------

def plot_line_emission_map(
    cube: SpectralCube,
    line_center_micron: float,
    half_width_micron: float,
    output_path: str | Path,
    line_name: str | None = None,
) -> None:
    """Continuum-subtracted spatial emission map around a spectral line."""
    _apply_science_style()
    out = _prepare_output_path(output_path)
    wl = cube.wavelength_micron
    data = cube.data

    line_window = np.abs(wl - line_center_micron) <= half_width_micron
    sideband = (np.abs(wl - line_center_micron) > half_width_micron) & (
        np.abs(wl - line_center_micron) <= 2.0 * half_width_micron
    )
    if not np.any(line_window):
        raise ValueError("No spectral channels in selected line window.")

    line_map = np.nansum(data[line_window], axis=0)
    if np.any(sideband):
        continuum = np.nanmedian(data[sideband], axis=0)
        line_map = line_map - continuum * max(1, np.count_nonzero(line_window))

    title = f"Emission Map: {line_name}" if line_name else f"Emission @ {line_center_micron:.3f} µm"

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(line_map, origin="lower", cmap="inferno")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("X [spaxel]")
    ax.set_ylabel("Y [spaxel]")
    fig.colorbar(im, ax=ax, label="Integrated Flux (a.u.)", shrink=0.85)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_line_diagnostic_grid(
    cube: SpectralCube,
    matches: list[LineMatch],
    output_path: str | Path,
    half_width_micron: float = 0.03,
    max_lines: int = 9,
) -> None:
    """Multi-panel spatial emission maps for top matched lines."""
    _apply_science_style()
    out = _prepare_output_path(output_path)

    if not matches:
        return

    top = sorted(matches, key=lambda m: m.confidence, reverse=True)[:max_lines]
    n = len(top)
    ncols = min(3, n)
    nrows = max(1, (n + ncols - 1) // ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes_flat = np.atleast_2d(axes).ravel()

    wl = cube.wavelength_micron
    data = cube.data

    for idx, m in enumerate(top):
        ax = axes_flat[idx]
        line_win = np.abs(wl - m.observed_wavelength_micron) <= half_width_micron
        sideband = (np.abs(wl - m.observed_wavelength_micron) > half_width_micron) & (
            np.abs(wl - m.observed_wavelength_micron) <= 2.0 * half_width_micron
        )

        if not np.any(line_win):
            ax.set_visible(False)
            continue

        line_map = np.nansum(data[line_win], axis=0)
        if np.any(sideband):
            cont = np.nanmedian(data[sideband], axis=0)
            line_map = line_map - cont * max(1, np.count_nonzero(line_win))

        color = _get_ion_color(m.ionization_class)
        im = ax.imshow(line_map, origin="lower", cmap="inferno")
        ax.set_title(f"{m.line_name} ({m.reliability})", fontsize=9,
                     color=color, fontweight="bold")
        ax.set_xlabel("X", fontsize=7)
        ax.set_ylabel("Y", fontsize=7)
        ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=ax, shrink=0.8)

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Spatial Emission Maps — Top Matched Lines", fontweight="bold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. Interactive HTML (Plotly)
# ---------------------------------------------------------------------------

def export_interactive_html(
    wavelength_micron: np.ndarray,
    processed_spectrum: np.ndarray,
    peaks: list[Peak],
    matches: list[LineMatch],
    output_path: str | Path,
) -> None:
    """Generate standalone interactive HTML spectrum viewer with Plotly.

    Uses Plotly's CDN for zero additional dependency weight.
    Falls back gracefully if Plotly is not installed.
    """
    out = _prepare_output_path(output_path)

    try:
        import plotly.graph_objects as go
    except ImportError:
        # Write a minimal HTML note instead
        with open(out, "w", encoding="utf-8") as f:
            f.write("<html><body><p>Install plotly for interactive HTML export: "
                    "<code>pip install plotly</code></p></body></html>")
        return

    fig = go.Figure()

    # Spectrum trace
    fig.add_trace(go.Scatter(
        x=wavelength_micron.tolist(),
        y=processed_spectrum.tolist(),
        mode="lines",
        name="Processed Spectrum",
        line=dict(color="#58A6FF", width=1),
    ))

    # Peak markers
    if peaks:
        fig.add_trace(go.Scatter(
            x=[p.wavelength_micron for p in peaks],
            y=[p.flux for p in peaks],
            mode="markers",
            name="Detected Peaks",
            marker=dict(color="#F78166", size=6, symbol="circle"),
            text=[f"SNR={p.snr:.1f}" for p in peaks],
            hoverinfo="text+x+y",
        ))

    # Matched line annotations
    for m in sorted(matches, key=lambda m: m.confidence, reverse=True)[:25]:
        color = _IONIZATION_COLORS.get(m.ionization_class, "#C9D1D9")
        fig.add_vline(x=m.observed_wavelength_micron, line=dict(color=color, width=0.5, dash="dot"))
        fig.add_annotation(
            x=m.observed_wavelength_micron,
            y=float(np.nanmax(processed_spectrum)) * 0.9,
            text=f"{m.line_name}<br>c={m.confidence:.2f}",
            showarrow=False,
            font=dict(size=8, color=color),
            textangle=-45,
        )

    fig.update_layout(
        template="plotly_dark",
        title="JWST MIRI IFU — Interactive Spectrum Viewer",
        xaxis_title="Wavelength [µm]",
        yaxis_title="Flux",
        hovermode="x unified",
        width=1200,
        height=500,
    )

    fig.write_html(str(out), include_plotlyjs="cdn")
