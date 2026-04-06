"""Generate publication-quality images from the HI 21cm FITS cube.

Demonstrates the full pipeline's imaging capabilities on real data:
1. Moment-0 (integrated intensity) map
2. Moment-1 (velocity field) map  
3. Peak brightness temperature map
4. Channel maps (HI emission at different velocities)
5. Spectrum of the brightest region
6. RGB composite (velocity-encoded color)
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib import gridspec
from astropy.utils.data import download_file
from pathlib import Path

from ifu_spectral_cube.io import load_miri_ifu_cube
from ifu_spectral_cube.extraction import circular_mask, extract_region_spectrum
from ifu_spectral_cube.preprocessing import (
    subtract_continuum_1d,
    apply_savgol_smoothing,
    estimate_noise_sigma,
)
from ifu_spectral_cube.peaks import detect_peaks_1d

# ── Configuration ────────────────────────────────────────────────────
HI_URL = "http://data.astropy.org/tutorials/FITS-cubes/reduced_TAN_C14.fits"
OUTPUT_DIR = Path(__file__).parent / "hi_images"
OUTPUT_DIR.mkdir(exist_ok=True)

# Custom colormaps
ASTRO_CMAP = LinearSegmentedColormap.from_list("astro_hot", [
    "#050810", "#0d1b3e", "#1a2f6e", "#3366cc", "#4daaff",
    "#7ecfff", "#b8e6ff", "#ffffff", "#ffe066", "#ffaa33",
    "#ff6622", "#cc2200",
])
VELOCITY_CMAP = LinearSegmentedColormap.from_list("velocity", [
    "#2166ac", "#4393c3", "#92c5de", "#d1e5f0",
    "#f7f7f7",
    "#fddbc7", "#f4a582", "#d6604d", "#b2182b",
])

# ── Dark science style ───────────────────────────────────────────────
plt.style.use("dark_background")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.facecolor": "#0A0E1A",
    "figure.facecolor": "#050810",
    "axes.edgecolor": "#30363D",
    "axes.labelcolor": "#C9D1D9",
    "text.color": "#C9D1D9",
    "xtick.color": "#8B949E",
    "ytick.color": "#8B949E",
    "grid.color": "#21262D",
    "grid.alpha": 0.5,
})


def main():
    # ── Load data ────────────────────────────────────────────────────
    print("📥 Downloading HI cube...")
    cube_path = download_file(HI_URL, cache=True, show_progress=True)
    cube = load_miri_ifu_cube(cube_path)

    data = cube.data  # (450, 150, 150) — brightness temperature in K
    wl = cube.wavelength_micron  # ~211000 µm (21cm)

    # Convert wavelength back to velocity for physical labeling
    # v = c × (1 - f_obs/f_rest) and f = c/λ
    # so v = c × (1 - λ_rest/λ_obs)
    HI_REST_WL = 211061.14  # µm (21.1 cm)
    c_kms = 299792.458  # km/s
    velocity_kms = c_kms * (1.0 - HI_REST_WL / wl)

    # Replace NaN with 0 for imaging
    data_clean = np.where(np.isfinite(data), data, 0.0)

    print(f"   Cube shape: {data.shape}")
    print(f"   Velocity range: {velocity_kms[0]:.1f} to {velocity_kms[-1]:.1f} km/s")
    print(f"   Peak Tb: {np.nanmax(data):.1f} K")

    # ═══════════════════════════════════════════════════════════════════
    # 1. Moment-0: Integrated Intensity Map
    # ═══════════════════════════════════════════════════════════════════
    print("🖼️  Generating Moment-0 map...")
    dv = np.abs(velocity_kms[1] - velocity_kms[0])
    moment0 = np.nansum(data_clean, axis=0) * dv  # K·km/s

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(moment0, origin="lower", cmap=ASTRO_CMAP, aspect="equal")
    cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label("Integrated Intensity [K·km/s]", fontsize=12)
    ax.set_xlabel("X [pixels]", fontsize=12)
    ax.set_ylabel("Y [pixels]", fontsize=12)
    ax.set_title("HI 21cm — Moment 0 (Integrated Intensity)", fontsize=15, fontweight="bold",
                 color="#58A6FF", pad=12)
    ax.grid(True, alpha=0.15, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "moment0_integrated_intensity.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"   ✓ Saved moment0_integrated_intensity.png")

    # ═══════════════════════════════════════════════════════════════════
    # 2. Moment-1: Velocity Field Map
    # ═══════════════════════════════════════════════════════════════════
    print("🖼️  Generating Moment-1 velocity field...")
    vel_grid = velocity_kms[:, np.newaxis, np.newaxis]
    mask = moment0 > 0.05 * np.nanmax(moment0)
    moment1 = np.nansum(data_clean * vel_grid, axis=0)
    moment1[mask] /= np.nansum(data_clean[:, mask], axis=0)
    moment1[~mask] = np.nan

    fig, ax = plt.subplots(figsize=(10, 9))
    vmin, vmax = np.nanpercentile(moment1[mask], [5, 95])
    im = ax.imshow(moment1, origin="lower", cmap=VELOCITY_CMAP, aspect="equal",
                   vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label("Velocity [km/s]", fontsize=12)
    ax.set_xlabel("X [pixels]", fontsize=12)
    ax.set_ylabel("Y [pixels]", fontsize=12)
    ax.set_title("HI 21cm — Moment 1 (Velocity Field)", fontsize=15, fontweight="bold",
                 color="#58A6FF", pad=12)
    ax.grid(True, alpha=0.15, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "moment1_velocity_field.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"   ✓ Saved moment1_velocity_field.png")

    # ═══════════════════════════════════════════════════════════════════
    # 3. Peak Brightness Temperature Map
    # ═══════════════════════════════════════════════════════════════════
    print("🖼️  Generating peak brightness map...")
    peak_tb = np.nanmax(data_clean, axis=0)

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(peak_tb, origin="lower", cmap="inferno", aspect="equal")
    cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label("Peak Brightness Temperature [K]", fontsize=12)
    ax.set_xlabel("X [pixels]", fontsize=12)
    ax.set_ylabel("Y [pixels]", fontsize=12)
    ax.set_title("HI 21cm — Peak Brightness Temperature", fontsize=15, fontweight="bold",
                 color="#58A6FF", pad=12)
    ax.grid(True, alpha=0.15, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "peak_brightness_temperature.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"   ✓ Saved peak_brightness_temperature.png")

    # ═══════════════════════════════════════════════════════════════════
    # 4. Channel Maps (6-panel grid)
    # ═══════════════════════════════════════════════════════════════════
    print("🖼️  Generating channel maps...")
    n_channels = 6
    # Pick channels with significant emission
    channel_flux = np.nansum(data_clean.reshape(data_clean.shape[0], -1), axis=1)
    emission_channels = np.where(channel_flux > 0.1 * np.nanmax(channel_flux))[0]
    
    if len(emission_channels) >= n_channels:
        sel = np.linspace(emission_channels[0], emission_channels[-1], n_channels, dtype=int)
    else:
        sel = np.linspace(0, data_clean.shape[0] - 1, n_channels, dtype=int)

    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    vmax = np.nanpercentile(data_clean[sel], 99)

    for ax, ch_idx in zip(axes.flat, sel):
        im = ax.imshow(data_clean[ch_idx], origin="lower", cmap=ASTRO_CMAP,
                       vmin=0, vmax=vmax, aspect="equal")
        v = velocity_kms[ch_idx]
        ax.set_title(f"v = {v:.1f} km/s", fontsize=11, color="#79C0FF")
        ax.set_xlabel("X", fontsize=9)
        ax.set_ylabel("Y", fontsize=9)
        ax.tick_params(labelsize=8)

    fig.suptitle("HI 21cm — Channel Maps", fontsize=17, fontweight="bold",
                 color="#58A6FF", y=0.98)
    cbar = fig.colorbar(im, ax=axes, pad=0.02, shrink=0.6, location="right")
    cbar.set_label("Brightness Temperature [K]", fontsize=11)
    fig.tight_layout(rect=[0, 0, 0.92, 0.95])
    fig.savefig(OUTPUT_DIR / "channel_maps.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"   ✓ Saved channel_maps.png")

    # ═══════════════════════════════════════════════════════════════════
    # 5. RGB Composite (velocity-encoded color)
    # ═══════════════════════════════════════════════════════════════════
    print("🖼️  Generating RGB velocity composite...")
    # Split cube into 3 velocity bands: blue (approaching), green (systemic), red (receding)
    n_spec = data_clean.shape[0]
    third = n_spec // 3
    blue_band = np.nansum(data_clean[:third], axis=0)
    green_band = np.nansum(data_clean[third:2*third], axis=0)
    red_band = np.nansum(data_clean[2*third:], axis=0)

    # Normalize each band
    def norm_band(band):
        vmin, vmax = np.nanpercentile(band, [2, 99])
        b = (band - vmin) / (vmax - vmin + 1e-10)
        return np.clip(b, 0, 1)

    rgb = np.stack([norm_band(red_band), norm_band(green_band), norm_band(blue_band)], axis=-1)
    # Apply sqrt stretch for better contrast
    rgb = np.sqrt(rgb)

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.imshow(rgb, origin="lower", aspect="equal")
    ax.set_xlabel("X [pixels]", fontsize=12)
    ax.set_ylabel("Y [pixels]", fontsize=12)
    ax.set_title("HI 21cm — RGB Velocity Composite", fontsize=15, fontweight="bold",
                 color="#58A6FF", pad=12)

    # Add velocity band labels
    ax.text(0.02, 0.97, f"● Blue: {velocity_kms[0]:.0f} to {velocity_kms[third]:.0f} km/s",
            transform=ax.transAxes, fontsize=10, color="#6699ff", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="#0A0E1A", ec="#30363D", alpha=0.8))
    ax.text(0.02, 0.91, f"● Green: {velocity_kms[third]:.0f} to {velocity_kms[2*third]:.0f} km/s",
            transform=ax.transAxes, fontsize=10, color="#66cc88", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="#0A0E1A", ec="#30363D", alpha=0.8))
    ax.text(0.02, 0.85, f"● Red: {velocity_kms[2*third]:.0f} to {velocity_kms[-1]:.0f} km/s",
            transform=ax.transAxes, fontsize=10, color="#ff6644", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="#0A0E1A", ec="#30363D", alpha=0.8))

    ax.grid(True, alpha=0.1, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "rgb_velocity_composite.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"   ✓ Saved rgb_velocity_composite.png")

    # ═══════════════════════════════════════════════════════════════════
    # 6. Brightest Region Spectrum (using pipeline extraction)
    # ═══════════════════════════════════════════════════════════════════
    print("🖼️  Generating spectrum of brightest region...")
    bright_y, bright_x = np.unravel_index(np.nanargmax(moment0), moment0.shape)
    mask_bright = circular_mask((150, 150), (float(bright_x), float(bright_y)), radius=5.0)
    spectrum, n_spx = extract_region_spectrum(data, mask_bright, statistic="mean")
    clean_spec = np.where(np.isfinite(spectrum), spectrum, 0.0)
    residual, continuum = subtract_continuum_1d(clean_spec, window_length=51)
    smoothed = apply_savgol_smoothing(residual, window_length=11, polyorder=3)
    sigma = estimate_noise_sigma(residual)

    # Detect peaks
    peaks, properties, _ = detect_peaks_1d(
        wl, smoothed, prominence_sigma=3.0, width=(2, 50), distance=5,
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), height_ratios=[1, 1],
                                     gridspec_kw={"hspace": 0.25})

    # Top: Raw spectrum with continuum
    ax1.plot(velocity_kms, clean_spec, color="#58A6FF", lw=0.8, alpha=0.7, label="Extracted")
    ax1.plot(velocity_kms, continuum, color="#F78166", lw=1.5, label="Continuum")
    ax1.fill_between(velocity_kms, clean_spec, alpha=0.15, color="#58A6FF")
    ax1.set_ylabel("Brightness Temperature [K]", fontsize=12)
    ax1.set_title(f"HI Spectrum at Brightest Region (x={bright_x}, y={bright_y}, r=5 spaxels)",
                  fontsize=14, fontweight="bold", color="#58A6FF", pad=10)
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(True, alpha=0.2)

    # Bottom: Continuum-subtracted with peaks
    ax2.plot(velocity_kms, smoothed, color="#7EE787", lw=1.0, label="Continuum-subtracted (SG)")
    ax2.axhline(0, color="#30363D", lw=0.5)
    ax2.axhline(3 * sigma, color="#FF7B72", lw=0.8, ls="--", alpha=0.5, label=f"3σ = {3*sigma:.3f} K")
    ax2.axhline(-3 * sigma, color="#FF7B72", lw=0.8, ls="--", alpha=0.5)

    if len(peaks) > 0:
        peak_indices = np.array([p.index for p in peaks])
        peak_vel = velocity_kms[peak_indices]
        peak_flux = smoothed[peak_indices]
        ax2.scatter(peak_vel, peak_flux, color="#FFA657", s=60, marker="v", zorder=5,
                    edgecolors="white", linewidths=0.5, label=f"{len(peaks)} peaks detected")
        for v, f in zip(peak_vel, peak_flux):
            ax2.annotate(f"{v:.1f} km/s", (v, f), textcoords="offset points",
                        xytext=(0, 12), fontsize=8, color="#FFA657", ha="center")

    ax2.set_xlabel("Velocity [km/s]", fontsize=12)
    ax2.set_ylabel("Tb − Continuum [K]", fontsize=12)
    ax2.legend(loc="upper right", fontsize=10)
    ax2.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "brightest_region_spectrum.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"   ✓ Saved brightest_region_spectrum.png")

    # ═══════════════════════════════════════════════════════════════════
    # 7. Combined dashboard figure
    # ═══════════════════════════════════════════════════════════════════
    print("🖼️  Generating combined dashboard...")
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.28, wspace=0.3)

    # Moment 0
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(moment0, origin="lower", cmap=ASTRO_CMAP)
    fig.colorbar(im1, ax=ax1, shrink=0.8, pad=0.02)
    ax1.set_title("Moment 0 — Integrated Intensity", color="#58A6FF", fontsize=11, fontweight="bold")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")

    # Velocity field
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(moment1, origin="lower", cmap=VELOCITY_CMAP, vmin=vmin, vmax=vmax)
    fig.colorbar(im2, ax=ax2, shrink=0.8, pad=0.02)
    ax2.set_title("Moment 1 — Velocity Field", color="#58A6FF", fontsize=11, fontweight="bold")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")

    # RGB composite
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(rgb, origin="lower")
    ax3.set_title("RGB Velocity Composite", color="#58A6FF", fontsize=11, fontweight="bold")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")

    # Peak Tb
    ax4 = fig.add_subplot(gs[1, 0])
    im4 = ax4.imshow(peak_tb, origin="lower", cmap="inferno")
    fig.colorbar(im4, ax=ax4, shrink=0.8, pad=0.02)
    ax4.set_title("Peak Brightness Temp", color="#58A6FF", fontsize=11, fontweight="bold")
    ax4.set_xlabel("X")
    ax4.set_ylabel("Y")

    # Spectrum
    ax5 = fig.add_subplot(gs[1, 1:])
    ax5.plot(velocity_kms, clean_spec, color="#58A6FF", lw=0.8, alpha=0.6, label="Raw")
    ax5.plot(velocity_kms, smoothed + continuum, color="#7EE787", lw=1.2, label="Smoothed")
    if len(peaks) > 0:
        pidx = np.array([p.index for p in peaks])
        ax5.scatter(velocity_kms[pidx], (smoothed + continuum)[pidx],
                    color="#FFA657", s=50, marker="v", edgecolors="white",
                    linewidths=0.5, zorder=5, label=f"{len(peaks)} peaks")
    ax5.fill_between(velocity_kms, clean_spec, alpha=0.1, color="#58A6FF")
    ax5.set_xlabel("Velocity [km/s]", fontsize=12)
    ax5.set_ylabel("Brightness Temperature [K]", fontsize=12)
    ax5.set_title("Spectrum — Brightest Region", color="#58A6FF", fontsize=11, fontweight="bold")
    ax5.legend(fontsize=9, loc="upper right")
    ax5.grid(True, alpha=0.2)

    fig.suptitle("JWST/IFU Pipeline — HI 21cm Data Analysis Dashboard",
                 fontsize=18, fontweight="bold", color="#58A6FF", y=0.99)
    fig.savefig(OUTPUT_DIR / "hi_analysis_dashboard.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"   ✓ Saved hi_analysis_dashboard.png")

    print(f"\n✅ All images saved to: {OUTPUT_DIR}/")
    print(f"   Generated 7 publication-quality figures.")


if __name__ == "__main__":
    main()
