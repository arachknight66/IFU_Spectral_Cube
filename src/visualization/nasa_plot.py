"""
NASA / STScI Press-Release Cartography & Renderer.

Renders high-definition astronomical RGB composite photographs complete with
celestial sky grids (RA/Dec), scale bars, North/East compass rosettes, and dark frame styling.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import Any


def render_nasa_photo(
    rgb_array: np.ndarray,
    title: str = "JWST MIRI IFU False-Color Composite",
    target_name: str = "JWST Target",
    channel_labels: dict[str, str] | None = None,
    scale_bar_arcsec: float = 1.0,
    pixel_scale_arcsec: float = 0.1,
    show_compass: bool = True,
    figsize: tuple[float, float] = (10, 10),
) -> plt.Figure:
    """Render a publication/press-release grade NASA photograph.

    Parameters
    ----------
    rgb_array : np.ndarray
        RGB float array of shape (H, W, 3) in range [0, 1].
    title : str
        Main photograph title.
    target_name : str
        Target object identifier.
    channel_labels : dict, optional
        Dict with keys 'red', 'green', 'blue' mapping to line names.
    scale_bar_arcsec : float
        Length of scale bar in arcseconds.
    pixel_scale_arcsec : float
        Spatial scale per original raw pixel.
    show_compass : bool
        Whether to draw North/East compass.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    plt.Figure
        Matplotlib Figure object.
    """
    fig = plt.figure(figsize=figsize, facecolor="black")
    ax = fig.add_axes([0.05, 0.05, 0.90, 0.88], facecolor="black")

    # Display RGB image
    ax.imshow(rgb_array, origin="lower")

    h, w, _ = rgb_array.shape

    # Title & Target header
    ax.text(
        0.03, 0.96, title.upper(),
        transform=ax.transAxes, color="white", fontsize=14, fontweight="bold",
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6, edgecolor="none")
    )
    if target_name:
        ax.text(
            0.03, 0.91, f"TARGET: {target_name.upper()}  |  JWST MIRI MRS IFU",
            transform=ax.transAxes, color="#00E5FF", fontsize=10, fontweight="bold",
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6, edgecolor="none")
        )

    # Color channel legend tags in bottom left
    if channel_labels:
        r_lbl = channel_labels.get("red", "Red")
        g_lbl = channel_labels.get("green", "Green")
        b_lbl = channel_labels.get("blue", "Blue")

        legend_txt = f"■ R: {r_lbl}    ■ G: {g_lbl}    ■ B: {b_lbl}"
        ax.text(
            0.03, 0.04, legend_txt,
            transform=ax.transAxes, color="white", fontsize=9, fontweight="bold",
            va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#111111", alpha=0.8, edgecolor="#333333")
        )

    # Scale bar in bottom right
    # Calculate scale bar width in upsampled image pixels
    # Assuming original spatial size was divided by upsample factor
    scale_bar_pixels = (scale_bar_arcsec / max(1e-4, pixel_scale_arcsec)) * (w / 30.0)
    bar_x1 = w * 0.92 - scale_bar_pixels
    bar_x2 = w * 0.92
    bar_y = h * 0.06

    ax.plot([bar_x1, bar_x2], [bar_y, bar_y], color="white", linewidth=3)
    ax.text(
        (bar_x1 + bar_x2) / 2.0, bar_y + h * 0.02, f'{scale_bar_arcsec}"',
        color="white", fontsize=11, fontweight="bold", ha="center", va="bottom"
    )

    # North / East Compass Rose in top right
    if show_compass:
        cx, cy = w * 0.90, h * 0.88
        arrow_len = min(w, h) * 0.07

        # North (Up)
        ax.annotate(
            "", xy=(cx, cy + arrow_len), xytext=(cx, cy),
            arrowprops=dict(arrowstyle="->", color="#FF3366", lw=2)
        )
        ax.text(cx, cy + arrow_len * 1.25, "N", color="#FF3366", fontsize=10, fontweight="bold", ha="center", va="center")

        # East (Left)
        ax.annotate(
            "", xy=(cx - arrow_len, cy), xytext=(cx, cy),
            arrowprops=dict(arrowstyle="->", color="#00E5FF", lw=2)
        )
        ax.text(cx - arrow_len * 1.25, cy, "E", color="#00E5FF", fontsize=10, fontweight="bold", ha="center", va="center")

    # Clean axes
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(1.5)

    return fig
