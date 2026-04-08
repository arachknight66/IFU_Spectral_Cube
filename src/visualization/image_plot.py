"""
Publication-Quality 2D Image Plotting for IFU Data Products.

Provides functions for displaying:
    - Single 2D images (wavelength slices, integrated maps)
    - Multi-panel image grids for comparison
    - Proper colorbars with scientific formatting
    - Coordinate axes (pixel or WCS)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import Normalize, LogNorm, SymLogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable


matplotlib.rcParams.update({
    "font.size": 11,
    "axes.linewidth": 0.8,
    "figure.dpi": 150,
})


def plot_image(
    image: np.ndarray,
    title: str = "",
    colormap: str = "inferno",
    colorbar_label: str = "Flux (MJy/sr)",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    log_scale: bool = False,
    symmetric: bool = False,
    save_path: Optional[str] = None,
    figsize: tuple[float, float] = (7, 6),
    return_figure: bool = False,
) -> Figure | None:
    """Display a single 2D image with colorbar.

    Parameters
    ----------
    image : np.ndarray
        2D image array.
    title : str
        Plot title.
    colormap : str
        Matplotlib colormap name.
    colorbar_label : str
        Label for the colorbar.
    vmin, vmax : float, optional
        Colorbar limits. If None, derived from data.
    log_scale : bool
        If True, use logarithmic color scale.
    symmetric : bool
        If True, use symmetric log scale (for emission maps that
        can go negative). Overrides log_scale.
    save_path : str, optional
        Path to save the figure.
    figsize : tuple
        Figure size in inches.
    return_figure : bool
        If True, return the Figure object.

    Returns
    -------
    Figure or None
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Handle NaN values for display
    display_img = np.ma.masked_invalid(image)

    # Choose normalization
    if symmetric:
        abs_max = np.nanmax(np.abs(display_img))
        norm = SymLogNorm(linthresh=abs_max * 0.01, vmin=-abs_max, vmax=abs_max)
        colormap = "RdBu_r"
    elif log_scale:
        pos_data = display_img[display_img > 0]
        if len(pos_data) > 0:
            norm = LogNorm(
                vmin=vmin or np.nanpercentile(pos_data, 1),
                vmax=vmax or np.nanpercentile(pos_data, 99),
            )
        else:
            norm = None
    else:
        norm = Normalize(
            vmin=vmin if vmin is not None else np.nanpercentile(display_img.compressed(), 1),
            vmax=vmax if vmax is not None else np.nanpercentile(display_img.compressed(), 99),
        ) if display_img.count() > 0 else None

    im = ax.imshow(
        display_img,
        origin="lower",
        cmap=colormap,
        norm=norm,
        aspect="equal",
        interpolation="nearest",
    )

    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    if title:
        ax.set_title(title, fontweight="bold", fontsize=13)

    # Colorbar with proper sizing
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.08)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(colorbar_label, fontsize=10)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if return_figure:
        return fig
    else:
        plt.close(fig)
        return None


def plot_image_grid(
    images: list[np.ndarray],
    titles: list[str],
    colormap: str = "inferno",
    colorbar_label: str = "Flux",
    ncols: int = 3,
    save_path: Optional[str] = None,
    figsize_per_panel: tuple[float, float] = (4.5, 4.0),
    return_figure: bool = False,
) -> Figure | None:
    """Display a grid of 2D images.

    Parameters
    ----------
    images : list of np.ndarray
        List of 2D image arrays.
    titles : list of str
        Titles for each panel.
    colormap : str
        Matplotlib colormap.
    colorbar_label : str
        Label for colorbars.
    ncols : int
        Number of columns in the grid.
    save_path : str, optional
        Save path.
    figsize_per_panel : tuple
        Size per panel.
    return_figure : bool
        Return figure if True.

    Returns
    -------
    Figure or None
    """
    n = len(images)
    nrows = max(1, (n + ncols - 1) // ncols)
    figsize = (figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for i in range(nrows * ncols):
        row, col = divmod(i, ncols)
        ax = axes[row, col]

        if i < n:
            display_img = np.ma.masked_invalid(images[i])
            if display_img.count() > 0:
                vmin = np.nanpercentile(display_img.compressed(), 2)
                vmax = np.nanpercentile(display_img.compressed(), 98)
            else:
                vmin, vmax = 0, 1

            im = ax.imshow(
                display_img,
                origin="lower",
                cmap=colormap,
                vmin=vmin, vmax=vmax,
                aspect="equal",
                interpolation="nearest",
            )

            if i < len(titles):
                ax.set_title(titles[i], fontsize=10, fontweight="bold")

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax)

            ax.set_xlabel("X", fontsize=8)
            ax.set_ylabel("Y", fontsize=8)
        else:
            ax.set_visible(False)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if return_figure:
        return fig
    else:
        plt.close(fig)
        return None
