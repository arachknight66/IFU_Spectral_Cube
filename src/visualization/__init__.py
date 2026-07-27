"""Visualization utilities and Phase 6 false-colour rendering engine."""

from .spectra_plot import plot_spectrum
from .image_plot import plot_image, plot_image_grid
from .false_color import FalseColorResult, FalseColorError, IncompatibleMapError, render_false_color

__all__ = [
    "plot_spectrum",
    "plot_image",
    "plot_image_grid",
    "FalseColorResult",
    "FalseColorError",
    "IncompatibleMapError",
    "render_false_color",
]
