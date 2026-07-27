"""Spectral imaging utilities, Phase 1 image-recipe models, and Phase 5 component map generation."""

from .slice import wavelength_slice, wavelength_slice_at
from .integrate import integrated_image
from .emission import emission_line_map
from .recipes import ImageRecipe
from .component_map import ComponentMap, ComponentMapError, WavelengthCoverageError, extract_component_map, extract_recipe_component_maps

__all__ = [
    "ImageRecipe",
    "ComponentMap",
    "ComponentMapError",
    "WavelengthCoverageError",
    "extract_component_map",
    "extract_recipe_component_maps",
    "emission_line_map",
    "integrated_image",
    "wavelength_slice",
    "wavelength_slice_at",
]
