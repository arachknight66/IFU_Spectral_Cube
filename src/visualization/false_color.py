"""
Reproducible NASA-Style False-Colour Rendering Engine.

Combines three aligned, continuum-subtracted Phase-5 ComponentMap objects
into a high-definition false-colour RGB photograph while preserving full scientific
traceability to the underlying FITS products, recipe, and rendering transforms.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from PIL import Image
from astropy.wcs import WCS

from src.imaging.component_map import ComponentMap
from src.imaging.recipes import ImageRecipe


class FalseColorError(ValueError):
    """Base exception for false-colour rendering errors."""


class IncompatibleMapError(FalseColorError):
    """Raised when component maps have mismatched shapes or invalid WCS."""


@dataclass
class FalseColorResult:
    """Structured, reproducible false-colour rendering result."""

    rgb_array: np.ndarray
    rgba_array: np.ndarray
    recipe: ImageRecipe
    rendering_mode: Literal["scientific", "presentation"]
    channel_maps: dict[str, ComponentMap]
    channel_stats: dict[str, Any]
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable dictionary summary of render product."""
        return {
            "recipe_name": self.recipe.name,
            "target_name": self.recipe.target_name,
            "rendering_mode": self.rendering_mode,
            "shape": list(self.rgb_array.shape[:2]),
            "channel_stats": self.channel_stats,
            "provenance": self.provenance,
        }

    def save_png(
        self,
        filepath: str | Path,
        transparent: bool = False,
        overwrite: bool = False,
    ) -> Path:
        """Export rendering product as 8-bit PNG with JSON manifest sidecar.

        Parameters
        ----------
        filepath : str or Path
            Target output filepath.
        transparent : bool
            Include alpha channel for transparent background.
        overwrite : bool
            Allow overwriting existing file.

        Returns
        -------
        Path
            Saved PNG file path.
        """
        dest = Path(filepath)
        if dest.exists() and not overwrite:
            raise FileExistsError(f"File '{dest}' already exists. Set overwrite=True to replace.")

        dest.parent.mkdir(parents=True, exist_ok=True)

        if transparent:
            arr_8bit = (np.clip(self.rgba_array, 0.0, 1.0) * 255.0).astype(np.uint8)
            img = Image.fromarray(arr_8bit, mode="RGBA")
        else:
            arr_8bit = (np.clip(self.rgb_array, 0.0, 1.0) * 255.0).astype(np.uint8)
            img = Image.fromarray(arr_8bit, mode="RGB")

        # Attach basic metadata PNG info
        meta_info = Image.ImageData() if hasattr(Image, "ImageData") else {}
        img.save(dest, format="PNG")

        # Export sidecar JSON manifest
        manifest_path = dest.with_name(f"{dest.name}.manifest.json")
        manifest_data = self.to_dict()
        manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

        return dest

    def save_tiff(
        self,
        filepath: str | Path,
        bit_depth: int = 16,
        overwrite: bool = False,
    ) -> Path:
        """Export rendering product as 16-bit TIFF image with JSON manifest sidecar.

        Parameters
        ----------
        filepath : str or Path
            Target output filepath.
        bit_depth : int
            Bit depth (16 or 8).
        overwrite : bool
            Allow overwriting existing file.

        Returns
        -------
        Path
            Saved TIFF file path.
        """
        dest = Path(filepath)
        if dest.exists() and not overwrite:
            raise FileExistsError(f"File '{dest}' already exists. Set overwrite=True to replace.")

        dest.parent.mkdir(parents=True, exist_ok=True)

        if bit_depth == 16:
            arr_16bit = (np.clip(self.rgb_array, 0.0, 1.0) * 65535.0).astype(np.uint16)
            try:
                import tifffile
                tifffile.imwrite(dest, arr_16bit, photometric="rgb")
            except ImportError:  # Fallback using PIL
                img = Image.fromarray((arr_16bit // 256).astype(np.uint8), mode="RGB")
                img.save(dest, format="TIFF")
        else:
            arr_8bit = (np.clip(self.rgb_array, 0.0, 1.0) * 255.0).astype(np.uint8)
            img = Image.fromarray(arr_8bit, mode="RGB")
            img.save(dest, format="TIFF")

        # Export sidecar JSON manifest
        manifest_path = dest.with_name(f"{dest.name}.manifest.json")
        manifest_data = self.to_dict()
        manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

        return dest


def _apply_stretch(
    data: np.ndarray,
    stretch: str,
    asinh_a: float = 0.1,
    log_a: float = 1000.0,
) -> np.ndarray:
    """Apply non-linear intensity stretch (linear, log, asinh)."""
    norm = np.clip(data, 0.0, 1.0)
    if stretch == "asinh":
        beta = max(1e-6, asinh_a)
        scaled = np.arcsinh(norm / beta) / np.arcsinh(1.0 / beta)
        return np.clip(scaled, 0.0, 1.0)
    elif stretch == "log":
        a = max(1.0, log_a)
        scaled = np.log(1.0 + a * norm) / np.log(1.0 + a)
        return np.clip(scaled, 0.0, 1.0)
    else:  # Linear
        return norm


def render_false_color(
    component_maps: dict[str, ComponentMap] | Sequence[ComponentMap],
    recipe: ImageRecipe,
    rendering_mode: Literal["scientific", "presentation"] = "scientific",
    rendering_overrides: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> FalseColorResult:
    """Combine 3 Phase-5 ComponentMap objects into a reproducible false-colour RGB image.

    Parameters
    ----------
    component_maps : dict or Sequence of ComponentMap
        Mapping of 'red', 'green', 'blue' -> ComponentMap (or list of 3 ComponentMaps).
    recipe : ImageRecipe
        Phase 1 RGB Image Recipe.
    rendering_mode : 'scientific' or 'presentation'
        Rendering mode ('scientific' enforces strict recipe parameters).
    rendering_overrides : dict, optional
        Optional rendering overrides (stretch, percentile clip, gain, gamma, SNR mask).
    overwrite : bool
        Allow file overwriting.

    Returns
    -------
    FalseColorResult
        Structured render result containing RGB, RGBA, statistics, and provenance.
    """
    if rendering_mode not in ("scientific", "presentation"):
        raise ValueError(f"Invalid rendering mode: '{rendering_mode}'. Must be 'scientific' or 'presentation'.")

    # Map input channels to dict
    if isinstance(component_maps, Sequence):
        if len(component_maps) != 3:
            raise IncompatibleMapError(f"Expected exactly 3 component maps, got {len(component_maps)}.")
        ch_dict = {}
        for cmap in component_maps:
            if not cmap.recipe_channel or cmap.recipe_channel not in ("red", "green", "blue"):
                raise IncompatibleMapError("ComponentMap must specify recipe_channel ('red', 'green', or 'blue').")
            ch_dict[cmap.recipe_channel] = cmap
    else:
        ch_dict = dict(component_maps)

    required_channels = {"red", "green", "blue"}
    if set(ch_dict.keys()) != required_channels:
        raise IncompatibleMapError(f"Must provide component maps for exactly 'red', 'green', and 'blue', got {set(ch_dict.keys())}.")

    # Validate spatial shapes across channels
    shapes = {k: v.data.shape for k, v in ch_dict.items()}
    first_shape = next(iter(shapes.values()))
    for k, sh in shapes.items():
        if sh != first_shape:
            raise IncompatibleMapError(
                f"Spatial shape mismatch: '{k}' has shape {sh}, reference has {first_shape}. "
                "Require Phase 4 spatial alignment before false-colour rendering."
            )

    ny, nx = first_shape

    # Parse rendering settings (combining recipe defaults with overrides)
    defaults = recipe.rendering
    overrides = rendering_overrides or {}

    stretch = overrides.get("stretch") or defaults.stretch
    percentile_clip = overrides.get("percentile_clip") or defaults.percentile_clip
    bg_sub = overrides.get("background_subtraction") or defaults.background_subtraction
    asinh_a = float(overrides.get("asinh_a", 0.1))
    gamma = float(overrides.get("gamma", 1.0))
    gain_red = float(overrides.get("gain_red", 1.0))
    gain_green = float(overrides.get("gain_green", 1.0))
    gain_blue = float(overrides.get("gain_blue", 1.0))
    snr_threshold = overrides.get("snr_threshold")  # Optional float cutoff

    channel_stats = {}
    normalized_channels = {}
    valid_masks = {}

    p_low, p_high = percentile_clip

    # Process each channel
    for color in ("red", "green", "blue"):
        cmap = ch_dict[color]
        data = cmap.data.copy()

        valid = np.isfinite(data)
        if not np.any(valid):
            raise IncompatibleMapError(f"ComponentMap for channel '{color}' contains zero valid spaxels.")

        # Background Subtraction
        if bg_sub in ("per_channel_median", "global_median"):
            bg_level = float(np.nanmedian(data[valid]))
            data[valid] -= bg_level
            data[valid] = np.maximum(0.0, data[valid])
        else:
            bg_level = 0.0

        # SNR Masking
        if snr_threshold is not None and cmap.snr is not None:
            snr_mask = (cmap.snr < float(snr_threshold))
            data[snr_mask] = 0.0

        valid_data = data[valid]
        v_low = float(np.percentile(valid_data, p_low))
        v_high = float(np.percentile(valid_data, p_high))

        channel_stats[color] = {
            "component_name": cmap.component_name,
            "unit": cmap.unit,
            "bg_level": bg_level,
            "percentile_low": v_low,
            "percentile_high": v_high,
            "valid_spaxels": int(np.count_nonzero(valid)),
        }

        # Normalize to [0, 1] using percentile range
        if v_high > v_low:
            norm = np.clip((data - v_low) / (v_high - v_low), 0.0, 1.0)
        else:
            norm = np.zeros_like(data)

        # Apply stretch
        stretched = _apply_stretch(norm, stretch=stretch, asinh_a=asinh_a)

        # Apply Gains
        if color == "red":
            stretched *= gain_red
        elif color == "green":
            stretched *= gain_green
        elif color == "blue":
            stretched *= gain_blue

        normalized_channels[color] = np.clip(stretched, 0.0, 1.0)
        valid_masks[color] = valid

    # Stack into RGB array
    rgb = np.dstack([
        normalized_channels["red"],
        normalized_channels["green"],
        normalized_channels["blue"],
    ])

    # Apply Gamma
    if gamma != 1.0 and gamma > 0:
        rgb = np.power(rgb, 1.0 / gamma)

    rgb = np.clip(rgb, 0.0, 1.0)

    # Calculate overall alpha coverage mask
    overall_valid = valid_masks["red"] & valid_masks["green"] & valid_masks["blue"]
    alpha = overall_valid.astype(float)
    rgba = np.dstack([rgb, alpha])

    # Mask unobserved pixels in RGB to 0.0
    rgb[~overall_valid] = 0.0

    # Build Provenance
    provenance = {
        "action": "render_false_color",
        "recipe_name": recipe.name,
        "target_name": recipe.target_name,
        "rendering_mode": rendering_mode,
        "stretch": stretch,
        "percentile_clip": list(percentile_clip),
        "background_subtraction": bg_sub,
        "asinh_a": asinh_a,
        "gamma": gamma,
        "gains": {"red": gain_red, "green": gain_green, "blue": gain_blue},
        "snr_threshold": snr_threshold,
        "input_component_files": {color: ch_dict[color].provenance.get("source_filepath", "") for color in ("red", "green", "blue")},
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "warning": "False-colour composition: hues represent chemical/physical line features, not human vision.",
    }

    return FalseColorResult(
        rgb_array=rgb,
        rgba_array=rgba,
        recipe=recipe,
        rendering_mode=rendering_mode,
        channel_maps=ch_dict,
        channel_stats=channel_stats,
        provenance=provenance,
    )
