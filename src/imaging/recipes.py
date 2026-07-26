"""Typed, validated image recipes for future JWST MIRI IFU RGB rendering.

This module deliberately describes *what* to measure and show.  It does not
download MAST products, open FITS files, or render pixels; those are Phase 2+
responsibilities.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping


ValidationError = ValueError
StretchMethod = Literal["linear", "sqrt", "log", "asinh"]
MappingIntent = Literal["scientific", "presentation"]

_STRETCHES = frozenset({"linear", "sqrt", "log", "asinh"})
_BACKGROUND_METHODS = frozenset({"none", "global_median", "per_channel_median"})
_COLOUR_BALANCES = frozenset({"preserve_flux_ratios", "equalize_channels", "manual"})
_CONTINUUM_METHODS = frozenset({"adjacent_sidebands", "local_linear_fit"})
_RGB_COLOURS = frozenset({"red", "green", "blue"})


def _require_non_empty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{field_name} must be a non-empty string")


@dataclass(frozen=True)
class MastSearchParameters:
    """Optional constraints to apply when Phase 2 queries MAST."""

    proposal_id: str | None = None
    obs_id: str | None = None
    program_id: str | None = None

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if value is not None:
                _require_non_empty(value, f"mast_search.{name}")


@dataclass(frozen=True)
class ProductExpectation:
    """The archive product this recipe is intended to consume."""

    instrument: str
    product_type: str
    observing_mode: str

    def __post_init__(self) -> None:
        _require_non_empty(self.instrument, "expected_product.instrument")
        _require_non_empty(self.product_type, "expected_product.product_type")
        _require_non_empty(self.observing_mode, "expected_product.observing_mode")
        if self.instrument.upper() != "MIRI":
            raise ValidationError("expected_product.instrument must be 'MIRI'")
        if self.product_type.lower() != "s3d":
            raise ValidationError("expected_product.product_type must be 's3d'")
        if self.observing_mode.upper() != "MRS IFU":
            raise ValidationError("expected_product.observing_mode must be 'MRS IFU'")


@dataclass(frozen=True)
class ContinuumSubtraction:
    """Explicit local-continuum measurement settings for one spectral band."""

    enabled: bool
    method: str | None
    sideband_width_um: float | None
    gap_um: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValidationError("continuum_subtraction.enabled must be boolean")
        if not self.enabled:
            if any(value is not None for value in (self.method, self.sideband_width_um, self.gap_um)):
                raise ValidationError("disabled continuum subtraction must use null method and widths")
            return
        if self.method not in _CONTINUUM_METHODS:
            raise ValidationError(f"continuum_subtraction.method must be one of {sorted(_CONTINUUM_METHODS)}")
        for name, value in (("sideband_width_um", self.sideband_width_um), ("gap_um", self.gap_um)):
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
                raise ValidationError(f"continuum_subtraction.{name} must be positive when enabled")


@dataclass(frozen=True)
class RGBChannel:
    """One named spectral feature and its assigned output display colour."""

    feature_name: str
    central_wavelength_um: float
    integration_width_um: float
    continuum_subtraction: ContinuumSubtraction
    display_colour: str

    def __post_init__(self) -> None:
        _require_non_empty(self.feature_name, "channel.feature_name")
        for name, value in (("central_wavelength_um", self.central_wavelength_um),
                            ("integration_width_um", self.integration_width_um)):
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
                raise ValidationError(f"channel.{name} must be positive")
        if self.display_colour not in _RGB_COLOURS:
            raise ValidationError("channel.display_colour must be red, green, or blue")


@dataclass(frozen=True)
class RenderingDefaults:
    """Display-only choices, stored alongside the scientific measurement recipe."""

    stretch: StretchMethod
    percentile_clip: tuple[float, float]
    background_subtraction: str
    colour_balance: str
    output_resolution: tuple[int, int]
    output_filename: str

    def __post_init__(self) -> None:
        if self.stretch not in _STRETCHES:
            raise ValidationError(f"rendering.stretch must be one of {sorted(_STRETCHES)}")
        if len(self.percentile_clip) != 2:
            raise ValidationError("rendering.percentile_clip must contain [low, high]")
        low, high = self.percentile_clip
        if any(not isinstance(x, (int, float)) or isinstance(x, bool) for x in (low, high)) or not (0 <= low < high <= 100):
            raise ValidationError("rendering.percentile_clip must satisfy 0 <= low < high <= 100")
        if self.background_subtraction not in _BACKGROUND_METHODS:
            raise ValidationError(f"rendering.background_subtraction must be one of {sorted(_BACKGROUND_METHODS)}")
        if self.colour_balance not in _COLOUR_BALANCES:
            raise ValidationError(f"rendering.colour_balance must be one of {sorted(_COLOUR_BALANCES)}")
        if len(self.output_resolution) != 2 or any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in self.output_resolution):
            raise ValidationError("rendering.output_resolution must contain two positive integers")
        _require_non_empty(self.output_filename, "rendering.output_filename")


@dataclass(frozen=True)
class ImageRecipe:
    """Full, serializable specification for a three-channel MIRI false-colour image."""

    name: str
    target_name: str
    mast_search: MastSearchParameters | None
    expected_product: ProductExpectation
    mapping_intent: MappingIntent
    mapping_note: str
    channels: tuple[RGBChannel, RGBChannel, RGBChannel]
    rendering: RenderingDefaults

    def __post_init__(self) -> None:
        _require_non_empty(self.name, "name")
        _require_non_empty(self.target_name, "target_name")
        _require_non_empty(self.mapping_note, "mapping_note")
        if self.mapping_intent not in {"scientific", "presentation"}:
            raise ValidationError("mapping_intent must be 'scientific' or 'presentation'")
        if len(self.channels) != 3:
            raise ValidationError("recipes must define exactly three RGB channels")
        colours = [channel.display_colour for channel in self.channels]
        if set(colours) != _RGB_COLOURS or len(set(colours)) != 3:
            raise ValidationError("recipes must assign one each of red, green, and blue")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ImageRecipe":
        """Build a recipe from parsed JSON, rejecting absent or invalid fields."""
        if not isinstance(data, Mapping):
            raise ValidationError("recipe root must be an object")
        try:
            mast = data.get("mast_search")
            channels = tuple(
                RGBChannel(
                    feature_name=channel["feature_name"],
                    central_wavelength_um=channel["central_wavelength_um"],
                    integration_width_um=channel["integration_width_um"],
                    continuum_subtraction=ContinuumSubtraction(**channel["continuum_subtraction"]),
                    display_colour=channel["display_colour"],
                )
                for channel in data["channels"]
            )
            rendering = data["rendering"]
            return cls(
                name=data["name"], target_name=data["target_name"],
                mast_search=MastSearchParameters(**mast) if mast is not None else None,
                expected_product=ProductExpectation(**data["expected_product"]),
                mapping_intent=data["mapping_intent"], mapping_note=data["mapping_note"],
                channels=channels, rendering=RenderingDefaults(
                    stretch=rendering["stretch"], percentile_clip=tuple(rendering["percentile_clip"]),
                    background_subtraction=rendering["background_subtraction"],
                    colour_balance=rendering["colour_balance"],
                    output_resolution=tuple(rendering["output_resolution"]),
                    output_filename=rendering["output_filename"],
                ),
            )
        except (KeyError, TypeError) as exc:
            raise ValidationError(f"recipe is missing or has an invalid field: {exc}") from exc

    @classmethod
    def load(cls, path: str | Path) -> "ImageRecipe":
        """Load a JSON recipe file. JSON is used to avoid an undeclared YAML dependency."""
        source = Path(path)
        if source.suffix.lower() != ".json":
            raise ValidationError("recipe files must use the .json extension")
        try:
            with source.open(encoding="utf-8") as handle:
                return cls.from_dict(json.load(handle))
        except json.JSONDecodeError as exc:
            raise ValidationError(f"invalid JSON in {source}: {exc.msg}") from exc

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation without losing recipe detail."""
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False) + "\n"

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        if destination.suffix.lower() != ".json":
            raise ValidationError("recipe files must use the .json extension")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.to_json(), encoding="utf-8")
