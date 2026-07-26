"""Validation and serialization tests for Phase 1 false-colour recipes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.imaging.recipes import ImageRecipe


RECIPE_DIR = Path(__file__).parents[1] / "src" / "imaging" / "recipes"


def test_starter_recipes_load() -> None:
    recipes = [ImageRecipe.load(path) for path in sorted(RECIPE_DIR.glob("*.json"))]
    assert len(recipes) == 3
    assert {recipe.mapping_intent for recipe in recipes} == {"scientific", "presentation"}
    assert all({channel.display_colour for channel in recipe.channels} == {"red", "green", "blue"} for recipe in recipes)


def test_invalid_recipe_is_rejected() -> None:
    data = json.loads((RECIPE_DIR / "miri_emission_line_rgb.json").read_text(encoding="utf-8"))
    data["channels"][0]["integration_width_um"] = 0
    with pytest.raises(ValueError, match="integration_width_um must be positive"):
        ImageRecipe.from_dict(data)


def test_recipe_round_trip(tmp_path: Path) -> None:
    source = ImageRecipe.load(RECIPE_DIR / "miri_pah_rgb.json")
    destination = tmp_path / "round_trip.json"
    source.save(destination)
    assert ImageRecipe.load(destination) == source
