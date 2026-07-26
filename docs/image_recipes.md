# MIRI IFU false-colour image recipes (Phase 1)

Phase 1 defines reproducible image specifications only. It does **not** query MAST, download observations, read FITS cubes, or make images.

False colour is not natural visible-light colour. MIRI records mid-infrared emission, so RGB is a documented visual encoding of selected wavelength bands. Each JSON recipe records the feature name, wavelength, integration width, continuum model, and display colour needed to reproduce that encoding.

## Starter recipes

The bundled JSON files live in `src/imaging/recipes/`:

| Recipe | Blue | Green | Red | Mapping type |
| --- | --- | --- | --- | --- |
| `miri_emission_line_rgb.json` | [S IV] 10.51 um | [Ne II] 12.81 um | [S III] 18.71 um | Scientific |
| `miri_pah_rgb.json` | PAH 7.7 um | PAH 11.3 um | [Ne II] 12.81 um | Scientific |
| `miri_nasa_style_false_colour.json` | [S IV] 10.51 um | PAH 11.3 um | [Ne III] 15.55 um | Presentation |

**Scientific mapping** assigns declared RGB colours directly to named physical emission features, making the resulting colours interpretable as feature mixtures. **Presentation mapping** may apply deliberate colour balance or stretches for visual clarity. Its source bands are still fully documented in the recipe and must be stated with any published image.

## Creating a recipe

Copy a starter JSON file, set a real `target_name` (and optional `mast_search` constraints), then update all three channel objects. Every channel needs a feature name, positive central wavelength and integration width in microns, explicit continuum-subtraction settings, and exactly one of `blue`, `green`, or `red`. Set `mapping_intent` to `scientific` or `presentation`, explain the choice in `mapping_note`, and explicitly choose the rendering controls.

Validate and serialize with the typed API:

```python
from src.imaging.recipes import ImageRecipe

recipe = ImageRecipe.load("src/imaging/recipes/miri_emission_line_rgb.json")
recipe.save("my_recipe.json")
```

Invalid science settings are rejected rather than repaired: there are no implicit wavelengths, widths, sidebands, or RGB assignments. Supported stretches are `linear`, `sqrt`, `log`, and `asinh`.

## Phase 2 MAST inputs

Phase 2 must retrieve calibrated JWST **MIRI MRS IFU Level-3 `s3d` spectral cubes** matching `expected_product` and any `mast_search` constraints. It must retain archive provenance (observation/product identifiers, source URL, retrieval date, and calibration/pipeline metadata), spectral WCS and units, science data, and the associated uncertainty and data-quality arrays when available. The renderer can then integrate the documented bands, estimate the specified continuum, and apply the recipe's display controls.
