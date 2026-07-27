# JWST MIRI IFU Spectral Cube Alignment, Component Mapping & False-Colour Pipeline

[![JWST MIRI IFU Pipeline CI](https://github.com/arachknight66/IFU_Spectral_Cube/actions/workflows/ci.yml/badge.svg)](https://github.com/arachknight66/IFU_Spectral_Cube/actions/workflows/ci.yml)

An uncertainty-aware, scientifically traceable Python pipeline for **JWST MIRI Medium Resolution Spectroscopy (MRS) IFU** data cubes.

Processes FITS spectral cubes from **MAST discovery** to **3D celestial WCS alignment**, **continuum-subtracted component mapping**, **reproducible NASA-style false-colour rendering**, and **JSON manifest sidecar export**.

---

## 🌟 Features & Scientific Architecture

- **MAST Archive Integration**: Search real STScI MAST data by Target Name, Proposal ID, Observation ID, or RA/Dec coordinates. Verified FITS caching with full `.provenance.json` sidecars.
- **Uncertainty-Aware Alignment (Phase 4)**: Reproject and stitch multiple overlapping MIRI sub-band cubes (`ch1-short`, `ch2-medium`, etc.) onto a unified spatial WCS grid using Astropy `reproject` with inverse-variance flux weighting and bitwise OR DQ mask propagation.
- **Traceable Component Mapping (Phase 5)**: Physical $\Delta \lambda$ flux integration and local linear continuum fitting ($m \lambda + b$) across blue/red sidebands. Generates 2D on-band, continuum, subtracted feature, uncertainty, and SNR ($F_{\text{sub}} / \sigma_{\text{sub}}$) maps in multi-extension FITS format.
- **NASA-Style False-Colour Synthesis (Phase 6)**: Combine three Phase-5 component maps into high-definition false-colour RGB photographs. Supports `scientific` and `presentation` rendering modes with non-linear intensity stretches (`asinh`, `log`, `linear`), percentile clipping, and background estimation.
- **Full Reproducibility (Phase 7 & 8)**: Generates a `<image>.manifest.json` sidecar for every rendered image recording input FITS hashes, WCS keywords, stretch cutoffs, gains, timestamp, and false-colour disclaimers.

---

## 🚀 Quick Start

### Installation

Requires Python 3.10+. Install using standard `pip`:

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package with development dependencies
pip install -e .[dev]
```

### Interactive Streamlit Workflow UI (Recommended)

Launch the guided 7-stage web application:

```bash
streamlit run frontend/app.py
```

### Command-Line Interface (CLI)

```bash
# 1. Process single FITS cube
python main.py data/raw/jw01288-o001_t001_miri_ch2-medium_s3d.fits -x 15 -y 20 -o output/

# 2. Align and stitch multiple overlapping MIRI cubes
python main.py --align-files data/raw/cube1.fits data/raw/cube2.fits --reference union -o output/

# 3. Extract component maps using an ImageRecipe
python main.py data/raw/cube.fits --extract-recipe src/imaging/recipes/miri_emission_line_rgb.json --export-maps

# 4. Render false-colour image from component maps
python main.py render \
  --recipe src/imaging/recipes/miri_emission_line_rgb.json \
  --red output/component_miri_emission_line_rgb_red.fits \
  --green output/component_miri_emission_line_rgb_green.fits \
  --blue output/component_miri_emission_line_rgb_blue.fits \
  --render-output output/crab_false_colour.png \
  --mode presentation

# 5. Validate a rendering provenance manifest sidecar
python main.py validate-manifest output/crab_false_colour.png.manifest.json

# 6. Run built-in synthetic test cube
python main.py --synthetic --x 15 --y 15
```

---

## 🎨 Scientific vs Presentation Rendering Modes

> [!WARNING]
> **FALSE-COLOUR DISCLAIMER**: All rendered outputs map mid-infrared spectral line features ([S III], [Ne II], PAH dust, etc.) to visible Red, Green, Blue channels. They represent physical chemical distributions, **NOT human visible-light colour**.

- **`scientific` Mode**: Strictly enforces declared `ImageRecipe` channel mappings and display transforms without arbitrary aesthetic overrides.
- **`presentation` Mode**: Permits aesthetic per-channel gain, gamma, and colour-balance adjustments for press-release graphics, while recording all overrides in the output manifest sidecar.

---

## 🔬 Repository Structure

```text
IFU_Spectral_Cube/
├── pyproject.toml              # Modern PEP 621 package & tooling configuration
├── .gitignore                  # Clean repository ignore rules
├── pipeline.py                 # High-level pipeline orchestration layer
├── main.py                     # CLI entry point (align, extract, render, validate)
├── src/
│   ├── core/
│   │   ├── cube.py             # SpectralCube dataclass & validation report
│   │   ├── loader.py           # FITS reader & 3D WCS dimension parser
│   │   ├── mast.py             # STScI MAST discovery, query & verified download
│   │   ├── mosaic.py           # Master WCS grid footprint calculations
│   │   └── stitch.py           # Reprojection & inverse-variance stitching
│   ├── imaging/
│   │   ├── component_map.py    # Physical Δλ integration & continuum fitting
│   │   └── recipes.py          # Phase-1 ImageRecipe JSON parser & schemas
│   ├── visualization/
│   │   ├── false_color.py      # Reproducible false-colour rendering engine
│   │   └── nasa_plot.py        # Press-release photograph cartography
│   └── utils/
│       ├── logger.py           # Centralized structured logger
│       └── provenance_validator.py # Provenance manifest validator
├── frontend/
│   └── app.py                  # Guided 7-stage Streamlit web application
├── tests/                      # 100% offline unit test suite (52+ tests)
└── .github/workflows/ci.yml    # GitHub Actions CI workflow
```

---

## 🧪 Testing & Code Quality

Run the complete offline test suite:

```bash
# Run standard offline unit tests
pytest

# Run linter checks
ruff check .
```

---

## ⚠️ Data Limitations & Troubleshooting

1. **WCS Compatibility**: MIRI cubes must possess valid 3D celestial WCS coordinates (`RA---TAN`, `DEC--TAN`, `WAVE`). Cubes with missing WCS will raise `WCSAlignmentError`.
2. **Uncertainty Data**: If a FITS file lacks an `ERR` extension, uncertainty maps will be marked as unauthoritative (`has_authoritative_uncertainty = False`) and SNR maps will be suppressed.
3. **MAST Network Requirements**: MAST search & download requires internet access. All other pipeline phases (alignment, extraction, rendering, validation) operate 100% offline.
