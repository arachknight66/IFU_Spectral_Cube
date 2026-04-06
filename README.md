# IFU Spectral Cube

Research-grade pipeline for **spectral line identification in JWST MIRI IFU cubes**.

## Project Structure

```text
IFU_Spectral_Cube/
├── pyproject.toml
├── config.example.yaml
├── README.md
├── src/
│   └── ifu_spectral_cube/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py            # Command-line interface
│       ├── config.py          # Pipeline configuration dataclasses
│       ├── extraction.py      # Spectrum extraction (aperture, optimal, k-means)
│       ├── fitting.py         # Gaussian/Voigt fitting, deblending, diagnostics
│       ├── io.py              # FITS I/O with WCS-aware wavelength inference
│       ├── lines.py           # Line database, matching, confidence, reliability
│       ├── models.py          # Data models (SpectralCube, Peak, LineMatch, etc.)
│       ├── peaks.py           # Peak detection (find_peaks + CWT + local SNR)
│       ├── pipeline.py        # Pipeline orchestrator with logging & timing
│       ├── preprocessing.py   # Continuum, smoothing, DQ masking, defringing
│       ├── visualization.py   # Publication-quality plots + interactive HTML
│       └── data/
│           └── lines_mid_ir.csv   # 43-line MIR database
└── tests/
    ├── test_extraction.py
    ├── test_fitting.py
    ├── test_io.py
    ├── test_peaks.py
    ├── test_pipeline_integration.py
    ├── test_preprocessing.py
    └── test_signal_and_lines.py
```

## Scientific Pipeline Design

### 1. Data Handling (JWST FITS Cube + WCS)

- `io.load_miri_ifu_cube` reads `SCI` cube (or first 3D image HDU), optional `ERR` and `DQ`.
- WCS is parsed to infer the spectral world axis and convert wavelength to **micron**.
- Cube is normalized to canonical shape: **`(n_lambda, ny, nx)`**.
- Optional wavelength clipping via `io.clip_cube_wavelength`.

Why this matters:
- A fixed spectral-first convention avoids axis mistakes in downstream filtering, extraction, and fitting.
- Using WCS avoids hard-coding axis assumptions across different MIRI cube products.

### 2. Data Quality Masking

- `apply_dq_mask(data, dq, bad_dq_flags)` replaces flagged voxels with NaN using bitwise DQ flags.
- The JWST pipeline sets DQ bit 0 (`DO_NOT_USE`) for dead pixels, cosmic ray hits, saturated channels.
- Masking **before** any filtering prevents hardware artifacts from propagating into continuum estimates and peak detection.

### 3. Continuum Subtraction + Noise Reduction

Two continuum estimation modes:
- **Standard**: Wide 1D median filter (`estimate_continuum_1d`).
- **Iterative sigma-clipped** (`estimate_continuum_iterative`): iteratively rejects emission features above Nσ before re-estimating, following IRAF's `continuum` task methodology.

Two smoothers applied to continuum-subtracted signal:
- **Savitzky-Golay** (`apply_savgol_smoothing`): local polynomial least-squares fitting.  Preserves polynomial moments up to degree *p*, maintaining line centroid and amplitude fidelity for narrow features.
- **Gaussian** (`apply_gaussian_smoothing`): symmetric low-pass filter. More aggressive noise suppression but attenuates peaks.

`compare_smoothing_metrics` reports robust noise reduction factors and flux-preservation proxies.

**Defringing** (`defringe_1d`): Uses Lomb-Scargle periodogram to identify and subtract residual Fabry-Perot etalon fringes that persist in MIRI MRS Level-3 cubes at the 1–5% level.

### 4. Spectrum Extraction

Six extraction modes in `extraction.py`:
- **Circular aperture** (`circular_mask` + `extract_region_spectrum`)
- **Rectangular aperture** (`rectangular_mask`)
- **Annular background subtraction** (`annular_mask` + `extract_with_background`) — subtracts median local sky spectrum
- **Optimal (inverse-variance weighted)** (`extract_optimal_spectrum`) — Horne (1986) formalism adapted for IFU, weights by 1/σ² from ERR extension
- **Percentile-based** (`extract_percentile_spectrum`) — selects brightest spaxels by moment-0
- **k-means clustering** (`cluster_spaxels_kmeans`) — unsupervised spatial segmentation by spectral moments

### 5. Peak Detection

Two complementary algorithms:

- **`detect_peaks_1d`** — `scipy.signal.find_peaks` with prominence/width thresholds calibrated against robust MAD noise.  Prominence is SNR-scaled (`N_sigma × σ_noise`); width bounds are tied to MIRI resolving power R.
- **`detect_peaks_cwt`** — Continuous wavelet transform peak finder (`find_peaks_cwt`).  Searches across multiple scales simultaneously, robust for blended features and varying line widths.

`merge_peak_lists` deduplicates results from both methods, keeping the higher-SNR detection.

`refine_peak_local_snr` recomputes SNR in a local window around each peak, handling bandpass-dependent noise variations.

### 6. Spectral Line Identification

Built on a 43-line MIR database (`data/lines_mid_ir.csv`) covering:
- Atomic fine-structure lines ([Ne II], [Ne III], [S IV], [O IV], [Fe II], etc.)
- Molecular hydrogen S(0)–S(7) pure rotational series
- PAH features (6.2, 7.7, 8.6, 11.3, 12.7, 16.4, 17.0 µm)
- Hydrogen recombination lines (Pfund-α, Humphreys-α/β)
- Coronal lines ([Mg V], [Mg VII], [Si VII], [Ca V], [Ar V], [Ne V])
- CO₂ ice absorption (15.2 µm)

Identification workflow:
1. Coarse redshift from SNR-weighted peak-line consensus histogram.
2. Adaptive tolerance matching: `tol = max(abs_floor, σ_tolerance × λ/(R·2.355))`.
3. Multi-factor confidence scoring: SNR term × wavelength residual term × ambiguity penalty × fit quality term + multi-line consistency bonus.
4. Reliability classification: `secure`, `probable`, `tentative`, `marginal` following spectroscopic survey conventions.

### 7. Advanced Fitting + Redshift Refinement

- **Gaussian + linear baseline** (`fit_peak_gaussian`) with `curve_fit`.
- **Voigt + linear baseline** (`fit_peak_voigt`) — Gaussian+Lorentzian convolution via `scipy.special.voigt_profile`.  BIC-based model selection decides if the extra Lorentzian parameter is justified.
- **Multi-Gaussian deblending** (`fit_multiplet`) — simultaneous fit of 2–4 Gaussians on shared baseline for blended multiplets.
- **Fit quality diagnostics**: reduced χ², Shapiro-Wilk normality test on residuals, Durbin-Watson autocorrelation statistic.
- **Weighted redshift refinement** from fitted line centers with proper uncertainty propagation.

### 8. Visualization

Publication-quality diagnostic plots with dark science theme:

| Plot | File | Description |
|------|------|-------------|
| Smoothing comparison | `smoothing_comparison.png` | Raw spectrum, continuum, SG, Gaussian |
| Peaks and matches | `peaks_and_matches.png` | Color-coded line labels by ionization class |
| Gaussian fit panels | `gaussian_fit_panels.png` | Data + model + residuals per line |
| Redshift histogram | `redshift_histogram.png` | SNR-weighted z-consensus |
| Noise profile | `noise_profile.png` | Local noise vs wavelength |
| Strongest line map | `strongest_line_map.png` | Continuum-subtracted spatial emission |
| Diagnostic grid | `line_diagnostic_grid.png` | Multi-panel maps for top matched lines |
| Interactive viewer | `interactive_spectrum.html` | Plotly HTML (optional) |

## Outputs

`pipeline.run_pipeline` writes:
- `detected_peaks.csv`
- `line_matches.csv`
- `gaussian_fits.csv`
- `voigt_fits.csv` (if any Voigt model was preferred)
- `summary.json` (includes timing, reliability counts, redshift estimates)
- `intermediate_spectra.npz` (wavelength, raw, continuum, smoothed spectra)
- All diagnostic plots listed above

## Installation

```bash
cd IFU_Spectral_Cube
python -m pip install -e .[dev]
```

## Run

CLI mode:

```bash
ifu-line-id \
  --cube /path/to/jwst_miri_cube.fits \
  --output-dir outputs \
  --mode circular \
  --x 25 --y 30 --radius 3.0 \
  --prominence-sigma 4.5
```

With background subtraction:

```bash
ifu-line-id \
  --cube /path/to/cube.fits \
  --output-dir outputs \
  --x 25 --y 30 --radius 3.0 \
  --bg-inner 4.0 --bg-outer 7.0
```

With advanced features:

```bash
ifu-line-id \
  --cube /path/to/cube.fits \
  --output-dir outputs \
  --iterative-continuum \
  --defringe \
  --use-cwt \
  --try-voigt \
  --interactive-html
```

YAML config mode:

```bash
ifu-line-id --config config.example.yaml
```

## Testing

```bash
pytest -v
```

69 tests covering:
- Preprocessing (noise estimation, DQ masking, continuum, smoothing, defringing)
- Extraction (all mask types, background subtraction, optimal extraction)
- Peak detection (find_peaks, CWT, merging, local SNR)
- Fitting (Gaussian parameter recovery, Voigt, deblending, diagnostics)
- Line identification (database loading, matching, reliability classification)
- I/O (synthetic FITS cube loading, wavelength handling, ERR/DQ extensions)
- End-to-end integration (synthetic cube → full pipeline → validate outputs)

## Notes for Real JWST Data

- Calibrated Level-3 products can contain residual fringing/continuum curvature; use `--defringe` and `--iterative-continuum` for crowded spectra.
- For crowded fields, use `--bg-inner` / `--bg-outer` for local background subtraction.
- Use `--mode optimal` with ERR extension for SNR-optimal extraction.
- Inspect ambiguous matches and enforce multi-line consistency for final astrophysical claims.
- Use fitted line centers (not smoothed peaks) for final redshift reporting and uncertainty propagation.
- The reliability classification (`secure`/`probable`/`tentative`/`marginal`) provides a quick filter for publication-quality identifications.
