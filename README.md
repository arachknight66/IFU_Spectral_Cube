# JWST MIRI IFU Spectral Analysis Pipeline

A modern, production-grade Python pipeline designed specifically to process and analyze **JWST MIRI Medium Resolution Spectroscopy (MRS)** Integral Field Unit (IFU) data cubes. It conducts fully automated spectral alignment, denoising, peak detection, emission line identification, and complex spatial mapping through either a CLI or an interactive **Streamlit frontend dashboard**.

---

## 🌟 Key Features

- **Seamless Spectral Stitching** — Upload multiple diverse channel cubes (such as `ch1-short` and `ch2-medium`). The pipeline uses `astropy.wcs` to dynamically warp varying field-of-view arrays and pixel CDELT resolutions onto a mathematically exact unified spatial grid.
- **Robust Preprocessing** — Applies Savitzky-Golay denoising, deliberately avoiding standard Gaussian algorithms which catastrophically blend tight spectral pairs in MIRI datasets (e.g., [Ne II] 12.814 µm and PAH 12.7 µm).
- **Automated Detection & Line Identification** — Uses standard-deviation-based adaptive height profiling algorithms across iteratively subtracted continuum baselines to spot valid spectral spikes, natively checking results against a baked-in JWST molecular/atomic line database.
- **Resilient Degenerate HDU Handling** — Built in fallback geometry parsing. JWST Level-3 (`s3d`) ASDF pipelines often strip standard linear astropy headers, causing silent 0-dimension failures. The pipeline catches these structure anomalies and maps fallback 1:1 boundaries natively.
- **Publication Imaging** — Render monochromatic slices, band-integrated fluxes, and robust continuum-subtracted emission line maps ready for paper publication.

## 🏗️ Architecture Stack

All pure scientific logic is abstracted from the UI elements. 

```text
FITS File(s) → loader.py & stitch.py → Master SpectralCube
    → Preprocessing (SG Denoise + Sigma-Clipping Baseline Subtraction)
    → Analysis (Prominence Detection + Gaussian Fitting + Line Matching)
    → Imaging (Monochromatic Slices + Band Integration + Emission Maps)
    → Output (Matplotlib publication renderings + Streamlit)
```

## 🚀 Quick Start

### Installation

Requires a modern Python 3.10+ environment. Follow standard package installations via `pip`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Streamlit Dashboard (Recommended)

To launch the interactive GUI, load the dashboard via Streamlit. You can drag and drop multiple FITS files into the sidebar, and the server will dynamically stitch and analyze them while preserving previous run caching.

```bash
streamlit run frontend/app.py
```

### Command-Line Interface (CLI)

Need batch processing capabilities or want to execute outputs into a directory directly? Drop to the console.

```bash
# Run on a real single FITS observation
python main.py path/to/your/cube.fits -x 15 -y 20 -o output_dir/

# Test the backend mechanics using the built-in synthetic cube simulation generator
python main.py --synthetic --x 15 --y 15

# Customize your extraction mechanics overrides
python main.py path/to/cube.fits --savgol-window 13 --prominence 3.0 --tolerance 0.08 --redshift 0.003
```

## 📂 Project Structure

```text
IFU_Spectral_Cube/
├── src/
│   ├── core/
│   │   ├── loader.py       # FITS Extractor & Degenerate Dimension Collider
│   │   ├── stitch.py       # Spatial WCS Warping & Spectral Aggregation 
│   │   └── cube.py         # Standardized Data Object Container
│   ├── preprocessing/
│   │   ├── denoise.py      # SG Moment Preservation Methods
│   │   └── baseline.py     # Polynomial iterative masking
│   ├── analysis/
│   │   ├── spectrum.py     # Data slicing heuristics 
│   │   ├── peaks.py        # Peak topography algorithms
│   │   ├── fitting.py      # Gaussian profile curve fits
│   │   ├── line_id.py      # Match algorithms for line verification
│   │   └── advanced.py     # Confidence bounds & Redshift calculations
│   ├── imaging/
│   │   ├── slice.py        # 2D projection slicing
│   │   ├── integrate.py    # Mean/Sum spatial band passes
│   │   └── emission.py     # Localized line mapping
│   ├── visualization/
│   │   ├── spectra_plot.py # Annotated chart layouts
│   │   └── image_plot.py   # Publication 2D renders
│   └── utils/
│       ├── config.py       # Core Configuration Variables
│       └── constants.py    # Complete MIRI 5–28 µm Line Database
├── frontend/
│   └── app.py              # Streamlit Web Framework & UI definitions
├── tests/
│   ├── synthetic.py        # Synthetic simulation cube payload generator
│   └── test_stitch.py      # Sub-band multi-file geometry verification
├── pipeline.py             # Internal Orchestration Layer
└── main.py                 # Core shell execution
```

## 🌌 Spectral Line Database Integration

The internal constants package actively guards against:
- **Major Atomic Lines**: [Ne II], [Ne III], [Ar II], [Ar III], [S III], [S IV].
- **Molecules**: Extensively handles H₂ chains (S(0) through S(7)) and common carbon configurations (CO₂, C₂H₂, HCN).
- **Dust/Ices**: Broad bands at 6.2, 7.7, 8.6, 11.3, and 12.7 µm (PAHs) and structural ices.

## License

MIT 
