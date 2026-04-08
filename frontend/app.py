"""
Streamlit Frontend for JWST MIRI IFU Spectral Pipeline.

A scientific workflow UI organized into tabbed panels:
    1. 📊 Spectrum Analysis — raw/processed overlay, peak markers, line IDs
    2. 🖼️ Imaging — wavelength slices, integrated maps, emission line maps
    3. 📋 Summary & Export — metadata, results table, downloads

All computation is delegated to the SpectralPipeline backend.
No scientific logic lives in this file.
"""

from __future__ import annotations

import sys
import io
import csv
from pathlib import Path

import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Streamlit
import matplotlib.pyplot as plt

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.utils.config import PipelineConfig
from src.core.cube import SpectralCube
from pipeline import SpectralPipeline, AnalysisResult

# ===========================================================================
# Page config
# ===========================================================================
st.set_page_config(
    page_title="JWST MIRI IFU Pipeline",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===========================================================================
# Custom CSS for a clean scientific aesthetic
# ===========================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    /* Global Typography & Background */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Base App Background */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #1a1c29, #0a0b10);
        color: #e2e8f0;
    }
    
    .main > div { padding-top: 1.5rem; }

    /* Top Header - Premium Glassmorphism */
    .pipeline-header {
        background: rgba(20, 22, 37, 0.4);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 2.5rem;
        border: 1px solid rgba(102, 252, 241, 0.15);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
        position: relative;
        overflow: hidden;
    }
    .pipeline-header::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%, #66fcf1 100%);
    }
    .pipeline-header h1 {
        font-size: 2.4rem;
        font-weight: 700;
        margin: 0;
        background: -webkit-linear-gradient(45deg, #ffffff, #8bc6ec);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.01em;
    }
    .pipeline-header p {
        color: #94a3b8;
        font-size: 0.95rem;
        margin: 0.5rem 0 0;
        font-weight: 300;
        letter-spacing: 0.05em;
    }

    /* Metric Cards - Interactive hover and glass */
    .metric-card {
        background: linear-gradient(145deg, rgba(30, 33, 48, 0.6), rgba(20, 22, 31, 0.8));
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 14px;
        padding: 1.2rem;
        text-align: center;
        transition: transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.3s ease, border-color 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .metric-card:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 12px 25px rgba(0,0,0,0.4);
        border-color: rgba(102, 252, 241, 0.4);
    }
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #66fcf1;
        text-shadow: 0 0 12px rgba(102, 252, 241, 0.3);
    }
    .metric-card .label {
        font-size: 0.72rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-top: 5px;
        font-weight: 600;
    }

    /* Sidebar Refinements */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #10121a 0%, #0a0b10 100%);
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.08);
    }
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #4facfe;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: rgba(20, 22, 31, 0.4);
        padding: 0.5rem 1rem 0;
        border-radius: 12px 12px 0 0;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        font-weight: 500;
        color: #94a3b8;
        padding: 0.8rem 1.5rem;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(79, 172, 254, 0.1);
        color: #66fcf1 !important;
        border-bottom-color: #66fcf1;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        color: #000;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        box-shadow: 0 0 15px rgba(79, 172, 254, 0.5);
        transform: scale(1.02);
        color: #000;
    }

    /* Dataframes/Tables */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.05);
    }

    /* Main Page Entry Animation */
    @keyframes fadeInScale {
        0% { opacity: 0; transform: translateY(10px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    .main > div {
        animation: fadeInScale 0.5s ease-out forwards;
    }
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# Header
# ===========================================================================
st.markdown("""
<div class="pipeline-header">
    <h1>🔭 JWST MIRI IFU Spectral Pipeline</h1>
    <p>Spectral extraction • Peak detection • Line identification • Imaging</p>
</div>
""", unsafe_allow_html=True)


# ===========================================================================
# Sidebar — Data source & configuration
# ===========================================================================

with st.sidebar:
    st.markdown("### 📁 Data Source")

    data_source = st.radio(
        "Source",
        ["🧪 Synthetic Test Cube", "📂 Upload FITS File", "📁 Load from Directory"],
        label_visibility="collapsed",
    )

    uploaded_files = []
    local_fits_path = None

    if data_source == "📂 Upload FITS File":
        uploaded_files = st.file_uploader(
            "Upload FITS cube(s)",
            type=["fits", "fits.gz"],
            accept_multiple_files=True,
            help="JWST MIRI IFU FITS file(s). Upload multiple to stitch spectrally.",
        )
    elif data_source == "📁 Load from Directory":
        fits_dir = Path("data/raw")
        if fits_dir.exists():
            fits_files = sorted(fits_dir.glob("*.fits")) + sorted(fits_dir.glob("*.fits.gz"))
            if fits_files:
                local_fits_path = st.selectbox(
                    "Select FITS file",
                    fits_files,
                    format_func=lambda p: p.name,
                )
            else:
                st.info("No FITS files found in `data/raw/`")
        else:
            st.info("Directory `data/raw/` does not exist")

    st.markdown("---")
    st.markdown("### ⚙️ Preprocessing")

    savgol_window = st.slider(
        "SG Window Length",
        min_value=5, max_value=31, value=11, step=2,
        help="Savitzky-Golay filter window (must be odd)",
    )
    savgol_polyorder = st.slider(
        "SG Polynomial Order",
        min_value=1, max_value=min(7, savgol_window - 1), value=3,
        help="Higher orders preserve more detail but suppress less noise",
    )
    continuum_order = st.slider(
        "Continuum Poly Order",
        min_value=1, max_value=7, value=3,
        help="Degree of polynomial for continuum fit",
    )
    sigma_clip = st.slider(
        "Sigma Clip",
        min_value=1.0, max_value=10.0, value=3.0, step=0.5,
        help="σ threshold for iterative clipping during continuum fitting",
    )

    st.markdown("---")
    st.markdown("### 🔍 Peak Detection")

    prominence = st.slider(
        "Sigma Threshold",
        min_value=1.0, max_value=10.0, value=3.0, step=0.5,
        help="Detection threshold in units of noise σ",
    )
    peak_distance = st.slider(
        "Min Peak Distance",
        min_value=1, max_value=20, value=5,
        help="Minimum separation between peaks (channels)",
    )

    st.markdown("---")
    st.markdown("### 🏷️ Line Identification")

    tolerance = st.slider(
        "Tolerance (µm)",
        min_value=0.01, max_value=0.2, value=0.05, step=0.01,
        help="Maximum |Δλ| for matching against database",
    )
    redshift = st.number_input(
        "Redshift (z)",
        min_value=0.0, max_value=5.0, value=0.0, step=0.001,
        format="%.4f",
        help="Source redshift for shifting line database",
    )


# ===========================================================================
# Build config from sidebar
# ===========================================================================

@st.cache_data
def build_config(sw, sp, co, sc, prom, pd, tol, z):
    """Create PipelineConfig from widget values (cached by args)."""
    return PipelineConfig(
        savgol_window=sw,
        savgol_polyorder=sp,
        continuum_poly_order=co,
        sigma_clip=sc,
        peak_prominence=prom,
        peak_distance=pd,
        tolerance_um=tol,
        redshift=z,
    )

config = build_config(
    savgol_window, savgol_polyorder, continuum_order,
    sigma_clip, prominence, peak_distance, tolerance, redshift,
)
pipeline = SpectralPipeline(config)


# ===========================================================================
# Load cube
# ===========================================================================

@st.cache_resource
def load_synthetic():
    from tests.synthetic import generate_synthetic_cube
    return generate_synthetic_cube()


@st.cache_resource(show_spinner="Loading and aligning cubes...")
def load_uploaded_files(files_data: list[tuple[bytes, str]]):
    """Load FITS cubes from uploaded bytes (handling multiple for stitching)."""
    import tempfile
    import os
    import streamlit as st
    from astropy.io import fits
    
    tmp_paths = []
    try:
        for data_bytes, filename in files_data:
            fd, path = tempfile.mkstemp(suffix=".fits")
            with os.fdopen(fd, 'wb') as f:
                f.write(data_bytes)
            tmp_paths.append(path)
            
        return pipeline.load_multiple(tmp_paths)
    except ValueError as e:
        st.error(f"Failed to load FITS cube: {str(e)}")
        try:
            with fits.open(tmp_paths[0]) as hdul:
                shapes = [f"'{ext.name}' (Shape: {ext.data.shape if ext.data is not None else 'Empty'})" for ext in hdul]
            st.warning(f"**Diagnostic Info:** I couldn't find a 3D data array in the first file. I found these structures:\n\n" + "\n".join(f"- {s}" for s in shapes))
            st.info("💡 **Tip:** Are you sure this is a 3D IFU cube? If it's a 2D image or 1D spectrum, this pipeline expects a 3D dataset. To apply the backend fix handling degenerate empty dimensions (like [1, 250, 30, 30]), please restart Streamlit.")
        except Exception as e_inner:
            st.error(f"Additionally, astropy failed to inspect the file: {str(e_inner)}")
        st.stop()
    finally:
        for path in tmp_paths:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass

@st.cache_resource
def load_from_path(path: str):
    return pipeline.load(path)


cube: SpectralCube | None = None

if data_source == "🧪 Synthetic Test Cube":
    cube = load_synthetic()
elif data_source == "📂 Upload FITS File" and uploaded_files:
    files_data = [(f.getvalue(), f.name) for f in uploaded_files]
    if len(files_data) > 1:
        st.sidebar.success(f"Stitching {len(files_data)} cubes...")
    cube = load_uploaded_files(files_data)
elif data_source == "📁 Load from Directory" and local_fits_path is not None:
    cube = load_from_path(str(local_fits_path))


if cube is None:
    st.info("👆 Select a data source in the sidebar to begin.")
    st.stop()


# ===========================================================================
# Pixel selector
# ===========================================================================
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📍 Pixel Selection")

    ny, nx = cube.spatial_shape
    col_x, col_y = st.columns(2)
    with col_x:
        px = st.number_input("X", min_value=0, max_value=nx - 1, value=nx // 2, step=1)
    with col_y:
        py = st.number_input("Y", min_value=0, max_value=ny - 1, value=ny // 2, step=1)


# ===========================================================================
# Run analysis
# ===========================================================================

@st.cache_data
def run_analysis(_cube, x, y, _config):
    """Run pipeline analysis (cached by pixel + config)."""
    p = SpectralPipeline(_config)
    return p.analyze(_cube, x, y)

analysis = run_analysis(cube, int(px), int(py), config)


# ===========================================================================
# Metrics bar
# ===========================================================================
st.markdown("")
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="value">{cube.shape[0]}</div>
        <div class="label">Channels</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="value">{cube.spatial_shape[0]}×{cube.spatial_shape[1]}</div>
        <div class="label">Spatial</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="value">{cube.wavelength_range[0]:.1f}–{cube.wavelength_range[1]:.1f}</div>
        <div class="label">λ Range (µm)</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="value">{analysis.peaks.n_peaks}</div>
        <div class="label">Peaks</div>
    </div>""", unsafe_allow_html=True)
with c5:
    n_id = sum(1 for m in analysis.line_matches if m is not None)
    st.markdown(f"""
    <div class="metric-card">
        <div class="value">{n_id}</div>
        <div class="label">Identified</div>
    </div>""", unsafe_allow_html=True)


# ===========================================================================
# Main panel — tabbed layout
# ===========================================================================
tab_spec, tab_img, tab_summary = st.tabs([
    "📊 Spectrum Analysis",
    "🖼️ Imaging",
    "📋 Summary & Export",
])


# ===== Tab 1: Spectrum Analysis =============================================
with tab_spec:
    # Build labels
    peak_labels = []
    for match in analysis.line_matches:
        if match is not None:
            peak_labels.append(f"{match.matched_species}")
        else:
            peak_labels.append("")

    from src.visualization.spectra_plot import plot_spectrum as _plot_spectrum

    fig = _plot_spectrum(
        wavelength=analysis.wavelength,
        raw=analysis.raw_spectrum,
        processed=analysis.processed_spectrum,
        continuum=analysis.continuum,
        peak_wavelengths=analysis.peaks.wavelengths if analysis.peaks.n_peaks > 0 else None,
        peak_labels=peak_labels if peak_labels else None,
        noise_rms=analysis.peaks.noise_rms,
        title=f"Spectrum at pixel ({int(px)}, {int(py)})",
        return_figure=True,
    )
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Peak results table
    if analysis.peaks.n_peaks > 0:
        st.markdown("#### Detected Peaks")

        rows = []
        for i in range(analysis.peaks.n_peaks):
            match = analysis.line_matches[i] if i < len(analysis.line_matches) else None
            gfit = analysis.gaussian_fits[i] if i < len(analysis.gaussian_fits) else None

            rows.append({
                "#": i + 1,
                "λ (µm)": f"{analysis.peaks.wavelengths[i]:.4f}",
                "SNR": f"{analysis.peaks.snr[i]:.1f}",
                "Prominence": f"{analysis.peaks.prominences[i]:.4e}",
                "Width (µm)": f"{analysis.peaks.widths_um[i]:.4f}",
                "Species": match.matched_species if match else "—",
                "Rest λ": f"{match.rest_wavelength_um:.4f}" if match else "—",
                "Confidence": f"{match.confidence:.3f}" if match else "—",
                "Gauss FWHM": f"{gfit.fwhm:.4f}" if gfit and gfit.fit_success else "—",
                "Gauss Flux": f"{gfit.integrated_flux:.4e}" if gfit and gfit.fit_success else "—",
            })

        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("No peaks detected with current parameters. "
                "Try lowering the sigma threshold or adjusting preprocessing.")


# ===== Tab 2: Imaging ======================================================
with tab_img:
    st.markdown("#### Wavelength Slice Viewer")

    wl_min, wl_max = cube.wavelength_range

    slice_col1, slice_col2 = st.columns([3, 1])

    with slice_col2:
        slice_idx = st.slider(
            "Channel Index",
            min_value=0,
            max_value=cube.n_wavelengths - 1,
            value=cube.n_wavelengths // 2,
            help="Select spectral channel",
        )
        st.markdown(f"**λ = {cube.get_wavelength_at(slice_idx):.4f} µm**")

        cmap_options = ["inferno", "viridis", "magma", "plasma", "cividis", "hot", "jet"]
        colormap = st.selectbox("Colormap", cmap_options, index=0)

    with slice_col1:
        from src.visualization.image_plot import plot_image as _plot_image

        slice_img = cube.get_slice(slice_idx)
        fig_slice = _plot_image(
            slice_img,
            title=f"λ = {cube.get_wavelength_at(slice_idx):.4f} µm (channel {slice_idx})",
            colormap=colormap,
            return_figure=True,
        )
        st.pyplot(fig_slice, use_container_width=True)
        plt.close(fig_slice)

    st.markdown("---")

    # Band-integrated image
    st.markdown("#### Band-Integrated Image")
    band_col1, band_col2 = st.columns([3, 1])

    with band_col2:
        band_start = st.number_input(
            "λ start (µm)", min_value=float(wl_min), max_value=float(wl_max),
            value=float(wl_min), step=0.1, format="%.2f",
        )
        band_end = st.number_input(
            "λ end (µm)", min_value=float(wl_min), max_value=float(wl_max),
            value=float(wl_max), step=0.1, format="%.2f",
        )
        band_method = st.selectbox("Method", ["mean", "sum"])

    with band_col1:
        if band_start < band_end:
            try:
                from src.imaging.integrate import integrated_image as _integrated_image
                band_img = _integrated_image(cube, band_start, band_end, method=band_method)
                fig_band = _plot_image(
                    band_img,
                    title=f"Integrated [{band_start:.2f}, {band_end:.2f}] µm ({band_method})",
                    colormap=colormap,
                    return_figure=True,
                )
                st.pyplot(fig_band, use_container_width=True)
                plt.close(fig_band)
            except ValueError as e:
                st.warning(str(e))
        else:
            st.warning("λ start must be less than λ end")

    # Emission line maps for identified lines
    if analysis.peaks.n_peaks > 0:
        identified = [m for m in analysis.line_matches if m is not None]
        if identified:
            st.markdown("---")
            st.markdown("#### Emission Line Maps")

            line_options = {
                f"{m.matched_species} ({m.peak_wavelength_um:.3f} µm)": m
                for m in identified
            }
            selected_line = st.selectbox("Select Line", list(line_options.keys()))

            if selected_line:
                match = line_options[selected_line]
                em_col1, em_col2 = st.columns([3, 1])

                with em_col2:
                    line_w = st.slider("Line Width (µm)", 0.02, 0.5, 0.1, 0.01)
                    cont_w = st.slider("Continuum Width (µm)", 0.05, 1.0, 0.2, 0.05)

                with em_col1:
                    from src.imaging.emission import emission_line_map as _emission_map
                    try:
                        em_img = _emission_map(
                            cube,
                            line_center=match.peak_wavelength_um,
                            line_width=line_w,
                            continuum_width=cont_w,
                        )
                        fig_em = _plot_image(
                            em_img,
                            title=f"{match.matched_species} Emission Map",
                            colormap="RdBu_r",
                            symmetric=True,
                            return_figure=True,
                        )
                        st.pyplot(fig_em, use_container_width=True)
                        plt.close(fig_em)
                    except ValueError as e:
                        st.warning(str(e))


# ===== Tab 3: Summary & Export ==============================================
with tab_summary:
    st.markdown("#### FITS Metadata")

    if cube.header:
        with st.expander("📄 Full Header", expanded=False):
            for key, val in cube.header.items():
                st.text(f"{key:>10s} = {val}")

    st.markdown("---")
    st.markdown("#### Analysis Summary")

    summary_cols = st.columns(3)
    with summary_cols[0]:
        st.metric("Cube Shape", f"{cube.shape}")
        st.metric("Wavelength Range", f"{wl_min:.2f} – {wl_max:.2f} µm")
    with summary_cols[1]:
        st.metric("Pixel", f"({int(px)}, {int(py)})")
        st.metric("Noise RMS", f"{analysis.peaks.noise_rms:.4e}")
    with summary_cols[2]:
        st.metric("Peaks Detected", analysis.peaks.n_peaks)
        n_id = sum(1 for m in analysis.line_matches if m is not None)
        st.metric("Lines Identified", n_id)

    # Download buttons
    st.markdown("---")
    st.markdown("#### 💾 Download Results")

    dl_col1, dl_col2 = st.columns(2)

    with dl_col1:
        # Download peak table as CSV
        if analysis.peaks.n_peaks > 0:
            csv_buffer = io.StringIO()
            writer = csv.writer(csv_buffer)
            writer.writerow([
                "wavelength_um", "snr", "prominence", "width_um",
                "species", "rest_wavelength", "confidence",
                "gaussian_fwhm", "gaussian_flux",
            ])
            for i in range(analysis.peaks.n_peaks):
                match = analysis.line_matches[i] if i < len(analysis.line_matches) else None
                gfit = analysis.gaussian_fits[i] if i < len(analysis.gaussian_fits) else None
                writer.writerow([
                    f"{analysis.peaks.wavelengths[i]:.5f}",
                    f"{analysis.peaks.snr[i]:.2f}",
                    f"{analysis.peaks.prominences[i]:.6e}",
                    f"{analysis.peaks.widths_um[i]:.5f}",
                    match.matched_species if match else "",
                    f"{match.rest_wavelength_um:.5f}" if match else "",
                    f"{match.confidence:.3f}" if match else "",
                    f"{gfit.fwhm:.5f}" if gfit and gfit.fit_success else "",
                    f"{gfit.integrated_flux:.6e}" if gfit and gfit.fit_success else "",
                ])

            st.download_button(
                "📊 Download Peak Table (CSV)",
                csv_buffer.getvalue(),
                file_name="peaks.csv",
                mime="text/csv",
            )

    with dl_col2:
        # Download spectrum plot
        fig_dl = _plot_spectrum(
            wavelength=analysis.wavelength,
            raw=analysis.raw_spectrum,
            processed=analysis.processed_spectrum,
            continuum=analysis.continuum,
            peak_wavelengths=analysis.peaks.wavelengths if analysis.peaks.n_peaks > 0 else None,
            peak_labels=peak_labels if peak_labels else None,
            noise_rms=analysis.peaks.noise_rms,
            title=f"Spectrum at ({int(px)}, {int(py)})",
            return_figure=True,
        )
        buf = io.BytesIO()
        fig_dl.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        plt.close(fig_dl)
        buf.seek(0)

        st.download_button(
            "📈 Download Spectrum Plot (PNG)",
            buf,
            file_name="spectrum.png",
            mime="image/png",
        )

    # Configuration display
    st.markdown("---")
    st.markdown("#### ⚙️ Current Configuration")

    with st.expander("View Config", expanded=False):
        config_items = {
            "SG Window": config.savgol_window,
            "SG Polyorder": config.savgol_polyorder,
            "Continuum Order": config.continuum_poly_order,
            "Sigma Clip": config.sigma_clip,
            "Peak Prominence": config.peak_prominence,
            "Peak Distance": config.peak_distance,
            "Line ID Tolerance": f"{config.tolerance_um} µm",
            "Redshift": config.redshift,
        }
        for k, v in config_items.items():
            st.text(f"{k:>20s}:  {v}")
