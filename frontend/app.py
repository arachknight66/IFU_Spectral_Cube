"""
Streamlit Frontend for JWST MIRI IFU Spectral Pipeline.

End-to-End Guided 7-Stage Workflow Interface:
    1. 🛰️ MAST & Data Source — Search real MAST archive, upload FITS files, or generate test cubes
    2. 📋 Cube Validation & QC — Inspect Phase 3 validation report, WCS status, DQ flags, and pixel quality
    3. 🧭 Alignment & Mosaicking — Align multiple MIRI cubes on a common sky-coordinate grid (Phase 4)
    4. 📊 Spectrum & Line ID — Denoise spectra, fit continuum, detect peaks, and identify atomic/ionic species
    5. 🗺️ Component Maps — Extract continuum-subtracted 2D science maps using physical Δλ integration (Phase 5)
    6. 🎨 NASA False-Colour — Compose reproducible 3-channel false-colour photographs (Phase 6)
    7. 💾 Reproducible Export — One-click export for PNG, 16-bit TIFF, FITS component maps, and JSON manifests
"""

from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from pipeline import AnalysisResult, SpectralPipeline
from src.core.cube import SpectralCube
from src.core.loader import CubeLoadError, load_fits_cube
from src.core.mast import (
    MastDownloadError,
    MastQuery,
    MastQueryError,
    download_mast_product,
    parse_coordinates,
    search_mast_jwst,
)
from src.core.stitch import AlignmentConfig, align_and_stitch_cubes
from src.imaging.component_map import (
    ComponentMap,
    WavelengthCoverageError,
    extract_component_map,
    extract_recipe_component_maps,
)
from src.imaging.recipes import ImageRecipe
from src.utils.config import PipelineConfig
from src.visualization.false_color import FalseColorResult, render_false_color
from src.visualization.image_plot import plot_image
from src.visualization.nasa_plot import render_nasa_photo
from src.visualization.spectra_plot import plot_spectrum

# ===========================================================================
# Page Configuration
# ===========================================================================
st.set_page_config(
    page_title="JWST MIRI IFU Pipeline",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    .stApp {
        background: radial-gradient(circle at 50% 0%, #1a1c29, #0a0b10);
        color: #e2e8f0;
    }
    .pipeline-header {
        background: rgba(20, 22, 37, 0.4);
        backdrop-filter: blur(16px);
        border-radius: 16px;
        padding: 1.8rem 2.2rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(102, 252, 241, 0.15);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
        position: relative;
    }
    .pipeline-header::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%, #66fcf1 100%);
    }
    .pipeline-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        background: -webkit-linear-gradient(45deg, #ffffff, #8bc6ec);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: linear-gradient(145deg, rgba(30, 33, 48, 0.6), rgba(20, 22, 31, 0.8));
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 14px;
        padding: 1rem;
        text-align: center;
    }
    .metric-card .value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #66fcf1;
    }
    .metric-card .label {
        font-size: 0.7rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 4px;
    }
    .false-color-disclaimer {
        background: rgba(255, 171, 0, 0.15);
        border-left: 4px solid #ffab00;
        padding: 0.8rem 1.2rem;
        border-radius: 6px;
        margin: 1rem 0;
        font-size: 0.9rem;
        color: #ffd54f;
    }
</style>
""", unsafe_allow_html=True)

# Header Banner
st.markdown("""
<div class="pipeline-header">
    <h1>🔭 JWST MIRI IFU End-to-End Science Pipeline</h1>
    <p>MAST Discovery • WCS Validation • Uncertainty Alignment • Component Mapping • False-Colour Synthesis</p>
</div>
""", unsafe_allow_html=True)

# ===========================================================================
# Session State Initialization
# ===========================================================================
if "active_cube" not in st.session_state:
    st.session_state["active_cube"] = None
if "loaded_cubes" not in st.session_state:
    st.session_state["loaded_cubes"] = []
if "extracted_comp_maps" not in st.session_state:
    st.session_state["extracted_comp_maps"] = {}
if "false_color_result" not in st.session_state:
    st.session_state["false_color_result"] = None

# Pipeline setup
config = PipelineConfig(output_dir="output")
pipeline = SpectralPipeline(config)

# ===========================================================================
# Main Guided Workflow Tabs
# ===========================================================================
tab_mast, tab_val, tab_align, tab_spec, tab_comp, tab_nasa, tab_export = st.tabs([
    "🛰️ 1. MAST & Data Source",
    "📋 2. Validation & QC",
    "🧭 3. Alignment & Mosaic",
    "📊 4. Spectrum & Line ID",
    "🗺️ 5. Component Maps",
    "🎨 6. NASA False-Colour",
    "💾 7. Reproducible Export",
])

# ===========================================================================
# Tab 1: MAST Search & Data Source
# ===========================================================================
with tab_mast:
    st.markdown("### 🛰️ Step 1: Discover & Load JWST MIRI Datasets")
    st.markdown(
        "Search the Space Telescope Science Institute (STScI) MAST archive for public JWST MIRI IFU (`s3d`) "
        "cubes, upload local FITS files, or initialize synthetic test data."
    )

    data_source_mode = st.radio(
        "Data Source Mode",
        ["🛰️ Search MAST Archive", "🦀 Load Crab Nebula (M1) Reference FITS", "📂 Upload Local FITS", "🧪 Synthetic Test Cube"],
        horizontal=True,
    )

    if data_source_mode == "🛰️ Search MAST Archive":
        st.markdown("#### Search MAST Archive")
        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            query_target = st.text_input("Target Name", value="NGC 7319")
        with sc2:
            query_prop = st.text_input("Proposal ID", value="")
        with sc3:
            query_obs = st.text_input("Observation ID", value="")
        with sc4:
            query_limit = st.number_input("Max Results", 1, 100, 20)

        use_coords = st.checkbox("Search by RA / Dec Coordinates")
        query_ra, query_dec = "", ""
        if use_coords:
            cc1, cc2 = st.columns(2)
            with cc1:
                query_ra = st.text_input("Right Ascension (RA)", value="339.967")
            with cc2:
                query_dec = st.text_input("Declination (Dec)", value="33.963")

        if st.button("🔍 Query MAST Archive", type="primary"):
            with st.spinner("Querying STScI MAST Archive API..."):
                try:
                    if use_coords and query_ra and query_dec:
                        results = search_mast_jwst(ra=query_ra, dec=query_dec, limit=query_limit)
                    else:
                        results = search_mast_jwst(
                            target_name=query_target or None,
                            proposal_id=query_prop or None,
                            observation_id=query_obs or None,
                            limit=query_limit,
                        )
                    st.session_state["mast_search_results"] = results
                    if not results:
                        st.warning("No matching MIRI IFU products found in MAST.")
                    else:
                        st.success(f"Found {len(results)} normalized MAST MIRI product(s)!")
                except Exception as exc:
                    st.error(f"MAST Query Error: {exc}")

        # Display Search Results
        results = st.session_state.get("mast_search_results", [])
        if results:
            st.markdown("##### Real MAST Products Table")
            display_rows = []
            for idx, r in enumerate(results):
                display_rows.append({
                    "Index": idx,
                    "Product Filename": r["product_filename"],
                    "Submode": r["submode"],
                    "Target": r["target_name"],
                    "Proposal": r["proposal_id"],
                    "Status": r["release_status"],
                    "Size (MB)": f"{r['size_bytes'] / 1e6:.1f}" if r.get("size_bytes") else "-",
                })
            st.dataframe(display_rows, use_container_width=True)

            sel_idx = st.selectbox("Select Product to Download & Load", [r["Index"] for r in display_rows])
            if st.button("📥 Download & Validate Product", type="primary"):
                sel_rec = results[sel_idx]
                with st.spinner(f"Downloading product `{sel_rec['product_filename']}`..."):
                    try:
                        cached_p = download_mast_product(sel_rec, "data/raw")
                        cube = load_fits_cube(cached_p)
                        st.session_state["active_cube"] = cube
                        st.session_state["loaded_cubes"] = [cube]
                        st.success(f"Loaded FITS Cube `{cached_p.name}`! Proceed to Step 2.")
                    except Exception as exc:
                        st.error(f"MAST Download Error: {exc}")

    elif data_source_mode == "🦀 Load Crab Nebula (M1) Reference FITS":
        if st.button("🦀 Load Crab Nebula (M1) Composite Cube", type="primary"):
            from scripts.generate_crab_fits import create_crab_nebula_fits_file
            crab_p = create_crab_nebula_fits_file("data/crab_nebula_miri_composite.fits")
            cube = load_fits_cube(crab_p)
            st.session_state["active_cube"] = cube
            st.session_state["loaded_cubes"] = [cube]
            st.success("Loaded Crab Nebula (M1) reference FITS cube! Proceed to Step 2.")

    elif data_source_mode == "📂 Upload Local FITS":
        up_files = st.file_uploader("Upload FITS Cube(s)", type=["fits", "fits.gz"], accept_multiple_files=True)
        if up_files and st.button("📂 Process Uploaded FITS"):
            loaded = []
            out_dir = Path("data/raw")
            out_dir.mkdir(parents=True, exist_ok=True)
            for uf in up_files:
                save_p = out_dir / uf.name
                save_p.write_bytes(uf.getvalue())
                cube = load_fits_cube(save_p)
                loaded.append(cube)
            if loaded:
                st.session_state["active_cube"] = loaded[0]
                st.session_state["loaded_cubes"] = loaded
                st.success(f"Loaded {len(loaded)} FITS cube(s)! Proceed to Step 2.")

    elif data_source_mode == "🧪 Synthetic Test Cube":
        if st.button("🧪 Generate & Load Synthetic Cube", type="primary"):
            from tests.synthetic import generate_synthetic_cube
            cube = generate_synthetic_cube()
            st.session_state["active_cube"] = cube
            st.session_state["loaded_cubes"] = [cube]
            st.success("Loaded Synthetic test cube! Proceed to Step 2.")

# ===========================================================================
# Tab 2: FITS Validation & Quality Control
# ===========================================================================
with tab_val:
    st.markdown("### 📋 Step 2: Phase 3 Cube Quality & WCS Validation")

    cube: SpectralCube | None = st.session_state.get("active_cube")
    if cube is None:
        st.warning("⚠️ No active cube loaded. Please complete Step 1 first.")
    else:
        st.markdown(f"**Loaded Cube:** `{cube.filepath}`")
        report = cube.validation_report

        # Summary Metrics Bar
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.markdown(f"""<div class="metric-card"><div class="value">{cube.n_wavelengths}</div><div class="label">Spectral Channels</div></div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""<div class="metric-card"><div class="value">{cube.spatial_shape[0]}×{cube.spatial_shape[1]}</div><div class="label">Spatial Grid</div></div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""<div class="metric-card"><div class="value">{cube.wavelength_range[0]:.2f}–{cube.wavelength_range[1]:.2f}</div><div class="label">λ Range (µm)</div></div>""", unsafe_allow_html=True)
        with m4:
            st.markdown(f"""<div class="metric-card"><div class="value">{cube.flux_unit}</div><div class="label">Flux Unit</div></div>""", unsafe_allow_html=True)
        with m5:
            st.markdown(f"""<div class="metric-card"><div class="value">{report.wcs_status}</div><div class="label">WCS Status</div></div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### Quality Audit Report")
        if report.wcs_status == "FATAL":
            st.error("❌ Fatal WCS Error: Valid 3D celestial WCS coordinates are unavailable. Downstream alignment blocked.")
        else:
            st.success(f"✓ WCS Status: `{report.wcs_status}` | Valid Pixels: `{report.valid_pixel_count}` | DQ Flagged: `{report.dq_flagged_pixel_count}`")

        if report.warnings:
            with st.expander("⚠️ Validation Warnings", expanded=True):
                for w in report.warnings:
                    st.warning(w)

        with st.expander("📄 Full FITS Header"):
            if cube.header:
                for k, v in cube.header.items():
                    st.text(f"{k:>12s} = {v}")

# ===========================================================================
# Tab 3: Alignment & Mosaicking
# ===========================================================================
with tab_align:
    st.markdown("### 🧭 Step 3: Multi-Cube Spatial & WCS Alignment")

    loaded_cubes = st.session_state.get("loaded_cubes", [])
    if len(loaded_cubes) < 2:
        st.info("ℹ️ Single cube active. Multi-cube spatial reprojection is optional.")
        if loaded_cubes:
            st.success("Single cube is ready for component extraction.")
    else:
        st.markdown(f"**Loaded Cubes for Mosaicking:** `{len(loaded_cubes)}` cubes")

        c_opt1, c_opt2, c_opt3 = st.columns(3)
        with c_opt1:
            align_ref = st.selectbox("Reference Grid", ["union", "ref_0"])
        with c_opt2:
            overlap_pol = st.selectbox("Overlap Policy", ["combine_overlaps", "resample_common"])
        with c_opt3:
            interp_mode = st.selectbox("Interpolation Mode", ["bilinear", "nearest"])

        if st.button("🚀 Align & Stitch Cubes", type="primary"):
            with st.spinner("Aligning celestial WCS grids & inverse-variance weighting..."):
                try:
                    align_cfg = AlignmentConfig(
                        reference=align_ref,
                        overlap_policy=overlap_pol,
                        interpolation=interp_mode,
                    )
                    align_res = pipeline.align_cubes(loaded_cubes, config=align_cfg)
                    st.session_state["active_cube"] = align_res.aligned_cube
                    st.session_state["aligned_cube"] = align_res.aligned_cube
                    st.success(f"Successfully aligned into target shape `{align_res.aligned_cube.shape}`!")
                except Exception as exc:
                    st.error(f"Alignment Error: {exc}")

# ===========================================================================
# Tab 4: Spectrum & Line Identification
# ===========================================================================
with tab_spec:
    st.markdown("### 📊 Step 4: Pixel Spectrum Analysis & Peak Detection")

    cube = st.session_state.get("active_cube")
    if cube is None:
        st.warning("⚠️ No active cube. Complete Step 1 & 2 first.")
    else:
        ny, nx = cube.spatial_shape
        sp1, sp2 = st.columns(2)
        with sp1:
            px = st.number_input("Pixel X", 0, nx - 1, nx // 2)
        with sp2:
            py = st.number_input("Pixel Y", 0, ny - 1, ny // 2)

        analysis = pipeline.analyze(cube, int(px), int(py))

        peak_labels = [m.matched_species if m else "" for m in analysis.line_matches]
        fig_spec = plot_spectrum(
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
        st.pyplot(fig_spec, use_container_width=True)
        plt.close(fig_spec)

# ===========================================================================
# Tab 5: Component Maps
# ===========================================================================
with tab_comp:
    st.markdown("### 🗺️ Step 5: Spectral Component Map Creation")

    cube = st.session_state.get("active_cube")
    if cube is None:
        st.warning("⚠️ No active cube. Complete Step 1 & 2 first.")
    else:
        recipes_dir = Path("src/imaging/recipes")
        recipe_files = sorted(recipes_dir.glob("*.json")) if recipes_dir.exists() else []

        if recipe_files:
            r1, r2 = st.columns([2, 1])
            with r1:
                sel_rec_p = st.selectbox("Select Image Recipe Preset", recipe_files, format_func=lambda p: p.name)
            with r2:
                allow_one_sided = st.checkbox("Allow One-Sided Continuum Fallback", value=True)

            if sel_rec_p:
                recipe = ImageRecipe.load(sel_rec_p)
                st.session_state["active_recipe"] = recipe
                st.markdown(f"**Target:** *{recipe.target_name}* | **Mapping Note:** {recipe.mapping_note}")

                # Coverage check
                wl_min, wl_max = cube.wavelength_range
                cov_cols = st.columns(3)
                for idx, channel in enumerate(recipe.channels):
                    ch_c = channel.central_wavelength_um
                    is_cov = (wl_min <= ch_c <= wl_max)
                    with cov_cols[idx]:
                        if is_cov:
                            st.success(f"**{channel.display_colour.upper()}**: {channel.feature_name} ({ch_c:.2f} µm) ✓")
                        else:
                            st.warning(f"**{channel.display_colour.upper()}**: {channel.feature_name} ({ch_c:.2f} µm) ⚠️ Out of range")

                if st.button("🚀 Extract Component Maps", type="primary", use_container_width=True):
                    with st.spinner("Performing physical Δλ integration & local continuum linear fitting..."):
                        try:
                            cmaps = pipeline.extract_recipe_component_maps(cube, recipe, allow_one_sided_continuum=allow_one_sided)
                            st.session_state["extracted_comp_maps"] = cmaps
                            st.success(f"Extracted {len(cmaps)} component maps!")
                        except Exception as exc:
                            st.error(f"Component Extraction Error: {exc}")

                cmaps = st.session_state.get("extracted_comp_maps", {})
                if cmaps:
                    st.markdown("---")
                    st.markdown("#### Component Map Previews & Diagnostics")
                    sel_color = st.radio("Channel Display", ["red", "green", "blue"], horizontal=True)
                    if sel_color in cmaps:
                        cmap: ComponentMap = cmaps[sel_color]
                        st.caption(f"Feature: **{cmap.component_name}** | Unit: `{cmap.unit}` | Authoritative Error: `{cmap.has_authoritative_uncertainty}`")

                        p1, p2, p3, p4 = st.columns(4)
                        with p1:
                            fig1 = plot_image(cmap.on_band_flux, title="Raw On-Band Flux", colormap="viridis", return_figure=True)
                            st.pyplot(fig1, use_container_width=True)
                            plt.close(fig1)
                        with p2:
                            fig2 = plot_image(cmap.continuum_map, title="Estimated Continuum", colormap="magma", return_figure=True)
                            st.pyplot(fig2, use_container_width=True)
                            plt.close(fig2)
                        with p3:
                            fig3 = plot_image(cmap.data, title="Subtracted Feature Map", colormap="RdBu_r", symmetric=True, return_figure=True)
                            st.pyplot(fig3, use_container_width=True)
                            plt.close(fig3)
                        with p4:
                            if cmap.snr is not None:
                                fig4 = plot_image(cmap.snr, title="Signal-to-Noise Ratio (SNR)", colormap="plasma", return_figure=True)
                                st.pyplot(fig4, use_container_width=True)
                                plt.close(fig4)
                            else:
                                st.info("No SNR Map")

                        if st.button(f"💾 Export {sel_color.upper()} FITS + JSON Sidecar"):
                            out_p = Path("output") / f"component_{recipe.name}_{sel_color}.fits"
                            saved_p = cmap.save_fits(out_p, export_sidecar=True)
                            st.success(f"Exported FITS: `{saved_p}` and `{saved_p}.provenance.json`!")

# ===========================================================================
# Tab 6: NASA False-Colour Composition
# ===========================================================================
with tab_nasa:
    st.markdown("### 🎨 Step 6: NASA-Style False-Colour Composition")

    st.markdown("""
    <div class="false-color-disclaimer">
        ⚠️ <strong>SCIENTIFIC FALSE-COLOUR DISCLAIMER:</strong> This rendering maps mid-infrared line features 
        ([S III], [Ne II], PAH dust, etc.) to visible Red, Green, Blue channels. It represents physical chemical distribution, 
        <strong>NOT human visible-light color</strong>.
    </div>
    """, unsafe_allow_html=True)

    cmaps = st.session_state.get("extracted_comp_maps", {})
    recipe = st.session_state.get("active_recipe")

    if not cmaps or len(cmaps) != 3:
        st.warning("⚠️ Requires 3 extracted Phase-5 Component Maps. Complete Step 5 first.")
    else:
        fc1, fc2, fc3, fc4 = st.columns(4)
        with fc1:
            render_mode = st.radio("Rendering Mode", ["scientific", "presentation"], horizontal=True)
        with fc2:
            stretch_choice = st.selectbox("Intensity Stretch", ["asinh", "log", "linear"])
        with fc3:
            p_low = st.number_input("Percentile Low", 0.0, 10.0, 1.0, 0.5)
        with fc4:
            p_high = st.number_input("Percentile High", 90.0, 100.0, 99.5, 0.5)

        bg_method = st.selectbox("Background Subtraction Method", ["none", "per_channel_median", "global_median"])

        if st.button("🎨 Render False-Colour Photograph", type="primary", use_container_width=True):
            with st.spinner("Synthesizing false-colour composite..."):
                try:
                    res = render_false_color(
                        cmaps,
                        recipe=recipe or ImageRecipe.load(Path("src/imaging/recipes/miri_emission_line_rgb.json")),
                        rendering_mode=render_mode,
                        rendering_overrides={
                            "stretch": stretch_choice,
                            "percentile_clip": (p_low, p_high),
                            "background_subtraction": bg_method,
                        },
                    )
                    st.session_state["false_color_result"] = res
                    st.success("Rendered false-colour composite!")
                except Exception as exc:
                    st.error(f"Rendering Error: {exc}")

        res: FalseColorResult | None = st.session_state.get("false_color_result")
        if res is not None:
            fig_photo = render_nasa_photo(
                res.rgb_array,
                title=f"JWST MIRI MRS — {res.recipe.target_name.upper()}",
                target_name="JWST MIRI MRS IFU",
                channel_labels={
                    "red": res.channel_stats["red"]["component_name"],
                    "green": res.channel_stats["green"]["component_name"],
                    "blue": res.channel_stats["blue"]["component_name"],
                },
            )
            st.pyplot(fig_photo, use_container_width=True)
            plt.close(fig_photo)

# ===========================================================================
# Tab 7: Reproducible Export & Provenance
# ===========================================================================
with tab_export:
    st.markdown("### 💾 Step 7: Reproducible Export & Provenance Manifests")

    res: FalseColorResult | None = st.session_state.get("false_color_result")
    cmaps = st.session_state.get("extracted_comp_maps", {})

    if res is None:
        st.warning("⚠️ No rendered false-colour product available. Complete Step 6 first.")
    else:
        st.markdown("#### Download Reproducible Products")
        ex1, ex2 = st.columns(2)

        with ex1:
            if st.button("💾 Export 8-Bit PNG + JSON Manifest Sidecar", use_container_width=True):
                out_p = Path("output") / f"false_colour_{res.recipe.name}_{res.rendering_mode}.png"
                saved_png = res.save_png(out_p, overwrite=True)
                st.success(f"Exported PNG: `{saved_png}` and `{saved_png}.manifest.json`!")

        with ex2:
            if st.button("💾 Export Publication-Grade 16-Bit TIFF + Manifest", use_container_width=True):
                out_t = Path("output") / f"false_colour_{res.recipe.name}_{res.rendering_mode}.tiff"
                saved_t = res.save_tiff(out_t, bit_depth=16, overwrite=True)
                st.success(f"Exported 16-bit TIFF: `{saved_t}` and `{saved_t}.manifest.json`!")

        st.markdown("---")
        st.markdown("#### Complete Execution Provenance")
        st.json(res.to_dict())
