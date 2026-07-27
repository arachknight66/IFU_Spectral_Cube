"""
CLI Entry Point for the JWST MIRI IFU Spectral Pipeline.

Usage
-----
    python main.py data/raw/cube.fits --x 15 --y 20
    python main.py cube.fits -x 10 -y 10 --savgol-window 13 --prominence 0.1
    python main.py --align-files cube1.fits cube2.fits --reference union
    python main.py render --recipe recipes/miri_emission_line_rgb.json --red red.fits --green green.fits --blue blue.fits --output false_colour.png --mode presentation
    python main.py --synthetic  # Run on synthetic test data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="jwst-ifu-pipeline",
        description=(
            "JWST MIRI IFU Spectral Analysis Pipeline — "
            "Processes FITS spectral cubes through alignment, denoising, "
            "peak detection, line identification, component mapping, and false-colour rendering."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py data/raw/cube.fits --x 15 --y 20
  python main.py --align-files cube1.fits cube2.fits --reference union
  python main.py render --recipe recipes/miri_emission_line_rgb.json --red red.fits --green green.fits --blue blue.fits --mode presentation
  python main.py --synthetic --x 15 --y 15
        """,
    )

    # Allow optional 'render' command as positional argument
    parser.add_argument(
        "command_or_fits",
        nargs="?",
        help="Command ('render') or path to FITS data cube.",
    )

    # Phase 6 False-Colour Rendering
    parser.add_argument("--recipe", type=str, default="", help="Path to ImageRecipe JSON file.")
    parser.add_argument("--red", type=str, default="", help="Path to Red FITS component map.")
    parser.add_argument("--green", type=str, default="", help="Path to Green FITS component map.")
    parser.add_argument("--blue", type=str, default="", help="Path to Blue FITS component map.")
    parser.add_argument("--mode", type=str, default="scientific", choices=["scientific", "presentation"], help="Rendering mode ('scientific' or 'presentation').")
    parser.add_argument("--render-output", type=str, default="", help="Output image file path (.png or .tiff).")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing output files.")

    # Multi-cube alignment
    parser.add_argument(
        "--align-files", nargs="+", default=[],
        help="Multiple FITS file paths to align, reproject, and stitch.",
    )
    parser.add_argument(
        "--reference", type=str, default="union",
        help="Reference celestial grid for alignment: 'union' (default) or 'ref_0'",
    )
    parser.add_argument(
        "--overlap-policy", type=str, default="combine_overlaps",
        help="Wavelength overlap policy: 'combine_overlaps' (default) or 'resample_common'",
    )
    parser.add_argument(
        "--interpolation", type=str, default="bilinear",
        help="Spatial reprojection interpolation: 'bilinear' (default) or 'nearest'",
    )

    # Synthetic mode
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use a synthetic test cube instead of a FITS file.",
    )

    # MAST Archive mode
    parser.add_argument(
        "--mast-target", type=str, default="",
        help="Search and download JWST dataset from MAST by target name (e.g., 'NGC 7319')",
    )
    parser.add_argument(
        "--mast-proposal", type=str, default="",
        help="Search and download JWST dataset from MAST by Proposal/Program ID (e.g., '1288')",
    )
    parser.add_argument("--mast-observation", type=str, default="", help="Search MAST by exact observation ID")
    parser.add_argument("--mast-limit", type=int, default=25, help="Maximum normalized MAST products to list")
    parser.add_argument("--mast-select", type=int, default=None, help="Zero-based matching product index to download")
    parser.add_argument("--mast-cache", type=str, default="data/raw", help="Verified MAST download cache directory")
    parser.add_argument(
        "--ra", type=str, default="",
        help="Search MAST by Right Ascension (decimal degrees or sexagesimal e.g. '339.967' or '22h39m52s')",
    )
    parser.add_argument(
        "--dec", type=str, default="",
        help="Search MAST by Declination (decimal degrees or sexagesimal e.g. '33.963' or '+33d57m46s')",
    )
    parser.add_argument(
        "--radius-arcsec", type=float, default=15.0,
        help="Search radius in arcseconds for MAST RA/Dec cone search (default: 15.0)",
    )

    # Pixel coordinates
    parser.add_argument("-x", "--x", type=int, default=0, help="X pixel coordinate (default: 0)")
    parser.add_argument("-y", "--y", type=int, default=0, help="Y pixel coordinate (default: 0)")

    # Preprocessing
    parser.add_argument("--savgol-window", type=int, default=11, help="Savitzky-Golay window (odd, default: 11)")
    parser.add_argument("--savgol-polyorder", type=int, default=3, help="SG polynomial order (default: 3)")
    parser.add_argument("--continuum-order", type=int, default=3, help="Continuum polynomial order (default: 3)")
    parser.add_argument("--sigma-clip", type=float, default=3.0, help="Sigma clipping for continuum (default: 3.0)")

    # Peak detection
    parser.add_argument("--prominence", type=float, default=3.0, help="Detection sigma threshold (default: 3.0)")
    parser.add_argument("--peak-distance", type=int, default=5, help="Min peak distance in channels (default: 5)")

    # Line ID
    parser.add_argument("--tolerance", type=float, default=0.05, help="Line ID tolerance in µm (default: 0.05)")
    parser.add_argument("--redshift", "-z", type=float, default=0.0, help="Source redshift (default: 0.0)")

    # Phase 5 Component Maps
    parser.add_argument(
        "--extract-recipe", type=str, default="",
        help="Path to JSON ImageRecipe file (e.g., 'src/imaging/recipes/miri_emission_line_rgb.json') to extract component maps.",
    )
    parser.add_argument(
        "--export-maps", action="store_true",
        help="Export extracted component maps to FITS + JSON sidecars in output directory.",
    )

    # Output
    parser.add_argument("-o", "--output", type=str, default="output", help="Output directory (default: output/)")

    return parser.parse_args(argv)


def _load_component_map_from_fits(fits_path: Path, channel_color: str) -> Any:
    """Helper to reload a ComponentMap object from a Phase 5 multi-extension FITS file."""
    from src.imaging.component_map import ComponentMap

    with fits.open(fits_path) as hdul:
        primary_hdr = hdul[0].header
        comp_name = primary_hdr.get("COMPONENT", fits_path.stem)
        unit_str = primary_hdr.get("BUNIT", "MJy/sr")
        has_auth_err = bool(primary_hdr.get("AUTH_ERR", False))

        wcs_2d = WCS(hdul["SCI"].header) if "SCI" in hdul else WCS(primary_hdr)

        sci_data = hdul["SCI"].data if "SCI" in hdul else hdul[0].data
        err_data = hdul["ERR"].data if "ERR" in hdul else None
        snr_data = hdul["SNR"].data if "SNR" in hdul else None
        dq_data = hdul["DQ"].data if "DQ" in hdul else None
        cov_data = hdul["COV"].data if "COV" in hdul else np.ones_like(sci_data, dtype=int)
        onband_data = hdul["ONBAND"].data if "ONBAND" in hdul else sci_data
        cont_data = hdul["CONTINUUM"].data if "CONTINUUM" in hdul else np.zeros_like(sci_data)

    provenance = {
        "action": "loaded_from_fits",
        "source_filepath": str(fits_path),
        "component_name": comp_name,
        "recipe_channel": channel_color,
    }

    return ComponentMap(
        data=sci_data,
        uncertainty=err_data,
        snr=snr_data,
        dq=dq_data,
        coverage=cov_data,
        wcs=wcs_2d,
        unit=unit_str,
        component_name=comp_name,
        recipe_channel=channel_color,
        on_band_flux=onband_data,
        continuum_map=cont_data,
        has_authoritative_uncertainty=has_auth_err,
        provenance=provenance,
    )


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args(argv)

    # Lazy import to keep argparse fast
    from src.utils.config import PipelineConfig
    from pipeline import SpectralPipeline
    from src.core.stitch import AlignmentConfig

    # Build config from CLI args
    config = PipelineConfig(
        savgol_window=args.savgol_window,
        savgol_polyorder=args.savgol_polyorder,
        continuum_poly_order=args.continuum_order,
        sigma_clip=args.sigma_clip,
        peak_prominence=args.prominence,
        peak_distance=args.peak_distance,
        tolerance_um=args.tolerance,
        redshift=args.redshift,
        output_dir=args.output,
    )

    pipeline = SpectralPipeline(config)

    # Phase 6 Render Subcommand / Command Mode
    if args.command_or_fits == "render" or (args.recipe and args.red and args.green and args.blue):
        print("=" * 60)
        print("  JWST MIRI IFU Spectral Pipeline — Phase 6 False-Colour Render")
        print("=" * 60)

        recipe_p = Path(args.recipe)
        red_p = Path(args.red)
        green_p = Path(args.green)
        blue_p = Path(args.blue)

        if not recipe_p.exists():
            print(f"Error: Recipe file not found: {recipe_p}", file=sys.stderr)
            return 1
        for color_name, p in [("Red", red_p), ("Green", green_p), ("Blue", blue_p)]:
            if not p.exists():
                print(f"Error: {color_name} component map FITS file not found: {p}", file=sys.stderr)
                return 1

        from src.imaging.recipes import ImageRecipe
        recipe = ImageRecipe.load(recipe_p)

        print(f"\n  Loading Component Maps...")
        c_red = _load_component_map_from_fits(red_p, "red")
        c_green = _load_component_map_from_fits(green_p, "green")
        c_blue = _load_component_map_from_fits(blue_p, "blue")

        print(f"  Rendering Mode: {args.mode.upper()}")
        render_res = pipeline.render_false_color(
            component_maps={"red": c_red, "green": c_green, "blue": c_blue},
            recipe=recipe,
            rendering_mode=args.mode,
            overwrite=args.overwrite,
        )

        out_img_path = Path(args.render_output or args.output)
        if out_img_path.is_dir() or not out_img_path.suffix:
            out_img_path = out_img_path / f"false_colour_{recipe.name}_{args.mode}.png"

        saved_img = render_res.save_png(out_img_path, overwrite=args.overwrite)
        print(f"  Saved False-Colour Image: {saved_img}")
        print(f"  Saved Manifest Sidecar: {saved_img}.manifest.json")
        return 0

    if args.align_files:
        print("=" * 60)
        print("  JWST MIRI IFU Spectral Pipeline — Alignment Mode")
        print("=" * 60)
        align_config = AlignmentConfig(
            reference=args.reference,
            overlap_policy=args.overlap_policy,
            interpolation=args.interpolation,
        )
        print(f"\n  Aligning {len(args.align_files)} cubes...")
        print(f"  Reference: {args.reference}")
        print(f"  Overlap Policy: {args.overlap_policy}")
        align_res = pipeline.align_cubes([pipeline.load(f) for f in args.align_files], config=align_config)
        cube = align_res.aligned_cube
        print(f"  Result Shape: {cube.shape}")
        print(f"  Wavelength: {cube.wavelength_range[0]:.4f} - {cube.wavelength_range[1]:.4f} µm")
        print(f"  WCS: {align_res.output_wcs.wcs.ctype}")
    elif args.synthetic:
        print("=" * 60)
        print("  JWST MIRI IFU Spectral Pipeline — Synthetic Test Mode")
        print("=" * 60)
        from tests.synthetic import generate_synthetic_cube
        cube = generate_synthetic_cube()
        print(f"\n  Cube: {cube}")
    elif args.mast_target or args.mast_proposal or args.mast_observation or (args.ra and args.dec):
        print("=" * 60)
        print("  JWST MIRI IFU Spectral Pipeline — MAST Archive Mode")
        print("=" * 60)
        from src.core.mast import MastDownloadError, MastQuery, MastQueryError, ProductSelectionError, download_mast_product, parse_coordinates, search_mast_jwst
        try:
            if args.ra and args.dec:
                print(f"\n  Searching MAST for RA={args.ra}, Dec={args.dec} (Radius={args.radius_arcsec}\")...")
                records = search_mast_jwst(ra=args.ra, dec=args.dec, radius_arcsec=args.radius_arcsec, limit=args.mast_limit)
            else:
                print(f"\n  Searching MAST for Target='{args.mast_target}', Proposal='{args.mast_proposal}'...")
                records = search_mast_jwst(target_name=args.mast_target, proposal_id=args.mast_proposal or None, observation_id=args.mast_observation or None, limit=args.mast_limit)
        except (MastQueryError, ValueError) as exc:
            print(f"MAST search error: {exc}", file=sys.stderr)
            return 1

        if not records:
            print("Error: No MAST observations found matching search parameters.", file=sys.stderr)
            return 1
        for index, rec in enumerate(records):
            print(f"  [{index}] {rec['obs_id']} | {rec['submode']} | {rec['product_filename']} | {rec['release_status']}")
        if args.mast_select is None:
            print("Select a product with --mast-select INDEX to download; archive mode does not process cubes yet.")
            return 0
        if not 0 <= args.mast_select < len(records):
            print("Error: --mast-select is outside the listed product range.", file=sys.stderr)
            return 2
        rec = records[args.mast_select]
        try:
            query_args = {"target_name": args.mast_target or None, "proposal_id": args.mast_proposal or None, "observation_id": args.mast_observation or None, "radius_arcsec": args.radius_arcsec, "limit": args.mast_limit}
            if args.ra and args.dec:
                query_args["ra_deg"], query_args["dec_deg"] = parse_coordinates(args.ra, args.dec)
            query = MastQuery(**query_args)
            local_path = download_mast_product(rec, args.mast_cache, query=query)
        except (MastDownloadError, ProductSelectionError, ValueError) as exc:
            print(f"MAST download error: {exc}", file=sys.stderr)
            return 1
        print(f"Downloaded verified FITS: {local_path}")
        print(f"Provenance: {local_path.name}.provenance.json")
        try:
            from src.core.loader import CubeLoadError, load_fits_cube
            validated_cube = load_fits_cube(local_path)
            report = validated_cube.validation_report
            print(f"Validation: {report.wcs_status}; shape={report.shape}; "
                  f"wavelength={report.wavelength_range_um[0]:.5g}-{report.wavelength_range_um[1]:.5g} um; "
                  f"invalid={report.invalid_pixel_count}; DQ-flagged={report.dq_flagged_pixel_count}")
            for warning in report.warnings:
                print(f"Validation warning: {warning}")
        except CubeLoadError as exc:
            print(f"FITS validation error: {exc}", file=sys.stderr)
            return 1
        return 0
    elif args.command_or_fits and Path(args.command_or_fits).exists():
        filepath = Path(args.command_or_fits)
        print("=" * 60)
        print("  JWST MIRI IFU Spectral Pipeline")
        print("=" * 60)
        print(f"\n  Loading: {filepath}")
        cube = pipeline.load(str(filepath))
        print(f"  Cube: {cube}")
    else:
        print("Error: Provide a FITS file, 'render' command, --align-files, or use --synthetic", file=sys.stderr)
        return 1

    # Validate pixel coordinates
    ny, nx = cube.spatial_shape
    x, y = args.x, args.y
    if not (0 <= x < nx and 0 <= y < ny):
        print(f"\n  Warning: pixel ({x}, {y}) out of bounds "
              f"(nx={nx}, ny={ny}). Using (0, 0).")
        x, y = 0, 0

    print(f"  Pixel: ({x}, {y})")
    print(f"  Output: {config.output_dir}/")
    print()

    # Extract Recipe Component Maps if requested
    if args.extract_recipe:
        from src.imaging.recipes import ImageRecipe
        recipe_path = Path(args.extract_recipe)
        if recipe_path.exists():
            print(f"\n  [Phase 5] Extracting component maps for recipe: {recipe_path.name}...")
            recipe = ImageRecipe.load(recipe_path)
            try:
                comp_maps = pipeline.extract_recipe_component_maps(cube, recipe, allow_one_sided_continuum=True)
                print(f"         Extracted {len(comp_maps)} channel maps: {list(comp_maps.keys())}")
                if args.export_maps:
                    out_dir = Path(config.output_dir)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    for ch_name, cmap in comp_maps.items():
                        fits_p = out_dir / f"component_{recipe.name}_{ch_name}.fits"
                        saved = cmap.save_fits(fits_p)
                        print(f"         Saved component map: {saved}")
            except Exception as exc:
                print(f"         Component extraction error: {exc}", file=sys.stderr)

    # Generate images
    print("\n  [2/4] Generating image products...")
    images = pipeline.generate_images(cube, analysis.line_matches)
    print(f"         Generated {len(images)} images")

    # Run analysis
    print("  [1/4] Extracting and preprocessing spectrum...")
    analysis = pipeline.analyze(cube, x, y)
    print(f"         Noise RMS: {analysis.peaks.noise_rms:.4e}")
    print(f"         Peaks detected: {analysis.peaks.n_peaks}")

    n_id = sum(1 for m in analysis.line_matches if m is not None)
    print(f"         Lines identified: {n_id}")

    if analysis.peaks.n_peaks > 0:
        print(f"\n  {'#':>3}  {'λ (µm)':>10}  {'SNR':>7}  {'Species':>15}")
        print("  " + "-" * 45)
        for i in range(analysis.peaks.n_peaks):
            match = analysis.line_matches[i] if i < len(analysis.line_matches) else None
            species = match.matched_species if match else "-"
            print(f"  {i+1:>3}  {analysis.peaks.wavelengths[i]:>10.4f}  "
                  f"{analysis.peaks.snr[i]:>7.1f}  {species:>15}")

    # Build result
    from pipeline import PipelineResult
    result = PipelineResult(
        cube=cube,
        analysis=analysis,
        images=images,
        config=config,
    )

    # Save
    print("\n  [3/4] Saving results...")
    out_path = pipeline.save_results(result)
    print(f"         Saved to: {out_path}/")

    print("\n  [4/4] Done!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
