"""
CLI Entry Point for the JWST MIRI IFU Spectral Pipeline.

Usage
-----
    python main.py data/raw/cube.fits --x 15 --y 20
    python main.py cube.fits -x 10 -y 10 --savgol-window 13 --prominence 0.1
    python main.py cube.fits -x 5 -y 5 -o results/ --redshift 0.003
    python main.py --synthetic  # Run on synthetic test data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="jwst-ifu-pipeline",
        description=(
            "JWST MIRI IFU Spectral Analysis Pipeline — "
            "Processes FITS spectral cubes through denoising, "
            "peak detection, line identification, and imaging."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py data/raw/cube.fits --x 15 --y 20
  python main.py cube.fits -x 10 -y 10 --savgol-window 13
  python main.py --synthetic --x 15 --y 15
        """,
    )

    # Positional / required
    parser.add_argument(
        "fits_file",
        nargs="?",
        help="Path to the FITS data cube.",
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

    # Output
    parser.add_argument("-o", "--output", type=str, default="output", help="Output directory (default: output/)")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args(argv)

    # Lazy import to keep argparse fast
    from src.utils.config import PipelineConfig
    from pipeline import SpectralPipeline

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

    if args.synthetic:
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
    elif args.fits_file:
        filepath = Path(args.fits_file)
        if not filepath.exists():
            print(f"Error: FITS file not found: {filepath}", file=sys.stderr)
            return 1
        print("=" * 60)
        print("  JWST MIRI IFU Spectral Pipeline")
        print("=" * 60)
        print(f"\n  Loading: {filepath}")
        cube = pipeline.load(str(filepath))
        print(f"  Cube: {cube}")
    else:
        print("Error: Provide a FITS file or use --synthetic", file=sys.stderr)
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

    # Generate images
    print("\n  [2/4] Generating image products...")
    images = pipeline.generate_images(cube, analysis.line_matches)
    print(f"         Generated {len(images)} images")

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
