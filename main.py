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
        print("═" * 60)
        print("  JWST MIRI IFU Spectral Pipeline — Synthetic Test Mode")
        print("═" * 60)
        from tests.synthetic import generate_synthetic_cube
        cube = generate_synthetic_cube()
        print(f"\n  Cube: {cube}")
    elif args.fits_file:
        filepath = Path(args.fits_file)
        if not filepath.exists():
            print(f"Error: FITS file not found: {filepath}", file=sys.stderr)
            return 1
        print("═" * 60)
        print("  JWST MIRI IFU Spectral Pipeline")
        print("═" * 60)
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
            species = match.matched_species if match else "—"
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
    print("═" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
