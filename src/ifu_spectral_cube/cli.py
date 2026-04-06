"""Command-line interface for the JWST MIRI IFU pipeline."""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace

from .config import (
    ExtractionConfig,
    FitConfig,
    IdentificationConfig,
    PeakConfig,
    PipelineConfig,
    PreprocessConfig,
    VisualizationConfig,
    load_pipeline_config,
)
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="JWST MIRI IFU spectral line identification pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick run with defaults
  ifu-line-id --cube /path/to/cube.fits --output-dir results

  # Full config file
  ifu-line-id --config config.yaml

  # Override config from CLI
  ifu-line-id --config config.yaml --prominence-sigma 5.0 --try-voigt
""",
    )

    # Input/output
    p.add_argument("--config", type=str, help="YAML configuration file")
    p.add_argument("--cube", type=str, help="Input FITS cube path")
    p.add_argument("--output-dir", type=str, default=None, help="Output directory")
    p.add_argument("--wmin", type=float, default=None, help="Minimum wavelength (µm)")
    p.add_argument("--wmax", type=float, default=None, help="Maximum wavelength (µm)")

    # Extraction
    p.add_argument("--x", type=float, default=None, help="Aperture center x (spaxels)")
    p.add_argument("--y", type=float, default=None, help="Aperture center y (spaxels)")
    p.add_argument("--radius", type=float, default=2.5, help="Aperture radius (spaxels)")
    p.add_argument("--mode", type=str, default="circular",
                   choices=["circular", "percentile", "optimal"])
    p.add_argument("--percentile", type=float, default=90.0)
    p.add_argument("--bg-inner", type=float, default=None, help="Background annulus inner radius")
    p.add_argument("--bg-outer", type=float, default=None, help="Background annulus outer radius")

    # Preprocessing
    p.add_argument("--iterative-continuum", action="store_true", help="Use iterative sigma-clipped continuum")
    p.add_argument("--defringe", action="store_true", help="Apply spectral defringing")

    # Peak detection
    p.add_argument("--prominence-sigma", type=float, default=4.5)
    p.add_argument("--use-cwt", action="store_true", help="Enable CWT peak detection")

    # Line matching
    p.add_argument("--line-db", type=str, default=None, help="Custom line DB CSV path")
    p.add_argument("--min-confidence", type=float, default=0.2)

    # Fitting
    p.add_argument("--try-voigt", action="store_true", help="Try Voigt profile fits")
    p.add_argument("--no-fit", action="store_true", help="Skip profile fitting")

    # Visualization
    p.add_argument("--interactive-html", action="store_true", help="Generate interactive HTML viewer")

    # Verbosity
    p.add_argument("--verbose", "-v", action="store_true", help="Increase logging verbosity")
    p.add_argument("--quiet", "-q", action="store_true", help="Suppress log output")

    return p


def _config_from_args(args: argparse.Namespace) -> PipelineConfig:
    if args.config:
        cfg = load_pipeline_config(args.config)
        # Command-line overrides
        if args.cube:
            cfg = replace(cfg, cube_path=args.cube)
        if args.output_dir is not None:
            cfg = replace(cfg, output_dir=args.output_dir)
        if args.wmin is not None or args.wmax is not None:
            cfg = replace(cfg, wmin=args.wmin, wmax=args.wmax)
        return cfg

    if not args.cube:
        raise ValueError("Either --config or --cube must be provided.")

    extraction = ExtractionConfig(
        mode=args.mode,
        center_x=args.x,
        center_y=args.y,
        radius=args.radius,
        percentile=args.percentile,
        background_inner_radius=args.bg_inner,
        background_outer_radius=args.bg_outer,
    )
    preprocess = PreprocessConfig(
        iterative_continuum=args.iterative_continuum,
        defringe=args.defringe,
    )
    peaks = PeakConfig(
        prominence_sigma=args.prominence_sigma,
        use_cwt=args.use_cwt,
    )
    ident = IdentificationConfig(
        line_db_path=args.line_db,
        min_confidence=args.min_confidence,
    )
    fitting = FitConfig(
        enabled=not args.no_fit,
        try_voigt=args.try_voigt,
    )
    viz = VisualizationConfig(
        interactive_html=args.interactive_html,
    )
    return PipelineConfig(
        cube_path=args.cube,
        output_dir=args.output_dir or "outputs",
        wmin=args.wmin,
        wmax=args.wmax,
        preprocessing=preprocess,
        extraction=extraction,
        peaks=peaks,
        identification=ident,
        fitting=fitting,
        visualization=viz,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Configure logging
    level = logging.WARNING if args.quiet else (logging.DEBUG if args.verbose else logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    config = _config_from_args(args)
    summary = run_pipeline(config)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
