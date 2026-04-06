"""Flask web server for the JWST MIRI IFU pipeline dashboard.

Serves the static frontend and provides API endpoints for:
- Pipeline execution on uploaded FITS cubes
- Loading and serving pipeline results
- Serving generated plot images
"""
from __future__ import annotations

import json
import logging
import math
import os
import shutil
import tempfile
import traceback
from pathlib import Path
from threading import Thread

from flask import Flask, jsonify, request, send_from_directory, send_file, Response

logger = logging.getLogger("ifu_web")

# Resolve paths
_WEB_DIR = Path(__file__).parent
_STATIC_DIR = _WEB_DIR / "static"

app = Flask(__name__, static_folder=str(_STATIC_DIR))

# Pipeline state
_state = {
    "status": "idle",       # idle | running | complete | error
    "progress": "",
    "output_dir": None,
    "summary": None,
    "error": None,
}


def _sanitize(obj):
    """Recursively replace NaN/Inf with None for JSON serialization.

    JavaScript's JSON.parse() does not support NaN/Infinity which are
    Python-specific extensions.  This ensures all API responses are
    valid RFC 8259 JSON.
    """
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def _json_response(data, status=200):
    """Create a JSON Response with NaN/Inf sanitized to null."""
    clean = _sanitize(data)
    body = json.dumps(clean, ensure_ascii=False)
    return Response(body, status=status, mimetype="application/json")


# ---------------------------------------------------------------------------
# Static file serving
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(str(_STATIC_DIR), "index.html")


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(str(_STATIC_DIR), filename)


# ---------------------------------------------------------------------------
# API: Pipeline status
# ---------------------------------------------------------------------------

@app.route("/api/status")
def api_status():
    return jsonify({
        "status": _state["status"],
        "progress": _state["progress"],
        "error": _state["error"],
    })


# ---------------------------------------------------------------------------
# API: Run pipeline
# ---------------------------------------------------------------------------

def _run_pipeline_thread(cube_path: str, output_dir: str, params: dict):
    """Run pipeline in background thread."""
    try:
        _state["status"] = "running"
        _state["progress"] = "Initializing pipeline..."
        _state["error"] = None

        from ..config import PipelineConfig, PreprocessConfig, ExtractionConfig, PeakConfig, IdentificationConfig, FitConfig, VisualizationConfig
        from ..pipeline import run_pipeline

        config = PipelineConfig(
            cube_path=cube_path,
            output_dir=output_dir,
        )

        # Apply user parameters
        if params.get("wmin"):
            config.wmin = float(params["wmin"])
        if params.get("wmax"):
            config.wmax = float(params["wmax"])

        # Extraction
        config.extraction = ExtractionConfig(
            mode=params.get("extraction_mode", "circular"),
            center_x=float(params["center_x"]) if params.get("center_x") else None,
            center_y=float(params["center_y"]) if params.get("center_y") else None,
            radius=float(params.get("radius", 3.0)),
            percentile=float(params.get("percentile", 90.0)),
        )

        # Preprocessing
        config.preprocessing = PreprocessConfig(
            continuum_window=int(params.get("continuum_window", 101)),
            iterative_continuum=params.get("iterative_continuum") == "true",
            defringe=params.get("defringe") == "true",
            savgol_window=int(params.get("savgol_window", 11)),
            savgol_polyorder=int(params.get("savgol_polyorder", 3)),
        )

        # Peak detection
        config.peaks = PeakConfig(
            prominence_sigma=float(params.get("prominence_sigma", 4.5)),
            min_snr=float(params.get("min_snr", 4.5)),
            resolving_power=float(params.get("resolving_power", 2500.0)),
            use_cwt=params.get("use_cwt") == "true",
        )

        # Identification
        config.identification = IdentificationConfig(
            min_confidence=float(params.get("min_confidence", 0.2)),
        )

        # Fitting
        config.fitting = FitConfig(
            enabled=params.get("fitting_enabled") != "false",
            try_voigt=params.get("try_voigt") == "true",
        )

        # Visualization
        config.visualization = VisualizationConfig(
            smoothing_comparison=True,
            peaks_and_matches=True,
            gaussian_fits=True,
            redshift_histogram=True,
            noise_profile=True,
            emission_maps=True,
            line_diagnostic_grid=True,
            interactive_html=False,
        )

        _state["progress"] = "Running pipeline..."
        summary = run_pipeline(config)
        _state["summary"] = summary
        _state["output_dir"] = output_dir
        _state["status"] = "complete"
        _state["progress"] = "Pipeline complete!"

    except Exception as e:
        _state["status"] = "error"
        _state["error"] = str(e)
        _state["progress"] = f"Error: {e}"
        logger.error("Pipeline error: %s\n%s", e, traceback.format_exc())


@app.route("/api/run", methods=["POST"])
def api_run():
    if _state["status"] == "running":
        return jsonify({"error": "Pipeline already running"}), 409

    # Handle file upload
    if "cube" not in request.files:
        return jsonify({"error": "No FITS file uploaded"}), 400

    cube_file = request.files["cube"]
    if not cube_file.filename:
        return jsonify({"error": "Empty filename"}), 400

    # Save uploaded file
    upload_dir = Path(app.instance_path) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    cube_path = str(upload_dir / cube_file.filename)
    cube_file.save(cube_path)

    # Output directory
    output_dir = str(Path(app.instance_path) / "outputs" / Path(cube_file.filename).stem)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Collect parameters from form data
    params = dict(request.form)

    # Start pipeline in background thread
    thread = Thread(target=_run_pipeline_thread, args=(cube_path, output_dir, params), daemon=True)
    thread.start()

    return jsonify({"message": "Pipeline started", "output_dir": output_dir})


# ---------------------------------------------------------------------------
# API: Load existing results
# ---------------------------------------------------------------------------

@app.route("/api/load", methods=["POST"])
def api_load():
    """Load results from an existing output directory."""
    data = request.get_json()
    results_dir = data.get("path", "")

    if not results_dir or not Path(results_dir).exists():
        return jsonify({"error": f"Directory not found: {results_dir}"}), 404

    summary_path = Path(results_dir) / "summary.json"
    if not summary_path.exists():
        return jsonify({"error": "No summary.json found in directory"}), 404

    with open(summary_path, "r") as f:
        summary = json.load(f)

    _state["summary"] = summary
    _state["output_dir"] = str(results_dir)
    _state["status"] = "complete"
    _state["progress"] = "Results loaded"
    _state["error"] = None

    return _json_response({"message": "Results loaded", "summary": _sanitize(summary)})


# ---------------------------------------------------------------------------
# API: Get results data
# ---------------------------------------------------------------------------

@app.route("/api/summary")
def api_summary():
    if _state["summary"] is None:
        return jsonify({"error": "No results available"}), 404
    return _json_response(_state["summary"])


@app.route("/api/spectra")
def api_spectra():
    """Return intermediate spectra as JSON arrays."""
    if not _state["output_dir"]:
        return jsonify({"error": "No results"}), 404

    npz_path = Path(_state["output_dir"]) / "intermediate_spectra.npz"
    if not npz_path.exists():
        return jsonify({"error": "No spectra file"}), 404

    import numpy as np
    data = np.load(npz_path)

    def _safe_list(arr):
        """Convert numpy array to list, replacing NaN with None."""
        return [None if (isinstance(v, float) and math.isnan(v)) else v
                for v in arr.tolist()]

    return _json_response({
        "wavelength": _safe_list(data["wavelength_micron"]),
        "raw": _safe_list(data["raw_spectrum"]),
        "continuum": _safe_list(data["continuum"]),
        "continuum_subtracted": _safe_list(data["continuum_subtracted"]),
        "savgol": _safe_list(data["savgol_smoothed"]),
        "gaussian": _safe_list(data["gaussian_smoothed"]),
    })


@app.route("/api/peaks")
def api_peaks():
    if not _state["output_dir"]:
        return jsonify({"error": "No results"}), 404

    import csv
    path = Path(_state["output_dir"]) / "detected_peaks.csv"
    if not path.exists():
        return jsonify([])

    with open(path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return jsonify(rows)


@app.route("/api/matches")
def api_matches():
    if not _state["output_dir"]:
        return jsonify({"error": "No results"}), 404

    import csv
    path = Path(_state["output_dir"]) / "line_matches.csv"
    if not path.exists():
        return jsonify([])

    with open(path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return jsonify(rows)


@app.route("/api/fits")
def api_fits():
    if not _state["output_dir"]:
        return jsonify({"error": "No results"}), 404

    import csv
    path = Path(_state["output_dir"]) / "gaussian_fits.csv"
    if not path.exists():
        return jsonify([])

    with open(path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return jsonify(rows)


@app.route("/api/plot/<filename>")
def api_plot(filename):
    """Serve generated plot images."""
    if not _state["output_dir"]:
        return jsonify({"error": "No results"}), 404

    safe_name = Path(filename).name
    plot_path = Path(_state["output_dir"]) / safe_name
    if not plot_path.exists():
        return jsonify({"error": f"Plot not found: {safe_name}"}), 404

    return send_file(str(plot_path), mimetype="image/png")


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

def run_server(host: str = "0.0.0.0", port: int = 5000, debug: bool = False):
    """Start the web dashboard server."""
    print(f"\n  🔭  JWST MIRI IFU Pipeline Dashboard")
    print(f"  ────────────────────────────────────")
    print(f"  Open in browser: http://localhost:{port}")
    print(f"  Press Ctrl+C to stop\n")
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_server(debug=True)
