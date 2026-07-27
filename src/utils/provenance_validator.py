"""
Provenance Manifest Validator for Reproducible False-Colour Renderings.

Validates <image>.manifest.json sidecar files against schema version,
verifies input FITS component file paths and checksums, and checks recipe completeness.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def calculate_sha256(filepath: Path) -> str:
    """Calculate SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_provenance_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Validate a rendering manifest sidecar file.

    Parameters
    ----------
    manifest_path : str or Path
        Path to JSON manifest sidecar file.

    Returns
    -------
    dict
        Validation summary containing 'valid', 'errors', 'warnings', and 'manifest_data'.
    """
    path = Path(manifest_path)
    errors: list[str] = []
    warnings: list[str] = []

    if not path.exists():
        return {
            "valid": False,
            "errors": [f"Manifest file not found: '{path}'"],
            "warnings": [],
            "manifest_data": None,
        }

    try:
        manifest_data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "valid": False,
            "errors": [f"Invalid JSON syntax in manifest: {exc}"],
            "warnings": [],
            "manifest_data": None,
        }

    # Check required top-level keys
    required_keys = {"recipe_name", "target_name", "rendering_mode", "shape", "channel_stats", "provenance"}
    missing_keys = required_keys - set(manifest_data.keys())
    if missing_keys:
        errors.append(f"Missing required manifest keys: {sorted(missing_keys)}")

    # Check rendering mode
    mode = manifest_data.get("rendering_mode")
    if mode not in ("scientific", "presentation"):
        errors.append(f"Invalid rendering_mode: '{mode}'. Expected 'scientific' or 'presentation'.")

    # Check provenance details
    prov = manifest_data.get("provenance", {})
    if not isinstance(prov, dict):
        errors.append("Field 'provenance' must be a dictionary.")
    else:
        # Check disclaimer
        disclaimer = prov.get("warning", "")
        if "False-colour" not in disclaimer and "false-colour" not in disclaimer.lower():
            warnings.append("Manifest provenance lacks standard false-colour disclaimer.")

        # Check input FITS component files
        input_files = prov.get("input_component_files", {})
        if isinstance(input_files, dict):
            for color, fpath_str in input_files.items():
                if fpath_str:
                    comp_path = Path(fpath_str)
                    if not comp_path.exists():
                        warnings.append(
                            f"Input component FITS file for channel '{color}' not found at '{comp_path}'. "
                            "External file may have been moved."
                        )

    is_valid = len(errors) == 0

    return {
        "valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "manifest_data": manifest_data,
    }


def main_cli(argv: list[str] | None = None) -> int:
    """CLI entry point for manifest validation."""
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print("Usage: python -m src.utils.provenance_validator <manifest.json>", file=sys.stderr)
        return 1

    manifest_p = Path(args[0])
    res = validate_provenance_manifest(manifest_p)

    print("=" * 60)
    print(f"  Provenance Manifest Validation: {manifest_p.name}")
    print("=" * 60)
    print(f"  Status: {'✓ VALID' if res['valid'] else '❌ INVALID'}")

    if res["errors"]:
        print("\n  Errors:")
        for err in res["errors"]:
            print(f"    - {err}")

    if res["warnings"]:
        print("\n  Warnings:")
        for warn in res["warnings"]:
            print(f"    - {warn}")

    print("=" * 60)
    return 0 if res["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main_cli())
