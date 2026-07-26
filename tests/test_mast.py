"""
Unit tests for MAST Archive search and download functions.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from src.core.mast import search_mast_jwst, download_mast_product


def test_search_mast_jwst():
    """Verify searching MAST returns valid observation metadata dicts."""
    results = search_mast_jwst(target_name="NGC 7319", limit=3)
    assert isinstance(results, list)
    assert len(results) > 0

    rec = results[0]
    assert "obs_id" in rec
    assert "target_name" in rec
    assert "download_url" in rec
    assert "product_filename" in rec


def test_download_mast_product():
    """Verify product download writes a valid file to target directory."""
    results = search_mast_jwst(target_name="NGC 7319", limit=1)
    assert len(results) > 0

    rec = results[0]
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = download_mast_product(rec["download_url"], rec["product_filename"], output_dir=tmp_dir)
        assert Path(path).exists()
        assert Path(path).stat().st_size > 0
