"""Offline tests for MAST discovery, selection, caching, and provenance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.core import mast

OBSERVATION = {"obs_id": "jw01288-o001_t001_miri", "proposal_id": "1288", "target_name": "Example target", "instrument_name": "MIRI", "instrument_configuration": "MRS"}
PRODUCT = {"productFilename": "jw01288-o001_t001_miri_ch2-medium_s3d.fits", "dataURI": "mast:JWST/product/jw01288-o001_t001_miri_ch2-medium_s3d.fits", "releaseStatus": "PUBLIC"}


def test_normalized_search_results_from_mocked_astroquery(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeObservations:
        @staticmethod
        def query_criteria(**kwargs):
            assert kwargs["obs_collection"] == "JWST"
            return [OBSERVATION]
        @staticmethod
        def get_product_list(observation):
            return [PRODUCT]
    monkeypatch.setattr(mast, "Observations", FakeObservations)
    results = mast.search_mast_jwst(target_name="Example target")
    assert len(results) == 1
    assert results[0]["mast_uri"] == PRODUCT["dataURI"]
    assert results[0]["product_filename"].endswith("_s3d.fits")


def test_empty_and_proprietary_products_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeObservations:
        @staticmethod
        def query_criteria(**kwargs): return [OBSERVATION]
        @staticmethod
        def get_product_list(observation): return [{"productFilename": "wrong_x1d.fits", "dataURI": "mast:JWST/product/wrong_x1d.fits"}]
    monkeypatch.setattr(mast, "Observations", FakeObservations)
    assert mast.search_mast_jwst(target_name="Example target") == []
    product = mast._normalize_product(OBSERVATION, PRODUCT)
    assert product is not None
    private = mast.MastProduct(**{**product.__dict__, "release_status": "PROPRIETARY"})
    with pytest.raises(mast.ProductSelectionError, match="proprietary"):
        mast.validate_selected_product(private)


def test_failed_download_leaves_no_final_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    product = mast._normalize_product(OBSERVATION, PRODUCT)
    assert product is not None
    monkeypatch.setattr(mast.urllib.request, "urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("offline")))
    with pytest.raises(mast.MastDownloadError, match="download failed"):
        mast.download_mast_product(product, tmp_path, retries=0)
    assert not (tmp_path / product.product_filename).exists()
    assert not list(tmp_path.glob("*.part"))


def test_verified_cached_file_reuse_and_provenance(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    product = mast._normalize_product(OBSERVATION, PRODUCT)
    assert product is not None
    cached = tmp_path / product.product_filename
    cached.write_bytes(b"verified test fits")
    monkeypatch.setattr(mast, "_validate_fits", lambda path: None)
    monkeypatch.setattr(mast.urllib.request, "urlopen", lambda *args, **kwargs: pytest.fail("cache must be reused"))
    result = mast.download_mast_product(product, tmp_path, query=mast.MastQuery(target_name="Example target"))
    assert result == cached
    provenance = json.loads((tmp_path / f"{cached.name}.provenance.json").read_text())
    assert provenance["reused_cache"] is True
    assert provenance["mast_product"]["mast_uri"] == product.mast_uri


def test_archive_module_has_no_synthetic_fallback() -> None:
    assert "generate_synthetic_cube" not in Path(mast.__file__).read_text(encoding="utf-8")
