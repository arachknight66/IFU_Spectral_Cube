"""
Unit tests for MAST celestial coordinate (RA/Dec) parsing and cone search.
"""

from __future__ import annotations

from src.core.mast import parse_coordinates, search_mast_jwst


def test_parse_coordinates_decimal():
    """Verify parsing decimal degrees."""
    ra_deg, dec_deg = parse_coordinates(339.967, 33.963)
    assert abs(ra_deg - 339.967) < 1e-4
    assert abs(dec_deg - 33.963) < 1e-4


def test_parse_coordinates_sexagesimal():
    """Verify parsing sexagesimal string format."""
    ra_deg, dec_deg = parse_coordinates("22h39m52.08s", "+33d57m46.8s")
    assert 339.0 < ra_deg < 340.5
    assert 33.5 < dec_deg < 34.5


def test_search_mast_jwst_coords():
    """Verify searching MAST using RA/Dec coordinates."""
    results = search_mast_jwst(ra=339.967, dec=33.963, radius_arcsec=15.0, limit=2)
    assert isinstance(results, list)
    assert len(results) > 0

    rec = results[0]
    assert "obs_id" in rec
    assert "download_url" in rec
    assert "product_filename" in rec
