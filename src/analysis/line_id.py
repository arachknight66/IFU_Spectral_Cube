"""
Spectral Line Identification.

Matches detected peak wavelengths against a database of known spectral
lines within a configurable tolerance. Returns structured LineMatch
objects with confidence scores.

Confidence score: C = 1 - |Δλ| / tolerance
    where Δλ = λ_observed - λ_database (accounting for redshift).
    C = 1.0 means exact match, C = 0.0 means at the tolerance boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..utils.constants import get_line_database


@dataclass
class LineMatch:
    """A matched spectral line identification.

    Attributes
    ----------
    peak_wavelength_um : float
        Observed peak wavelength.
    matched_species : str
        Chemical species of the matched line.
    rest_wavelength_um : float
        Rest-frame wavelength of the database line.
    observed_wavelength_um : float
        Redshift-corrected database wavelength.
    transition : str
        Transition label.
    category : str
        Line category (atomic, molecular, dust, ice).
    error_um : float
        Absolute wavelength difference |peak - database|.
    confidence : float
        Match confidence in [0, 1].
    """
    peak_wavelength_um: float
    matched_species: str
    rest_wavelength_um: float
    observed_wavelength_um: float
    transition: str
    category: str
    error_um: float
    confidence: float


def identify_lines(
    peak_wavelengths: np.ndarray,
    tolerance_um: float = 0.05,
    redshift: float = 0.0,
    categories: list[str] | None = None,
) -> list[LineMatch | None]:
    """Match detected peak wavelengths against the spectral line database.

    For each peak, finds the closest database line within tolerance.
    If multiple lines are within tolerance, the best match (smallest
    error) is returned.

    Parameters
    ----------
    peak_wavelengths : np.ndarray
        Observed wavelengths of detected peaks (µm).
    tolerance_um : float
        Maximum |Δλ| for a valid match (µm).
    redshift : float
        Source redshift for shifting database to observed frame.
    categories : list of str or None
        Filter database to specific categories.

    Returns
    -------
    list of (LineMatch or None)
        One entry per input peak. None if no match within tolerance.
    """
    line_db = get_line_database(redshift=redshift, categories=categories)

    if len(line_db) == 0:
        return [None] * len(peak_wavelengths)

    # Pre-extract observed wavelengths for vectorized comparison
    db_wavelengths = np.array([l["observed_wavelength_um"] for l in line_db])

    results: list[LineMatch | None] = []

    for peak_wl in peak_wavelengths:
        errors = np.abs(db_wavelengths - peak_wl)
        min_idx = int(np.argmin(errors))
        min_error = errors[min_idx]

        if min_error <= tolerance_um:
            line = line_db[min_idx]
            confidence = 1.0 - (min_error / tolerance_um)
            results.append(LineMatch(
                peak_wavelength_um=float(peak_wl),
                matched_species=line["species"],
                rest_wavelength_um=line["rest_wavelength_um"],
                observed_wavelength_um=line["observed_wavelength_um"],
                transition=line["transition"],
                category=line["category"],
                error_um=float(min_error),
                confidence=float(confidence),
            ))
        else:
            results.append(None)

    return results
