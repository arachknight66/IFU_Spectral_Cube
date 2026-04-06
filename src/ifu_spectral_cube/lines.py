"""Spectral line identification: database loading, matching, and confidence.

Identification workflow
-----------------------
1. **Coarse redshift** — estimate z from peak-line consensus histogram.
2. **Matching** — for each detected peak, find all rest-frame lines whose
   redshifted wavelength falls within an adaptive tolerance window:
   ``tol = max(abs_tolerance, σ_tolerance · σ_inst)`` where
   ``σ_inst = λ / (R·2.355)`` is the instrumental Gaussian width.
3. **Confidence scoring** — multi-factor model incorporating:
   - SNR term (exponential saturation)
   - Wavelength residual term (Gaussian penalty)
   - Ambiguity penalty (multiple candidate lines)
   - Fit quality term (from Gaussian/Voigt χ²)
   - Multi-line consistency (lines at same z are boosted)
4. **Reliability classification** — ``secure/probable/tentative/marginal``
   following spectroscopic survey conventions.
"""
from __future__ import annotations

import csv
import math
from importlib import resources
from pathlib import Path

import numpy as np

from .models import GaussianFit, LineMatch, Peak


# ---------------------------------------------------------------------------
# Line database
# ---------------------------------------------------------------------------

def load_line_database(
    path: str | Path | None = None,
    wmin_micron: float | None = None,
    wmax_micron: float | None = None,
) -> list[dict[str, str | float]]:
    """Load mid-IR line list from CSV.

    The bundled database covers the full MIRI 5–28 µm range with atomic
    fine-structure, molecular hydrogen, PAH features, hydrogen recombination,
    coronal lines, and CO₂ ice absorption.

    Returns list of dicts with keys:
    ``line_name``, ``rest_wavelength_micron``, ``ionization_class``,
    ``species_type``, ``notes``.
    """
    if path is None:
        db_path = resources.files("ifu_spectral_cube").joinpath("data/lines_mid_ir.csv")
    else:
        db_path = Path(path)

    rows: list[dict[str, str | float]] = []
    with open(db_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            lam = float(row["rest_wavelength_micron"])
            if wmin_micron is not None and lam < wmin_micron:
                continue
            if wmax_micron is not None and lam > wmax_micron:
                continue
            rows.append(
                {
                    "line_name": row["line_name"],
                    "rest_wavelength_micron": lam,
                    "ionization_class": row.get("ionization_class", ""),
                    "species_type": row.get("species_type", ""),
                    "notes": row.get("notes", ""),
                }
            )
    return sorted(rows, key=lambda r: float(r["rest_wavelength_micron"]))


# ---------------------------------------------------------------------------
# Redshift estimation
# ---------------------------------------------------------------------------

def estimate_redshift_from_line_consensus(
    peaks: list[Peak],
    line_db: list[dict[str, str | float]],
    z_min: float = -0.02,
    z_max: float = 0.2,
    dz: float = 5e-4,
) -> float:
    """Estimate coarse redshift from an SNR-weighted peak-line histogram.

    For each (peak, line) pair, compute z = λ_obs/λ_rest − 1.  Bin all
    candidate z-values into a histogram weighted by peak SNR.  The bin
    with maximum weight gives the consensus redshift.

    This is a brute-force but robust method that works even with partial
    line identification and moderate false-positive rates.
    """
    if not peaks or not line_db:
        return 0.0

    z_candidates: list[float] = []
    weights: list[float] = []
    for p in peaks:
        for line in line_db:
            lam0 = float(line["rest_wavelength_micron"])
            z = p.wavelength_micron / lam0 - 1.0
            if z_min <= z <= z_max:
                z_candidates.append(z)
                weights.append(max(0.0, p.snr))

    if not z_candidates:
        return 0.0

    bins = np.arange(z_min, z_max + dz, dz)
    hist, edges = np.histogram(z_candidates, bins=bins, weights=weights)
    best = int(np.argmax(hist))
    z_center = 0.5 * (edges[best] + edges[best + 1])
    return float(z_center)


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

def _line_match_confidence(
    snr: float,
    delta: float,
    tol: float,
    ambiguity_count: int,
    fit_chi2: float | None = None,
    z_consistency_bonus: float = 0.0,
) -> float:
    """Multi-factor confidence score for a line identification.

    Components
    ----------
    1. **SNR term**: 1 − exp(−SNR/5).  Saturates for strong lines,
       penalises marginal detections.

    2. **Wavelength residual term**: exp(−0.5·(Δ/tol)²).  Gaussian
       penalty on the wavelength offset relative to tolerance.

    3. **Ambiguity penalty**: 1/N_candidates.  Multiple candidate lines
       reduce confidence because identification is uncertain.

    4. **Fit quality term** (optional): exp(−max(0, χ²_r − 1)/3).  Good
       fits (χ²_r ≈ 1) contribute ~1; poor fits are penalised.

    5. **Multi-line consistency bonus**: additive term [0, 0.15] for
       lines consistent with the same redshift as other detections.
    """
    # SNR term: exponential saturation
    snr_term = 1.0 - math.exp(-max(0.0, snr) / 5.0)

    # Wavelength residual term: Gaussian penalty
    wav_term = math.exp(-0.5 * (delta / max(tol, 1e-12)) ** 2)

    # Ambiguity penalty
    ambiguity_penalty = 1.0 / max(1, ambiguity_count)

    # Fit quality term
    fit_term = 1.0
    if fit_chi2 is not None and np.isfinite(fit_chi2):
        fit_term = math.exp(-max(0.0, fit_chi2 - 1.0) / 3.0)

    base = snr_term * wav_term * ambiguity_penalty * fit_term
    boosted = min(1.0, base + z_consistency_bonus)
    return float(max(0.0, boosted))


# ---------------------------------------------------------------------------
# Line matching
# ---------------------------------------------------------------------------

def match_peaks_to_lines(
    peaks: list[Peak],
    line_db: list[dict[str, str | float]],
    redshift: float = 0.0,
    resolving_power: float = 2500.0,
    abs_tolerance_micron: float = 0.008,
    sigma_tolerance: float = 2.5,
    min_confidence: float = 0.2,
    gaussian_fits: list[GaussianFit] | None = None,
) -> list[LineMatch]:
    """Match detected peaks to known rest-frame lines with adaptive tolerance.

    Tolerance strategy
    ~~~~~~~~~~~~~~~~~~
    The matching window combines:

    1. **Absolute calibration floor** (``abs_tolerance_micron``): accounts for
       systematic wavelength calibration uncertainty in the JWST pipeline.

    2. **Instrumental profile scale**: σ_inst = λ / (R·2.355), and the
       acceptance window is ``sigma_tolerance × σ_inst``.  This naturally
       widens at longer wavelengths where MIRI's spectral resolution is
       lower.

    The final tolerance is max(floor, profile_term), ensuring we never
    use a window narrower than the calibration uncertainty.

    False-positive control
    ~~~~~~~~~~~~~~~~~~~~~~
    * Minimum confidence threshold rejects weak/ambiguous matches.
    * Instrument-aware tolerance prevents over-matching at low R.
    * Ambiguous matches (multiple candidate lines) are flagged.
    """
    fit_chi2_by_peak = {}
    if gaussian_fits:
        fit_chi2_by_peak = {f.peak_index: f.reduced_chi2 for f in gaussian_fits}

    # Compute z-consistency bonus: peaks whose z is close to the supplied
    # redshift get a small boost, rewarding multi-line agreement.
    z_consistent_peaks: set[int] = set()
    if len(peaks) > 1:
        for p in peaks:
            for line in line_db:
                lam0 = float(line["rest_wavelength_micron"])
                expected = lam0 * (1.0 + redshift)
                sigma_inst = expected / (resolving_power * 2.355)
                if abs(p.wavelength_micron - expected) < 2.0 * max(abs_tolerance_micron, sigma_inst):
                    z_consistent_peaks.add(p.index)
                    break

    n_consistent = len(z_consistent_peaks)

    out: list[LineMatch] = []
    for p in peaks:
        candidates: list[tuple[dict[str, str | float], float, float]] = []
        for line in line_db:
            lam0 = float(line["rest_wavelength_micron"])
            expected = lam0 * (1.0 + redshift)
            delta = p.wavelength_micron - expected
            sigma_inst = expected / (resolving_power * 2.355)
            tol = max(abs_tolerance_micron, sigma_tolerance * sigma_inst)
            if abs(delta) <= tol:
                candidates.append((line, delta, tol))

        if not candidates:
            continue

        candidates.sort(key=lambda x: abs(x[1]))
        best_line, best_delta, best_tol = candidates[0]
        ambiguous = len(candidates) > 1

        # Multi-line consistency bonus
        z_bonus = 0.0
        if n_consistent >= 2 and p.index in z_consistent_peaks:
            z_bonus = 0.1 * min(1.0, n_consistent / 5.0)

        confidence = _line_match_confidence(
            snr=p.snr,
            delta=abs(best_delta),
            tol=best_tol,
            ambiguity_count=len(candidates),
            fit_chi2=fit_chi2_by_peak.get(p.index),
            z_consistency_bonus=z_bonus,
        )
        if confidence < min_confidence:
            continue

        out.append(
            LineMatch(
                peak_index=p.index,
                observed_wavelength_micron=p.wavelength_micron,
                line_name=str(best_line["line_name"]),
                rest_wavelength_micron=float(best_line["rest_wavelength_micron"]),
                ionization_class=str(best_line.get("ionization_class", "")),
                delta_micron=float(best_delta),
                redshift=float(redshift),
                confidence=confidence,
                ambiguous=ambiguous,
                notes=str(best_line.get("notes", "")),
            )
        )
    return sorted(out, key=lambda m: m.confidence, reverse=True)


# ---------------------------------------------------------------------------
# Reliability classification
# ---------------------------------------------------------------------------

def classify_detection_reliability(
    matches: list[LineMatch],
    gaussian_fits: list[GaussianFit] | None = None,
) -> list[LineMatch]:
    """Assign human-readable reliability classes to line identifications.

    Classification follows spectroscopic survey conventions (SDSS, zCOSMOS):

    * **secure** (confidence ≥ 0.75, not ambiguous, good fit χ²):
      Publication-ready identification.
    * **probable** (confidence ≥ 0.50, or secure but ambiguous):
      Likely correct but warrants visual inspection.
    * **tentative** (confidence ≥ 0.30):
      Possible detection; needs corroboration from other diagnostics.
    * **marginal** (confidence < 0.30):
      Low-confidence; could be false positive.

    The ``reliability`` field on each ``LineMatch`` is updated in place.
    """
    fit_chi2 = {}
    if gaussian_fits:
        fit_chi2 = {f.peak_index: f.reduced_chi2 for f in gaussian_fits}

    for m in matches:
        chi2 = fit_chi2.get(m.peak_index)
        good_fit = chi2 is not None and chi2 < 2.0

        if m.confidence >= 0.75 and not m.ambiguous and good_fit:
            m.reliability = "secure"
        elif m.confidence >= 0.50:
            m.reliability = "probable"
        elif m.confidence >= 0.30:
            m.reliability = "tentative"
        else:
            m.reliability = "marginal"

    return matches
