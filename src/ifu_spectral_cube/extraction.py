"""Spectrum extraction from 3-D IFU cubes.

Supported modes
---------------
* **Circular aperture** — simple spatial mask with configurable radius.
* **Rectangular aperture** — axis-aligned box extraction.
* **Annular background subtraction** — source aperture minus local sky from
  a surrounding annulus, essential for fields with spatially varying continuum.
* **Optimal (inverse-variance weighted)** extraction — when an ERR extension
  is available, weight each spaxel by 1/σ² to maximise SNR.  This follows
  the IFU analogue of the Horne (1986) optimal extraction formalism.
* **Percentile-based bright-spaxel averaging** — selects spaxels above a
  flux percentile threshold; useful when source morphology is unknown.
* **k-means spectral clustering** — unsupervised spatial grouping by
  spectral moments (moment-0, -1, -2) for exploratory segmentation of
  mixed kinematic or ionization regions.
"""
from __future__ import annotations

import numpy as np
from scipy.cluster.vq import kmeans2, whiten


# ---------------------------------------------------------------------------
# Spatial masks
# ---------------------------------------------------------------------------

def circular_mask(
    shape: tuple[int, int],
    center_xy: tuple[float, float],
    radius: float,
) -> np.ndarray:
    """Boolean mask selecting spaxels within ``radius`` of ``center_xy``."""
    ny, nx = shape
    yy, xx = np.indices((ny, nx), dtype=float)
    cx, cy = center_xy
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2


def rectangular_mask(
    shape: tuple[int, int],
    x_min: int,
    x_max: int,
    y_min: int,
    y_max: int,
) -> np.ndarray:
    """Boolean mask selecting an axis-aligned rectangular region."""
    ny, nx = shape
    mask = np.zeros((ny, nx), dtype=bool)
    mask[max(0, y_min): min(ny, y_max), max(0, x_min): min(nx, x_max)] = True
    return mask


def annular_mask(
    shape: tuple[int, int],
    center_xy: tuple[float, float],
    inner_radius: float,
    outer_radius: float,
) -> np.ndarray:
    """Boolean mask selecting an annulus for local background estimation.

    The annulus is defined as all spaxels with inner_radius < r ≤ outer_radius
    from ``center_xy``.  Subtracting the median spectrum of this annulus from
    the source aperture removes spatially varying continuum and diffuse
    emission, improving line detection in crowded fields.
    """
    ny, nx = shape
    yy, xx = np.indices((ny, nx), dtype=float)
    cx, cy = center_xy
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    return (r2 > inner_radius ** 2) & (r2 <= outer_radius ** 2)


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_region_spectrum(
    cube_data: np.ndarray,
    mask: np.ndarray,
    statistic: str = "mean",
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """Extract a 1-D spectrum by combining spaxels under a spatial mask.

    Parameters
    ----------
    cube_data : (n_lambda, ny, nx) array
    mask : (ny, nx) boolean array
    statistic : ``"mean"`` | ``"median"`` | ``"sum"``
    weights : optional (ny, nx) weight map (e.g., 1/σ² for optimal extraction)

    Returns
    -------
    spectrum : 1-D array of length n_lambda
    n_selected : number of spaxels in the mask
    """
    cube = np.asarray(cube_data, dtype=float)
    if cube.ndim != 3:
        raise ValueError("Expected cube shape (n_lambda, ny, nx).")
    if mask.shape != cube.shape[1:]:
        raise ValueError("Mask shape does not match spatial shape of cube.")

    n_selected = int(mask.sum())
    if n_selected == 0:
        raise ValueError("Mask selects zero spaxels.")

    spectra = cube[:, mask]  # shape: (n_lambda, n_selected)

    if statistic == "mean":
        if weights is None:
            spectrum = np.nanmean(spectra, axis=1)
        else:
            w = np.asarray(weights, dtype=float)[mask]
            w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
            denom = np.sum(w)
            if denom <= 0:
                raise ValueError("Weights are non-positive over selected spaxels.")
            spectrum = np.nansum(spectra * w[None, :], axis=1) / denom
    elif statistic == "median":
        spectrum = np.nanmedian(spectra, axis=1)
    elif statistic == "sum":
        spectrum = np.nansum(spectra, axis=1)
    else:
        raise ValueError(f"Unsupported statistic '{statistic}'.")
    return spectrum, n_selected


# ---------------------------------------------------------------------------
# Background-subtracted extraction
# ---------------------------------------------------------------------------

def extract_with_background(
    cube_data: np.ndarray,
    source_mask: np.ndarray,
    background_mask: np.ndarray,
    statistic: str = "mean",
) -> tuple[np.ndarray, np.ndarray, int]:
    """Extract source spectrum with local background subtraction.

    Computes:  spectrum_source − median(spectrum_background)

    This removes spatially varying continuum (zodiacal background, diffuse
    nebular emission) that would otherwise bias continuum subtraction and
    line detection.

    Returns
    -------
    net_spectrum : background-subtracted source spectrum
    background_spectrum : the background estimate
    n_source : number of source spaxels
    """
    cube = np.asarray(cube_data, dtype=float)

    source_spec, n_source = extract_region_spectrum(cube, source_mask, statistic=statistic)

    n_bg = int(background_mask.sum())
    if n_bg == 0:
        # No background spaxels → return source spectrum unchanged
        return source_spec, np.zeros_like(source_spec), n_source

    bg_spec, _ = extract_region_spectrum(cube, background_mask, statistic="median")
    net = source_spec - bg_spec
    return net, bg_spec, n_source


# ---------------------------------------------------------------------------
# Optimal (inverse-variance weighted) extraction
# ---------------------------------------------------------------------------

def extract_optimal_spectrum(
    cube_data: np.ndarray,
    err_data: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Inverse-variance weighted extraction for maximum SNR.

    Following the IFU analogue of Horne (1986), at each wavelength channel
    the combined flux is:

        F(λ) = Σ_i w_i · f_i(λ)  /  Σ_i w_i

    where w_i = 1/σ_i(λ)² and σ_i(λ) is the per-spaxel uncertainty from
    the ERR extension.  The combined uncertainty is:

        σ_F(λ) = 1 / √(Σ_i w_i)

    This produces the minimum-variance unbiased estimate of the mean flux
    and is particularly valuable when spaxel noise varies across the IFU
    FOV (edge effects, vignetting).

    Returns
    -------
    spectrum : optimally-weighted 1-D spectrum
    uncertainty : 1-D uncertainty array
    n_selected : number of spaxels
    """
    cube = np.asarray(cube_data, dtype=float)
    err = np.asarray(err_data, dtype=float)
    if cube.ndim != 3 or err.ndim != 3:
        raise ValueError("Expected 3D cube and error arrays.")
    if mask.shape != cube.shape[1:]:
        raise ValueError("Mask shape does not match spatial shape.")

    n_selected = int(mask.sum())
    if n_selected == 0:
        raise ValueError("Mask selects zero spaxels.")

    flux = cube[:, mask]      # (n_lambda, n_sel)
    sigma = err[:, mask]      # (n_lambda, n_sel)

    # Avoid division by zero or negative errors
    sigma = np.where((sigma > 0) & np.isfinite(sigma), sigma, np.nan)
    var = sigma ** 2
    inv_var = np.where(np.isfinite(var) & (var > 0), 1.0 / var, 0.0)

    # Weighted mean at each wavelength
    sum_w = np.nansum(inv_var, axis=1)
    safe_sum_w = np.where(sum_w > 0, sum_w, 1.0)

    spectrum = np.nansum(flux * inv_var, axis=1) / safe_sum_w
    uncertainty = np.where(sum_w > 0, 1.0 / np.sqrt(safe_sum_w), np.nan)

    return spectrum, uncertainty, n_selected


# ---------------------------------------------------------------------------
# Percentile extraction
# ---------------------------------------------------------------------------

def extract_percentile_spectrum(
    cube_data: np.ndarray,
    percentile: float = 90.0,
    channel_min: int | None = None,
    channel_max: int | None = None,
    statistic: str = "mean",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract spectrum by averaging the brightest spaxels (by moment-0).

    Useful when target morphology is unknown or irregular — selects spaxels
    whose integrated flux exceeds the requested percentile threshold.
    """
    cube = np.asarray(cube_data, dtype=float)
    sl = slice(channel_min, channel_max)
    moment0 = np.nansum(cube[sl], axis=0)
    threshold = np.nanpercentile(moment0, percentile)
    mask = moment0 >= threshold
    spectrum, _ = extract_region_spectrum(cube, mask=mask, statistic=statistic)
    return spectrum, mask


# ---------------------------------------------------------------------------
# Spectral clustering
# ---------------------------------------------------------------------------

def cluster_spaxels_kmeans(
    cube_data: np.ndarray,
    wavelength_micron: np.ndarray,
    n_clusters: int = 3,
    random_state: int = 0,
) -> np.ndarray:
    """Unsupervised spatial grouping by spectral moments.

    Features per spaxel:

    1. **Moment-0** (integrated flux) — total line emission intensity.
    2. **Moment-1** (flux-weighted centroid) — peak emission wavelength.
    3. **Moment-2** (spectral dispersion) — line-width / velocity spread.

    These moments capture the dominant spectral variation across the FOV
    and are sufficient to separate regions with different line ratios,
    kinematics, or excitation conditions.
    """
    cube = np.asarray(cube_data, dtype=float)
    wl = np.asarray(wavelength_micron, dtype=float)
    ny, nx = cube.shape[1], cube.shape[2]

    flat = cube.reshape(cube.shape[0], -1).T  # (n_spaxels, n_lambda)
    flux_sum = np.nansum(flat, axis=1)
    safe_sum = np.where(np.abs(flux_sum) < 1e-12, 1e-12, flux_sum)
    centroid = np.nansum(flat * wl[None, :], axis=1) / safe_sum
    variance = np.nansum(flat * (wl[None, :] - centroid[:, None]) ** 2, axis=1) / safe_sum
    variance = np.where(variance < 0, 0, variance)

    features = np.column_stack([flux_sum, centroid, np.sqrt(variance)])
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    features = whiten(features)

    _, labels = kmeans2(features, k=n_clusters, minit="points", seed=random_state)
    return labels.reshape(ny, nx)
