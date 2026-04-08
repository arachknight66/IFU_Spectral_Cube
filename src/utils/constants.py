"""
Physical Constants and Spectral Line Database for MIRI IFU Analysis.

This module provides:
    1. Fundamental physical constants (sourced from astropy where possible).
    2. A curated database of spectral lines commonly observed in the
       JWST MIRI MRS wavelength range (4.9–28.1 µm).

The line database is organized as a list of dictionaries for structured
querying. Each entry contains:
    - species : str — chemical species or feature name
    - rest_wavelength_um : float — laboratory rest wavelength in microns
    - transition : str — quantum mechanical transition label
    - category : str — one of 'atomic', 'molecular', 'dust', 'ice'

Line data sources:
    - NIST Atomic Spectra Database (https://www.nist.gov/pml/atomic-spectra-database)
    - Molecular line lists: HITRAN/HITEMP for CO, CO₂
    - H₂ rotational lines: Rosenthal et al. (2000)
    - PAH features: Draine & Li (2007), Tielens (2008)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Physical constants (SI units unless noted)
# ---------------------------------------------------------------------------
SPEED_OF_LIGHT_M_S: float = 2.99792458e8       # m/s
PLANCK_CONSTANT_J_S: float = 6.62607015e-34     # J·s
BOLTZMANN_CONSTANT_J_K: float = 1.380649e-23    # J/K
SPEED_OF_LIGHT_KM_S: float = 299792.458         # km/s


# ---------------------------------------------------------------------------
# MIRI MRS spectral line database
# ---------------------------------------------------------------------------
MIRI_SPECTRAL_LINES: list[dict] = [
    # ==================== Hydrogen Recombination ====================
    {
        "species": "H I",
        "rest_wavelength_um": 7.460,
        "transition": "Pfund-α (6→5)",
        "category": "atomic",
    },
    {
        "species": "H I",
        "rest_wavelength_um": 12.372,
        "transition": "Humphreys-α (7→6)",
        "category": "atomic",
    },

    # ==================== Molecular Hydrogen (Rotational) ====================
    # Pure rotational transitions S(J) = J+2 → J, ground vibrational state
    {
        "species": "H₂",
        "rest_wavelength_um": 28.219,
        "transition": "S(0) 0-0",
        "category": "molecular",
    },
    {
        "species": "H₂",
        "rest_wavelength_um": 17.035,
        "transition": "S(1) 0-0",
        "category": "molecular",
    },
    {
        "species": "H₂",
        "rest_wavelength_um": 12.279,
        "transition": "S(2) 0-0",
        "category": "molecular",
    },
    {
        "species": "H₂",
        "rest_wavelength_um": 9.665,
        "transition": "S(3) 0-0",
        "category": "molecular",
    },
    {
        "species": "H₂",
        "rest_wavelength_um": 8.025,
        "transition": "S(4) 0-0",
        "category": "molecular",
    },
    {
        "species": "H₂",
        "rest_wavelength_um": 6.909,
        "transition": "S(5) 0-0",
        "category": "molecular",
    },
    {
        "species": "H₂",
        "rest_wavelength_um": 6.108,
        "transition": "S(6) 0-0",
        "category": "molecular",
    },
    {
        "species": "H₂",
        "rest_wavelength_um": 5.511,
        "transition": "S(7) 0-0",
        "category": "molecular",
    },

    # ==================== Neon ====================
    {
        "species": "[Ne II]",
        "rest_wavelength_um": 12.814,
        "transition": "²P₁/₂ → ²P₃/₂",
        "category": "atomic",
    },
    {
        "species": "[Ne III]",
        "rest_wavelength_um": 15.555,
        "transition": "³P₁ → ³P₂",
        "category": "atomic",
    },
    {
        "species": "[Ne V]",
        "rest_wavelength_um": 14.322,
        "transition": "³P₁ → ³P₂",
        "category": "atomic",
    },
    {
        "species": "[Ne V]",
        "rest_wavelength_um": 24.318,
        "transition": "³P₂ → ³P₁",
        "category": "atomic",
    },
    {
        "species": "[Ne VI]",
        "rest_wavelength_um": 7.652,
        "transition": "²P₃/₂ → ²P₁/₂",
        "category": "atomic",
    },

    # ==================== Argon ====================
    {
        "species": "[Ar II]",
        "rest_wavelength_um": 6.985,
        "transition": "²P₁/₂ → ²P₃/₂",
        "category": "atomic",
    },
    {
        "species": "[Ar III]",
        "rest_wavelength_um": 8.991,
        "transition": "³P₁ → ³P₂",
        "category": "atomic",
    },
    {
        "species": "[Ar V]",
        "rest_wavelength_um": 13.102,
        "transition": "³P₁ → ³P₂",
        "category": "atomic",
    },

    # ==================== Sulfur ====================
    {
        "species": "[S III]",
        "rest_wavelength_um": 18.713,
        "transition": "³P₂ → ³P₁",
        "category": "atomic",
    },
    {
        "species": "[S IV]",
        "rest_wavelength_um": 10.511,
        "transition": "²P₃/₂ → ²P₁/₂",
        "category": "atomic",
    },

    # ==================== Iron ====================
    {
        "species": "[Fe II]",
        "rest_wavelength_um": 5.340,
        "transition": "a⁶D₇/₂ → a⁶D₉/₂",
        "category": "atomic",
    },
    {
        "species": "[Fe II]",
        "rest_wavelength_um": 17.936,
        "transition": "a⁴F₇/₂ → a⁴F₉/₂",
        "category": "atomic",
    },
    {
        "species": "[Fe II]",
        "rest_wavelength_um": 25.988,
        "transition": "a⁶D₅/₂ → a⁶D₇/₂",
        "category": "atomic",
    },
    {
        "species": "[Fe III]",
        "rest_wavelength_um": 22.925,
        "transition": "³F₃ → ³F₂",
        "category": "atomic",
    },

    # ==================== Silicon / Magnesium ====================
    {
        "species": "[Si II]",
        "rest_wavelength_um": 34.815,
        "transition": "²P₃/₂ → ²P₁/₂",
        "category": "atomic",
    },
    {
        "species": "[Mg V]",
        "rest_wavelength_um": 5.610,
        "transition": "³P₁ → ³P₂",
        "category": "atomic",
    },
    {
        "species": "[Mg VII]",
        "rest_wavelength_um": 5.503,
        "transition": "³P₁ → ³P₂",
        "category": "atomic",
    },

    # ==================== Oxygen ====================
    {
        "species": "[O IV]",
        "rest_wavelength_um": 25.890,
        "transition": "²P₃/₂ → ²P₁/₂",
        "category": "atomic",
    },

    # ==================== PAH Features ====================
    # Polycyclic Aromatic Hydrocarbon emission bands
    {
        "species": "PAH",
        "rest_wavelength_um": 6.2,
        "transition": "C-C stretch",
        "category": "dust",
    },
    {
        "species": "PAH",
        "rest_wavelength_um": 7.7,
        "transition": "C-C stretch complex",
        "category": "dust",
    },
    {
        "species": "PAH",
        "rest_wavelength_um": 8.6,
        "transition": "C-H in-plane bend",
        "category": "dust",
    },
    {
        "species": "PAH",
        "rest_wavelength_um": 11.3,
        "transition": "C-H out-of-plane bend (solo)",
        "category": "dust",
    },
    {
        "species": "PAH",
        "rest_wavelength_um": 12.7,
        "transition": "C-H out-of-plane bend (duo/trio)",
        "category": "dust",
    },
    {
        "species": "PAH",
        "rest_wavelength_um": 16.4,
        "transition": "C-C-C bend",
        "category": "dust",
    },

    # ==================== Ices ====================
    {
        "species": "H₂O ice",
        "rest_wavelength_um": 6.0,
        "transition": "ν₂ bending mode",
        "category": "ice",
    },
    {
        "species": "CO₂ ice",
        "rest_wavelength_um": 15.2,
        "transition": "ν₂ bending mode",
        "category": "ice",
    },
    {
        "species": "CO₂ ice",
        "rest_wavelength_um": 4.27,
        "transition": "ν₃ asymmetric stretch",
        "category": "ice",
    },
    {
        "species": "CH₃OH ice",
        "rest_wavelength_um": 9.75,
        "transition": "C-O stretch",
        "category": "ice",
    },

    # ==================== Molecules (gas phase) ====================
    {
        "species": "CO₂ (gas)",
        "rest_wavelength_um": 14.97,
        "transition": "ν₂ Q-branch",
        "category": "molecular",
    },
    {
        "species": "C₂H₂",
        "rest_wavelength_um": 13.7,
        "transition": "ν₅ bend",
        "category": "molecular",
    },
    {
        "species": "HCN",
        "rest_wavelength_um": 14.0,
        "transition": "ν₂ bend",
        "category": "molecular",
    },
]


def get_line_database(
    redshift: float = 0.0,
    categories: list[str] | None = None,
) -> list[dict]:
    """Return spectral line database, optionally redshifted and filtered.

    Parameters
    ----------
    redshift : float
        Source redshift. Lines are shifted to observed frame via
        λ_obs = λ_rest × (1 + z).
    categories : list of str, optional
        Filter to specific categories (e.g. ['atomic', 'molecular']).
        None returns all lines.

    Returns
    -------
    list of dict
        Each dict has keys: species, rest_wavelength_um,
        observed_wavelength_um, transition, category.
    """
    lines = MIRI_SPECTRAL_LINES
    if categories is not None:
        lines = [l for l in lines if l["category"] in categories]

    result = []
    for line in lines:
        entry = line.copy()
        entry["observed_wavelength_um"] = (
            line["rest_wavelength_um"] * (1.0 + redshift)
        )
        result.append(entry)

    return result
