# Scientifically safe FITS ingestion (Phase 3)

`load_fits_cube()` accepts a downloaded `.fits` or `.fits.gz` product and requires a named 3D `SCI` extension plus an unambiguous spectral Astropy WCS. Wavelengths are converted from the declared WCS unit to microns. Missing, non-spectral, non-monotonic, or ambiguous WCS values are errors; channel indices are available only with explicit `allow_placeholder_wavelength=True`, which marks the result unsuitable for science analysis.

SCI values are copied exactly: NaN and Inf remain present and are represented by `cube.validity_mask`. DQ is copied bit-for-bit, with `dq_mask()` and `valid_science_mask()` helpers. `masked_flux()` and explicit `filled_flux(fill_value)` create derived arrays without altering source data.

`ERR` is retained as standard deviation. `VAR_POISSON` and `VAR_RNOISE` are retained as native variances; `standard_deviation()` quadrature-combines them only when ERR is absent. The cube retains full primary and science FITS headers, WCS, BUNIT, source path, and its Phase-2 provenance sidecar. `cube.validation_report` summarizes the safe-ingestion decision for the CLI and UI.
