# MAST discovery and ingestion (Phase 2)

Phase 2 discovers and caches only real, public JWST MIRI MRS Level-3 `*_s3d.fits` products. It does not create mock archive records, synthesize FITS files after a failure, or alter FITS science data.

`src.core.mast.search_mast_jwst()` accepts a target name, JWST program ID, observation ID, or an RA/Dec cone with a radius in arcseconds. It uses `astroquery.mast.Observations` first and falls back to MAST's official Mashup endpoint, `https://mast.stsci.edu/api/v0/invoke`, when necessary. Both routes are treated as untrusted input and only normalized MIRI MRS/IFU candidates with a real `dataURI` and `*_s3d.fits` filename are returned.

Use the CLI to inspect products without downloading them:

```text
python main.py --mast-target "NGC 7319" --mast-limit 10
python main.py --mast-target "NGC 7319" --mast-select 0 --mast-cache data/raw
```

Downloads require a selected public product, stream to a temporary file, verify that it is readable FITS, then atomically move it into the cache. An existing cached file is reused only after the same FITS readability check. Failures leave no final replacement file and report the archive/network error; retries use bounded backoff.

Every successful download (including a verified cache reuse) gets a `<product>.provenance.json` sidecar recording the canonical MAST URI and URL, normalized identifiers, query, local byte size, SHA-256, timestamp, and available package versions. Phase 3 is responsible for validating cube semantics, WCS, and data arrays.

Network tests are deliberately optional. Set `RUN_MAST_INTEGRATION=1` to opt into the live archive check; normal tests mock astroquery/HTTP.
