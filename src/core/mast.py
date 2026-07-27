"""Reliable MAST discovery and download support for JWST MIRI MRS cubes.

Archive operations never fabricate observations or FITS products.  Synthetic
test cubes remain in :mod:`tests.synthetic` and are only used by explicit demo
commands outside this module.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    from astroquery.mast import Observations
    import astroquery
except ImportError:  # pragma: no cover - exercised when optional dependency missing
    Observations = None  # type: ignore[assignment]
    astroquery = None  # type: ignore[assignment]

try:
    from astropy.io import fits
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    import astropy
except ImportError:  # pragma: no cover
    fits = None  # type: ignore[assignment]
    SkyCoord = None  # type: ignore[assignment]
    u = None  # type: ignore[assignment]
    astropy = None  # type: ignore[assignment]


MAST_INVOKE_URL = "https://mast.stsci.edu/api/v0/invoke"
MAST_DOWNLOAD_URL = "https://mast.stsci.edu/api/v0/download/file"
_SAFE_FILENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*_s3d\.fits(?:\.gz)?$", re.IGNORECASE)


class MastError(RuntimeError):
    """Base class for an honest MAST archive failure."""


class MastQueryError(MastError):
    """MAST could not perform or return a trustworthy query."""


class ProductSelectionError(MastError):
    """A candidate cannot be used as a public MIRI MRS Level-3 cube."""


class MastDownloadError(MastError):
    """A product could not be safely cached and verified."""


@dataclass(frozen=True)
class MastQuery:
    target_name: str | None = None
    proposal_id: str | None = None
    observation_id: str | None = None
    ra_deg: float | None = None
    dec_deg: float | None = None
    radius_arcsec: float = 10.0
    limit: int = 25

    def __post_init__(self) -> None:
        if not any((self.target_name, self.proposal_id, self.observation_id, self.ra_deg is not None and self.dec_deg is not None)):
            raise ValueError("provide target_name, proposal_id, observation_id, or both ra_deg and dec_deg")
        if (self.ra_deg is None) != (self.dec_deg is None):
            raise ValueError("RA and Dec must be supplied together")
        if self.ra_deg is not None and not (0 <= self.ra_deg < 360 and -90 <= self.dec_deg <= 90):
            raise ValueError("RA must be in [0, 360) and Dec in [-90, 90]")
        if self.radius_arcsec <= 0 or self.limit <= 0:
            raise ValueError("radius_arcsec and limit must be positive")


@dataclass(frozen=True)
class MastProduct:
    observation_id: str
    program_id: str | None
    target_name: str | None
    instrument: str | None
    mode: str | None
    product_filename: str
    mast_uri: str
    download_url: str
    release_status: str
    metadata: dict[str, Any] = field(default_factory=dict)
    checksum: str | None = None

    @property
    def obs_id(self) -> str:  # Compatibility for current callers.
        return self.observation_id

    @property
    def proposal_id(self) -> str | None:
        return self.program_id

    @property
    def submode(self) -> str:
        return self.mode or "MIRI MRS IFU"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"obs_id": self.observation_id, "proposal_id": self.program_id, "submode": self.submode}


def parse_coordinates(ra: float | str, dec: float | str) -> tuple[float, float]:
    """Parse decimal or sexagesimal coordinates; reject rather than guess."""
    if SkyCoord is None or u is None:
        try:
            return float(ra), float(dec)
        except (TypeError, ValueError) as exc:
            raise ValueError("sexagesimal coordinates require astropy") from exc
    try:
        if isinstance(ra, str) and (":" in ra or "h" in ra.lower()):
            coord = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
        else:
            coord = SkyCoord(ra, dec, unit=(u.deg, u.deg))
        return float(coord.ra.deg), float(coord.dec.deg)
    except Exception as exc:
        raise ValueError(f"invalid celestial coordinates: RA={ra!r}, Dec={dec!r}") from exc


def search_mast_jwst(
    target_name: str = "", proposal_id: str | None = None, observation_id: str | None = None,
    ra: float | str | None = None, dec: float | str | None = None, radius_arcsec: float = 10.0,
    limit: int = 25,
) -> list[dict[str, Any]]:
    """Find actual public MIRI MRS `*_s3d.fits` products and normalize them.

    Astroquery is used first.  If it is unavailable or MAST rejects that query,
    the documented MAST Mashup API is queried; failures raise ``MastQueryError``.
    """
    if (ra is None) != (dec is None):
        raise ValueError("RA and Dec must be supplied together")
    ra_deg, dec_deg = parse_coordinates(ra, dec) if ra is not None else (None, None)
    query = MastQuery(target_name=target_name.strip() or None, proposal_id=proposal_id or None,
                      observation_id=observation_id or None, ra_deg=ra_deg, dec_deg=dec_deg,
                      radius_arcsec=radius_arcsec, limit=limit)
    errors: list[str] = []
    if Observations is not None:
        try:
            return [product.to_dict() for product in _search_astroquery(query)]
        except Exception as exc:
            errors.append(f"astroquery: {exc}")
    try:
        return [product.to_dict() for product in _search_rest_api(query)]
    except Exception as exc:
        errors.append(f"MAST API: {exc}")
    raise MastQueryError("MAST search failed; " + " | ".join(errors))


def _search_astroquery(query: MastQuery) -> list[MastProduct]:
    assert Observations is not None
    if query.ra_deg is not None:
        position: Any = f"{query.ra_deg} {query.dec_deg}" if SkyCoord is None else SkyCoord(query.ra_deg * u.deg, query.dec_deg * u.deg)
        observations = Observations.query_region(position, radius=f"{query.radius_arcsec} arcsec")
    else:
        criteria: dict[str, Any] = {"obs_collection": "JWST"}
        if query.target_name: criteria["target_name"] = query.target_name
        if query.proposal_id: criteria["proposal_id"] = query.proposal_id
        if query.observation_id: criteria["obs_id"] = query.observation_id
        observations = Observations.query_criteria(**criteria)
    products: list[MastProduct] = []
    for observation in list(observations)[:query.limit]:
        obs = _row_dict(observation)
        if str(obs.get("obs_collection", "JWST")).upper() != "JWST":
            continue
        for product_row in Observations.get_product_list(observation):
            product = _normalize_product(obs, _row_dict(product_row))
            if product is not None:
                products.append(product)
                if len(products) >= query.limit:
                    return products
    return products


def _search_rest_api(query: MastQuery) -> list[MastProduct]:
    """Fallback using MAST's official `/api/v0/invoke` Mashup interface."""
    filters = [{"paramName": "obs_collection", "values": ["JWST"]}]
    for name, value in (("target_name", query.target_name), ("proposal_id", query.proposal_id), ("obs_id", query.observation_id)):
        if value:
            filters.append({"paramName": name, "values": [value]})
    request: dict[str, Any] = {"service": "Mast.Caom.Filtered", "format": "json", "params": {"columns": "*", "filters": filters}}
    if query.ra_deg is not None:
        request["service"] = "Mast.Caom.Cone"
        request["params"] = {"ra": query.ra_deg, "dec": query.dec_deg, "radius": query.radius_arcsec / 3600}
    observations = _mast_invoke(request).get("data", [])[:query.limit]
    products: list[MastProduct] = []
    for obs in observations:
        obs_dict = dict(obs)
        obsid = obs_dict.get("obsid") or obs_dict.get("obs_id")
        if obsid is None:
            continue
        product_response = _mast_invoke({"service": "Mast.Caom.Products", "format": "json", "params": {"obsid": str(obsid)}})
        for row in product_response.get("data", []):
            product = _normalize_product(obs_dict, dict(row))
            if product is not None:
                products.append(product)
                if len(products) >= query.limit:
                    return products
    return products


def _mast_invoke(payload: Mapping[str, Any]) -> dict[str, Any]:
    encoded = urllib.parse.urlencode({"request": json.dumps(payload)}).encode("utf-8")
    request = urllib.request.Request(MAST_INVOKE_URL, data=encoded, headers={"User-Agent": "jwst-ifu-pipeline/phase-2"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise MastQueryError(f"official MAST API request failed: {exc}") from exc
    if not isinstance(body, dict) or not isinstance(body.get("data", []), list):
        raise MastQueryError("official MAST API returned an unexpected response")
    return body


def _row_dict(row: Any) -> dict[str, Any]:
    if isinstance(row, Mapping):
        return dict(row)
    colnames = getattr(row, "colnames", None) or getattr(getattr(row, "table", None), "colnames", [])
    return {name: row[name] for name in colnames}


def _text(value: Any) -> str | None:
    if value is None or str(value).strip() in {"", "--", "None"}:
        return None
    return str(value).strip()


def _normalize_product(observation: Mapping[str, Any], product: Mapping[str, Any]) -> MastProduct | None:
    filename = _text(product.get("productFilename") or product.get("product_filename") or product.get("filename"))
    uri = _text(product.get("dataURI") or product.get("data_uri") or product.get("uri"))
    instrument = _text(observation.get("instrument_name") or observation.get("instrument"))
    mode = _text(observation.get("instrument_configuration") or observation.get("submode") or observation.get("obs_mode"))
    # MRS filenames are the stable product-level signal; metadata can be inconsistent.
    if not filename or not uri or not filename.lower().endswith("_s3d.fits"):
        return None
    searchable = " ".join(value or "" for value in (instrument, mode, filename)).lower()
    if "miri" not in searchable or not any(marker in searchable for marker in ("mrs", "ifu", "ch1", "ch2", "ch3", "ch4")):
        return None
    status = _text(product.get("releaseStatus") or product.get("release_status") or observation.get("dataRights")) or "UNKNOWN"
    canonical_url = f"{MAST_DOWNLOAD_URL}?{urllib.parse.urlencode({'uri': uri})}"
    return MastProduct(
        observation_id=_text(observation.get("obs_id") or observation.get("obsid") or observation.get("observation_id")) or "UNKNOWN",
        program_id=_text(observation.get("proposal_id") or observation.get("proposal") or observation.get("program")),
        target_name=_text(observation.get("target_name")), instrument=instrument, mode=mode,
        product_filename=filename, mast_uri=uri, download_url=canonical_url, release_status=status,
        checksum=_text(product.get("md5") or product.get("checksum")),
        metadata={"obsid": _text(observation.get("obsid")), "productType": _text(product.get("productType") or product.get("product_type")), "calib_level": _text(product.get("calib_level"))},
    )


def validate_selected_product(product: Mapping[str, Any] | MastProduct) -> MastProduct:
    """Validate untrusted product metadata before a download starts."""
    value = product if isinstance(product, MastProduct) else MastProduct(
        observation_id=str(product.get("observation_id") or product.get("obs_id") or ""), program_id=_text(product.get("program_id") or product.get("proposal_id")),
        target_name=_text(product.get("target_name")), instrument=_text(product.get("instrument")), mode=_text(product.get("mode") or product.get("submode")),
        product_filename=str(product.get("product_filename") or ""), mast_uri=str(product.get("mast_uri") or ""),
        download_url=str(product.get("download_url") or ""), release_status=str(product.get("release_status") or "UNKNOWN"),
        metadata=dict(product.get("metadata") or {}), checksum=_text(product.get("checksum")),
    )
    if not _SAFE_FILENAME.fullmatch(value.product_filename):
        raise ProductSelectionError("unsupported product: expected a MIRI Level-3 '*_s3d.fits' filename")
    if value.release_status.upper() in {"PROPRIETARY", "PRIVATE", "EXCLUSIVE"}:
        raise ProductSelectionError("product is proprietary and cannot be downloaded anonymously")
    if value.release_status.upper() not in {"PUBLIC", "RELEASED", "ARCHIVED"}:
        raise ProductSelectionError(f"product is unavailable for verified public download (release status: {value.release_status})")
    if not value.mast_uri.startswith("mast:") or not value.download_url.startswith("https://mast.stsci.edu/"):
        raise ProductSelectionError("product lacks a canonical MAST URI or download URL")
    searchable = " ".join(item or "" for item in (value.instrument, value.mode, value.product_filename)).lower()
    if "miri" not in searchable or not any(mark in searchable for mark in ("mrs", "ifu", "ch1", "ch2", "ch3", "ch4")):
        raise ProductSelectionError("product is not identified as a MIRI MRS/IFU cube")
    return value


def _validate_fits(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise MastDownloadError("download is empty")
    if fits is None:
        raise MastDownloadError("astropy is required to verify FITS downloads")
    try:
        with fits.open(path, memmap=False):
            pass
    except Exception as exc:
        raise MastDownloadError(f"download is not readable FITS: {exc}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_mast_product(product: Mapping[str, Any] | MastProduct, output_dir: str | Path = "data/raw", *, query: MastQuery | None = None, retries: int = 2) -> Path:
    """Download one selected MAST product transactionally, with a provenance sidecar."""
    selected = validate_selected_product(product)
    if retries < 0:
        raise ValueError("retries must be non-negative")
    cache = Path(output_dir).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / selected.product_filename
    if destination.exists():
        try:
            _validate_fits(destination)
            _write_provenance(destination, selected, query, reused_cache=True)
            return destination
        except MastDownloadError:
            # Do not overwrite a suspect cache; leave it for inspection.
            raise MastDownloadError(f"cached file failed FITS verification: {destination}")
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        temporary: Path | None = None
        try:
            fd, temp_name = tempfile.mkstemp(prefix=f".{selected.product_filename}.", suffix=".part", dir=cache)
            temporary = Path(temp_name)
            with os.fdopen(fd, "wb") as handle:
                request = urllib.request.Request(selected.download_url, headers={"User-Agent": "jwst-ifu-pipeline/phase-2"})
                with urllib.request.urlopen(request, timeout=120) as response:
                    if getattr(response, "status", 200) >= 400:
                        raise MastDownloadError(f"MAST returned HTTP {response.status}")
                    while chunk := response.read(1024 * 1024):
                        handle.write(chunk)
            _validate_fits(temporary)
            os.replace(temporary, destination)
            _write_provenance(destination, selected, query, reused_cache=False)
            return destination
        except Exception as exc:
            last_error = exc
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(min(2 ** attempt, 4))
    raise MastDownloadError(f"download failed after {retries + 1} attempt(s): {last_error}") from last_error


def _write_provenance(path: Path, product: MastProduct, query: MastQuery | None, *, reused_cache: bool) -> Path:
    record = {"schema_version": 1, "retrieved_at": datetime.now(timezone.utc).isoformat(), "reused_cache": reused_cache,
              "local_path": str(path), "byte_size": path.stat().st_size, "sha256": _sha256(path),
              "mast_product": product.to_dict(), "query": asdict(query) if query else None,
              "packages": {"astroquery": getattr(astroquery, "__version__", None), "astropy": getattr(astropy, "__version__", None)}}
    sidecar = path.with_name(path.name + ".provenance.json")
    sidecar.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sidecar
