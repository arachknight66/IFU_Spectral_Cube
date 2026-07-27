"""Generated JWST-like FITS fixtures for strict Phase 3 ingestion."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pytest
from astropy.io import fits
from src.core.loader import CubeLoadError, load_fits_cube

def _write_cube(path: Path, *, unit="um", cdelt=0.1, sci=None, err_shape=None, dq_shape=None, include_sci=True, valid_wcs=True) -> Path:
    data = np.arange(24, dtype=float).reshape(3, 2, 4) if sci is None else sci
    hdus = [fits.PrimaryHDU()]
    if include_sci:
        h = fits.Header(); h["BUNIT"]="MJy/sr"
        if valid_wcs:
            h.update({"CTYPE1":"RA---TAN","CTYPE2":"DEC--TAN","CTYPE3":"WAVE","CUNIT1":"deg","CUNIT2":"deg","CUNIT3":unit,"CRPIX1":1.,"CRPIX2":1.,"CRPIX3":1.,"CRVAL1":12.,"CRVAL2":-3.,"CRVAL3":10. if unit=="um" else (1e-5 if unit=="m" else 10000.),"CDELT1":-.001,"CDELT2":.001,"CDELT3":cdelt})
        hdus.extend([fits.ImageHDU(data=data,header=h,name="SCI"),fits.ImageHDU(data=np.ones(err_shape or data.shape),name="ERR"),fits.ImageHDU(data=np.zeros(dq_shape or data.shape,dtype=np.uint32),name="DQ")])
    fits.HDUList(hdus).writeto(path); return path

@pytest.mark.parametrize(("unit","increment"), [("um",.1),("nm",100.),("m",1e-7)])
def test_wavelength_units_and_wcs(unit, increment, tmp_path):
    cube=load_fits_cube(_write_cube(tmp_path/f"cube_{unit}.fits",unit=unit,cdelt=increment))
    assert np.allclose(cube.wavelength,[10.,10.1,10.2])
    assert cube.flux_unit=="MJy/sr" and cube.wcs is not None and cube.wcs.celestial.pixel_n_dim==2

def test_descending_nan_inf_dq_preservation(tmp_path):
    sci=np.arange(24,dtype=float).reshape(3,2,4); sci[0,0,0]=np.nan; sci[1,0,1]=np.inf
    path=_write_cube(tmp_path/"descending.fits",cdelt=-.1,sci=sci)
    with fits.open(path,mode="update") as hdul: hdul["DQ"].data[0,1,1]=8
    cube=load_fits_cube(path)
    assert np.all(np.diff(cube.wavelength)>0) and np.isinf(cube.data[-2,0,1]) and np.isnan(cube.data[-1,0,0])
    assert cube.dq[-1,1,1]==8 and cube.dq_mask(8)[-1,1,1]

@pytest.mark.parametrize("kwargs,message", [({"include_sci":False},"required SCI"),({"err_shape":(2,2,4)},"ERR shape"),({"dq_shape":(3,4,2)},"DQ shape"),({"valid_wcs":False},"spectral WCS")])
def test_malformed_products_fail(kwargs, message, tmp_path):
    with pytest.raises(CubeLoadError,match=message): load_fits_cube(_write_cube(tmp_path/"bad.fits",**kwargs))

def test_provenance_and_variances(tmp_path):
    path=_write_cube(tmp_path/"provenance.fits")
    with fits.open(path,mode="update") as hdul:
        del hdul["ERR"]; hdul.append(fits.ImageHDU(data=np.full((3,2,4),4.),name="VAR_POISSON")); hdul.append(fits.ImageHDU(data=np.full((3,2,4),9.),name="VAR_RNOISE"))
    sidecar=path.with_name(path.name+".provenance.json"); sidecar.write_text(json.dumps({"mast_product":{"mast_uri":"mast:JWST/product/example_s3d.fits","observation_id":"jw00001"}}))
    cube=load_fits_cube(path)
    assert np.allclose(cube.standard_deviation(),np.sqrt(13.)) and cube.provenance_path==str(sidecar)

def test_placeholder_is_explicit(tmp_path):
    path=_write_cube(tmp_path/"nowcs.fits",valid_wcs=False)
    with pytest.raises(CubeLoadError): load_fits_cube(path)
    assert load_fits_cube(path,allow_placeholder_wavelength=True).validation_report.wcs_status.startswith("placeholder")
