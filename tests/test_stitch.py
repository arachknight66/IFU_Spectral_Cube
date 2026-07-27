"""
Test script for the Spectral Stitching module.
Generates two synthetic MIRI channel cubes (e.g., Short and Medium)
with overlapping fields of view and wavelengths, then stitches them.
"""

from astropy.wcs import WCS
from src.core.stitch import stitch_cubes, align_and_stitch_cubes, AlignmentConfig
from src.core.cube import SpectralCube
from tests.synthetic import generate_synthetic_cube

def test_stitching():
    print("Generating Cube 1 (Channel 1 Short: 5-10 µm)")
    base1 = generate_synthetic_cube(
        n_wavelength=250,
        wl_start=5.0,
        wl_end=10.0,
        ny=30, nx=30,
    )

    wcs1 = WCS(naxis=3)
    wcs1.wcs.crval = [150.0, 2.0, 5.0]
    wcs1.wcs.crpix = [15.0, 15.0, 1.0]
    wcs1.wcs.cdelt = [-0.0001, 0.0001, 0.02]
    wcs1.wcs.ctype = ["RA---TAN", "DEC--TAN", "WAVE"]

    cube1 = SpectralCube(
        data=base1.data,
        wavelength=base1.wavelength,
        header=base1.header,
        wcs=wcs1,
        err=base1.err,
        dq=base1.dq,
        flux_unit="MJy/sr",
        filepath="cube1.fits",
    )

    print("Generating Cube 2 (Channel 1 Medium: 9-16 µm)")
    base2 = generate_synthetic_cube(
        n_wavelength=350,
        wl_start=9.0,
        wl_end=16.0,
        ny=20, nx=20,
    )

    wcs2 = WCS(naxis=3)
    wcs2.wcs.crval = [150.0, 2.0, 9.0]
    wcs2.wcs.crpix = [10.0, 10.0, 1.0]
    wcs2.wcs.cdelt = [-0.00015, 0.00015, 0.02]
    wcs2.wcs.ctype = ["RA---TAN", "DEC--TAN", "WAVE"]

    cube2 = SpectralCube(
        data=base2.data,
        wavelength=base2.wavelength,
        header=base2.header,
        wcs=wcs2,
        err=base2.err,
        dq=base2.dq,
        flux_unit="MJy/sr",
        filepath="cube2.fits",
    )

    print(f"\nCube 1 Shape: {cube1.shape}")
    print(f"Cube 2 Shape: {cube2.shape}")

    print("\nStitching cubes to reference (Cube 1)...")
    stitched_cube = stitch_cubes([cube1, cube2], reference_index=0)

    print("\nStitched Cube Final Spec:")
    print(f"Shape: {stitched_cube.shape}")
    print(f"Wavelengths: {stitched_cube.wavelength_range[0]:.2f} - {stitched_cube.wavelength_range[1]:.2f} µm")
    
    assert stitched_cube.spatial_shape == (30, 30), "Spatial shape should match reference cube"
    
    # Check that wavelength is sorted
    wl_diff = stitched_cube.wavelength[1:] - stitched_cube.wavelength[:-1]
    assert (wl_diff >= 0).all(), "Wavelength array is not strictly monotonic!"

    print("\n[✔] Success! Stitched cube preserves geometries and is sorted correctly.")

if __name__ == "__main__":
    test_stitching()
