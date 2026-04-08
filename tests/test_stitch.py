"""
Test script for the Spectral Stitching module.
Generates two synthetic MIRI channel cubes (e.g., Short and Medium)
with overlapping fields of view and wavelengths, then stitches them.
"""

from src.core.stitch import stitch_cubes
from src.core.cube import SpectralCube
from tests.synthetic import generate_synthetic_cube

def test_stitching():
    print("Generating Cube 1 (Channel 1 Short: 5-10 µm)")
    cube1 = generate_synthetic_cube(
        n_wavelength=250,
        wl_start=5.0,
        wl_end=10.0,
        ny=30, nx=30,
    )
    # Add fake WCS header metadata to force astropy path
    cube1.header.update({
        "CRVAL1": 150.0, "CRPIX1": 15.0, "CDELT1": 0.1, "CTYPE1": 'RA---TAN',
        "CRVAL2": 2.0, "CRPIX2": 15.0, "CDELT2": 0.1, "CTYPE2": 'DEC--TAN',
    })

    print("Generating Cube 2 (Channel 1 Medium: 9-16 µm)")
    cube2 = generate_synthetic_cube(
        n_wavelength=350,
        wl_start=9.0,
        wl_end=16.0,
        ny=20, nx=20,  # Smaller spatial array
    )
    # Offset center slightly, larger pixels (0.15)
    cube2.header.update({
        "CRVAL1": 150.0, "CRPIX1": 10.0, "CDELT1": 0.15, "CTYPE1": 'RA---TAN',
        "CRVAL2": 2.0, "CRPIX2": 10.0, "CDELT2": 0.15, "CTYPE2": 'DEC--TAN',
    })

    print(f"\nCube 1 Shape: {cube1.shape}")
    print(f"Cube 2 Shape: {cube2.shape}")

    print("\nStitching cubes to reference (Cube 1)...")
    stitched_cube = stitch_cubes([cube1, cube2], reference_index=0)

    print("\nStitched Cube Final Spec:")
    print(f"Shape: {stitched_cube.shape}")
    print(f"Wavelengths: {stitched_cube.wavelength_range[0]:.2f} - {stitched_cube.wavelength_range[1]:.2f} µm")
    
    # Assertions
    assert stitched_cube.shape == (600, 30, 30), "Spatial shape should match reference cube, spectral should be sum"
    
    # Check that wavelength is sorted
    wl_diff = stitched_cube.wavelength[1:] - stitched_cube.wavelength[:-1]
    assert (wl_diff >= 0).all(), "Wavelength array is not strictly monotonic!"

    print("\n[✔] Success! Stitched cube preserves geometries and is sorted correctly.")

if __name__ == "__main__":
    test_stitching()
