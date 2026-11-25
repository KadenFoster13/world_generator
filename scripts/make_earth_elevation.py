"""Rebuild worldgen/data/earth_elevation.npz (Earth's elevation for seed 0) from ETOPO 2022.

Source: ETOPO 2022 15 arc-second global relief model, ice surface version (NOAA National
Centers for Environmental Information, DOI 10.25921/fd45-gt74). US-government data.

To get the input: on NOAA's ERDDAP server (coastwatch / NCEI ERDDAP; search "ETOPO 2022"),
open the griddap page of the 15 arc-second ice-surface dataset, request the elevation
variable over the whole globe with a stride of 27 in both latitude and longitude (3200 x
1600 points, ~10 MB) and choose the ".nc" (NetCDF-3) file type. Then:

    python scripts/make_earth_elevation.py path/to/download.nc

This takes 2 x 2 block means (to 1600 x 800, ~25 km cells at the equator), orders rows
from the north and columns from 180 W, and saves int16 metres.
"""
import os
import sys

import numpy as np
from scipy.io import netcdf_file

OUT = os.path.join(os.path.dirname(__file__), "..", "worldgen", "data", "earth_elevation.npz")
WIDTH, HEIGHT = 1600, 800


def main(path):
    with netcdf_file(path, "r", mmap=False) as f:
        names = list(f.variables)
        var = next(n for n in names if n not in ("latitude", "longitude", "lat", "lon"))
        lat_name = "latitude" if "latitude" in names else "lat"
        lon_name = "longitude" if "longitude" in names else "lon"
        z = np.array(f.variables[var][:], dtype=np.float64)
        lat = np.array(f.variables[lat_name][:])
        lon = np.array(f.variables[lon_name][:])
    if lat[0] < lat[-1]:                      # rows from the north
        z, lat = z[::-1], lat[::-1]
    order = np.argsort((lon + 180) % 360)     # columns from 180 W
    z = z[:, order]
    h, w = z.shape
    fy, fx = h // HEIGHT, w // WIDTH
    z = z[:HEIGHT * fy, :WIDTH * fx].reshape(HEIGHT, fy, WIDTH, fx).mean(axis=(1, 3))
    source = (f"ETOPO 2022 15 arc-second ice surface (NOAA NCEI, DOI 10.25921/fd45-gt74), "
              f"ERDDAP stride {round(43200 / h) if h else '?'}, {fy}x{fx} block mean")
    np.savez_compressed(OUT, elevation_m=np.round(z).astype(np.int16), source=np.array(source))
    print(f"saved {os.path.normpath(OUT)}: {WIDTH} x {HEIGHT}, {z.min():.0f} to {z.max():.0f} m")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
