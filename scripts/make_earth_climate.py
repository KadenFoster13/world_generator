"""Rebuild worldgen/data/earth_climate.npz (Earth's climate for seed 0) from the
University of Delaware monthly climatology.

Source: Willmott, C. J. and K. Matsuura, Terrestrial Air Temperature and Precipitation:
Monthly and Annual Time Series (V5.01), University of Delaware; long-term monthly means
for 1981-2010, as provided by the NOAA PSL, Boulder, Colorado, USA, from their website
at https://psl.noaa.gov (files air.mon.v501.ltm.1981-2010.nc and
precip.mon.v501.ltm.1981-2010.nc under
https://psl.noaa.gov/thredds/fileServer/Datasets/udel.airt.precip/). Then:

    python scripts/make_earth_climate.py path/to/air.mon.v501.ltm.1981-2010.nc path/to/precip.mon.v501.ltm.1981-2010.nc

Stores, on the data's own 0.5 degree grid (360 x 720, rows from the north, columns from
180 W): monthly mean temperature (0.1 C, int16) and monthly precipitation (mm, int16),
with sea cells filled from the nearest land so values can be interpolated up to the
coast; the land mask; and the average elevation of each half-degree cell (from the stored
Earth elevation), so finer maps can correct temperature for altitude.
"""
import os
import sys

import numpy as np
from scipy.io import netcdf_file
from scipy.ndimage import distance_transform_edt

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "worldgen", "data", "earth_climate.npz")


def read(path, name):
    with netcdf_file(path, "r", mmap=False) as f:
        v = f.variables[name]
        data = np.array(v[:], dtype=np.float64)
        missing = float(v._attributes.get("missing_value", np.nan))
        lat = np.array(f.variables["lat"][:])
        lon = np.array(f.variables["lon"][:])
    data[np.abs(data - missing) < 1e-6 * abs(missing)] = np.nan
    data[np.abs(data) > 1e30] = np.nan
    if lat[0] < lat[-1]:
        data, lat = data[:, ::-1], lat[::-1]
    order = np.argsort((lon + 180) % 360)          # columns from 180 W
    return data[:, :, order]


def main(air_path, precip_path):
    temp = read(air_path, "air")                   # C
    rain = read(precip_path, "precip") * 10.0      # cm -> mm
    land = np.isfinite(temp).all(axis=0) & np.isfinite(rain).all(axis=0)
    _, (iy, ix) = distance_transform_edt(~land, return_indices=True)    # nearest land cell
    temp, rain = temp[:, iy, ix], rain[:, iy, ix]

    import importlib.util
    spec = importlib.util.spec_from_file_location("earth", os.path.join(HERE, "..", "worldgen", "earth.py"))
    earth = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(earth)
    h, w = land.shape
    sub = 4                                        # average 4 x 4 points per half-degree cell
    rows = 90 - (np.arange(h * sub) + 0.5) * (180 / (h * sub))
    cols = -180 + (np.arange(w * sub) + 0.5) * (360 / (w * sub))
    lon, lat = np.meshgrid(cols, rows)
    elev = earth.elevation_at(lat, lon).reshape(h, sub, w, sub).mean(axis=(1, 3))

    np.savez_compressed(
        OUT, temp_c10=np.round(temp * 10).astype(np.int16), precip_mm=np.round(rain).astype(np.int16),
        land=land, elev_ref_m=np.round(elev).astype(np.int16),
        source=np.array("Willmott & Matsuura, Univ. of Delaware V5.01, 1981-2010 monthly means, via NOAA PSL"))
    print(f"saved {os.path.normpath(OUT)}: {h} x {w}, land {land.mean():.0%}, "
          f"temp {np.nanmin(temp):.0f} to {np.nanmax(temp):.0f} C, rain up to {np.nanmax(rain):.0f} mm/month")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
