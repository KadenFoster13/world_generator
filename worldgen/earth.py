"""Seed 0 is Earth: real elevation in place of the plate generator.

On land, temperature and rain are Earth's real 1981-2010 monthly means (earth_climate;
the model's climate stays at sea). Everything else (sea level at 0 m, rivers, lakes,
biomes, soils, floodplains, ice-age coasts, disease) follows as for any seed. Wild species
start from Earth's real homelands and ores are richest in Earth's known ore regions (both
in earth_data.py; used by nature.py).

The elevation is NOAA's ETOPO 2022 relief model, stored at 1600 x 800 as
data/earth_elevation.npz (see scripts/make_earth_elevation.py and the README); other map
widths are resampled from it.
"""
import os
from functools import lru_cache

import numpy as np

EARTH_SEED = 0
DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "earth_elevation.npz")

# Over the sea, air takes up moisture over about this many km (instead of at once), so
# narrow seas (the Red Sea, the Persian Gulf) don't water the deserts beside them.
SEA_RECHARGE_KM = 1500.0
# Enclosed seas smaller than this share of the globe become land. Small enough to keep the
# Black Sea and the Caspian (which lies below sea level), large enough to fill the Dead
# Sea, the Qattara depression and other dry hollows below sea level.
INLAND_SEA_MIN_AREA = 0.0004
# Outside Earth's known ore regions, the generator's geology-based ore fields count only
# this much (iron excepted: it is found almost everywhere).
MINERAL_DAMPING = 0.35
EARTH_RADIUS_KM = 6371.0


def is_earth(cfg):
    return getattr(cfg, "seed", None) == EARTH_SEED


@lru_cache(maxsize=1)
def _stored():
    """The stored elevation (int16 m, 800 rows x 1600 columns, rows from the north)."""
    with np.load(DATA_FILE) as data:
        return data["elevation_m"].astype(np.float32)


def elevation_at(lat, lon):
    """Earth's elevation (m) at any latitudes and longitudes (bilinear, wrapping east-west)."""
    from scipy.ndimage import map_coordinates

    z = _stored()
    h, w = z.shape
    row = (90.0 - np.asarray(lat)) / 180.0 * h - 0.5
    col = ((np.asarray(lon) + 180.0) % 360.0) / 360.0 * w - 0.5
    padded = np.concatenate([z[:, -1:], z, z[:, :1]], axis=1)      # wrap one column each side
    coords = np.array([np.clip(row, 0, h - 1), col + 1])
    return map_coordinates(padded, coords.reshape(2, -1), order=1, mode="nearest").reshape(np.shape(lat))


def earth_elevation(height, width):
    """Earth's elevation on a whole-globe grid of this size (m, float64): block means when
    the stored grid divides evenly, otherwise interpolated."""
    from .generator import Grid

    z = _stored()
    h, w = z.shape
    grid = Grid(height, width)
    if w % width == 0 and h % height == 0 and w // width == h // height:
        f = w // width
        out = z.reshape(height, f, width, f).mean(axis=(1, 3)).astype(np.float64)
    else:
        out = elevation_at(grid.lat, grid.lon).astype(np.float64)
    return carve_gorges(out, grid)


def earth_env(raw, grid):
    """The large-scale fields the rest of the generator reads from the terrain, worked out
    from Earth's relief instead of plates: continental crust, how far inland, mountain
    belts (high, rugged ground), volcanic arcs (mountain belts near the sea) and ore
    belts (mountains, young and old)."""
    from .generator import blur

    kpp = grid.km_per_px
    wraps = grid.wraps
    land = (raw >= 0).astype(np.float64)
    crust = (raw > -1500).astype(np.float64)                      # shelves count as continent
    landness = np.clip(blur(land, 300 / kpp, wraps), 0, 1)
    high = blur(np.clip((raw - 800) / 3000, 0, 1), 60 / kpp, wraps)
    rugged = blur(np.abs(raw - blur(raw, 40 / kpp, wraps)) / 600, 60 / kpp, wraps)
    collision = np.clip(0.6 * high + 0.6 * np.clip(rugged, 0, 1), 0, 1) * landness
    old = np.clip(blur(np.clip((raw - 300) / 1200, 0, 1), 80 / kpp, wraps) - high, 0, 1)
    zeros = np.zeros_like(raw)
    return dict(
        crust=crust, landness=landness, collision=collision, suture=zeros, rift=zeros, ridge=zeros,
        volcanic=np.clip(1.5 * collision * (1 - 0.6 * landness), 0, 1),
        ore=np.clip(1.2 * collision * landness + 0.7 * old * landness, 0, 1),
    )


def earth_terrain(cfg):
    """Seed 0's terrain: Earth's real elevation (m, sea level 0), one plate, no stamps."""
    from .generator import Grid, Terrain

    grid = Grid(cfg.height, cfg.width)
    raw = earth_elevation(cfg.height, cfg.width)
    env = earth_env(raw, grid)
    shape = raw.shape
    return Terrain(cfg=cfg, raw=raw, plate=np.zeros(shape, int), piece=np.zeros(shape, int),
                   plate_is_continental=np.array([True]), boundary=np.zeros(shape, bool),
                   convergence=np.zeros(shape), crust=env["crust"], lat=grid.lat, lon=grid.lon,
                   xyz=grid.xyz, target_continents=7, grid=grid, env=env, noises={}, hot_stamps=[],
                   arch_stamps=[], arch_dist=None, arch_sea=0.0, layout=None)


def region_elevation(grid):
    """Earth's elevation on a zoomed-in region's grid: interpolated from the stored grid,
    with a little fine relief on land (more in rough country) so Enhance shows texture
    the stored grid is too coarse to hold. The coastline is kept as it is."""
    from .noise import Perlin3, fbm

    z = elevation_at(grid.lat, grid.lon).astype(np.float64)
    land = z > 0
    relief = np.abs(z - elevation_at(grid.lat + 0.2, grid.lon + 0.2))      # local roughness, m
    detail = fbm(Perlin3(EARTH_SEED * 7919 + 424242), grid.xyz, 400, 3)
    bump = detail * np.clip(0.4 * relief, 5, 300)
    return carve_gorges(np.where(land, np.maximum(z + bump, 1.0), z), grid)


def _cell(grid, lat, lon):
    """The (row, col) of a point on a grid (whole globe or region), or None if outside."""
    row = int(np.floor((grid.lat0 - lat) / grid.dpp))
    dlon = lon - grid.lon0
    if grid.wraps:
        dlon %= 360.0
    col = int(np.floor(dlon / grid.dpp))
    h, w = grid.lat.shape
    return (row, col) if 0 <= row < h and 0 <= col < w else None


def carve_gorges(z, grid):
    """Cut Earth's narrow gorges (earth_data.GORGES) into elevation `z` on `grid`: along
    each river course, ground above the river's level is lowered to it, in a path of
    cells joined edge to edge (so water can follow it)."""
    from .earth_data import GORGES

    z = np.array(z, dtype=np.float64)
    for points in GORGES.values():
        prev = None
        for (la0, lo0, h0), (la1, lo1, h1) in zip(points[:-1], points[1:]):
            n = max(2, int(np.ceil(max(abs(la1 - la0), abs(lo1 - lo0)) / (0.25 * grid.dpp))))
            for t in np.linspace(0, 1, n):
                cell = _cell(grid, la0 + t * (la1 - la0), lo0 + t * (lo1 - lo0))
                if cell is None:
                    prev = None
                    continue
                level = h0 + t * (h1 - h0)
                if prev is not None and prev[0] != cell[0] and prev[1] != cell[1]:   # diagonal step
                    z[prev[0], cell[1]] = min(z[prev[0], cell[1]], level)
                z[cell] = min(z[cell], level)
                prev = cell
    return z


def earth_lakes(grid, elevation_m, ocean):
    """Earth's real lakes (earth_data.LAKES) on a grid. Each grows outward from the points
    along its length until it covers its real area, taking next whichever neighbouring
    land cell is nearest those points, counting ground above the lake's surface as
    farther (each LAKE_RISE_M_PER_KM metres up as a km), so it fills its own basin and
    keeps its real shape instead of running down a valley. Returns (lake mask, lake level
    m (nan elsewhere), mask of lakes with no outlet)."""
    import heapq

    from .earth_data import LAKES

    h, w = ocean.shape
    area = (grid.km_per_px ** 2) * np.maximum(np.cos(np.radians(grid.lat)), 0.01)
    lake = np.zeros(ocean.shape, bool)
    level = np.full(ocean.shape, np.nan)
    closed = np.zeros(ocean.shape, bool)
    for name, axis, surface, area_km2, endorheic in LAKES:
        starts = [c for c in (_cell(grid, la, lo) for la, lo in axis) if c is not None and not ocean[c]]
        if not starts:
            continue

        def cost(r, c):
            d = min(_segment_km(grid.lat[r, c], grid.lon[r, c], a, b) for a, b in zip(axis, axis[1:] or axis))
            return d + max(float(elevation_m[r, c]) - surface, 0.0) / LAKE_RISE_M_PER_KM

        heap = [(cost(*c), c) for c in starts]
        heapq.heapify(heap)
        seen, taken, covered = set(starts), [], 0.0
        while heap and (covered < area_km2 or not taken):
            _, (r, c) = heapq.heappop(heap)
            taken.append((r, c))
            covered += area[r, c]
            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                rr, cc = r + dr, c + dc
                if grid.wraps:
                    cc %= w
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in seen and not ocean[rr, cc]:
                    seen.add((rr, cc))
                    heapq.heappush(heap, (cost(rr, cc), (rr, cc)))
        rows, cols = zip(*taken)
        this = np.zeros(ocean.shape, bool)
        this[rows, cols] = True
        lake |= this
        level[this] = surface
        if endorheic:
            closed |= this
    return lake, level, closed


LAKE_RISE_M_PER_KM = 2.0


def _segment_km(lat, lon, a, b):
    """Distance (km) from a point to the segment a-b (lat, lon pairs), on a local flat map."""
    kx = 111.2 * np.cos(np.radians(lat))
    ax, ay = (a[1] - lon + 540) % 360 - 180, a[0] - lat
    bx, by = (b[1] - lon + 540) % 360 - 180, b[0] - lat
    ax, bx, ay, by = ax * kx, bx * kx, ay * 111.2, by * 111.2
    dx, dy = bx - ax, by - ay
    t = 0.0 if dx == dy == 0 else min(max(-(ax * dx + ay * dy) / (dx * dx + dy * dy), 0.0), 1.0)
    return float(np.hypot(ax + t * dx, ay + t * dy))


CLIMATE_FILE = os.path.join(os.path.dirname(__file__), "data", "earth_climate.npz")


@lru_cache(maxsize=1)
def _climate_data():
    """Monthly mean temperature (C) and precipitation (mm), 1981-2010, on a 0.5 degree grid
    (Willmott & Matsuura, University of Delaware V5.01, via NOAA PSL), with the average
    elevation of each cell; see scripts/make_earth_climate.py."""
    with np.load(CLIMATE_FILE) as d:
        return (d["temp_c10"].astype(np.float32) / 10, d["precip_mm"].astype(np.float32),
                d["elev_ref_m"].astype(np.float32))


def _sample(field, lat, lon):
    """Bilinear sample of a whole-globe grid (rows from the north, columns from 180 W)."""
    from scipy.ndimage import map_coordinates

    h, w = field.shape
    row = (90.0 - np.asarray(lat)) / 180.0 * h - 0.5
    col = ((np.asarray(lon) + 180.0) % 360.0) / 360.0 * w - 0.5
    padded = np.concatenate([field[:, -1:], field, field[:, :1]], axis=1)
    coords = np.array([np.clip(row, 0, h - 1), col + 1]).reshape(2, -1)
    return map_coordinates(padded, coords, order=1, mode="nearest").reshape(np.shape(lat))


SOIL_WATER_MM = 100.0          # water the soil holds for crops between rains
EVAPORATION_MM_PER_C_MONTH = 55.0 / 12   # what crops could use per month, per degree C above 0


SNOWMELT_MM_PER_C_MONTH = 90.0           # snow melted per month per degree C above 0


def _water_balance(temps, rains, growing_rain_mm):
    """A month-by-month water balance: snow piles up below 0 C and melts above it; the
    soil holds up to SOIL_WATER_MM for plants, which use what they can; what the soil can't
    hold runs off into rivers. Returns (how wet enough each month is for crops, 0..1;
    yearly runoff, mm).

    Wet enough is FAO's length of growing period: rain plus stored water covering at least
    half of what crops could use (or plenty of rain in the month itself). Winter-rain
    lands grow wheat on stored water into spring (the Fertile Crescent, Anatolia, the
    Mediterranean); counted on rain alone they look barren. Runoff by months matters where
    rain and snow come in the cool season: eastern Anatolia's snowmelt feeds the Euphrates,
    though a whole year's evaporation would use up all its rain."""
    need = EVAPORATION_MM_PER_C_MONTH * np.maximum(temps, 0)
    shape = temps.shape[1:]
    store, snow, runoff = np.zeros(shape), np.zeros(shape), np.zeros(shape)
    wet = np.zeros(temps.shape)
    for lap in range(2):                         # the second lap starts from settled stores
        for m in range(12):
            frozen = temps[m] < 0
            snow = snow + np.where(frozen, rains[m], 0.0)
            melt = np.where(frozen, 0.0, np.minimum(snow, SNOWMELT_MM_PER_C_MONTH * np.maximum(temps[m], 0)))
            snow = snow - melt
            water = np.where(frozen, 0.0, rains[m]) + melt + store
            use = np.minimum(water, need[m])
            left = water - use
            store = np.minimum(left, SOIL_WATER_MM)
            if lap:
                runoff = runoff + (left - store)
                ratio = water / np.maximum(need[m], 1e-6)
                wet[m] = np.maximum(np.clip((ratio - 0.3) / 0.4, 0, 1),
                                    np.clip((rains[m] - growing_rain_mm / 2) / growing_rain_mm, 0, 1))
    return wet, runoff


def earth_climate(grid, elevation_m, lapse_c_per_km, growing_temp_c, growing_rain_mm):
    """Earth's real climate on any grid: the 1981-2010 monthly means, corrected for the
    difference between each cell's height and its half-degree cell's average (finer maps
    and Enhance), and the season figures the generator uses, from the real months: the
    warmest and coldest month, the wettest and driest month, how much of the rain falls in
    the wetter half of the year, the months warm enough to grow crops and the months both
    warm and wet enough (so monsoon summers and Mediterranean dry summers come out right).

    Returns (temperature C, precipitation mm/yr, seasons dict), meant for land cells."""
    temp_m, rain_m, elev_ref = _climate_data()
    lat, lon = grid.lat, grid.lon
    height_fix = -lapse_c_per_km * (np.maximum(elevation_m, 0) - np.maximum(_sample(elev_ref, lat, lon), 0)) / 1000
    temps = np.stack([_sample(temp_m[m], lat, lon) for m in range(12)]) + height_fix
    rains = np.maximum(np.stack([_sample(rain_m[m], lat, lon) for m in range(12)]), 0)
    temperature = temps.mean(axis=0)
    precip = rains.sum(axis=0)
    wet_half = np.sort(rains, axis=0)[6:].mean(axis=0)                 # the six wettest months
    seasonality = np.clip(12 * wet_half / np.maximum(precip, 1e-6) - 1, 0, 0.9)
    warm = np.clip((temps - growing_temp_c) / 3 + 0.5, 0, 1)            # per month, 0..1
    wet, runoff = _water_balance(temps, rains, growing_rain_mm)
    seasons = dict(warmest_month_c=temps.max(axis=0), coldest_month_c=temps.min(axis=0),
                   wettest_month_mm=rains.max(axis=0), driest_month_mm=rains.min(axis=0),
                   rain_seasonality=seasonality, warm_months=warm.sum(axis=0),
                   growing_months=np.minimum(warm, wet).sum(axis=0),
                   runoff_mm=runoff)    # taken out by the generator for its rivers
    return temperature, precip, seasons


def great_circle_km(lat, lon, lat0, lon0):
    """Distance over the globe (km) from (lat0, lon0) to points (arrays of degrees)."""
    la, lo = np.radians(lat), np.radians(lon)
    la0, lo0 = np.radians(lat0), np.radians(lon0)
    c = np.sin(la) * np.sin(la0) + np.cos(la) * np.cos(la0) * np.cos(lo - lo0)
    return EARTH_RADIUS_KM * np.arccos(np.clip(c, -1, 1))


def _disc(grid, lat, lon, radius_km):
    """1 at (lat, lon), falling smoothly to 0 at `radius_km` (on any grid)."""
    d = great_circle_km(grid.lat, grid.lon, lat, lon) / radius_km
    t = np.clip(1 - d, 0, 1)
    return t * t * (3 - 2 * t)
