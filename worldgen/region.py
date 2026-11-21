"""Enhanced regions: regenerate part of a world in much more detail.

A region is a rectangle of the world map, regenerated with pixels 2-8x smaller.
It stays consistent with the world it came from:

- large-scale shapes (continents, mountain belts, rifts, prevailing winds) are
  taken from the world and sampled finer;
- fine detail (coastline wiggles, ridges, hills) is regenerated with the same
  noise plus extra finer octaves, so it adds detail instead of changing things;
- islands and hotspot volcanoes are redrawn from the world's recorded positions;
- lakes keep the world's water levels, with shorelines traced on the finer terrain;
- rivers are recomputed at the finer scale, and rivers flowing in from outside
  the region enter with the discharge they had on the world map.
"""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label, maximum_filter

from . import hydrology
from .generator import (BIOME_ID, Grid, Terrain, World, _apply_archipelagos, _climate,
                      _piece_map, _plate_boundaries, _rivers, _synthesize, _to_metres,
                      detail_octaves, fill_pockets)

MAX_REGION_PX = 1600         # widest a region can be, in its own pixels
FACTORS = (8, 4, 2)          # how many times finer than the world map


@dataclass
class RegionBox:
    """Which part of the world map a region covers, in world-map pixels."""
    row0: int
    col0: int      # may be any value; columns wrap around the globe
    rows: int
    cols: int
    factor: int    # region pixels per world pixel, along each side


def box_for_view(world: World, col_start: float, col_end: float, row_start: float, row_end: float):
    """The region box for a visible area of the world map (pixel coordinates, columns
    already including any scroll). Returns None if the area is too large to enhance."""
    h, w = world.elevation_m.shape
    c0 = int(np.floor(col_start + 0.5))
    c1 = int(np.ceil(col_end + 0.5))
    r0 = max(0, int(np.floor(row_start + 0.5)))
    r1 = min(h, int(np.ceil(row_end + 0.5)))
    # Round sizes up to multiples of 4 so the finer grid divides evenly for noise.
    cols = -(-(c1 - c0) // 4) * 4
    rows = -(-(r1 - r0) // 4) * 4
    if rows > h:
        rows = h - h % 4
    r0 = min(r0, h - rows)
    factor = next((f for f in FACTORS if cols * f <= MAX_REGION_PX), None)
    if factor is None or cols >= w:
        return None
    return RegionBox(r0, c0 % w, rows, cols, factor)


def region_box(world: World, north: float, south: float, west: float, east: float):
    """The region box covering a latitude/longitude rectangle (degrees; east of 180 wraps,
    and west > east means the box crosses the date line). Returns None if it's too large:
    a region can be at most MAX_REGION_PX / 2 world-map cells across."""
    g = world.grid
    if east < west:
        east += 360
    return box_for_view(world, (west + 180) / g.dpp - 0.5, (east + 180) / g.dpp - 0.5,
                        (90 - north) / g.dpp - 0.5, (90 - south) / g.dpp - 0.5)


def generate_region(world: World, box: RegionBox) -> World:
    cfg = world.cfg
    t = world.terrain
    g = world.grid
    f = box.factor
    grid = Grid(box.rows * f, box.cols * f, lat0=g.lat0 - box.row0 * g.dpp,
                lon0=g.lon0 + box.col0 * g.dpp, dpp=g.dpp / f, wraps=False)
    extra = detail_octaves(grid.equivalent_width)
    sample, nearest = _samplers(world, box)

    # --- terrain: the world's large-scale fields + finer detail noise
    env = {name: sample(field) for name, field in t.env.items()}
    raw = _synthesize(cfg, grid, extra, env, t.noises, t.hot_stamps)
    if t.arch_stamps:
        raw = _apply_archipelagos(raw, cfg, grid, t.arch_stamps, sample(t.arch_dist), t.arch_sea,
                                  t.noises["arch"], extra)
    elevation_m, ocean = _to_metres(raw, world.sea_raw, world.land_scale, world.depth_scale, cfg)
    elevation_m, ocean = _keep_real_sea(elevation_m, ocean, nearest(world.ocean), grid)

    # --- climate, with the world's winds
    wind = {name: sample(field) for name, field in world.wind.items()}
    temperature, precip, biome, _ = _climate(cfg, grid, elevation_m, ocean, wind)

    # --- lakes: the world's water levels on the finer terrain
    level = world.lake_level_m
    # Spread each lake's level one world pixel outward so shorelines can follow the
    # finer terrain instead of the blocky world pixels.
    padded = maximum_filter(np.nan_to_num(level, nan=-1e9), size=3, mode=("nearest", "wrap"))
    lake_level = nearest(padded)
    lake = ~ocean & (lake_level > elevation_m)
    salt_flat = nearest(world.salt_flat) & ~lake & ~ocean
    closed = nearest(world.closed) & ~ocean

    # --- rivers: drain to the sea or off the region's edge; add rivers flowing in
    edge = np.zeros(ocean.shape, bool)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    outlet = ocean | edge
    filled, receiver, order = hydrology.priority_flood(elevation_m, outlet)
    inflow = _inflow(world, box, grid)
    discharge, catchment = _rivers(receiver, order, outlet, closed, precip, temperature,
                                   grid, cfg, inflow=inflow)
    stream = (catchment >= cfg.river_min_catchment_km2) & ~ocean & ~lake
    river = stream & (discharge >= cfg.river_min_discharge_m3s)

    biome[salt_flat] = BIOME_ID["Salt flat"]
    biome[lake & (temperature >= -10)] = BIOME_ID["Lake"]

    # Plates, for the Plates layer.
    lay = t.layout
    piece, _ = _piece_map(grid.xyz, lay["centres"], lay["size_bias"], lay["warp_noise"],
                          cfg.plate_boundary_warp)
    plate = lay["plate_of"][piece]
    boundary, convergence, _ = _plate_boundaries(plate, lay["omega"], grid.xyz)

    terrain = Terrain(cfg=cfg, raw=raw, plate=plate, piece=piece,
                      plate_is_continental=t.plate_is_continental, boundary=boundary,
                      convergence=convergence, crust=env["crust"], lat=grid.lat, lon=grid.lon,
                      xyz=grid.xyz, target_continents=t.target_continents, grid=grid)
    region = World(terrain=terrain, cfg=cfg, elevation_m=elevation_m, ocean=ocean,
                   temperature_c=temperature, precip_mm=precip, biome=biome,
                   landmass_id=np.zeros(ocean.shape, int), landmass_area=np.zeros(1),
                   major_landmasses=0, islands=0, lake=lake, salt_flat=salt_flat,
                   filled_elevation_m=filled, lakes=label(lake)[1], receiver=receiver,
                   discharge_m3s=discharge, river=river, catchment_km2=catchment, stream=stream,
                   grid=grid, sea_raw=world.sea_raw, land_scale=world.land_scale,
                   depth_scale=world.depth_scale, wind=wind,
                   lake_level_m=np.where(lake, lake_level, np.nan), closed=closed)
    region.box = box
    return region


# ------------------------------------------------------------------ helpers
def _samplers(world, box):
    """Functions that read world-map arrays at the region's pixels.

    sample   bilinear: for smooth fields
    nearest  the world pixel each region pixel falls in: for masks
    """
    h, w = world.elevation_m.shape
    f = box.factor
    gy = box.row0 + (np.arange(box.rows * f) + 0.5) / f - 0.5
    gx = box.col0 + (np.arange(box.cols * f) + 0.5) / f - 0.5
    y0 = np.floor(gy).astype(int)
    x0 = np.floor(gx).astype(int)
    ty = (gy - y0)[:, None]
    tx = (gx - x0)[None, :]
    ya, yb = np.clip(y0, 0, h - 1), np.clip(y0 + 1, 0, h - 1)
    xa, xb = x0 % w, (x0 + 1) % w
    ny = np.clip(box.row0 + np.arange(box.rows * f) // f, 0, h - 1)
    nx = (box.col0 + np.arange(box.cols * f) // f) % w

    def sample(a):
        top = a[ya][:, xa] * (1 - tx) + a[ya][:, xb] * tx
        bottom = a[yb][:, xa] * (1 - tx) + a[yb][:, xb] * tx
        return top * (1 - ty) + bottom * ty

    def nearest(a):
        return a[ny][:, nx]

    return sample, nearest


def _keep_real_sea(elevation_m, ocean, world_ocean, grid):
    """Finer coastline noise can open little pockets below sea level inside the land.
    Keep only water connected to the world's sea; fill the rest in as land."""
    lbl, n = label(ocean)
    if n == 0:
        return elevation_m, ocean
    connected = np.zeros(n + 1, bool)
    connected[np.unique(lbl[ocean & world_ocean])] = True
    connected[0] = False
    pockets = ocean & ~connected[lbl]
    if not pockets.any():
        return elevation_m, ocean
    return fill_pockets(elevation_m, pockets, ~ocean, wraps=False), ocean & ~pockets


def _inflow(world, box, grid):
    """Water and catchment entering the region from rivers outside it.

    Finds world-map pixels outside the box that drain into it, and adds their
    discharge where they enter (one world pixel in from the edge, so the finer
    drainage picks it up rather than sending it straight back out).
    """
    h, w = world.elevation_m.shape
    f = box.factor
    inside = np.zeros((h, w), bool)
    cols = (box.col0 + np.arange(box.cols)) % w
    inside[box.row0:box.row0 + box.rows][:, cols] = True
    rec = world.receiver
    flat_inside = inside.ravel()
    src = np.flatnonzero(~flat_inside & (rec >= 0))
    src = src[flat_inside[rec[src]]]
    q = np.zeros(grid.lat.shape)
    area = np.zeros(grid.lat.shape)
    if len(src) == 0:
        return q, area
    ry, rx = np.divmod(rec[src], w)
    r = np.clip((ry - box.row0) * f + f // 2, 1, grid.height - 2)
    c = np.clip(((rx - box.col0) % w) * f + f // 2, 1, grid.width - 2)
    np.add.at(q, (r, c), world.discharge_m3s.ravel()[src])
    np.add.at(area, (r, c), world.catchment_km2.ravel()[src])
    return q, area

