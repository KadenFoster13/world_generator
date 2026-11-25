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
  the region enter with the discharge they had on the world map;
- the world's rivers keep their courses: water is routed over the finer ground with
  their channels cut into it, leaves the region only where the world's drainage does,
  and ends in closed basins where it does on the world map.
"""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label, maximum_filter

from . import hydrology
from .earth import earth_env, is_earth, region_elevation
from .generator import (BIOME_ID, Grid, Terrain, World, _apply_archipelagos, _climate,
                      _piece_map, _plate_boundaries, _rivers, _synthesize, _to_metres,
                      detail_octaves, erode, fill_pockets, finish_world)

MAX_REGION_PX = 1600         # widest a region can be, in its own pixels
CHANNEL_DEPTH_M = 1000.0     # world rivers are cut this deep into the ground water is routed
                             # over (routing only), so no nearby outlet can draw them away
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
    if is_earth(cfg):   # Earth: the stored real elevation, finer, with a little extra relief
        raw = region_elevation(grid)
        env = earth_env(raw, grid)
        elevation_m, ocean = raw.copy(), raw < 0
    else:
        env = {name: sample(field) for name, field in t.env.items()}
        raw = _synthesize(cfg, grid, extra, env, t.noises, t.hot_stamps)
        if t.arch_stamps:
            raw = _apply_archipelagos(raw, cfg, grid, t.arch_stamps, sample(t.arch_dist), t.arch_sea,
                                      t.noises["arch"], extra, t.noises.get("arch_fine"))
        elevation_m, ocean = _to_metres(raw, world.sea_raw, world.land_scale, world.depth_scale, cfg)
    elevation_m, ocean = _keep_real_sea(elevation_m, ocean, nearest(world.ocean), grid)

    # --- climate, with the world's winds
    wind = {name: sample(field) for name, field in world.wind.items()}
    temperature, precip, biome, _, seasons = _climate(cfg, grid, elevation_m, ocean, wind)
    runoff = seasons.pop("runoff_mm", None)       # Earth: from its months

    # --- rivers carve valleys (rivers flowing in from outside count too)
    outlet = ocean | _edge_outlets(world, box, grid) | _basin_sinks(world, box, grid)
    inflow = _inflow(world, box, grid)
    elevation_m = erode(elevation_m, ocean, outlet, precip, temperature, grid, cfg, inflow, runoff)

    # --- lakes: the world's water levels on the finer terrain
    level = world.lake_level_m
    # Spread each lake's level one world pixel outward so shorelines can follow the
    # finer terrain instead of the blocky world pixels.
    padded = maximum_filter(np.nan_to_num(level, nan=-1e9), size=3, mode=("nearest", "wrap"))
    lake_level = nearest(padded)
    lake = ~ocean & (lake_level > elevation_m)
    salt_flat = nearest(world.salt_flat) & ~lake & ~ocean
    closed = nearest(world.closed) & ~ocean

    # --- rivers: drain to the sea or off the region's edge; add rivers flowing in.
    # The world's rivers are burned into the ground the water is routed over, so they keep
    # their courses (the finer terrain alone can tip a big river into another valley).
    routed = np.where(ocean, -1e7, _burn_rivers(world, box, grid, elevation_m, ocean))   # sea below any channel
    _, receiver, order = hydrology.priority_flood(routed, outlet, wraps=False)
    filled = hydrology.priority_flood(elevation_m, outlet, wraps=False)[0]   # water levels on the real ground
    discharge, catchment = _rivers(receiver, order, outlet, closed, precip, temperature,
                                   grid, cfg, inflow=inflow, runoff=runoff)
    stream = (catchment >= cfg.river_min_catchment_km2) & ~ocean & ~lake
    river = stream & (discharge >= cfg.river_min_discharge_m3s)

    biome[salt_flat] = BIOME_ID["Salt flat"]
    biome[lake & (temperature >= -10)] = BIOME_ID["Lake"]

    # Plates, for the Plates layer.
    lay = t.layout
    if lay is None:     # Earth: no generated plates
        piece = plate = np.zeros(ocean.shape, int)
        boundary, convergence = np.zeros(ocean.shape, bool), np.zeros(ocean.shape)
    else:
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
    finish_world(region, seasons, env)
    # Facts that need the whole globe come from the world.
    region.species_native = np.stack([nearest(layer) for layer in world.species_native]) & (
        region.species_suitability >= 0.35)
    for name in ("wind_east", "wind_north", "current_east", "current_north", "glacial_land", "glacial_ice"):
        setattr(region, name, nearest(getattr(world, name)))
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


def _burn_rivers(world, box, grid, elevation_m, ocean):
    """Region elevation (for routing water only) with a channel cut along every world
    river inside the region and into it: from each river cell's centre to the centre of
    the cell it drains to. The channel lies deep below the world's ground and always falls
    a little from cell to cell downstream, cutting through the rims of basins the world
    map filled in, so the water keeps the world's course (the finer terrain alone can tip
    a big river into another valley). The ground shown is not changed."""
    h, w = world.elevation_m.shape
    f = box.factor
    H, W = grid.lat.shape
    cfg = world.cfg
    rr = np.arange(max(box.row0 - 1, 0), min(box.row0 + box.rows + 1, h))
    cols = (box.col0 + np.arange(-1, box.cols + 1)) % w
    sub = np.ix_(rr, cols)
    river = (world.discharge_m3s[sub] >= cfg.river_min_discharge_m3s) & ~world.ocean[sub]   # on through lakes
    ys, xs = np.nonzero(river)
    if len(ys) == 0:
        return elevation_m
    cells = rr[ys] * w + cols[xs]
    in_set = set(int(c) for c in cells)
    rec = world.receiver
    ground = world.elevation_m.ravel()
    sea = world.ocean.ravel()
    lake = world.lake.ravel()
    lake_level = np.nan_to_num(world.lake_level_m.ravel(), nan=0.0) if world.lake_level_m is not None else ground
    DROP = 0.5                                     # metres per world cell, at least

    def end_level(c):
        if c < 0:
            return None
        if sea[c]:
            return min(float(ground[c]), 0.0)
        if lake[c]:
            return float(lake_level[c])
        return float(ground[c])

    # From the sources downstream: never above the ground, always a little below the
    # cell upstream (rivers cut through the rims of basins the world map filled in).
    cell_list = [int(c) for c in cells]
    donors_left = {c: 0 for c in cell_list}
    for c in cell_list:
        r = int(rec[c])
        if r in donors_left:
            donors_left[r] += 1
    level = {c: float(ground[c]) for c in cell_list}
    ready = [c for c in cell_list if donors_left[c] == 0]
    while ready:
        c = ready.pop()
        r = int(rec[c])
        if r in donors_left:
            level[r] = min(level[r], level[c] - DROP)
            donors_left[r] -= 1
            if donors_left[r] == 0:
                ready.append(r)
    out = elevation_m.copy()
    n = 2 * f + 1
    t = np.linspace(0.0, 1.0, n)
    for a in cells:
        a = int(a)
        b = int(rec[a])
        if b < 0:
            continue
        za = level[a]
        zb = level[b] if b in level else min(za - DROP, end_level(b))
        ay, ax = divmod(a, w)
        by, bx = divmod(b, w)
        fy0 = (ay - box.row0) * f + (f - 1) / 2
        fx0 = ((ax - box.col0 + w // 2) % w - w // 2) * f + (f - 1) / 2
        fy1 = fy0 + (by - ay) * f
        fx1 = fx0 + (((bx - ax + w // 2) % w) - w // 2) * f
        py = np.rint(fy0 + (fy1 - fy0) * t).astype(int)
        px = np.rint(fx0 + (fx1 - fx0) * t).astype(int)
        z = za + (zb - za) * t - CHANNEL_DEPTH_M
        inside = (py >= 0) & (py < H) & (px >= 0) & (px < W)
        if inside.any():
            np.minimum.at(out, (py[inside], px[inside]), z[inside])
    return np.where(ocean, elevation_m, out)


def _basin_sinks(world, box, grid):
    """Where the water of a closed basin ends (and evaporates) on the world map: the centre
    of each such world cell in the region. Rivers flowing into a closed lake end there,
    as on the world map, instead of draining out of the basin another way."""
    h, w = world.elevation_m.shape
    f = box.factor
    out = np.zeros(grid.lat.shape, bool)
    if world.closed is None or not world.closed.any():
        return out
    rows = np.arange(box.row0, box.row0 + box.rows)
    cols = (box.col0 + np.arange(box.cols)) % w
    closed = world.closed[np.ix_(rows, cols)]
    rec = world.receiver[(rows[:, None] * w + cols[None, :])]
    rec_closed = np.where(rec >= 0, world.closed.ravel()[np.maximum(rec, 0)], False)
    ys, xs = np.nonzero(closed & ~rec_closed)
    a, b = (f - 1) // 2, f // 2 + 1
    for y, x in zip(ys, xs):
        out[y * f + a:y * f + b, x * f + a:x * f + b] = True
    return out


def _edge_outlets(world, box, grid):
    """Where water may leave the region over its edge: only where the world's own rivers
    and drainage leave it, so a river running near the edge stays on its course instead
    of spilling out over the nearest bit of edge. (If nothing drains out that way, the
    whole edge is open, as for a closed inland region.)"""
    h, w = world.elevation_m.shape
    f = box.factor
    inside = np.zeros((h, w), bool)
    cols = (box.col0 + np.arange(box.cols)) % w
    inside[box.row0:box.row0 + box.rows][:, cols] = True
    rec = world.receiver
    flat_inside = inside.ravel()
    leave = np.flatnonzero(flat_inside & (rec >= 0))
    leave = leave[~flat_inside[rec[leave]]]
    out = np.zeros(grid.lat.shape, bool)
    if len(leave) == 0:
        out[0, :] = out[-1, :] = out[:, 0] = out[:, -1] = True
        return out
    ry, rx = np.divmod(leave, w)
    by, bx = ry - box.row0, (rx - box.col0) % w           # the world cell, in the box
    ty, tx = np.divmod(rec[leave], w)
    dy = np.sign(ty - ry)
    dx = np.sign(((tx - rx + w // 2) % w) - w // 2)
    H, W = grid.lat.shape
    for y, x, sy, sx in zip(by, bx, dy, dx):
        rows = slice(y * f, (y + 1) * f)
        cols_ = slice(x * f, (x + 1) * f)
        if sy < 0 and y == 0:
            out[0, cols_] = True
        if sy > 0 and y == box.rows - 1:
            out[H - 1, cols_] = True
        if sx < 0 and x == 0:
            out[rows, 0] = True
        if sx > 0 and x == box.cols - 1:
            out[rows, W - 1] = True
    return out


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

