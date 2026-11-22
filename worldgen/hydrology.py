"""Water flow: drainage directions, river discharge, and river paths for drawing.

Everything works on the equirectangular grid and wraps east-west.
Pixels are addressed by flat index i = y * width + x.
"""
import heapq

import numpy as np

from . import _fast

EARTH_RADIUS_M = 6_371_000
SECONDS_PER_YEAR = 31_557_600
_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def priority_flood(elevation_m, ocean):
    """Work out where water drains from every land pixel.

    Starts at the coast and floods inland, always expanding from the lowest pixel
    reached so far (Barnes et al., 2014). Each pixel drains to the neighbour it was
    reached from, so water can always find a way to the sea, even out of a closed
    basin: it fills up to the basin's spill point and overflows there.

    Ties (pixels at the same filled level, i.e. inside a filled basin) are broken by
    the original ground height, so paths across a basin follow its real valley floor
    instead of running in straight lines.

    Returns
      filled    elevation with every basin filled to its spill point
      receiver  flat index of the pixel each pixel drains into (-1 for ocean)
      order     land pixels from the coast inward; reversed, it visits every pixel
                before the pixel it drains into
    """
    h, w = elevation_m.shape
    n = h * w
    z = elevation_m.ravel().tolist()
    filled = list(z)
    receiver = [-1] * n
    visited = bytearray(ocean.ravel().astype(np.uint8).tobytes())

    # Seeds: ocean pixels touching land.
    land = ~ocean
    touches_land = np.zeros_like(ocean)
    for dy, dx in _OFFSETS:
        nb = np.roll(land, (-dy, -dx), axis=(0, 1))
        if dy == 1:
            nb[-1] = False
        elif dy == -1:
            nb[0] = False
        touches_land |= nb
    seeds = np.flatnonzero(ocean & touches_land)
    if _fast.HAVE_NUMBA and len(seeds):
        filled, receiver, order = _fast.flood_kernel(
            elevation_m.ravel().astype(np.float64), seeds.astype(np.int64),
            ocean.ravel().astype(np.uint8), h, w)
        return filled.reshape(h, w), receiver, order
    heap = [(z[i], z[i], int(i)) for i in seeds]
    heapq.heapify(heap)

    order = []
    push, pop = heapq.heappush, heapq.heappop
    while heap:
        fc, _, c = pop(heap)
        y, x = divmod(c, w)
        for dy, dx in _OFFSETS:
            yy = y + dy
            if yy < 0 or yy >= h:
                continue
            j = yy * w + (x + dx) % w
            if visited[j]:
                continue
            visited[j] = 1
            receiver[j] = c
            fj = z[j] if z[j] > fc else fc
            filled[j] = fj
            order.append(j)
            push(heap, (fj, z[j], j))

    return (np.array(filled).reshape(h, w), np.array(receiver, dtype=np.int64),
            np.array(order, dtype=np.int64))


def pixel_area_m2(lat_deg, width):
    """Area of each pixel on an Earth-sized sphere."""
    side = 2 * np.pi * EARTH_RADIUS_M / width
    return side * side * np.cos(np.radians(lat_deg))


def runoff_mm(precip_mm, temperature_c, cfg):
    """Rain that isn't evaporated or used by plants, so it ends up in rivers."""
    potential_evap = cfg.evaporation_per_degree_mm * np.maximum(temperature_c, 0)
    return precip_mm - np.minimum(precip_mm * 0.9, potential_evap)


def accumulate(local_flow, receiver, order, passes):
    """Sum flow downstream. `passes[i]` False means water at i goes no further (it evaporates)."""
    if _fast.HAVE_NUMBA:
        flow = _fast.accumulate_kernel(local_flow.ravel().astype(np.float64).copy(),
                                       receiver.astype(np.int64), order.astype(np.int64),
                                       passes.ravel().astype(np.bool_))
        return flow.reshape(local_flow.shape)
    flow = local_flow.ravel().astype(float).tolist()
    recv = receiver.tolist()
    ok = passes.ravel().tolist()
    for c in reversed(order.tolist()):
        r = recv[c]
        if r >= 0 and ok[c]:
            flow[r] += flow[c]
    return np.array(flow).reshape(local_flow.shape)


def corner_grid(values, wraps=True, how="mean"):
    """Values at cell corners. Corner (r, c) is the top-left corner of cell (r, c), shared by
    cells (r-1, c-1), (r-1, c), (r, c-1) and (r, c) (fewer at the map's edges)."""
    left = np.roll(values, 1, axis=1) if wraps else np.concatenate([values[:, :1], values[:, :-1]], axis=1)
    if how == "any":
        pair = values | left
        return pair | np.concatenate([pair[:1], pair[:-1]], axis=0)
    pair = (values + left) / 2
    return (pair + np.concatenate([pair[:1], pair[:-1]], axis=0)) / 2


def river_network(discharge, receiver, channel, water, surface_m, seasonality, lat, km_per_px,
                  wraps=True):
    """Route rivers along cell edges.

    Each river step between two cells (including diagonal ones) becomes a walk along
    cell corners; a diagonal step turns at whichever of the two possible corners is
    lower. Rivers end at the first corner that touches the sea or a lake, and rivers
    leaving a lake start at the last corner touching it.

    channel     bool: cells whose step to their receiver carries a river
    water       bool: sea and lakes
    surface_m   water-surface height (filled terrain, 0 at sea) for river slopes
    seasonality 0..1 rain seasonality, for dry-season flow

    Returns (paths, edges):
      paths  list of (rows, cols, flows) corner paths for drawing; cols may run past
             the map edge where a river crosses the east-west seam
      edges  dict of (H, W) arrays. "_east" arrays describe the edge between cell (r, c)
             and (r, c+1), "_south" arrays the edge between (r, c) and (r+1, c):
             river_flow_*      mean flow, m3/s (0 = no river)
             river_dry_flow_*  dry-season flow, m3/s
             river_slope_*     fall along the river, m per km
    """
    h, w = discharge.shape
    corner_z = corner_grid(surface_m, wraps)
    wet = corner_grid(water, wraps, how="any").ravel()
    zc = corner_z.ravel()
    zs = surface_m.ravel()
    q = discharge.ravel()
    dry_factor = (1 - 0.8 * seasonality).ravel()
    ch = channel.ravel()
    is_water = water.ravel()
    edges = {f"{name}_{side}": np.zeros((h, w)) for name in ("river_flow", "river_dry_flow", "river_slope")
             for side in ("east", "south")}
    cos_row = np.maximum(np.cos(np.radians(lat[:, 0])), 0.05)

    # Cells that carry a river onward: channel cells, and lake cells draining into one.
    idx = np.flatnonzero(ch)
    donors = np.zeros(h * w, dtype=np.int32)
    r_idx = receiver[idx]
    np.add.at(donors, r_idx[r_idx >= 0], 1)
    outlet_of = np.full(h * w, -1, dtype=np.int64)
    lakes_out = np.flatnonzero(is_water & (receiver >= 0))
    lakes_out = lakes_out[ch[receiver[lakes_out]]]
    outlet_of[receiver[lakes_out]] = lakes_out

    paths = []
    for s in idx[donors[idx] != 1].tolist():          # river heads and confluences
        cells = [s]
        if donors[s] == 0 and outlet_of[s] >= 0:      # leaves a lake
            cells = [int(outlet_of[s]), s]
        cur = s
        while True:
            nxt = int(receiver[cur])
            if nxt < 0:
                break
            cells.append(nxt)
            if is_water[nxt] or not ch[nxt] or donors[nxt] != 1:
                break
            cur = nxt
        if len(cells) < 2:
            continue
        # Walk the corners. Each corner keeps the flow (and slope) of the cell step that
        # reached it. Slope is measured between the river's own cells, not the corners,
        # whose heights also average in the hillsides beside the river.
        corners = [cells[0]]
        flows = [q[cells[1]] if is_water[cells[0]] else q[cells[0]]]
        slopes = [0.0]
        for a, b in zip(cells[:-1], cells[1:]):
            ra, ca = divmod(a, w)
            rb, cb = divmod(b, w)
            dc = cb - ca
            if wraps:
                dc = (dc + w // 2) % w - w // 2
            dr = rb - ra
            step_flow = q[b] if is_water[a] else q[a]
            step_km = km_per_px * np.hypot(dr, dc * cos_row[ra])
            step_slope = max(zs[a] - zs[b], 0.0) / max(step_km, 1e-6)
            if dr and dc:  # diagonal: turn at the lower of the two corners
                m1 = rb * w + ca
                m2 = ra * w + (ca + dc) % w
                corners.append(m1 if zc[m1] <= zc[m2] else m2)
                flows.append(step_flow)
                slopes.append(step_slope)
            corners.append(b)
            flows.append(step_flow)
            slopes.append(step_slope)
        # Trim to the shoreline: start at the last wet corner of a lake outlet,
        # end at the first wet corner after that.
        start = 0
        while start + 1 < len(corners) and wet[corners[start + 1]]:
            start += 1
        end = start + 1
        while end < len(corners) - 1 and not wet[corners[end]]:
            end += 1
        corners, flows, slopes = corners[start:end + 1], flows[start:end + 1], slopes[start:end + 1]
        if len(corners) < 2:
            continue
        rows, cols = np.divmod(np.array(corners), w)
        _record_edges(edges, rows, cols, np.array(flows, float), np.array(slopes, float),
                      dry_factor, w, wraps)
        cols = cols.astype(float)
        if wraps:  # unwrap across the seam so the line stays continuous
            steps = (np.diff(cols) + w / 2) % w - w / 2
            cols = np.concatenate([[cols[0]], cols[0] + np.cumsum(steps)])
        paths.append((rows.astype(float), cols, np.array(flows, float)))
    return paths, edges


def _record_edges(edges, rows, cols, flows, slopes, dry_factor, w, wraps):
    """Write one corner path's moves into the edge arrays."""
    h = edges["river_flow_east"].shape[0]
    for i in range(len(rows) - 1):
        r0, c0, r1, c1 = rows[i], cols[i], rows[i + 1], cols[i + 1]
        q, slope = flows[i + 1], slopes[i + 1]
        if r0 != r1:   # vertical move along column line c0: edge between cells (r, c0-1) and (r, c0)
            r, c, side = min(r0, r1), c0 - 1, "east"
        else:          # horizontal move along row line r0: edge between cells (r0-1, c) and (r0, c)
            dc = c1 - c0
            if wraps:
                dc = (dc + w // 2) % w - w // 2
            r, c, side = r0 - 1, c0 if dc > 0 else c0 - 1, "south"
        if r < 0 or r >= h:
            continue
        if c < 0:
            if not wraps:
                continue
            c %= w
        if c >= w:
            if not wraps:
                continue
            c %= w
        cell = (r, c)
        edges[f"river_flow_{side}"][cell] = max(edges[f"river_flow_{side}"][cell], q)
        edges[f"river_dry_flow_{side}"][cell] = max(edges[f"river_dry_flow_{side}"][cell],
                                                     q * dry_factor[r * w + c])
        edges[f"river_slope_{side}"][cell] = max(edges[f"river_slope_{side}"][cell], slope)


def river_width_m(flow):
    """Typical river width for a flow (m3/s)."""
    return 7.0 * np.sqrt(np.maximum(flow, 0))


def river_depth_m(flow):
    """Typical river depth for a flow (m3/s)."""
    return 0.27 * np.maximum(flow, 0) ** 0.39


def edge_properties(world, side):
    """Derived river facts for every edge on one side ("east" or "south") of each cell.

    Returns a dict of arrays: flow, dry_flow, width_m, depth_m, dry_depth_m, slope_m_per_km,
    river (bool), fordable (shallow enough to wade in the dry season), navigable (deep
    enough for river boats, and no rapids) and rapids.
    """
    cfg = world.cfg
    flow = getattr(world, f"river_flow_{side}")
    dry = getattr(world, f"river_dry_flow_{side}")
    slope = getattr(world, f"river_slope_{side}")
    river = flow > 0
    depth = river_depth_m(flow)
    dry_depth = river_depth_m(dry)
    rapids = river & (slope >= cfg.rapids_min_slope_m_per_km)
    return dict(flow=flow, dry_flow=dry, width_m=np.where(river, river_width_m(flow), 0.0),
                depth_m=np.where(river, depth, 0.0), dry_depth_m=np.where(river, dry_depth, 0.0),
                slope_m_per_km=slope, river=river,
                # Fords are at riffles, where the river runs about half its typical depth.
                fordable=river & (0.5 * dry_depth < cfg.ford_max_depth_m),
                navigable=river & (depth >= cfg.navigable_min_depth_m) & ~rapids,
                rapids=rapids)


def cell_river_flow(world):
    """Largest river flow along any of each cell's four edges (0 = no river touches it)."""
    e, s = world.river_flow_east, world.river_flow_south
    wraps = world.grid.wraps
    west = np.roll(e, 1, axis=1) if wraps else np.concatenate([np.zeros_like(e[:, :1]), e[:, :-1]], axis=1)
    north = np.concatenate([np.zeros_like(s[:1]), s[:-1]], axis=0)
    return np.maximum.reduce([e, west, s, north])


def _chaikin(xs, ys, vals, iterations):
    """Corner-cutting smoothing that keeps both end points fixed."""
    for _ in range(iterations):
        if len(xs) < 3:
            break
        qx = 0.75 * xs[:-1] + 0.25 * xs[1:]
        rx = 0.25 * xs[:-1] + 0.75 * xs[1:]
        qy = 0.75 * ys[:-1] + 0.25 * ys[1:]
        ry = 0.25 * ys[:-1] + 0.75 * ys[1:]
        qv = 0.75 * vals[:-1] + 0.25 * vals[1:]
        rv = 0.25 * vals[:-1] + 0.75 * vals[1:]
        xs = np.concatenate([[xs[0]], np.column_stack([qx, rx]).ravel()[1:-1], [xs[-1]]])
        ys = np.concatenate([[ys[0]], np.column_stack([qy, ry]).ravel()[1:-1], [ys[-1]]])
        vals = np.concatenate([[vals[0]], np.column_stack([qv, rv]).ravel()[1:-1], [vals[-1]]])
    return xs, ys, vals
