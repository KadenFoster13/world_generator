"""Water flow: drainage directions, river discharge, and river paths for drawing.

Everything works on the equirectangular grid and wraps east-west.
Pixels are addressed by flat index i = y * width + x.
"""
import heapq

import numpy as np

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
    flow = local_flow.ravel().astype(float).tolist()
    recv = receiver.tolist()
    ok = passes.ravel().tolist()
    for c in reversed(order.tolist()):
        r = recv[c]
        if r >= 0 and ok[c]:
            flow[r] += flow[c]
    return np.array(flow).reshape(local_flow.shape)


def river_paths(discharge, receiver, river, stop, width, smooth=2):
    """Turn river pixels into smooth polylines for drawing.

    river  bool mask of pixels that carry a river
    stop   bool mask where rivers end (ocean, lakes)
    Returns a list of (xs, ys, flows) arrays in pixel coordinates. x may run past
    the map edge when a river crosses the east-west seam.
    """
    h, w = river.shape
    rv = river.ravel()
    st = stop.ravel()
    q = discharge.ravel()
    idx = np.flatnonzero(rv)
    donors = np.zeros(h * w, dtype=np.int32)
    r = receiver[idx]
    valid = r >= 0
    np.add.at(donors, r[valid], 1)

    # Lake pixels that drain into a river pixel: the river leaves the lake there.
    outlet_of = np.full(h * w, -1, dtype=np.int64)
    water = np.flatnonzero(st & (receiver >= 0))
    leaving = water[rv[receiver[water]]]
    outlet_of[receiver[leaving]] = leaving

    starts = idx[(donors[idx] != 1)]  # river heads and confluences
    paths = []
    for s in starts.tolist():
        from_lake = donors[s] == 0 and outlet_of[s] >= 0
        pts = [int(outlet_of[s]), s] if from_lake else [s]
        cur = s
        while True:
            nxt = int(receiver[cur])
            if nxt < 0:
                break
            pts.append(nxt)
            if st[nxt] or not rv[nxt] or donors[nxt] != 1:
                break
            cur = nxt
        if len(pts) < 2:
            continue
        ys, xs = np.divmod(np.array(pts), w)
        xs = xs.astype(float)
        # Unwrap across the seam so the line stays continuous.
        steps = (np.diff(xs) + w / 2) % w - w / 2
        xs = np.concatenate([[xs[0]], xs[0] + np.cumsum(steps)])
        ys = ys.astype(float)
        flows = q[pts].astype(float)
        # Meet the water exactly at the shoreline: the edge between the last land pixel
        # and the lake or sea (instead of running on to the water pixel's centre), and
        # likewise start lake outlets at the lake's edge.
        if st[pts[-1]]:
            xs[-1], ys[-1] = (xs[-1] + xs[-2]) / 2, (ys[-1] + ys[-2]) / 2
            flows[-1] = flows[-2]
        if from_lake and len(pts) > 2:
            xs[0], ys[0] = (xs[0] + xs[1]) / 2, (ys[0] + ys[1]) / 2
            flows[0] = flows[1]
        xs, ys, flows = _chaikin(xs, ys, flows, smooth)
        paths.append((xs, ys, flows))
    return paths


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
