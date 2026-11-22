"""Turns a World into RGB images for the different map layers."""
import numpy as np
from matplotlib import colormaps

from .generator import BIOMES, BIOME_ID, World, blur

LAYERS = ["Biomes", "Elevation", "Plates", "Temperature", "Precipitation",
          "Growing season", "Farmland", "Movement", "Rivers & harbours", "Wild species", "Minerals"]
MINERAL_COLOURS = {  # strongest mineral in each cell on the Minerals layer
    "copper": (0.85, 0.45, 0.15), "tin": (0.55, 0.75, 0.95), "iron": (0.55, 0.15, 0.1),
    "gold": (1.0, 0.85, 0.1), "silver": (0.85, 0.85, 0.9), "coal": (0.1, 0.1, 0.1),
    "oil": (0.5, 0.15, 0.6), "saltpeter": (0.95, 0.95, 0.75),
}
_BIOME_RGB = np.array([c for _, c in BIOMES])


def hillshade(elevation_m, strength, grid):
    """Light from the north-west; returns a multiplier around 1.

    Uses slope in metres per km, so relief looks the same at every resolution."""
    if grid.wraps:
        gx = (np.roll(elevation_m, -1, axis=1) - np.roll(elevation_m, 1, axis=1)) / 2
    else:
        gx = np.gradient(elevation_m, axis=1)
    gy = np.gradient(elevation_m, axis=0)
    kpp = grid.km_per_px
    return np.clip(1 + strength * (gx + gy) / kpp / 18.0, 0.45, 1.45)


def render(world: World, layer: str = "Biomes") -> np.ndarray:
    hs = world.cfg.hillshade
    if layer == "Biomes":
        img = _BIOME_RGB[world.biome].copy()
        # Ocean shading by depth so shelves, ridges and arcs show up.
        depth = np.clip(-world.elevation_m / world.cfg.max_depth_m, 0, 1)
        deep, shallow = np.array(BIOMES[0][1]), np.array(BIOMES[1][1])
        sea = world.ocean & (world.biome != BIOME_ID["Sea ice"])
        mix = depth[sea][:, None] ** 0.6
        img[sea] = shallow * (1 - mix) + deep * mix
        land = ~world.ocean & ~world.lake
        # Blend neighbouring land colours slightly so biome borders read as gradual transitions.
        if world.cfg.biome_blend > 0:
            wraps = world.grid.wraps
            weight = blur(land.astype(float), world.cfg.biome_blend, wraps)
            for ch in range(3):
                soft = (blur(np.where(land, img[..., ch], 0.0), world.cfg.biome_blend, wraps)
                        / np.maximum(weight, 1e-6))
                img[..., ch] = np.where(land, soft, img[..., ch])
        shade = hillshade(world.elevation_m, hs, world.grid)
        img[land] *= shade[land, None]
        return np.clip(img, 0, 1)

    if layer == "Elevation":
        e = world.elevation_m
        img = np.zeros(e.shape + (3,))
        land = ~world.ocean
        # Fixed scales (not this map's own max), so regions match the world map.
        img[land] = colormaps["terrain"](0.25 + 0.75 * np.clip(e[land] / world.cfg.max_elevation_m, 0, 1) ** 0.6)[:, :3]
        depth = np.clip(e[~land] / -world.cfg.max_depth_m, 0, 1) ** 0.7
        img[~land] = colormaps["Blues"](0.35 + 0.55 * depth)[:, :3]
        img *= hillshade(e, max(hs, 0.6), world.grid)[..., None]
        return np.clip(img, 0, 1)

    if layer == "Plates":
        t = world.terrain
        rng = np.random.default_rng(3)
        palette = rng.uniform(0.35, 0.9, (t.plate.max() + 1, 3))
        img = palette[t.plate]
        img[t.crust > 0.5] *= 0.75
        img[~world.ocean] = img[~world.ocean] * 0.55 + 0.45 * np.array([0.55, 0.45, 0.3])
        b = t.boundary
        c = t.convergence
        img[b & (c > 0.1)] = (0.85, 0.1, 0.1)    # colliding: mountains / island arcs
        img[b & (c < -0.1)] = (0.1, 0.3, 0.95)   # pulling apart: rifts / ridges
        img[b & (np.abs(c) <= 0.1)] = (0.2, 0.2, 0.2)  # sliding past
        return img

    if layer == "Temperature":
        return colormaps["RdYlBu_r"](np.clip((world.temperature_c + 30) / 60, 0, 1))[..., :3]

    if layer == "Precipitation":
        img = colormaps["YlGnBu"](np.clip(world.precip_mm / 3000, 0, 1))[..., :3]
        img[world.ocean] *= 0.6
        return img

    water = world.ocean | world.lake
    shade = hillshade(world.elevation_m, max(hs, 0.4), world.grid)[..., None]
    if layer == "Growing season":   # months warm and wet enough for crops, 0..12
        img = colormaps["YlGn"](0.1 + 0.9 * np.clip(world.growing_months / 12, 0, 1))[..., :3]
        return _with_water(img * shade, world, water)

    if layer == "Farmland":         # arable suitability 0..1
        img = colormaps["RdYlGn"](0.05 + 0.9 * np.clip(world.arable, 0, 1))[..., :3]  # red poor, green good
        return _with_water(img * shade, world, water)

    if layer == "Movement":         # effort to cross on foot: light = easy, dark = hard
        cost = np.log(np.where(water, 1.0, world.movement_cost)) / np.log(20)
        img = colormaps["magma_r"](0.05 + 0.85 * np.clip(cost, 0, 1))[..., :3]
        return _with_water(img, world, water)

    if layer == "Rivers & harbours":
        from .hydrology import edge_properties

        grey = _BIOME_RGB[world.biome].mean(axis=-1, keepdims=True)
        img = np.repeat(0.55 + 0.4 * grey, 3, axis=-1) * shade
        img = _with_water(img, world, water)
        # Harbours: coastal cells from pale yellow (open coast) to red (fine natural harbour).
        coast = world.harbour > 0.05
        img[coast] = colormaps["autumn_r"](0.2 + 0.8 * world.harbour[coast])[:, :3]
        # Rapids (red) and dry-season fords (gold) on the river edges beside each cell.
        for side in ("east", "south"):
            props = edge_properties(world, side)
            img[props["rapids"]] = (0.85, 0.1, 0.1)
            img[props["fordable"] & ~props["rapids"]] = (0.95, 0.75, 0.1)
        return img

    if layer == "Wild species":     # how many domesticable species live wild here
        count = world.species_native.sum(axis=0)
        img = colormaps["viridis"](np.clip(count / 10, 0, 1))[..., :3] * 0.85 + 0.1
        img[count == 0] = (0.55, 0.55, 0.55)
        return _with_water(img * shade, world, water)

    if layer == "Minerals":         # the strongest deposit in each cell
        grey = _BIOME_RGB[world.biome].mean(axis=-1, keepdims=True)
        img = np.repeat(0.6 + 0.35 * grey, 3, axis=-1) * shade
        img = _with_water(img, world, world.lake | (world.ocean & (world.oil < 0.3)))
        names = list(MINERAL_COLOURS)
        stack = np.stack([getattr(world, n) for n in names])
        best = stack.argmax(axis=0)
        strength = stack.max(axis=0)
        colours = np.array([MINERAL_COLOURS[n] for n in names])
        rich = strength > 0.3
        mix = np.clip((strength[rich] - 0.3) / 0.4, 0.4, 1)[:, None]
        img[rich] = img[rich] * (1 - mix) + colours[best[rich]] * mix
        return np.clip(img, 0, 1)

    raise ValueError(layer)


def _with_water(img, world, water):
    img = img.copy()
    depth = np.clip(-world.elevation_m / world.cfg.max_depth_m, 0, 1) ** 0.6
    sea = np.array([0.35, 0.5, 0.7]) * (1 - depth[..., None]) + np.array([0.15, 0.25, 0.45]) * depth[..., None]
    img[water] = sea[water]
    img[world.lake] = (0.4, 0.6, 0.85)
    return np.clip(img, 0, 1)


# --------------------------------------------------------------------- rivers
RIVER_COLOUR = (0.24, 0.47, 0.78)
RIVER_LAYERS = ("Biomes", "Elevation", "Growing season", "Farmland", "Movement", "Rivers & harbours")


def _river_network(world: World):
    """All river segments down to the smallest stream shown when zoomed in (cached per world)."""
    cache = getattr(world, "_river_network", None)
    if cache is None:
        from .hydrology import _chaikin

        # Rivers run along cell edges: corner (r, c) sits at (c - 0.5, r - 0.5) in pixel
        # coordinates. Region pixels are already fine, so one smoothing pass is enough there.
        smooth = 2 if world.grid.wraps else 1
        detailed = not world.grid.wraps   # enhanced regions get meanders and deltas
        segs, flows = [], []

        def add(xs, ys, q):
            pts = np.column_stack([xs, ys])
            segs.append(np.stack([pts[:-1], pts[1:]], axis=1))
            flows.append(0.5 * (q[:-1] + q[1:]))

        for i, (rows, cols, q) in enumerate(world.river_paths):
            xs, ys, q = _chaikin(cols - 0.5, rows - 0.5, q, smooth)
            if detailed:
                rng = np.random.default_rng((world.cfg.seed, int(rows[0]), int(cols[0]) % 100000, i))
                xs, ys = _meander(xs, ys, q, world.grid.km_per_px, rng)
                if _reaches_sea(world, rows[-1], cols[-1]):
                    for branch in _delta(xs, ys, q, world.grid.km_per_px, world.cfg, rng):
                        add(*branch)
            add(xs, ys, q)
        cache = ((np.concatenate(segs), np.concatenate(flows)) if segs
                 else (np.zeros((0, 2, 2)), np.zeros(0)))
        world._river_network = cache
    return cache


def _meander(xs, ys, q, km_per_px, rng):
    """Bend a river into meanders (drawing only; the river edge data is unchanged).

    Real meanders repeat about every 12 river widths and swing about a third of that to
    each side, so only big rivers on detailed maps get visible loops. The swing is capped
    at 1.5 cells so the drawn line stays close to the cell edges the river follows.
    """
    if len(xs) < 4:
        return xs, ys
    step = np.hypot(np.diff(xs), np.diff(ys))
    s = np.concatenate([[0], np.cumsum(step)])
    width_px = 7 * np.sqrt(np.maximum(q, 0)) / 1000 / km_per_px
    wavelength = 12 * width_px
    if wavelength.max() < 3:  # too small to see at this scale
        return xs, ys
    # Direction across the river (unit normal), smoothed so loops don't kink at corners.
    tx = np.gradient(xs)
    ty = np.gradient(ys)
    norm = np.maximum(np.hypot(tx, ty), 1e-9)
    nx, ny = -ty / norm, tx / norm
    phase = 2 * np.pi * np.cumsum(np.concatenate([[0], step / np.maximum(wavelength[1:], 1e-6)]))
    phase += rng.uniform(0, 2 * np.pi) + 0.6 * np.sin(phase * 0.37 + rng.uniform(0, 6))  # irregular loops
    swing = np.clip(0.3 * wavelength, 0, 1.5) * np.where(wavelength >= 3, 1.0, 0.0)
    taper = np.clip(np.minimum(s, s[-1] - s) / np.maximum(wavelength, 1), 0, 1)  # straight at the ends
    offset = swing * taper * np.sin(phase)
    return xs + nx * offset, ys + ny * offset


def _reaches_sea(world, row, col):
    """Whether a river path's end corner touches the sea (not a lake)."""
    h, w = world.ocean.shape
    r, c = int(row), int(col)
    cells = [(rr, cc) for rr in (r - 1, r) for cc in (c - 1, c) if 0 <= rr < h and 0 <= cc < w]
    return any(world.ocean[rr, cc] for rr, cc in cells)


def _delta(xs, ys, q, km_per_px, cfg, rng):
    """Distributary channels fanning out where a big river reaches the sea."""
    if q[-1] < cfg.delta_min_flow_m3s or len(xs) < 6:
        return []
    size = np.clip(np.sqrt(q[-1] / cfg.delta_min_flow_m3s) * 40 / km_per_px, 2, 12)  # ~40 km+ fans
    back = len(xs) - 1
    travelled = 0.0
    while back > 0 and travelled < size:  # split point, about `size` px upstream
        travelled += np.hypot(xs[back] - xs[back - 1], ys[back] - ys[back - 1])
        back -= 1
    sx, sy = xs[back], ys[back]
    dx, dy = xs[-1] - sx, ys[-1] - sy
    length = max(np.hypot(dx, dy), 1e-6)
    ux, uy = dx / length, dy / length
    branches = []
    n = int(rng.integers(2, 5))
    for k in range(n):
        spread = (k - (n - 1) / 2) / max(n - 1, 1) * size * 0.9 + rng.normal(0, size * 0.08)
        ex = xs[-1] - uy * spread + ux * rng.uniform(0, size * 0.25)
        ey = ys[-1] + ux * spread + uy * rng.uniform(0, size * 0.25)
        t = np.linspace(0, 1, 8)
        bend = np.sin(np.pi * t) * rng.normal(0, size * 0.08)
        bx = sx + (ex - sx) * t - uy * bend
        by = sy + (ey - sy) * t + ux * bend
        branches.append((bx, by, np.full(8, q[-1] / n)))
    return branches


def river_segments(world: World, offset: int = 0, zoom: float = 1.0, xlim=None):
    """Line segments and widths (points) of the rivers to draw.

    offset  horizontal scroll in pixels (the map wraps)
    zoom    how far the view is magnified; smaller streams appear as you zoom in,
            like on a real atlas
    xlim    visible x range; segments outside it are skipped to keep drawing fast
    """
    segs, flows = _river_network(world)
    cfg = world.cfg
    threshold = max(cfg.river_detail_min_m3s, cfg.river_min_discharge_m3s / zoom ** 1.5)
    keep = flows >= threshold
    segs, flows = segs[keep].copy(), flows[keep]
    widths = np.clip(0.2 * (flows / 600.0) ** 0.35, 0.12, 1.6) * zoom ** 0.4
    if not world.grid.wraps:  # a region: no scrolling or wrapping
        return segs, widths
    w = world.grid.width
    # Shift for scrolling, keeping each (tiny) segment in one piece across the seam.
    segs[..., 0] -= offset
    segs[..., 0] -= (np.floor(segs[:, 0, 0] / w) * w)[:, None]
    all_segs = np.concatenate([segs + [[s, 0]] for s in (-w, 0, w)])
    all_widths = np.tile(widths, 3)
    if xlim is not None:
        lo, hi = min(xlim) - 2, max(xlim) + 2
        x = all_segs[..., 0]
        visible = (x.max(axis=1) >= lo) & (x.min(axis=1) <= hi)
        all_segs, all_widths = all_segs[visible], all_widths[visible]
    return all_segs, all_widths


def save_map(world: World, layer: str, path: str, offset: int = 0, rivers: bool = True, px_per_cell: int = 2):
    """Save a layer as a PNG, with rivers drawn as smooth lines."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    h, w = world.elevation_m.shape
    dpi = 100
    fig = plt.figure(figsize=(w * px_per_cell / dpi, h * px_per_cell / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.imshow(np.roll(render(world, layer), -offset, axis=1), interpolation="nearest")
    if rivers and layer in RIVER_LAYERS:
        segs, widths = river_segments(world, offset)
        # Same line width relative to the map as in the viewer (11.4 inch wide map).
        scale = (w * px_per_cell / dpi) / 11.4
        ax.add_collection(LineCollection(segs, colors=[RIVER_COLOUR], linewidths=widths * scale,
                                         capstyle="round", joinstyle="round"))
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------- globe
GLOBE_BACKGROUND = (0.03, 0.04, 0.08)


def _globe_radius(size, zoom):
    return size / 2 * 0.95 * zoom


def globe_view(img, lon0, lat0, size, zoom=1.0, width=None):
    """Show a map image as a globe (orthographic projection, as seen from space).

    img          an equirectangular image of the whole world (H x W x 3)
    lon0, lat0   the point at the centre of the view, degrees
    zoom         1 = the whole visible hemisphere fits the picture's height
    width        picture width (default: square); the globe stays centred
    Returns (image size x width x 3, lat, lon, inside) where lat/lon give the map
    position under every pixel and `inside` marks pixels on the globe.
    """
    width = width or size
    r = _globe_radius(size, zoom)
    x, y = np.meshgrid((np.arange(width) + 0.5 - width / 2) / r, -(np.arange(size) + 0.5 - size / 2) / r)
    rho2 = x * x + y * y
    inside = rho2 <= 1
    z = np.sqrt(np.clip(1 - rho2, 0, 1))
    phi0, lam0 = np.radians(lat0), np.radians(lon0)
    lat = np.degrees(np.arcsin(np.clip(z * np.sin(phi0) + y * np.cos(phi0), -1, 1)))
    lon = (lon0 + np.degrees(np.arctan2(x, z * np.cos(phi0) - y * np.sin(phi0))) + 180) % 360 - 180

    # Bilinear sample of the map image (wrapping east-west).
    h, w = img.shape[:2]
    fy = np.clip((90 - lat) / 180 * h - 0.5, 0, h - 1)
    fx = (lon + 180) / 360 * w - 0.5
    y0 = np.floor(fy).astype(int)
    x0 = np.floor(fx).astype(int)
    ty, tx = (fy - y0)[..., None], (fx - x0)[..., None]
    y1 = np.minimum(y0 + 1, h - 1)
    x0w, x1w = x0 % w, (x0 + 1) % w
    out = ((img[y0, x0w] * (1 - tx) + img[y0, x1w] * tx) * (1 - ty)
           + (img[y1, x0w] * (1 - tx) + img[y1, x1w] * tx) * ty)
    out = out * (0.55 + 0.45 * z)[..., None]  # darker toward the edge, like a lit sphere
    out[~inside] = GLOBE_BACKGROUND
    return np.clip(out, 0, 1), lat, lon, inside


def globe_project(lon, lat, lon0, lat0, size, zoom=1.0, width=None):
    """Pixel position (x, y) of map points on the globe view, and whether they face the viewer."""
    phi, lam = np.radians(lat), np.radians(lon - lon0)
    phi0 = np.radians(lat0)
    facing = np.sin(phi0) * np.sin(phi) + np.cos(phi0) * np.cos(phi) * np.cos(lam) > 0
    gx = np.cos(phi) * np.sin(lam)
    gy = np.cos(phi0) * np.sin(phi) - np.sin(phi0) * np.cos(phi) * np.cos(lam)
    r = _globe_radius(size, zoom)
    return (width or size) / 2 + gx * r - 0.5, size / 2 - gy * r - 0.5, facing


def globe_rivers(world, lon0, lat0, size, zoom=1.0, width=None):
    """River segments and widths for the globe view (only the side facing the viewer)."""
    segs, flows = _river_network(world)
    cfg = world.cfg
    map_zoom = 1.5 * zoom  # a whole hemisphere shows about as much as a 1.5x map zoom
    threshold = max(cfg.river_detail_min_m3s, cfg.river_min_discharge_m3s / map_zoom ** 1.5)
    keep = flows >= threshold
    segs, flows = segs[keep], flows[keep]
    h, w = world.elevation_m.shape
    lon = (segs[..., 0] + 0.5) / w * 360 - 180
    lat = 90 - (segs[..., 1] + 0.5) / h * 180
    gx, gy, facing = globe_project(lon, lat, lon0, lat0, size, zoom, width)
    visible = facing.all(axis=1)
    out = np.stack([gx, gy], axis=-1)[visible]
    widths = np.clip(0.2 * (flows[visible] / 600.0) ** 0.35, 0.12, 1.6) * map_zoom ** 0.4
    return out, widths
