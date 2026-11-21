"""Turns a World into RGB images for the different map layers."""
import numpy as np
from matplotlib import colormaps

from .generator import BIOMES, BIOME_ID, World, blur

LAYERS = ["Biomes", "Elevation", "Plates", "Temperature", "Precipitation"]
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

    raise ValueError(layer)


# --------------------------------------------------------------------- rivers
RIVER_COLOUR = (0.24, 0.47, 0.78)
RIVER_LAYERS = ("Biomes", "Elevation")


def _river_network(world: World):
    """All river segments down to the smallest stream shown when zoomed in (cached per world)."""
    cache = getattr(world, "_river_network", None)
    if cache is None:
        from . import hydrology

        detail = world.stream & (world.discharge_m3s >= world.cfg.river_detail_min_m3s)
        # Region pixels are already fine, so one smoothing pass is enough (and draws faster).
        paths = hydrology.river_paths(world.discharge_m3s, world.receiver, detail,
                                      world.ocean | world.lake, world.elevation_m.shape[1],
                                      smooth=2 if world.grid.wraps else 1)
        segs, flows = [], []
        for xs, ys, q in paths:
            pts = np.column_stack([xs, ys])
            segs.append(np.stack([pts[:-1], pts[1:]], axis=1))
            flows.append(0.5 * (q[:-1] + q[1:]))
        cache = ((np.concatenate(segs), np.concatenate(flows)) if segs
                 else (np.zeros((0, 2, 2)), np.zeros(0)))
        world._river_network = cache
    return cache


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
