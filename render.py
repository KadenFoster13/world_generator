"""Turns a World into RGB images for the different map layers."""
import numpy as np
from matplotlib import colormaps

from worldgen import BIOMES, BIOME_ID, World, blur

LAYERS = ["Biomes", "Elevation", "Plates", "Temperature", "Precipitation"]
_BIOME_RGB = np.array([c for _, c in BIOMES])


def hillshade(elevation_m, strength):
    """Light from the north-west; returns a multiplier around 1."""
    gx = (np.roll(elevation_m, -1, axis=1) - np.roll(elevation_m, 1, axis=1)) / 2
    gy = np.gradient(elevation_m, axis=0)
    return np.clip(1 + strength * (gx + gy) / 900.0, 0.45, 1.45)


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
            weight = blur(land.astype(float), world.cfg.biome_blend)
            for ch in range(3):
                soft = blur(np.where(land, img[..., ch], 0.0), world.cfg.biome_blend) / np.maximum(weight, 1e-6)
                img[..., ch] = np.where(land, soft, img[..., ch])
        shade = hillshade(world.elevation_m, hs)
        img[land] *= shade[land, None]
        return np.clip(img, 0, 1)

    if layer == "Elevation":
        e = world.elevation_m
        img = np.zeros(e.shape + (3,))
        land = ~world.ocean
        img[land] = colormaps["terrain"](0.25 + 0.75 * (e[land] / max(e.max(), 1)) ** 0.6)[:, :3]
        depth = (e[~land] / min(e.min(), -1)) ** 0.7
        img[~land] = colormaps["Blues"](0.35 + 0.55 * depth)[:, :3]
        img *= hillshade(e, max(hs, 0.6))[..., None]
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
RIVER_COLOUR = (0.17, 0.39, 0.70)
RIVER_LAYERS = ("Biomes", "Elevation")


def river_segments(world: World, offset: int = 0):
    """Line segments and widths (points) for every visible river, shifted by `offset` px.

    Each segment is drawn three times (shifted one map-width left and right) so rivers
    that cross the east-west seam still show up after scrolling.
    """
    import hydrology

    cache = getattr(world, "_river_paths", None)
    if cache is None:
        cache = hydrology.river_paths(world.discharge_m3s, world.receiver, world.river,
                                      world.ocean | world.lake, world.cfg.width)
        world._river_paths = cache
    w = world.cfg.width
    threshold = world.cfg.river_min_discharge_m3s
    segs, widths = [], []
    for xs, ys, q in cache:
        xs = xs - offset
        xs = xs - np.floor(xs[0] / w) * w  # move the whole path so it starts on the map
        pts = np.column_stack([xs, ys])
        seg = np.stack([pts[:-1], pts[1:]], axis=1)
        qq = 0.5 * (q[:-1] + q[1:])
        lw = np.clip(0.3 + 0.75 * np.log10(np.maximum(qq, threshold) / threshold), 0.3, 3.0)
        for shift in (-w, 0, w):
            s = seg.copy()
            s[..., 0] += shift
            segs.append(s)
            widths.append(lw)
    if not segs:
        return np.zeros((0, 2, 2)), np.zeros(0)
    return np.concatenate(segs), np.concatenate(widths)


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
        scale = px_per_cell * 72 / dpi / 1.03  # same width per map cell as in the viewer
        ax.add_collection(LineCollection(segs, colors=[RIVER_COLOUR], linewidths=widths * scale,
                                         capstyle="round", joinstyle="round"))
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
