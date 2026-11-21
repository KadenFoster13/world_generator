"""World generation: tectonic plates -> terrain -> climate -> biomes.

The map is an equirectangular projection of a sphere, so it wraps east-west.
Generation is split into two stages so the viewer can stay responsive:

  generate_terrain(cfg)      slow-ish: plates, continents, mountains, islands
  apply_climate(terrain, cfg) fast:    sea level, temperature, rain, biomes
"""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, label, maximum_filter

from . import hydrology
from .config import WorldConfig
from .noise import Perlin3, SphereXYZ, fbm, ridged


# --------------------------------------------------------------------- biomes
# (name, RGB colour). The index in this list is the biome id stored in World.biome.
BIOMES = [
    ("Deep ocean", (0.13, 0.27, 0.52)),
    ("Shallow sea", (0.26, 0.47, 0.72)),
    ("Sea ice", (0.86, 0.91, 0.95)),
    ("Ice sheet", (0.96, 0.97, 0.99)),
    ("Tundra", (0.62, 0.64, 0.56)),
    ("Boreal forest", (0.24, 0.40, 0.30)),
    ("Grassland", (0.60, 0.71, 0.39)),
    ("Temperate forest", (0.30, 0.52, 0.25)),
    ("Temperate rainforest", (0.16, 0.42, 0.28)),
    ("Desert", (0.93, 0.84, 0.62)),
    ("Scrubland", (0.79, 0.71, 0.49)),
    ("Savanna", (0.71, 0.69, 0.37)),
    ("Tropical dry forest", (0.47, 0.58, 0.24)),
    ("Tropical rainforest", (0.10, 0.40, 0.16)),
    ("Alpine", (0.55, 0.51, 0.47)),
    ("Lake", (0.32, 0.58, 0.82)),
    ("Salt flat", (0.93, 0.90, 0.82)),
]
BIOME_ID = {name: i for i, (name, _) in enumerate(BIOMES)}
EARTH_RADIUS_KM = 6371.0


@dataclass
class Grid:
    """A lat/lon pixel grid: the whole globe, or a rectangular region of it."""
    height: int
    width: int
    lat0: float = 90.0     # latitude of the top edge
    lon0: float = -180.0   # longitude of the left edge
    dpp: float = 0.0       # degrees per pixel (0 = whole globe, 360 / width)
    wraps: bool = True     # east and west edges meet (whole globe only)

    def __post_init__(self):
        if not self.dpp:
            self.dpp = 360.0 / self.width
        rows = self.lat0 - (np.arange(self.height) + 0.5) * self.dpp
        cols = self.lon0 + (np.arange(self.width) + 0.5) * self.dpp
        self.lon, self.lat = np.meshgrid((cols + 180) % 360 - 180, rows)
        la, lo = np.radians(self.lat), np.radians(self.lon)
        self.xyz = SphereXYZ(np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la),
                             wraps=self.wraps, px_rad=np.radians(self.dpp))

    @property
    def km_per_px(self):
        return np.radians(self.dpp) * EARTH_RADIUS_KM

    @property
    def equivalent_width(self):
        """Width a whole-globe map with this pixel size would have."""
        return int(round(360 / self.dpp))


@dataclass
class Terrain:
    cfg: WorldConfig
    raw: np.ndarray          # unscaled elevation, before sea level is chosen
    plate: np.ndarray        # tectonic plate id per pixel
    piece: np.ndarray        # crust piece id per pixel (plates and continents are built from pieces)
    plate_is_continental: np.ndarray  # per crust piece
    boundary: np.ndarray     # bool: pixel lies on a plate boundary
    convergence: np.ndarray  # at boundaries: >0 colliding, <0 pulling apart
    crust: np.ndarray        # 0 = oceanic, 1 = continental
    lat: np.ndarray          # degrees, +90 north
    lon: np.ndarray          # degrees, -180..180
    xyz: tuple
    target_continents: int
    grid: Grid = None
    # What a zoomed-in region needs to rebuild this terrain in more detail:
    env: dict = None         # smooth large-scale fields (continents, mountain belts, rifts)
    noises: dict = None      # the noise generators used for detail
    hot_stamps: list = None  # hotspot volcano bumps: (centre, radius, height)
    arch_stamps: list = None # archipelago island bumps: (centre, radius, height)
    arch_dist: np.ndarray = None  # km from land before the archipelagos were added
    arch_sea: float = 0.0
    layout: dict = None      # plate centres, motions and groupings


@dataclass
class World:
    terrain: Terrain
    cfg: WorldConfig
    elevation_m: np.ndarray   # metres above (+) or below (-) sea level
    ocean: np.ndarray         # bool
    temperature_c: np.ndarray # mean annual temperature
    precip_mm: np.ndarray     # annual precipitation
    biome: np.ndarray         # index into BIOMES
    landmass_id: np.ndarray   # connected land regions (0 = ocean), wraps east-west
    landmass_area: np.ndarray # area of each landmass as a share of the globe
    major_landmasses: int
    islands: int
    lake: np.ndarray          # bool: lake surface
    salt_flat: np.ndarray     # bool: dry basin that would be a lake in a wetter climate
    filled_elevation_m: np.ndarray  # elevation with every basin filled to its spill point (for rivers)
    lakes: int
    receiver: np.ndarray      # flat index of the pixel each pixel drains into (-1 = ocean)
    discharge_m3s: np.ndarray # river flow through each pixel
    catchment_km2: np.ndarray # land area draining through each pixel
    stream: np.ndarray        # bool: enough catchment to count as a river at all (any size)
    river: np.ndarray         # bool: pixel carries a river big enough to show
    grid: Grid = None
    # Kept so a zoomed-in region can match this world exactly:
    sea_raw: float = 0.0      # sea level in raw terrain units
    land_scale: float = 1.0   # raw height that maps to max_elevation_m
    depth_scale: float = 1.0  # raw depth that maps to max_depth_m
    wind: dict = None         # moisture carried by the winds (smooth, large-scale)
    lake_level_m: np.ndarray = None  # water surface height where there's a lake, else NaN
    closed: np.ndarray = None # bool: basins with no outflow (water evaporates)


# ------------------------------------------------------------------- helpers
def sphere_grid(width: int, height: int):
    """(lat, lon, xyz) of a whole-globe grid."""
    g = Grid(height, width)
    return g.lat, g.lon, g.xyz


def blur(a, sigma, wraps=True):
    """Gaussian blur that wraps east-west (on whole-globe maps) but not over the poles."""
    return gaussian_filter(a, sigma, mode=("nearest", "wrap" if wraps else "nearest"))


def shift(a, dy, dx):
    """out[y, x] = a[y + dy, x + dx], wrapping in x and clamping in y."""
    out = np.roll(a, (-dy, -dx), axis=(0, 1))
    if dy == 1:
        out[-1] = a[-1]
    elif dy == -1:
        out[0] = a[0]
    return out


def distance_to(mask):
    """Pixel distance to the nearest True pixel, plus that pixel's (y, x). Wraps east-west."""
    h, w = mask.shape
    if not mask.any():
        return np.full(mask.shape, np.inf), np.zeros(mask.shape, int), np.zeros(mask.shape, int)
    # Pad a quarter of the map on each side so distances wrap across the seam.
    pad = w // 4
    padded = np.concatenate([mask[:, -pad:], mask, mask[:, :pad]], axis=1)
    dist, (iy, ix) = distance_transform_edt(~padded, return_indices=True)
    sl = slice(pad, pad + w)
    return dist[:, sl], iy[:, sl], (ix[:, sl] - pad) % w


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def random_unit_vectors(rng, n):
    return unit(rng.normal(size=(n, 3)))


def weighted_quantile(values, weights, q):
    order = np.argsort(values, axis=None)
    v = values.ravel()[order]
    cw = np.cumsum(weights.ravel()[order])
    return v[min(np.searchsorted(cw, q * cw[-1]), v.size - 1)]


def chord_distance(xyz, point):
    """Straight-line distance through the sphere; ~ angle in radians for short distances."""
    x, y, z = xyz
    return np.sqrt((x - point[0]) ** 2 + (y - point[1]) ** 2 + (z - point[2]) ** 2)


def move_along(p, direction, angle):
    """Travel `angle` radians from unit vector p along tangent `direction`."""
    d = direction - np.dot(direction, p) * p
    d /= np.linalg.norm(d)
    return p * np.cos(angle) + d * np.sin(angle)


class _NoiseBank:
    """Hands out independent noise generators derived from one world seed."""

    def __init__(self, seed):
        self.seed = seed
        self.count = 0

    def next(self):
        self.count += 1
        return Perlin3(self.seed * 7919 + self.count * 104729)


# ------------------------------------------------------------------- plates
REFERENCE_WIDTH = 400   # big decisions (continents, plates) are made on this fixed grid
BASE_WIDTH = 800        # resolution the settings were tuned at; finer maps add detail octaves


def km_per_px(width):
    """Width of one pixel at the equator, on an Earth-sized planet."""
    return 2 * np.pi * EARTH_RADIUS_KM / width


def detail_octaves(width):
    """Extra noise octaves for maps finer than BASE_WIDTH, so they gain real detail."""
    return max(0, int(round(np.log2(width / BASE_WIDTH))))


def _piece_map(xyz, centres, size_bias, warp_noise, warp):
    """Nearest-centre (warped Voronoi) map of crust pieces on any grid."""
    q = unit(np.stack([xyz[i] + warp * fbm(warp_noise[i], xyz, 1.8, 4) for i in range(3)], axis=-1))
    return (q.reshape(-1, 3) @ centres.T + size_bias).argmax(axis=1).reshape(xyz[0].shape), q


def _coast_piece_map(xyz, q, centres, size_bias, coast_noise, warp):
    """A second, more strongly warped copy of the piece map that continents' coasts follow,
    so continents get peninsulas and bays instead of copying the outline exactly."""
    q2 = unit(np.stack([q[..., i] + warp * fbm(coast_noise[i], xyz, 3.0, 5, persistence=0.6)
                        for i in range(3)], axis=-1))
    return (q2.reshape(-1, 3) @ centres.T + size_bias).argmax(axis=1).reshape(xyz[0].shape)


def _neighbours(ids):
    """Boundary pixels between regions, and the id of the region on the other side."""
    neighbour = np.full(ids.shape, -1)
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        nb = shift(ids, dy, dx)
        neighbour = np.where(nb != ids, nb, neighbour)
    return neighbour >= 0, neighbour


def _adjacency(ids, n):
    boundary, neighbour = _neighbours(ids)
    adjacent = [set() for _ in range(n)]
    for a, b in set(zip(ids[boundary].tolist(), neighbour[boundary].tolist())):
        adjacent[a].add(b)
        adjacent[b].add(a)
    return adjacent


def _group_into_plates(cfg, rng, adjacent, area):
    """Merge crust pieces into larger tectonic plates by random flood fill.

    A plate ends up carrying both continental and oceanic crust, as on Earth
    (the South American plate reaches halfway across the Atlantic). So some
    coasts are quiet "passive margins" with no mountains, and some plate
    collisions happen deep inside continents (like the Himalayas).
    """
    n = len(adjacent)
    t = max(2, min(cfg.tectonic_plates, n))
    plate_of = np.full(n, -1)
    seeds = rng.choice(n, t, replace=False)
    plate_of[seeds] = np.arange(t)
    plate_area = area[seeds].copy()
    while (plate_of < 0).any():
        # Grow a smallish plate (randomly weighted) by one neighbouring piece.
        frontier = {}
        for p in np.nonzero(plate_of < 0)[0]:
            owners = [plate_of[q] for q in adjacent[p] if plate_of[q] >= 0]
            if owners:
                frontier[p] = owners
        if not frontier:  # isolated pieces (shouldn't happen) join plate 0
            plate_of[plate_of < 0] = 0
            break
        pieces = list(frontier)
        weights = np.array([1.0 / (min(plate_area[o] for o in frontier[p]) + 0.01) for p in pieces])
        p = pieces[int(rng.choice(len(pieces), p=weights / weights.sum()))]
        owner = min(frontier[p], key=lambda o: plate_area[o] * rng.uniform(0.5, 1.5))
        plate_of[p] = owner
        plate_area[owner] += area[p]
    return plate_of, t


def _plate_boundaries(plate, omega, xyz):
    """Find boundary pixels and how strongly the two plates collide there."""
    pos = np.stack(xyz, axis=-1)
    v_own = np.cross(omega[plate], pos)
    convergence = np.zeros(plate.shape)
    count = np.zeros(plate.shape)
    neighbour = np.full(plate.shape, -1)
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        nb = shift(plate, dy, dx)
        differs = nb != plate
        if not differs.any():
            continue
        normal = unit(shift(pos, dy, dx) - pos + 1e-12)
        v_nb = np.cross(omega[nb], pos)
        # Positive when this plate moves toward the neighbour faster than it moves away.
        c = np.sum((v_own - v_nb) * normal, axis=-1)
        convergence += np.where(differs, c, 0.0)
        count += differs
        neighbour = np.where(differs, nb, neighbour)
    boundary = count > 0
    convergence[boundary] /= count[boundary]
    typical = np.percentile(np.abs(convergence[boundary]), 85) if boundary.any() else 1.0
    return boundary, np.clip(convergence / max(typical, 1e-6), -1, 1), neighbour


def _choose_continents(cfg, rng, adjacent, area, centres):
    """Group crust pieces into K separate continents with roughly the right total area."""
    n = len(adjacent)

    k = cfg.continents or int(rng.integers(cfg.continents_min, cfg.continents_max + 1))
    k = max(1, min(k, n // 2))

    owner = np.full(n, -1)
    usable = area > 0.004  # tiny or empty plates can't anchor a continent
    # How welcome each plate is on a continent: polar plates less so, otherwise
    # most worlds end up with giant ice caps.
    polar_bias = cfg.polar_land_bias
    welcome = (1 - np.abs(centres[:, 2])) ** polar_bias + 0.02
    # Seed continents far apart, never adjacent to each other.
    first = int(rng.choice(np.nonzero(usable)[0], p=welcome[usable] / welcome[usable].sum()))
    seeds = [first]
    owner[first] = 0
    while len(seeds) < k:
        allowed = [p for p in range(n) if usable[p] and owner[p] < 0
                   and not any(owner[q] >= 0 for q in adjacent[p])]
        if not allowed:
            break
        sep = np.array([min(np.arccos(np.clip(centres[p] @ centres[s], -1, 1)) for s in seeds) for p in allowed])
        sep *= welcome[allowed]
        pick = allowed[int(rng.choice(np.argsort(-sep)[:3]))]
        owner[pick] = len(seeds)
        seeds.append(pick)
    k = len(seeds)

    # Grow continents plate by plate until they cover the target land area.
    # Some continents are "ambitious" so sizes vary like on Earth.
    ambition = rng.uniform(0.25, 1.0, k) ** 2
    cont_area = np.array([area[s] for s in seeds])
    target = cfg.land_fraction * 1.05
    while cont_area.sum() < target:
        options = []
        for c in range(k):
            for p in range(n):
                if owner[p] >= 0:
                    continue
                touching = {owner[q] for q in adjacent[p] if owner[q] >= 0}
                if touching == {c}:
                    options.append((c, p))
        if not options:
            break
        best_c = max({c for c, _ in options}, key=lambda c: ambition[c] / cont_area[c])
        plates_for_c = [p for c, p in options if c == best_c]
        wts = welcome[plates_for_c]
        p = plates_for_c[int(rng.choice(len(plates_for_c), p=wts / wts.sum()))]
        owner[p] = best_c
        cont_area[best_c] += area[p]
    return owner, k


# ------------------------------------------------------------------ terrain
def _pixel_of(p, w, h):
    """(row, col) of the pixel containing unit vector p on a global grid."""
    lat = np.arcsin(np.clip(p[2], -1, 1))
    lon = np.arctan2(p[1], p[0])
    row = int(np.clip((np.pi / 2 - lat) / np.pi * h, 0, h - 1))
    col = int((lon + np.pi) / (2 * np.pi) * w) % w
    return row, col


def _pick_point(rng, accept, tries=300):
    """A random spot on the globe where `accept(point)` holds.

    Always draws the same number of random numbers, and tests points rather than
    pixels, so the same seed picks the same spot at every map resolution.
    """
    candidates = random_unit_vectors(rng, tries)
    for p in candidates:
        if accept(p):
            return p
    return None


NOISE_NAMES = ("coast", "width", "dist", "gaps", "texture", "old", "old_mask", "hills", "arch")


def generate_terrain(cfg: WorldConfig) -> Terrain:
    """Whole-globe terrain.

    Built in two layers so a zoomed-in region can be regenerated in more detail
    and still match: smooth large-scale "envelope" fields (continents, mountain
    belts, rifts) computed here, then detail noise added by _synthesize().
    """
    grid = Grid(cfg.height, cfg.width)
    w, h = grid.width, grid.height
    xyz, lat = grid.xyz, grid.lat
    kpp = grid.km_per_px
    extra = detail_octaves(w)
    rng = np.random.default_rng(cfg.seed)
    noise = _NoiseBank(cfg.seed)
    pos = np.stack(xyz, axis=-1)

    # --- crust pieces. Continents and plates are chosen on a fixed coarse grid, so
    # every resolution of the same seed gets the same continents and plates.
    n = cfg.plates
    centres = random_unit_vectors(rng, n)
    size_bias = rng.uniform(0, cfg.plate_size_variation, n)
    warp_noise = [noise.next() for _ in range(3)]
    coast_noise = [noise.next() for _ in range(3)]
    noises = {name: noise.next() for name in NOISE_NAMES}
    ref_lat, _, ref_xyz = sphere_grid(REFERENCE_WIDTH, REFERENCE_WIDTH // 2)
    ref_piece, _ = _piece_map(ref_xyz, centres, size_bias, warp_noise, cfg.plate_boundary_warp)
    ref_cos = np.cos(np.radians(ref_lat))
    area = np.bincount(ref_piece.ravel(), weights=ref_cos.ravel(), minlength=n)
    area /= area.sum()
    adjacent = _adjacency(ref_piece, n)
    owner, k = _choose_continents(cfg, rng, adjacent, area, centres)
    plate_of, t = _group_into_plates(cfg, rng, adjacent, area)
    # Each plate rotates around its own Euler pole: velocity = omega x position.
    omega = random_unit_vectors(rng, t) * rng.uniform(0.4, 1.0, t)[:, None]
    layout = dict(centres=centres, size_bias=size_bias, warp_noise=warp_noise, owner=owner,
                  plate_of=plate_of, omega=omega)

    piece, q = _piece_map(xyz, centres, size_bias, warp_noise, cfg.plate_boundary_warp)
    coast_piece = _coast_piece_map(xyz, q, centres, size_bias, coast_noise, cfg.coast_warp)
    plate = plate_of[piece]
    boundary, convergence, neighbour = _plate_boundaries(plate, omega, xyz)

    # --- continental crust, softened, plus small fragments
    crust = (owner[coast_piece] >= 0).astype(float)
    for _ in range(cfg.microcontinents):
        centre = _pick_point(rng, lambda p: crust[_pixel_of(p, w, h)] < 0.5)
        heading = random_unit_vectors(rng, 1)[0]
        size = rng.uniform(0.035, 0.08)
        blobs = int(rng.integers(2, 5))
        if centre is None:
            continue
        for step in range(blobs):  # a few blobs in a line = elongated fragment
            stamp(crust, grid, move_along(centre, heading, step * size * 0.9), size, 1.0)
    crust = np.clip(blur(crust, cfg.continent_edge_blur_km / kpp), 0, 1)
    landness = smoothstep(0.3, 0.7, crust)

    # --- mountain belts along plate boundaries (distances in km)
    width_noise = fbm(noises["width"], xyz, 4, 3)
    dist_noise = fbm(noises["dist"], xyz, 9, 3)

    def belt(mask, width_km):
        """Cross-section of a mountain belt centred on `mask`: crest, foothills, parallel folds."""
        dist, iy, ix = distance_to(mask)
        # Round off the V-shaped distance right at the boundary; otherwise every belt
        # gets a knife-edge crease along its centre line at high resolution.
        dist_km = np.sqrt((dist * kpp) ** 2 + 120.0 ** 2) - 120.0
        d = np.maximum(dist_km + 150.0 * dist_noise, 0)
        wdt = width_km * (1 + 0.45 * width_noise)
        crest = np.exp(-(d / wdt) ** 2)
        foothills = 0.3 * np.exp(-(d / (2.6 * wdt)) ** 2)
        folds = 0.85 + 0.15 * np.cos(2 * np.pi * d / (0.8 * wdt))
        return (crest + foothills) * folds, d, wdt, iy, ix

    profile, d, wdt, iy, ix = belt(boundary, cfg.mountain_width_km)
    # Collision strength of the nearest boundary, smoothed so it doesn't jump from
    # pixel to pixel (which shows up as dotted streaks at high resolution).
    c_near = blur(convergence[iy, ix], 150.0 / kpp)
    colliding = np.clip(c_near, 0, 1) ** 0.6
    pulling = np.clip(-c_near, 0, 1)
    amp = cfg.island_arc_height + (cfg.mountain_height - cfg.island_arc_height) * landness
    # Rift valleys: a narrow trough with raised shoulders on both sides, broken into
    # segments, so water collects in long thin basins (like the East African lakes).
    rift_shape = np.exp(-(d / (0.35 * wdt)) ** 2) - 0.45 * np.exp(-((d - 0.9 * wdt) / (0.45 * wdt)) ** 2)

    # Old sutures: where two pieces of one continent welded together long ago.
    piece_edge, piece_nb = _neighbours(piece)
    nb = np.maximum(piece_nb, 0)
    suture_mask = (piece_edge & (owner[piece] >= 0) & (owner[piece] == owner[nb])
                   & (plate_of[piece] == plate_of[nb]))
    suture = np.zeros_like(crust)
    if suture_mask.any():
        old_profile, *_ = belt(suture_mask, cfg.mountain_width_km * 1.3)
        suture = cfg.suture_factor * cfg.mountain_height * 0.6 * old_profile * landness

    env = dict(
        crust=crust,
        landness=landness,
        collision=amp * colliding * profile,            # x gaps x ridge texture
        suture=suture,                                  # x gaps x ridge texture
        rift=cfg.rift_depth * pulling * rift_shape * landness,  # x segment gaps
        ridge=cfg.mid_ocean_ridge_height * pulling * np.exp(-(d / (2.0 * wdt)) ** 2) * (1 - landness),
    )

    # --- hotspot island chains, trailing behind the plate's motion
    hot_stamps = []
    for _ in range(cfg.hotspots):
        p0 = _pick_point(rng, lambda p: crust[_pixel_of(p, w, h)] < 0.1)
        count = int(rng.integers(4, 11))
        jitter = rng.normal(0, 0.15, (count, 3))
        spacing = rng.uniform(0.03, 0.045, count)
        sizes = rng.uniform(0.7, 1.3, count)
        if p0 is None:
            continue
        trail = -np.cross(omega[plate[_pixel_of(p0, w, h)]], p0)
        if np.linalg.norm(trail) < 1e-6:
            trail = np.cross(p0, [0.0, 0.0, 1.0]) + 1e-6
        for j in range(count):
            age = j / count
            p = move_along(p0, trail + jitter[j], j * spacing[j])
            hot_stamps.append((p, 0.016 * (1 - 0.5 * age) * sizes[j], 0.95 - 0.55 * age))

    raw = _synthesize(cfg, grid, extra, env, noises, hot_stamps)

    # --- coastal archipelagos (last, so turning them off doesn't change anything else)
    arch_stamps, arch_dist, arch_sea = [], None, 0.0
    if cfg.archipelagos:
        arch_stamps, arch_dist, arch_sea = _plan_archipelagos(raw, cfg, rng, grid)
        raw = _apply_archipelagos(raw, cfg, grid, arch_stamps, arch_dist, arch_sea, noises["arch"], extra)

    return Terrain(cfg=cfg, raw=raw, plate=plate, piece=piece, plate_is_continental=owner >= 0,
                   boundary=boundary, convergence=convergence, crust=crust,
                   lat=lat, lon=grid.lon, xyz=xyz, target_continents=k, grid=grid,
                   env=env, noises=noises, hot_stamps=hot_stamps, arch_stamps=arch_stamps,
                   arch_dist=arch_dist, arch_sea=arch_sea, layout=layout)


def _synthesize(cfg, grid, extra, env, noises, hot_stamps):
    """Combine the smooth envelope fields with detail noise on any grid.

    Used for the whole globe and, with finer noise, for enhanced regions.
    """
    xyz = grid.xyz
    landness = env["landness"]
    coast = fbm(noises["coast"], xyz, cfg.coast_noise_frequency, 6, persistence=0.55, extra=extra)
    raw = -0.55 + 0.8 * env["crust"] + cfg.coast_roughness * coast

    gaps = smoothstep(-0.75, -0.05, fbm(noises["gaps"], xyz, cfg.chain_gap_frequency, 3))
    texture = 0.5 + 0.7 * ridged(noises["texture"], xyz, 9, 5, extra=extra)
    raw += (env["collision"] + env["suture"]) * gaps * texture
    raw -= env["rift"] * (0.4 + 0.6 * gaps)
    raw += env["ridge"]

    # Worn-down inland hills and ranges away from any boundary.
    old = (ridged(noises["old"], xyz, 5, 5, extra=extra)
           * smoothstep(0.15, 0.55, fbm(noises["old_mask"], xyz, 1.7, 3)))
    raw += cfg.ancient_mountains * old * landness

    # Hills and general roughness.
    raw += cfg.hill_roughness * fbm(noises["hills"], xyz, 14, 4, extra=extra) * (0.3 + 0.7 * landness)

    for centre, radius, height in hot_stamps:
        stamp(raw, grid, centre, radius, height, op="add")
    return raw


def stamp(field, grid, centre, radius, height, op="max"):
    """Draw a Gaussian bump of `radius` radians on the sphere, touching only nearby pixels."""
    h, w = field.shape
    clat = np.degrees(np.arcsin(np.clip(centre[2], -1, 1)))
    clon = np.degrees(np.arctan2(centre[1], centre[0]))
    reach = np.degrees(3 * radius)
    row = (grid.lat0 - clat) / grid.dpp - 0.5
    r0 = max(0, int(row - reach / grid.dpp) - 1)
    r1 = min(h, int(row + reach / grid.dpp) + 2)
    if r0 >= r1:
        return
    worst_cos = max(np.cos(np.radians(min(abs(clat) + reach, 90.0))), 1e-3)
    half = reach / (grid.dpp * worst_cos) + 2
    if grid.wraps:
        col = ((clon - grid.lon0) % 360) / grid.dpp - 0.5
        cols = np.arange(w) if half >= w / 2 else np.arange(int(col - half), int(col + half) + 1) % w
    else:
        col = ((clon - grid.lon0 + 180) % 360 - 180) / grid.dpp - 0.5
        cols = np.arange(max(0, int(col - half)), min(w, int(col + half) + 2))
        if len(cols) == 0:
            return
    sub = tuple(c[r0:r1][:, cols] for c in grid.xyz)
    bump = height * np.exp(-(chord_distance(sub, centre) / radius) ** 2)
    if op == "add":
        field[r0:r1, cols] += bump
    else:
        field[r0:r1, cols] = np.maximum(field[r0:r1, cols], bump)


def _local_axes(p):
    """East and north unit vectors at point p on the sphere."""
    lat = np.arcsin(np.clip(p[2], -1, 1))
    lon = np.arctan2(p[1], p[0])
    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    north = np.array([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)])
    return east, north


def _plan_archipelagos(raw, cfg, rng, grid):
    """Lay out island chains along continental coasts, each with a few large islands.

    Think Japan, the Philippines, Indonesia or the Caribbean. Returns the island
    bumps, the distance from land (km) and the provisional sea level.
    """
    h, w = raw.shape
    lat = grid.lat
    sea = weighted_quantile(raw, np.cos(np.radians(lat)), 1 - cfg.land_fraction)
    land = raw >= sea
    dist = distance_to(land)[0] * grid.km_per_px
    gap = cfg.archipelago_coast_gap_km
    stamps, anchors = [], []

    def good_anchor(p):
        y, x = _pixel_of(p, w, h)
        return (not land[y, x] and gap + 300 < dist[y, x] < gap + 1100 and abs(lat[y, x]) < 58
                and all(np.linalg.norm(p - a) > 0.5 for a in anchors))

    for _ in range(cfg.archipelago_count):
        p0 = _pick_point(rng, good_anchor)
        length = rng.uniform(0.35, 0.8)  # radians; 0.5 rad is about 3200 km on Earth
        bend = rng.normal(0, 0.05)
        chain_rng = np.random.default_rng(rng.integers(2 ** 63))  # fixed draws per chain
        if p0 is None:
            continue
        anchors.append(p0)
        y, x = _pixel_of(p0, w, h)
        # Run parallel to the coast: perpendicular to the distance-from-land gradient.
        gx = dist[y, (x + 1) % w] - dist[y, x - 1]
        gy = dist[min(y + 1, h - 1), x] - dist[max(y - 1, 0), x]
        if abs(gx) + abs(gy) < 1e-6:
            gx, gy = 1.0, 0.0
        east, north = _local_axes(p0)
        heading = unit(-gy * east - gx * north)  # map direction (-gy east, +gx south)

        step = 0.02
        steps = max(int(length / step), 3)
        # Start half a chain-length back so the chain is centred on the anchor.
        p = move_along(p0, -heading, length / 2)
        heading = unit(p0 - p)
        large = set(chain_rng.choice(steps, min(steps, int(chain_rng.integers(2, 5))), replace=False).tolist())
        for s in range(steps):
            if s in large:
                size = chain_rng.uniform(0.028, 0.05)
                for j in range(int(chain_rng.integers(2, 5))):  # elongated along the chain
                    stamps.append((move_along(p, heading, j * size * 0.8), size, chain_rng.uniform(0.4, 0.6)))
            elif chain_rng.random() < 0.85:
                side = np.cross(p, heading)
                for _ in range(int(chain_rng.integers(1, 4))):
                    q = unit(p + side * chain_rng.normal(0, 0.02) + heading * chain_rng.normal(0, 0.008))
                    stamps.append((q, chain_rng.uniform(0.008, 0.02), chain_rng.uniform(0.22, 0.4)))
            nxt = move_along(p, heading, step)
            # Carry the heading forward and bend it a little, so chains curve.
            heading = unit(nxt - p)
            heading = unit(heading * np.cos(bend) + np.cross(nxt, heading) * np.sin(bend))
            p = nxt
    return stamps, dist, sea


def _apply_archipelagos(raw, cfg, grid, stamps, dist_km, sea, rough_noise, extra):
    """Raise the planned islands out of the sea (works on the globe or a region)."""
    field = np.zeros_like(raw)
    for centre, radius, height in stamps:
        stamp(field, grid, centre, radius, height)
    # Keep a strait of open water between the islands and the mainland, so chains
    # stay separate islands instead of growing out of the coast as peninsulas.
    gap = cfg.archipelago_coast_gap_km
    field *= smoothstep(gap, gap + 250, dist_km)
    rough = fbm(rough_noise, grid.xyz, 18, 4, extra=extra)
    target = sea - 0.08 + field * (1 + 0.8 * rough)
    blend = smoothstep(0.0, 0.06, field)   # also lifts a shallow shelf around the islands
    return raw + (np.maximum(raw, target) - raw) * blend


# ------------------------------------------------------------------ climate
def _wind_sweep(land_elev_m, ocean, lat, cfg, reach_km=None):
    """Carry moisture from the ocean along prevailing winds, losing it inland and over mountains.

    Trade winds (|lat| < 30) and polar easterlies (|lat| > 60) blow toward the west;
    the westerlies in between blow toward the east.
    """
    h, w = ocean.shape
    # Share of moisture kept per pixel of travel; pixels are narrower toward the poles.
    reach_km = cfg.moisture_reach_km if reach_km is None else reach_km
    px_km = km_per_px(w) * np.maximum(np.cos(np.radians(lat[:, 0])), 0.05)
    retention = np.exp(-px_km / reach_km)
    westward = (np.abs(lat[:, 0]) < 30) | (np.abs(lat[:, 0]) > 60)

    def orient(a):
        a = a.copy()
        a[westward] = a[westward, ::-1]
        return a

    elev = orient(land_elev_m)
    sea = orient(ocean)
    moisture = np.ones(h)
    out = np.zeros((h, w))
    uplift_rain = np.zeros((h, w))
    for step in range(2 * w):  # two laps so moisture can wrap around the globe
        c = step % w
        rise = np.clip(elev[:, c] - elev[:, c - 1], 0, None) / 1000.0
        lost = np.minimum(moisture, cfg.rain_shadow * rise)
        moisture = np.where(sea[:, c], 1.0, moisture * retention - lost)
        moisture = np.clip(moisture, 0, 1)
        if step >= w:
            out[:, c] = moisture
            uplift_rain[:, c] = lost
    return orient(out), orient(uplift_rain)


def label_wrapped(mask):
    """Connected regions of `mask` (4-connected), joined across the east-west seam. 0 = background."""
    lbl, n = label(mask)
    if n == 0:
        return lbl, 0
    parent = np.arange(n + 1)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    seam = (lbl[:, 0] > 0) & (lbl[:, -1] > 0)
    for a, b in set(zip(lbl[seam, 0].tolist(), lbl[seam, -1].tolist())):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([find(i) for i in range(n + 1)])
    uniq, compact = np.unique(roots, return_inverse=True)
    return compact[lbl], len(uniq) - 1


def _region_areas(lbl, n, cos_lat):
    """Share of the globe covered by each labelled region (index 0 = background, set to 0)."""
    area = np.bincount(lbl.ravel(), weights=cos_lat.ravel(), minlength=n + 1) / cos_lat.sum()
    area[0] = 0
    return area


def _remove_small_seas(elevation_m, ocean, cos_lat, cfg):
    """Turn small enclosed bodies of sea water into land.

    Coastline noise leaves scattered low spots below sea level inside continents.
    They aren't real lakes, so they're filled in here and lakes are made separately
    from the shape of the land (see _find_lakes). Big enclosed seas (like the
    Caspian or Black Sea) are kept.
    """
    lbl, n = label_wrapped(ocean)
    area = _region_areas(lbl, n, cos_lat)
    keep = area >= cfg.inland_sea_min_area
    keep[np.argmax(area)] = True  # the world ocean
    keep[0] = False
    fill = ocean & ~keep[lbl]
    if not fill.any():
        return elevation_m, ocean
    return fill_pockets(elevation_m, fill, ~ocean), ocean & ~fill


def fill_pockets(elevation_m, pockets, land, wraps=True):
    """Fill sea pockets in as low ground, level with the lowest point of their shore.

    That makes them read as natural lowland basins instead of raised plateaus with
    cliffs around them. (A deep enough basin around them can still hold a lake.)
    """
    lbl, n = label_wrapped(pockets) if wraps else label(pockets)
    if n == 0:
        return elevation_m
    grown = maximum_filter(lbl, size=3, mode=("nearest", "wrap" if wraps else "nearest"))
    shore = land & (grown > 0)
    lowest = np.full(n + 1, np.inf)
    np.minimum.at(lowest, grown[shore], elevation_m[shore])
    lowest[~np.isfinite(lowest)] = 5.0
    # Keep a faint copy of the pocket's original bowl shape (its depths were negative),
    # so the floor isn't perfectly flat and rivers wind across it instead of running
    # in straight lines.
    elevation_m = elevation_m.copy()
    floor = lowest[lbl[pockets]] - 20.0 + 0.15 * elevation_m[pockets]
    elevation_m[pockets] = np.maximum(floor, 1.0 + 0.002 * (elevation_m[pockets] + 6000))
    return elevation_m


def _find_lakes(elevation_m, filled, ocean, precip, temperature, cfg):
    """Lakes form in basins that are deep and wide enough; in dry climates they become salt flats.

    Returns lake and salt-flat masks, a mask of "closed" basins (lakes that never
    fill up to their rim, and salt flats: rivers flowing into those end there, because
    the water evaporates before it can overflow), and each lake's water level.
    """
    depth = filled - elevation_m
    basins, n = label_wrapped((depth > 0.5) & ~ocean)
    lake = np.zeros(ocean.shape, bool)
    salt = np.zeros(ocean.shape, bool)
    if n == 0:
        return lake, salt, lake.copy(), np.full(ocean.shape, np.nan)
    flat = basins.ravel()
    size = np.bincount(flat, minlength=n + 1)
    max_depth = np.zeros(n + 1)
    np.maximum.at(max_depth, flat, depth.ravel())
    bottom = np.full(n + 1, np.inf)
    np.minimum.at(bottom, flat, elevation_m.ravel())
    spill = np.zeros(n + 1)
    np.maximum.at(spill, flat, filled.ravel())
    wet = np.bincount(flat, weights=precip.ravel(), minlength=n + 1) / np.maximum(size, 1)

    # Most basins on a real continent have been drained by rivers cutting through
    # their rims; only the biggest, deepest ones keep their water.
    px_area = km_per_px(ocean.shape[1]) ** 2
    min_px = max(1, int(round(cfg.lake_min_area_km2 / px_area)))
    max_px = max(min_px, int(round(cfg.max_lake_area_km2 / px_area)))
    keep = (size >= min_px) & (max_depth >= cfg.lake_min_depth_m)
    keep[0] = False
    score = np.where(keep, size * max_depth, 0)
    if keep.sum() > cfg.max_lakes:
        keep &= score >= np.sort(score)[-cfg.max_lakes]

    # Wet basins fill to the brim; dry ones only partly, leaving salt flats around a salt lake.
    fullness = np.clip(wet / cfg.lake_full_precip_mm, 0, 1) ** 1.5
    level = bottom + fullness * (spill - bottom)
    # Cap each lake's size: in a huge, shallow basin only the lowest part holds water.
    for b in np.nonzero(keep)[0]:
        heights = np.sort(elevation_m[basins == b])
        if len(heights) > max_px:
            level[b] = min(level[b], heights[max_px])
    lake = keep[basins] & (elevation_m < level[basins])
    # A partly filled basin can leave scattered puddles; keep only real-sized lakes.
    pools, m = label_wrapped(lake)
    if m:
        lake &= (np.bincount(pools.ravel(), minlength=m + 1) >= min_px)[pools]
    salt_line = bottom + np.maximum(0.6, fullness) * (spill - bottom)
    salt = (keep[basins] & ~lake & (fullness < 0.5)[basins]
            & (elevation_m < salt_line[basins]) & (temperature > 0))
    closed = keep & (level < spill - 1.0)
    lake_level = np.where(lake, level[basins], np.nan)
    return lake, salt, closed[basins] & (basins > 0), lake_level


def _rivers(receiver, order, ocean, closed, precip, temperature, grid, cfg, inflow=None):
    """River discharge (m3/s) and upstream catchment area (km2) for every pixel.

    `inflow` optionally adds (discharge, catchment) arrays of water entering from
    outside, for regions.
    """
    px_area = hydrology.pixel_area_m2(grid.lat, grid.equivalent_width)
    local = hydrology.runoff_mm(precip, temperature, cfg) / 1000 * px_area / hydrology.SECONDS_PER_YEAR
    local = np.where(ocean, 0.0, local)
    area_km2 = np.where(ocean, 0.0, px_area / 1e6)
    if inflow is not None:
        local = local + inflow[0]
        area_km2 = area_km2 + inflow[1]
    # Water leaving a closed basin is lost to evaporation instead of flowing on.
    recv = np.where(receiver >= 0, receiver, np.arange(receiver.size))
    leaving_closed = closed.ravel() & ~closed.ravel()[recv]
    discharge = hydrology.accumulate(local, receiver, order, ~leaving_closed)
    catchment = hydrology.accumulate(area_km2, receiver, order, np.ones(ocean.shape, bool))
    return discharge, catchment


def _elevation_scale(raw, cos_lat, cfg):
    """Sea level (so that `land_fraction` of the globe is land) and the raw heights that
    map to the tallest peak and deepest trench."""
    sea = weighted_quantile(raw, cos_lat, 1 - cfg.land_fraction)
    above = raw - sea
    land_scale = max(np.percentile(np.where(above > 0, above, 0), 99.9), 1e-9)
    depth_scale = max(float(-above.min()), 1e-9)
    return sea, land_scale, depth_scale


def _to_metres(raw, sea, land_scale, depth_scale, cfg):
    above = raw - sea
    ocean = above < 0
    elevation_m = np.where(
        ocean,
        -cfg.max_depth_m * np.clip(-above / depth_scale, 0, None) ** 0.7,
        cfg.max_elevation_m * np.clip(above / land_scale, 0, None) ** 1.25,
    )
    return elevation_m, ocean


def _climate(cfg, grid, elevation_m, ocean, wind=None):
    """Temperature and precipitation on any grid.

    `wind` holds the moisture carried in by the prevailing winds. On the whole globe
    it's computed here; an enhanced region passes in the globe's (it depends on
    oceans and mountains far upwind, outside the region).
    """
    xyz, lat = grid.xyz, grid.lat
    noise = _NoiseBank(cfg.seed + 99991)
    t_n, tlat_n, wob_n, p_n, tj_n, pj_n = (noise.next() for _ in range(6))

    # Temperature: latitude + altitude + a little noise. Ocean currents and weather bend
    # the temperature bands instead of following latitude exactly.
    t_lat = np.clip(np.abs(lat + 5 * fbm(tlat_n, xyz, 2.0, 4)), 0, 90)
    temperature = (cfg.equator_temp_c - (cfg.equator_temp_c - cfg.pole_temp_c) * (t_lat / 90) ** 2
                   - cfg.lapse_rate_c_per_km * np.maximum(elevation_m, 0) / 1000
                   + 1.5 * fbm(t_n, xyz, 3, 3))

    # Precipitation: global circulation bands x moisture carried in by the wind.
    wobbly_lat = np.abs(lat + 7 * fbm(wob_n, xyz, 2.5, 3))
    band = np.interp(wobbly_lat, [0, 8, 18, 27, 38, 52, 65, 80, 90],
                     [1.0, 0.95, 0.5, 0.22, 0.5, 0.8, 0.55, 0.3, 0.2])
    if wind is None:
        kpp = grid.km_per_px
        smooth_relief = blur(np.maximum(elevation_m, 0), 200 / kpp)  # rain shadows come from ranges
        moisture, uplift = _wind_sweep(smooth_relief, ocean, lat, cfg)
        monsoon, _ = _wind_sweep(smooth_relief, ocean, lat, cfg, reach_km=cfg.monsoon_reach_km)
        wind = dict(
            # Extra north-south smoothing hides the row-by-row wind streaks.
            moisture=blur(moisture, (250 / kpp, 150 / kpp)),
            monsoon=blur(monsoon, (250 / kpp, 150 / kpp)),
            # Rain squeezed out on windward slopes is measured per pixel climbed; finer
            # pixels climb less each, so scale it back up to match every resolution.
            uplift=blur(uplift, 100 / kpp) * (50.0 / kpp),
        )
    # Wet bands rain a lot even far inland (convection, recycled moisture from forests).
    inland_floor = 0.15 + 0.35 * band
    precip = (cfg.max_precip_mm * band * (inland_floor + (1 - inland_floor) * wind["moisture"])
              * (1 + 0.3 * fbm(p_n, xyz, 4, 4)))
    precip += cfg.max_precip_mm * 1.5 * wind["uplift"] * band
    # Subtropical monsoon: the east coasts of continents around 15-35 degrees get summer
    # rain blown in from the ocean (Florida, southern China), so the dry subtropical belt
    # is mostly felt on west coasts and in interiors (Sahara, Atacama, Australia).
    subtropics = np.exp(-((wobbly_lat - 24) / 10) ** 2)
    precip += cfg.max_precip_mm * cfg.monsoon_strength * subtropics * wind["monsoon"]
    precip = np.maximum(precip, 0)

    # A little fine noise on the biome thresholds so borders wander like real ecotones.
    t_jitter = 1.2 * fbm(tj_n, xyz, 12, 3)
    p_jitter = 1 + 0.18 * fbm(pj_n, xyz, 12, 3)
    biome = _classify(elevation_m, ocean, temperature + t_jitter, precip * p_jitter)
    return temperature, precip, biome, wind


def apply_climate(terrain: Terrain, cfg: WorldConfig) -> World:
    grid = terrain.grid
    lat = grid.lat
    cos_lat = np.cos(np.radians(lat))
    sea, land_scale, depth_scale = _elevation_scale(terrain.raw, cos_lat, cfg)
    elevation_m, ocean = _to_metres(terrain.raw, sea, land_scale, depth_scale, cfg)
    elevation_m, ocean = _remove_small_seas(elevation_m, ocean, cos_lat, cfg)
    temperature, precip, biome, wind = _climate(cfg, grid, elevation_m, ocean)

    # Drainage: which way water flows from every pixel, and how full each basin can get.
    filled, receiver, order = hydrology.priority_flood(elevation_m, ocean)
    if cfg.lakes:
        lake, salt_flat, closed, lake_level = _find_lakes(elevation_m, filled, ocean, precip, temperature, cfg)
    else:
        lake = salt_flat = closed = np.zeros(ocean.shape, bool)
        lake_level = np.full(ocean.shape, np.nan)
    discharge, catchment = _rivers(receiver, order, ocean, closed, precip, temperature, grid, cfg)
    # A river needs both enough water and enough land draining into it; a single very
    # wet hillside makes a lot of water but isn't a river yet.
    stream = (catchment >= cfg.river_min_catchment_km2) & ~ocean & ~lake
    river = stream & (discharge >= cfg.river_min_discharge_m3s)

    biome[salt_flat] = BIOME_ID["Salt flat"]
    biome[lake & (temperature >= -10)] = BIOME_ID["Lake"]

    landmass_id, n = label_wrapped(~ocean)
    landmass_area = _region_areas(landmass_id, n, cos_lat)
    major = int(np.sum(landmass_area >= 0.01))   # >= 1% of the globe, about Australia-sized
    islands = int(np.sum((landmass_area > 0) & (landmass_area < 0.01)))
    lake_count = label_wrapped(lake)[1]
    return World(terrain=terrain, cfg=cfg, elevation_m=elevation_m, ocean=ocean,
                 temperature_c=temperature, precip_mm=precip, biome=biome,
                 landmass_id=landmass_id, landmass_area=landmass_area,
                 major_landmasses=major, islands=islands,
                 lake=lake, salt_flat=salt_flat, filled_elevation_m=filled, lakes=lake_count,
                 receiver=receiver, discharge_m3s=discharge, river=river,
                 catchment_km2=catchment, stream=stream, grid=grid,
                 sea_raw=sea, land_scale=land_scale, depth_scale=depth_scale, wind=wind,
                 lake_level_m=lake_level, closed=closed)


def _classify(elev, ocean, t, p):
    b = np.full(elev.shape, BIOME_ID["Temperate forest"], dtype=np.int16)
    tropical = t >= 19
    temperate = (t >= 4) & ~tropical
    cold = (t >= -2) & (t < 4)

    # Rainfall thresholds in mm/yr; scrubland is the semi-arid fringe around deserts.
    b[tropical & (p < 300)] = BIOME_ID["Desert"]
    b[tropical & (p >= 300) & (p < 550)] = BIOME_ID["Scrubland"]
    b[tropical & (p >= 550) & (p < 1000)] = BIOME_ID["Savanna"]
    b[tropical & (p >= 1000) & (p < 1700)] = BIOME_ID["Tropical dry forest"]
    b[tropical & (p >= 1700)] = BIOME_ID["Tropical rainforest"]

    b[temperate & (p < 200)] = BIOME_ID["Desert"]
    b[temperate & (p >= 200) & (p < 350)] = BIOME_ID["Scrubland"]
    b[temperate & (p >= 350) & (p < 600)] = BIOME_ID["Grassland"]
    b[temperate & (p >= 600) & (p < 1500)] = BIOME_ID["Temperate forest"]
    b[temperate & (p >= 1500)] = BIOME_ID["Temperate rainforest"]

    b[cold & (p < 250)] = BIOME_ID["Tundra"]
    b[cold & (p >= 250)] = BIOME_ID["Boreal forest"]
    b[(t >= -10) & (t < -2)] = BIOME_ID["Tundra"]
    b[(elev > 3300) & (t >= -10)] = BIOME_ID["Alpine"]
    b[t < -10] = BIOME_ID["Ice sheet"]

    b[ocean] = BIOME_ID["Deep ocean"]
    b[ocean & (elev > -300)] = BIOME_ID["Shallow sea"]
    b[ocean & (t < -9)] = BIOME_ID["Sea ice"]
    return b


def generate_world(cfg: WorldConfig) -> World:
    return apply_climate(generate_terrain(cfg), cfg)
