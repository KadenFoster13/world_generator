"""Wild species, minerals, disease and seas: the natural world a history unfolds on.

- Wild crops and animals that can be domesticated, each with a climate niche and a
  native homeland. Oceans, mountains and distance stop them spreading, so not every
  landmass has every species (as with horses, missing from the Americas).
- Mineral deposits (copper, tin, iron, gold, silver, coal, oil, saltpeter) from the
  geology that forms them.
- Disease environments (malaria, tsetse fly).
- Prevailing winds and ocean currents for sailing, and the land exposed when sea level
  was lower during the ice age.

Everything here is deterministic from the world seed and doesn't change the terrain.
"""
from dataclasses import dataclass

import numpy as np

from .earth import MINERAL_DAMPING, is_earth
from .earth_data import HOMELANDS, ORES, REACH_KM as EARTH_REACH_KM
from .generator import BIOME_ID, blur, distance_to, smoothstep
from .noise import Perlin3, fbm


# ------------------------------------------------------------------ species
@dataclass(frozen=True)
class Species:
    key: str
    label: str
    kind: str                 # "crop" or "animal"
    temp: tuple               # comfortable mean annual temperature range, C
    temp_tolerance: float     # how far outside that range it can still survive, C
    precip: tuple             # comfortable annual rainfall range, mm
    min_growing: float = 0    # months of growing season needed (crops)
    elevation: tuple = (-100, 3500)  # m
    needs_water: bool = False # needs rivers, lakes or wetland nearby (rice, dates)
    biomes: tuple = ()        # biomes it lives in (empty = any land biome)
    difficulty: float = 1.0   # how hard to domesticate (1 = easy, 3 = very slow)
    chance: float = 0.6       # chance a large landmass has it at all
    homelands: tuple = (1, 2) # number of native homeland areas on a landmass that has it
    reach_km: float = 1500    # how far its wild range spreads from a homeland
    uses: tuple = ()
    food: float = 1.0         # relative food value once domesticated


_OPEN = ("Grassland", "Savanna", "Scrubland", "Temperate forest", "Tropical dry forest")
SPECIES = (
    # Crops
    Species("temperate_grain", "wild grain (wheat/barley-like)", "crop", (8, 20), 6, (250, 900), 4,
            difficulty=1.0, chance=0.55, uses=("food", "fodder"), food=1.0),
    Species("cold_grain", "hardy grain (rye/oat-like)", "crop", (2, 12), 5, (300, 1000), 3,
            difficulty=1.4, chance=0.5, uses=("food", "fodder"), food=0.8),
    Species("wet_grain", "wetland grain (rice-like)", "crop", (18, 29), 4, (1000, 3000), 6,
            elevation=(-100, 1500), needs_water=True, difficulty=1.2, chance=0.45, uses=("food",), food=1.2),
    Species("tropical_grain", "dry grain (millet/sorghum-like)", "crop", (18, 30), 5, (300, 1200), 4,
            difficulty=1.1, chance=0.55, uses=("food", "fodder"), food=0.8),
    Species("slow_grain", "big grain, slow to tame (maize-like)", "crop", (14, 28), 6, (500, 2500), 4,
            difficulty=2.0, chance=0.35, homelands=(1, 1), uses=("food", "fodder"), food=1.3),
    Species("legume", "pulses (bean/lentil-like)", "crop", (6, 28), 6, (300, 1300), 3,
            difficulty=1.0, chance=0.65, uses=("food",), food=0.7),
    Species("highland_tuber", "highland tuber (potato-like)", "crop", (4, 14), 5, (400, 1200), 3,
            elevation=(1000, 4200), difficulty=1.3, chance=0.35, homelands=(1, 1), uses=("food",), food=1.1),
    Species("tropical_tuber", "tropical root crop (yam/taro-like)", "crop", (20, 30), 4, (1200, 3500), 6,
            elevation=(-100, 1800), difficulty=1.5, chance=0.55, uses=("food",), food=1.0),
    Species("tree_crop", "fruit and oil trees (olive/date/banana-like)", "crop", (12, 30), 5, (200, 2500), 4,
            difficulty=1.5, chance=0.6, uses=("food", "oil"), food=0.6),
    Species("fibre", "fibre plant (flax/cotton-like)", "crop", (10, 30), 6, (400, 1600), 4,
            difficulty=1.2, chance=0.55, uses=("fibre",), food=0.0),
    # Animals
    Species("cattle", "wild cattle", "animal", (2, 26), 8, (300, 1800), 0,
            biomes=_OPEN + ("Boreal forest", "Wetland"), difficulty=1.5, chance=0.5,
            reach_km=2500, uses=("meat", "milk", "traction", "leather"), food=1.0),
    Species("sheep_goat", "wild sheep and goats", "animal", (-2, 25), 8, (150, 1200), 0,
            biomes=_OPEN + ("Alpine", "Desert"), elevation=(-100, 4500), difficulty=1.0, chance=0.55,
            reach_km=2500, uses=("meat", "milk", "wool"), food=0.8),
    Species("pig", "wild pig", "animal", (2, 28), 6, (500, 2500), 0,
            biomes=("Temperate forest", "Tropical dry forest", "Tropical rainforest", "Wetland", "Savanna"),
            difficulty=1.5, chance=0.6, reach_km=2500, uses=("meat",), food=0.9),
    Species("horse", "wild horse", "animal", (-4, 18), 6, (200, 900), 0,
            biomes=("Grassland", "Scrubland", "Tundra", "Boreal forest"), difficulty=2.0, chance=0.3,
            homelands=(1, 1), reach_km=3000, uses=("riding", "traction", "war", "milk"), food=0.5),
    Species("camel", "wild camel", "animal", (10, 30), 8, (20, 400), 0,
            biomes=("Desert", "Scrubland", "Grassland"), difficulty=2.0, chance=0.3,
            homelands=(1, 1), reach_km=2500, uses=("pack", "riding", "milk", "war"), food=0.4),
    Species("llama", "wild highland camelid (llama-like)", "animal", (-4, 14), 6, (200, 1200), 0,
            biomes=("Alpine", "Grassland", "Scrubland", "Tundra"), elevation=(1500, 5000), difficulty=1.5,
            chance=0.3, homelands=(1, 1), reach_km=2000, uses=("pack", "wool", "meat"), food=0.4),
    Species("fowl", "wild fowl (chicken-like)", "animal", (16, 30), 5, (800, 3000), 0,
            biomes=("Tropical dry forest", "Tropical rainforest", "Savanna", "Temperate forest"),
            difficulty=0.8, chance=0.5, reach_km=2000, uses=("meat", "eggs"), food=0.4),
)
SPECIES_KEYS = tuple(s.key for s in SPECIES)


def _band(value, lo, hi, tolerance):
    """1 inside [lo, hi], falling to 0 at `tolerance` outside it."""
    below = np.clip((lo - value) / tolerance, 0, None)
    above = np.clip((value - hi) / tolerance, 0, None)
    return np.clip(1 - below - above, 0, 1)


def species_suitability(world):
    """How well each species could live in each cell (0..1), stacked (species, rows, cols)."""
    land = ~world.ocean & ~world.lake
    frozen = np.isin(world.biome, [BIOME_ID["Ice sheet"], BIOME_ID["Glacier"], BIOME_ID["Salt flat"]])
    usable = land & ~frozen
    log_p = np.log(np.maximum(world.precip_mm, 1))
    out = np.zeros((len(SPECIES),) + world.ocean.shape, np.float32)
    for i, sp in enumerate(SPECIES):
        s = _band(world.temperature_c, sp.temp[0], sp.temp[1], sp.temp_tolerance)
        s *= _band(log_p, np.log(sp.precip[0]), np.log(sp.precip[1]), 0.6)
        s *= _band(world.elevation_m, sp.elevation[0], sp.elevation[1], 400)
        if sp.min_growing:
            s *= np.clip((world.growing_months - sp.min_growing + 1) / 2, 0, 1)
        if sp.needs_water:
            s *= np.clip(np.maximum(world.fresh_water, world.wetland) / 0.6, 0, 1)
        if sp.biomes:
            s *= np.isin(world.biome, [BIOME_ID[b] for b in sp.biomes])
        out[i] = np.where(usable, s, 0)
    return out


def species_native(world, suitability):
    """Where each species lives wild at the start of history (species, rows, cols) bool.

    Each landmass independently may or may not have a species (smaller landmasses less
    often). If it does, the species spreads outward from one or two homelands *through
    its habitat*: easily across good habitat, slowly across poor habitat, and not at all
    across the sea, so ranges follow grasslands, valleys and coasts and stop at deserts
    and mountains instead of being circles. A little randomness in how hard each area is
    to cross keeps the edges irregular.
    """
    rng = np.random.default_rng((world.cfg.seed, 777))
    lm, area = world.landmass_id, world.landmass_area
    out = np.zeros(suitability.shape, bool)
    coarse = _CoarseGrid(world)
    # Homelands are chosen on the coarse grid (the same grid at every map width), and each
    # landmass's random draws come from one of its own coarse cells (the one with the
    # lowest of a fixed random number per cell), so a world's wildlife is the same
    # whatever its width.
    lm_c = coarse.sample(lm)
    big = [k for k in np.flatnonzero(area >= 0.0004) if k != 0]
    # Ragged, patchy edges: how far each species gets, and how poor a habitat it will put
    # up with, both vary from place to place. One shared pair of noise maps, shifted
    # east-west by a different amount for each species, is plenty.
    edge_seed = world.cfg.seed * 7919 + 515151
    edge_noise = (fbm(Perlin3(edge_seed + 2), world.grid.xyz, 10, 3),
                  fbm(Perlin3(edge_seed + 3), world.grid.xyz, 16, 2))
    earth = is_earth(world.cfg)
    for i, sp in enumerate(SPECIES):
        sp_seed = int(rng.integers(2 ** 63))                     # fixed draws per species
        sp_rng = np.random.default_rng(sp_seed)
        suit = suitability[i]
        suit_c = coarse._down(suit)
        homes = np.zeros(coarse.shape, bool)
        reach_km = sp.reach_km
        if earth:   # Earth: real homelands (earth_data.py) instead of picked ones
            homes = _earth_homes(coarse, suit_c, HOMELANDS[sp.key])
            reach_km = EARTH_REACH_KM.get(sp.key, reach_km)
            big = []
        marks = sp_rng.random(coarse.shape).ravel()
        tickets = sp_rng.random((max(sp.homelands),) + marks.shape)   # fixed per cell, per homeland
        for land_id in big:
            a = area[land_id]
            own = np.flatnonzero(lm_c.ravel() == land_id)
            if len(own) == 0:
                continue
            anchor = int(own[np.argmin(marks[own])])
            roll = 1 - (1 - marks[anchor]) ** len(own)          # the lowest of n: uniform again
            land_rng = np.random.default_rng((sp_seed, anchor))
            n_homes = int(land_rng.integers(sp.homelands[0], sp.homelands[1] + 1))
            if roll > sp.chance * np.clip(np.sqrt(a / 0.01), 0.25, 1.0):
                continue
            cells = np.flatnonzero((lm_c == land_id).ravel() & (suit_c.ravel() >= 0.6))
            if len(cells) == 0:
                continue
            # A pick weighted by suitability that small changes in suitability hardly move:
            # the cell with the highest ticket ** (1 / suitability).
            weight = suit_c.ravel()[cells]
            for k in range(n_homes):
                homes.ravel()[cells[np.argmax(np.log(tickets[k][cells]) / weight)]] = True
        if not homes.any():
            continue
        noise_seed = world.cfg.seed * 7919 + 515151 + i * 104729
        effort = coarse.spread(homes, suit, noise_seed=noise_seed, reach_km=reach_km * RANGE_EFFORT)
        offset = int(sp_rng.integers(lm.shape[1]))
        budget = reach_km * RANGE_EFFORT * (1 + 0.4 * np.roll(edge_noise[0], offset, axis=1))
        tolerance = 0.35 + 0.15 * np.roll(edge_noise[1], -offset, axis=1)
        out[i] = (effort <= budget) & (suit >= tolerance)
        if earth:
            # The real homelands themselves always count, wherever the habitat is at all
            # possible: the model's climate is only roughly Earth's (too dry or wet in places).
            out[i] |= _earth_cores(world, HOMELANDS[sp.key]) & (suit >= EARTH_CORE_MIN_SUIT)
    return out


EARTH_CORE_MIN_SUIT = 0.05


def _earth_cores(world, places):
    """Land within each of a species' real homelands, at full resolution (Earth only)."""
    from .earth import great_circle_km

    lat, lon = world.grid.lat, world.grid.lon
    core = np.zeros(lat.shape, bool)
    for plat, plon, radius_km in places:
        core |= great_circle_km(lat, lon, plat, plon) <= radius_km
    return core & ~world.ocean & ~world.lake


def _earth_homes(coarse, suit_c, places):
    """Coarse cells of Earth's homelands for one species: the suitable land within each
    place's radius, or (if the model's climate leaves none suitable there) the most
    suitable land cell within it, so every real homeland seeds a range."""
    from .earth import great_circle_km

    homes = np.zeros(coarse.shape, bool)
    lat, lon = coarse.lat, coarse.lon
    for plat, plon, radius_km in places:
        near = (great_circle_km(lat, lon, plat, plon) <= radius_km) & coarse.land
        if not near.any():
            continue
        good = near & (suit_c >= 0.5)
        if good.any():
            homes |= good
        else:
            homes.ravel()[np.argmax(np.where(near, suit_c, -1).ravel())] = True
    return homes


RANGE_EFFORT = 1.5          # effort budget relative to reach_km (good habitat costs ~1 per km)
COARSE_WIDTH = 400          # ranges are grown on a grid this wide (~100 km cells), then refined


class _CoarseGrid:
    """A coarser copy of the world for growing species ranges quickly."""

    def __init__(self, world):
        from .generator import Grid

        h, w = world.ocean.shape
        self.f = max(1, w // COARSE_WIDTH)
        self.shape = (h // self.f, w // self.f)
        self.full = (h, w)
        self.wraps = world.grid.wraps
        land = (~world.ocean & ~world.lake).astype(np.float32)
        self.land = self._down(land) > 0.5
        self.grid = Grid(*self.shape) if self.wraps else None
        km = world.grid.km_per_px * self.f
        self.lat, self.lon = self._down(world.grid.lat), self.sample(world.grid.lon) + 0.5 * world.grid.dpp * (self.f - 1)
        cos_row = np.maximum(np.cos(np.radians(self._down(world.grid.lat)[:, :1])), 0.05)
        self.km_ns, self.km_ew = km, km * cos_row

    def _down(self, a):
        f = self.f
        h, w = self.shape
        return a[:h * f, :w * f].reshape(h, f, w, f).mean(axis=(1, 3))

    def sample(self, a):
        """One full-resolution value per coarse cell (its first sub-cell)."""
        h, w = self.shape
        return a[:h * self.f:self.f, :w * self.f:self.f]

    def spread(self, homes, suit, noise_seed, reach_km):
        """Effort (km-equivalents) to reach each full-resolution cell from the homelands
        (`homes`: a bool array on the coarse grid)."""
        from .generator import shift

        homes_c = homes                                   # already on the coarse grid
        suit_c = self._down(suit)
        # Random difficulty at two scales: broad (~1000 km) and local (~300 km), so even
        # uniform habitat gets ragged edges, lobes and gaps.
        noise = (fbm(Perlin3(noise_seed), self.grid.xyz, 4, 3)
                 + 0.6 * fbm(Perlin3(noise_seed + 1), self.grid.xyz, 12, 2)) if self.grid is not None else 0.0
        # Effort per km of crossing a cell: 1 in ideal habitat, more in poor habitat,
        # a lot in unsuitable gaps (still crossable if narrow), impossible over water.
        cost = np.where(suit_c >= 0.2, 1 / (0.25 + 0.75 * suit_c), 8.0) * np.exp(0.9 * noise)
        cost = np.where(self.land, cost, np.inf)
        dist = np.where(homes_c, 0.0, np.inf)
        steps = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy or dx:
                    length = np.hypot(dy * self.km_ns, dx * self.km_ew) * np.ones(self.shape)
                    steps.append((dy, dx, length * 0.5 * (cost + shift(cost, dy, dx))))
        for _ in range(int(reach_km / self.km_ns) + 10):
            best = dist
            for dy, dx, step in steps:
                best = np.minimum(best, shift(dist, dy, dx) + step)
            if np.array_equal(best, dist):
                break
            dist = best
        f = self.f
        full = np.repeat(np.repeat(dist, f, axis=0), f, axis=1)
        out = np.full(self.full, np.inf)
        out[:full.shape[0], :full.shape[1]] = full
        return out


# ----------------------------------------------------------------- minerals
MINERALS = ("copper", "tin", "iron", "gold", "silver", "coal", "oil", "saltpeter")


def mineral_fields(cfg, grid, env, elevation_m, ocean, lake, precip, temperature, wetland):
    """Mineral richness per cell (0..1) from the geology that forms each mineral."""
    noise = [Perlin3(cfg.seed * 7919 + 424242 + k * 104729) for k in range(len(MINERALS))]
    xyz = grid.xyz
    land = ~ocean & ~lake
    landness, ore, volcanic = env["landness"], env["ore"], env["volcanic"]
    belts = np.clip(env["collision"] / max(float(np.max(env["collision"])), 1e-6), 0, 1)
    lowland = smoothstep(2000, 600, np.maximum(elevation_m, 0))
    shelf = ocean & (elevation_m > -300)
    kpp = grid.km_per_px
    foreland = np.clip(blur(belts * landness, 300 / kpp, grid.wraps) * 3, 0, 1)  # basins beside mountains
    arid = np.clip((450 - precip) / 350, 0, 1)

    def patches(k, freq, lo, hi):
        return smoothstep(lo, hi, fbm(noise[k], xyz, freq, 4))

    geology = {
        "copper": np.clip(belts + 0.6 * volcanic + 0.4 * ore, 0, 1),
        "tin": ore * landness,
        "iron": np.clip(0.3 * landness + 0.4 * ore + 0.3 * wetland, 0, 1),
        "gold": np.clip(0.7 * ore + 0.5 * volcanic, 0, 1),
        "silver": np.clip(0.6 * ore + 0.5 * volcanic, 0, 1),
        "coal": landness * lowland * np.clip(0.4 + foreland, 0, 1) * (0.5 + 0.5 * np.clip(precip / 1500, 0, 1)),
        "oil": np.clip(landness * lowland * (0.4 + foreland) + 0.8 * shelf, 0, 1),
        "saltpeter": arid * smoothstep(5, 15, temperature),
    }
    rarity = {  # (patch size frequency, threshold low, high): higher thresholds = rarer
        "copper": (6, 0.05, 0.45), "tin": (4, 0.45, 0.75), "iron": (5, -0.1, 0.35),
        "gold": (8, 0.35, 0.7), "silver": (8, 0.4, 0.75), "coal": (2.5, 0.15, 0.5),
        "oil": (2.5, 0.4, 0.7), "saltpeter": (6, -0.05, 0.4),
    }
    out = {}
    earth = is_earth(cfg)
    for k, name in enumerate(MINERALS):
        freq, lo, hi = rarity[name]
        value = geology[name] * patches(k, freq, lo, hi)
        allowed = land | (shelf if name == "oil" else np.zeros_like(land))
        if earth:   # Earth: known ore regions, plus a damped copy of the geology
            damp = 1.0 if name == "iron" else MINERAL_DAMPING
            value = np.maximum(_earth_ores(grid, ORES[name], patches(k, freq * 3, -0.6, 0.4)), damp * value)
        out[name] = np.where(allowed, np.clip(value, 0, 1), 0.0).astype(np.float32)
    return out


def _earth_ores(grid, places, patchiness):
    """Richness (0..1) of Earth's known ore regions on any grid."""
    from .earth import _disc

    out = np.zeros(grid.lat.shape)
    for lat, lon, radius_km, richness in places:
        out = np.maximum(out, richness * _disc(grid, lat, lon, radius_km))
    return out * (0.6 + 0.4 * patchiness)


# ------------------------------------------------------------------ disease
def disease_fields(elevation_m, ocean, lake, temperature, precip, wetland, warmest):
    """Disease environments (0..1) on land."""
    land = ~ocean & ~lake
    low = 1 - smoothstep(1200, 2000, elevation_m)
    # The deadly kind needs warmth most of the year, not just a hot summer: Japan, Korea and
    # the Po valley had mild (vivax) malaria; Bengal, the Congo and the Guinea coast the worst.
    malaria = (smoothstep(16, 24, warmest) * smoothstep(12, 21, temperature) * smoothstep(500, 1400, precip) * low
               * (0.6 + 0.4 * np.clip(wetland * 2, 0, 1)))
    # Tsetse flies need hot, seasonally wet woodland-savanna (deadly to cattle and horses).
    tsetse = (smoothstep(22, 25, temperature) * smoothstep(600, 900, precip)
              * (1 - smoothstep(1300, 1800, precip)) * (1 - smoothstep(1000, 1500, elevation_m)))
    return {"malaria": np.where(land, malaria, 0.0).astype(np.float32),
            "tsetse": np.where(land, tsetse, 0.0).astype(np.float32)}


# --------------------------------------------------------------------- seas
def sea_fields(world):
    """Prevailing winds and surface currents over the sea (m/s, east and north parts).

    Trade winds blow toward the west in the tropics (with calm doldrums at the equator),
    westerlies blow east in mid-latitudes, polar easterlies near the poles. Currents
    circle each ocean: west along the equator, east in the mid-latitudes, toward the
    poles along east coasts of continents and toward the equator along west coasts.
    """
    from .generator import _row_distance_to_land

    lat = world.grid.lat
    alat = np.abs(lat)
    sign = np.sign(lat)
    ocean = world.ocean
    # Winds: speed by latitude band; direction east (+) / west (-) plus a little toward the equator in the trades.
    east = np.interp(alat, [0, 5, 15, 28, 33, 45, 60, 65, 80, 90], [-2, -4, -7, -3, 2, 10, 8, -2, -5, -4])
    north = np.where(alat < 30, -sign * 2.0, np.where(alat < 60, sign * 1.0, -sign * 1.5))
    # Currents: zonal pattern plus boundary currents along coasts.
    px_km = world.grid.km_per_px * np.maximum(np.cos(np.radians(lat[:, 0])), 0.05)
    land_east = np.exp(-_row_distance_to_land(ocean, px_km, True) / 1000)
    land_west = np.exp(-_row_distance_to_land(ocean, px_km, False) / 1000)
    c_east = np.interp(alat, [0, 15, 25, 40, 50, 65, 90], [-0.5, -0.4, -0.1, 0.3, 0.3, 0.1, 0.0])
    poleward = np.where(alat < 45, land_west - land_east, land_east - land_west)
    c_north = sign * 0.5 * poleward
    zero = np.zeros_like(lat)
    return {name: np.where(ocean, value, zero).astype(np.float32)
            for name, value in (("wind_east", east), ("wind_north", north),
                                ("current_east", c_east), ("current_north", c_north))}


def glacial_fields(world, sea_drop_m):
    """Land exposed and land under ice at the height of the last ice age."""
    exposed = world.ocean & (world.elevation_m > -sea_drop_m)
    land_then = ~world.ocean | exposed
    ice = land_then & (world.temperature_c - 6 < -6)
    return {"glacial_land": exposed, "glacial_ice": ice}


# ------------------------------------------------------------------ hooks
def add_cell_nature(world, env):
    """Per-cell facts that can be worked out from a world or region on its own."""
    cfg, grid = world.cfg, world.grid
    world.species_suitability = species_suitability(world)
    for name, value in mineral_fields(cfg, grid, env, world.elevation_m, world.ocean, world.lake,
                                      world.precip_mm, world.temperature_c, world.wetland).items():
        setattr(world, name, value)
    for name, value in disease_fields(world.elevation_m, world.ocean, world.lake, world.temperature_c,
                                      world.precip_mm, world.wetland, world.warmest_month_c).items():
        setattr(world, name, value)
    from .earth import is_earth
    if is_earth(cfg):
        # Tsetse flies are African: India's and Brazil's hot savannas have the climate but
        # not the fly (zebu cattle there pull the plough).
        africa = (grid.lat > -32) & (grid.lat < 18) & (grid.lon > -20) & (grid.lon < 52)
        world.tsetse = np.where(africa, world.tsetse, 0.0).astype(np.float32)


def add_global_nature(world):
    """Facts that need the whole globe (landmasses, ocean basins); regions copy them."""
    world.species_native = species_native(world, world.species_suitability)
    for name, value in sea_fields(world).items():
        setattr(world, name, value)
    for name, value in glacial_fields(world, world.cfg.glacial_sea_drop_m).items():
        setattr(world, name, value)
