"""What the land is good for: soil, farmland, water, resources, coasts and travel.

All values are per cell and use simple, documented rules, so a simulation can read
them as facts about the land. Most are 0..1 scores; see DATA.md for every field.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt

from .generator import BIOME_ID, blur, distance_to, smoothstep

# Natural soil fertility by biome (0..1). Grasslands have the deepest, richest soils;
# rainforest soils are leached thin; deserts, tundra and ice are poor.
_FERTILITY = {
    "Grassland": 0.9, "Temperate forest": 0.75, "Tropical dry forest": 0.6, "Savanna": 0.55,
    "Temperate rainforest": 0.5, "Tropical rainforest": 0.35, "Scrubland": 0.3,
    "Boreal forest": 0.3, "Tundra": 0.1, "Alpine": 0.1, "Desert": 0.08,
}
# Soil on a fully flooded floodplain (fresh silt every year), whatever the biome.
ALLUVIAL_FERTILITY = 0.85
# How much usable timber grows, by biome (0..1).
_TIMBER = {
    "Tropical rainforest": 1.0, "Temperate rainforest": 1.0, "Temperate forest": 0.8,
    "Boreal forest": 0.8, "Tropical dry forest": 0.6, "Savanna": 0.2, "Scrubland": 0.1,
    "Grassland": 0.05, "Tundra": 0.02,
}
# Relative effort to cross a cell on foot (1 = open flat grassland), before slope.
_MOVEMENT = {
    "Grassland": 1.0, "Savanna": 1.0, "Scrubland": 1.2, "Salt flat": 1.3, "Temperate forest": 1.5,
    "Tundra": 1.5, "Desert": 1.6, "Tropical dry forest": 1.7, "Boreal forest": 1.8,
    "Temperate rainforest": 2.0, "Tropical rainforest": 2.5, "Alpine": 3.0, "Ice sheet": 4.0,
}


def _by_biome(biome, table, default=0.0):
    lookup = np.full(int(biome.max()) + 1 if biome.size else 1, default)
    for name, value in table.items():
        i = BIOME_ID[name]
        if i < len(lookup):
            lookup[i] = value
    return lookup[biome]


def land_fields(cfg, grid, elevation_m, ocean, lake, salt_flat, biome, temperature, precip,
                seasons, river_flow, volcanic, ore_potential):
    """All per-cell land facts. `river_flow` is the largest river flow on each cell's edges."""
    land = ~ocean & ~lake
    water = ocean | lake
    kpp = grid.km_per_px

    # --- slope (m per km), from the steepest of the two directions
    if grid.wraps:
        gx = (np.roll(elevation_m, -1, axis=1) - np.roll(elevation_m, 1, axis=1)) / 2
    else:
        gx = np.gradient(elevation_m, axis=1)
    gy = np.gradient(elevation_m, axis=0)
    cos_lat = np.maximum(np.cos(np.radians(grid.lat)), 0.05)
    slope = np.hypot(gx / (kpp * cos_lat), gy / kpp)
    slope = np.where(land, slope, 0.0)

    # --- fresh water: rivers on the cell's edges, lake shores, and rainfall
    near_lake = blur(lake.astype(float), 1.0, grid.wraps) > 0.05
    fresh = np.maximum.reduce([
        np.clip(np.log10(1 + river_flow) / 3.5, 0, 1),     # ~3,000 m3/s river = 1
        np.where(near_lake, 0.8, 0.0),
        0.6 * np.clip(precip / 1500, 0, 1),                # wells and streams too small to map
    ])
    fresh = np.where(land, fresh, 0.0)

    # --- frozen ground and waterlogged ground
    permafrost = np.where(land, np.clip((-1 - temperature) / 8, 0, 1), 0.0)
    surplus = precip - cfg.evaporation_per_degree_mm * np.maximum(temperature, 0)
    waterlogged = np.maximum.reduce([
        0.8 * np.clip(surplus / 1200, 0, 1),                                 # very wet lowlands
        np.clip(np.log10(np.maximum(river_flow, 1) / 500) / 1.3, 0, 1),      # big-river floodplains
        np.where(near_lake, 0.6, 0.0),                                       # lake shores
        0.7 * permafrost,                                                    # thawing bogs over permafrost
    ])
    flat = np.clip(1 - slope / 3, 0, 1)
    low = np.clip(1 - np.maximum(elevation_m, 0) / 500, 0, 1)
    wetland = np.where(land & ~salt_flat & (temperature > -10), flat * low * waterlogged, 0.0)
    # Marshes are areas, not scattered cells: smooth over ~25 km (matters on detailed maps).
    wetland = np.where(land, blur(wetland, 25 / kpp, grid.wraps), 0.0)

    # --- river water for farming
    # Floodplain: flat land a big river floods every year, leaving water and silt behind
    # (crops grow on the receding flood with no canals at all, as along the Nile).
    flood_flow = np.clip(np.log10(np.maximum(river_flow, 1) / 300) / 1.5, 0, 1)   # 300 -> 0, ~10,000 m3/s -> 1
    floodplain = np.where(land & ~salt_flat, flood_flow * np.clip(1 - slope / 6, 0, 1)
                          * np.clip(1 - np.maximum(elevation_m, 0) / 1500, 0, 1), 0.0)

    # --- soil fertility
    fertility = _by_biome(biome, _FERTILITY)
    # Cold, wet forests leach their soils (podzols: the Russian and Baltic forest, New
    # England): a temperate forest's soil grows poorer below ~9 C, to boreal-like at 4 C.
    cold_forest = np.isin(biome, [BIOME_ID["Temperate forest"], BIOME_ID["Temperate rainforest"]])
    fertility = np.where(cold_forest, fertility - 0.35 * np.clip((9 - temperature) / 5, 0, 1), fertility)
    from .earth import is_earth
    if is_earth(cfg):   # Earth: its real soil regions (earth_data.SOILS)
        from .earth import _disc
        from .earth_data import SOILS
        for lat, lon, radius_km, value in SOILS:
            w = _disc(grid, lat, lon, radius_km)
            fertility = fertility + (value - fertility) * w
    fertility = fertility + 0.25 * volcanic                # volcanic soils are rich
    fertility = fertility * np.clip(1 - slope / 80, 0.1, 1)
    # Each flood lays down fresh silt, so a floodplain's soil is rich whatever the climate
    # around it. In a desert the floodplain is all the farmland there is (the Nile, the
    # Euphrates, the Indus), so a valley cell's soil is the silt's. In a humid basin the
    # uplands between the rivers are farmed too, on their own (often poor) soil, so silt
    # counts for the share of the cell the river floods: about 0.2 x sqrt(discharge) km wide
    # (~6 km for 1,000 m3/s, ~20 km for 10,000, ~60 km for the Amazon). In between, a blend.
    full_silt = np.clip(floodplain / 0.4, 0, 1)
    share_silt = np.where(floodplain > 0, np.clip(0.2 * np.sqrt(np.maximum(river_flow, 0)) / kpp, 0, 1), 0.0)
    aridity = precip / (cfg.evaporation_per_degree_mm * np.maximum(temperature, 1.0))
    arid = np.clip((1.0 - aridity) / 0.5, 0, 1)            # 1 at rain <= half of evaporation, 0 at >= all of it
    silt = arid * full_silt + (1 - arid) * share_silt
    fertility = fertility + np.clip(ALLUVIAL_FERTILITY - fertility, 0, None) * silt
    fertility = np.where(land & ~salt_flat & (temperature > -10), np.clip(fertility, 0, 1), 0.0)
    # Irrigable: land canals could water from a river or lake (gentle slopes near water;
    # canals reach a few tens of km, so it spreads a little on detailed maps).
    source = np.maximum(np.clip(np.log10(1 + river_flow) / 3, 0, 1), np.where(near_lake, 0.6, 0.0))
    reach = blur(source, 15 / kpp, grid.wraps) if 15 / kpp >= 0.5 else source
    irrigable = np.where(land & ~salt_flat & (temperature > 0),
                         np.maximum(source, reach) * np.clip(1 - slope / 15, 0, 1), 0.0)
    irrigable = np.maximum(irrigable, floodplain)

    # --- arable share: fertile, flat enough, long enough growing season, enough water
    growing = seasons["growing_months"]
    irrigation = np.maximum(np.clip(precip / 700, 0, 1), np.clip(np.log10(1 + river_flow) / 3, 0, 1))
    arable = (fertility * np.clip(1 - slope / 60, 0, 1) * np.clip(growing / 6, 0, 1)
              * np.clip(irrigation, 0, 1) * (1 - 0.6 * wetland))     # marshes need draining first

    # --- resources
    timber = np.where(land, _by_biome(biome, _TIMBER), 0.0)
    ore = np.where(land, np.clip(ore_potential, 0, 1), 0.0)
    coastal = land & _touches(ocean, grid.wraps)
    salt = np.where(salt_flat, 1.0, 0.0)
    salt = np.maximum(salt, np.where(coastal & (precip < 350) & (temperature > 15), 0.6, 0.0))  # salt pans
    shelf = ocean & (elevation_m > -300)
    fish = np.where(coastal, np.clip(blur(shelf.astype(float), 100 / kpp, grid.wraps) * 2, 0, 1), 0.0)
    fish = np.maximum(fish, np.where(land & _touches(lake, grid.wraps), 0.5, 0.0))

    # --- coasts and harbours
    if grid.wraps:
        to_sea = distance_to(ocean)[0] * kpp
        to_land = distance_to(~ocean)[0] * kpp
    else:
        to_sea = distance_transform_edt(~ocean) * kpp if ocean.any() else np.full(ocean.shape, np.inf)
        to_land = distance_transform_edt(ocean) * kpp if (~ocean).any() else np.full(ocean.shape, np.inf)
    dist_to_coast = np.where(ocean, to_land, to_sea)
    # Sheltered water: little open sea around the cell (bays, inlets, behind islands).
    exposure = blur(ocean.astype(float), 120 / kpp, grid.wraps)
    shelter = np.clip((0.65 - exposure) / 0.35, 0, 1)
    river_mouth = np.clip(np.log10(1 + river_flow) / 3.5, 0, 1)
    harbour = np.where(coastal & (temperature > -8), np.clip(0.75 * shelter + 0.35 * river_mouth, 0, 1), 0.0)

    # --- movement: relative effort to cross the cell on foot (inf = water)
    movement = _by_biome(biome, _MOVEMENT, default=1.5) * (1 + slope / 15) * (1 + 2 * wetland)
    movement = np.where(elevation_m > 3000, movement * 1.5, movement)
    movement = np.where(water, np.inf, movement)

    return dict(slope_m_per_km=slope, soil_fertility=fertility, arable=arable, fresh_water=fresh,
                floodplain=floodplain, irrigable=irrigable,
                timber=timber, ore=ore, salt=salt, fish=fish, coastal=coastal,
                dist_to_coast_km=dist_to_coast, harbour=harbour, movement_cost=movement,
                wetland=wetland, permafrost=permafrost)


def _touches(mask, wraps):
    """Cells next to (4-neighbour) a cell in `mask`."""
    out = np.zeros_like(mask)
    for axis, step in ((0, 1), (0, -1), (1, 1), (1, -1)):
        shifted = np.roll(mask, step, axis=axis)
        if axis == 0:
            if step == 1:
                shifted[0] = False
            else:
                shifted[-1] = False
        elif not wraps:
            if step == 1:
                shifted[:, 0] = False
            else:
                shifted[:, -1] = False
        out |= shifted
    return out
