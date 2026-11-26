"""The world's data contract: every field, its units and meaning; export and caching.

FIELDS below is the single list of what a world provides. DATA.md is generated from it
(`python -m worldgen.export --docs`), and `save_world` writes exactly these fields.

Two ways to keep a world:
- `save_world` / `load_world_data`: a .npz file of plain arrays, readable by any program.
- `save_world_cache` / `load_world_cache` / `load_or_generate`: a compact snapshot of the
  full World object (so regions can still be enhanced), for reloading quickly.

A world can always be regenerated exactly from its settings (seed and config), so for
sharing a world the settings are enough.
"""
import gzip
import hashlib
import json
import os
import pickle
import sys
from dataclasses import asdict

import numpy as np

from .generator import BIOMES, DATA_VERSION
from .nature import MINERALS, SPECIES

# name: (units, meaning). "Cell" fields have one value per map cell (rows north to south,
# columns west to east from 180W). "Edge" fields describe the edge between a cell and its
# neighbour to the east ("_east") or south ("_south").
FIELDS = {
    # --- position
    "lat": ("degrees", "latitude of the cell centre (+ north)"),
    "lon": ("degrees", "longitude of the cell centre (-180..180)"),
    # --- terrain
    "elevation_m": ("m", "height above (+) or depth below (-) sea level"),
    "ocean": ("bool", "sea"),
    "slope_m_per_km": ("m/km", "steepness of the land (0 on water)"),
    "plate": ("id", "tectonic plate the cell sits on"),
    # --- climate
    "temperature_c": ("C", "mean annual temperature"),
    "warmest_month_c": ("C", "mean temperature of the warmest month"),
    "coldest_month_c": ("C", "mean temperature of the coldest month"),
    "precip_mm": ("mm/yr", "annual precipitation"),
    "wettest_month_mm": ("mm", "rain in the wettest month"),
    "driest_month_mm": ("mm", "rain in the driest month"),
    "rain_seasonality": ("0..1", "0 = rain all year, 1 = all rain in one season"),
    "warm_months": ("months", "months warmer than 5 C"),
    "growing_months": ("months", "months both warm (>5 C) and wet (>40 mm) enough for crops without irrigation"),
    "biome": ("index", "index into metadata['biomes']"),
    # --- water
    "lake": ("bool", "lake surface"),
    "salt_flat": ("bool", "dry lake bed"),
    "lake_level_m": ("m", "water surface height where there's a lake (NaN elsewhere)"),
    "discharge_m3s": ("m3/s", "water draining through the cell (rivers themselves are on edges)"),
    "catchment_km2": ("km2", "land area draining through the cell"),
    "receiver": ("flat index", "cell this cell drains into (row * width + col); -1 = none"),
    "river": ("bool", "a river big enough to show when zoomed out drains through this cell"),
    "stream": ("bool", "enough catchment to count as a river at all"),
    # --- rivers on cell edges
    "river_flow_east": ("m3/s", "river along the edge to the east neighbour (0 = none)"),
    "river_flow_south": ("m3/s", "river along the edge to the south neighbour (0 = none)"),
    "river_dry_flow_east": ("m3/s", "dry-season flow of that river"),
    "river_dry_flow_south": ("m3/s", "dry-season flow of that river"),
    "river_slope_east": ("m/km", "how steeply that river falls (rapids above rapids_min_slope_m_per_km)"),
    "river_slope_south": ("m/km", "how steeply that river falls"),
    # --- land use
    "soil_fertility": ("0..1", "natural soil quality (biome, floodplains, volcanic soil, slope)"),
    "arable": ("0..1", "suitability for farming (fertility, slope, growing season, water)"),
    "fresh_water": ("0..1", "access to fresh water (rivers, lakes, rainfall)"),
    "floodplain": ("0..1", "flat land a big river floods every year: farmable on the receding flood without rain or canals"),
    "irrigable": ("0..1", "land canals could water from a river or lake (includes floodplain)"),
    "timber": ("0..1", "usable wood"),
    "ore": ("0..1", "likelihood of metal ores (mountain belts, active or ancient)"),
    "salt": ("0..1", "salt from dry lake beds and coastal salt pans"),
    "fish": ("0..1", "fishing from shallow coastal seas and lakes"),
    # --- coasts and travel
    "coastal": ("bool", "land cell next to the sea"),
    "dist_to_coast_km": ("km", "distance to the coastline (land: to sea; sea: to land)"),
    "harbour": ("0..1", "natural harbour quality: sheltered water and river mouths (coastal cells)"),
    "wetland": ("0..1", "waterlogged ground (marsh, swamp, bog); cells above wetland_threshold get the Wetland biome"),
    "permafrost": ("0..1", "permanently frozen ground (0 = none, 1 = deep continuous permafrost)"),
    # --- nature (worldgen 1.1)
    "species_suitability": ("0..1 (species, rows, cols)", "how well each wild species in metadata['species'] could live in the cell"),
    "species_native": ("bool (species, rows, cols)", "the species lives wild here at the start of history (year 0); the same ranges at any map width"),
    "copper": ("0..1", "copper deposits (mountain belts, volcanic arcs)"),
    "tin": ("0..1", "tin deposits (rare; old and active mountain belts)"),
    "iron": ("0..1", "iron ore (common; also bog iron in wetlands)"),
    "gold": ("0..1", "gold deposits"),
    "silver": ("0..1", "silver deposits"),
    "coal": ("0..1", "coal (lowland basins beside mountain belts)"),
    "oil": ("0..1", "oil (sedimentary basins and continental shelves; may be under shallow sea)"),
    "saltpeter": ("0..1", "natural saltpeter (hot deserts); needed for gunpowder"),
    "malaria": ("0..1", "malaria-type disease risk (hot, wet, low ground)"),
    "tsetse": ("0..1", "tsetse-fly zone, deadly to cattle and horses (hot woodland savanna)"),
    "wind_east": ("m/s", "prevailing wind over the sea, toward the east (negative = toward the west)"),
    "wind_north": ("m/s", "prevailing wind over the sea, toward the north"),
    "current_east": ("m/s", "surface ocean current, toward the east"),
    "current_north": ("m/s", "surface ocean current, toward the north"),
    "glacial_land": ("bool", "sea today but dry land at the height of the last ice age (land bridges)"),
    "glacial_ice": ("bool", "under ice sheets at the height of the last ice age"),
    "movement_cost": ("x", "effort to cross the cell on foot, 1 = flat grassland (inf on water); "
                           "river crossings are extra, see the river edge fields"),
}

EDGE_NOTES = """River edges: rivers run along the edges between cells, so they can be borders and
must be crossed. Derived facts (see worldgen.hydrology.edge_properties):
width_m ~ 7 * sqrt(flow), depth_m ~ 0.27 * flow^0.39,
fordable = dry-season riffle depth (half the typical depth) < ford_max_depth_m,
navigable = depth >= navigable_min_depth_m and no rapids,
rapids = slope >= rapids_min_slope_m_per_km."""


def _metadata(world):
    from . import __version__

    g = world.grid
    return {
        "worldgen_version": __version__,
        "data_version": DATA_VERSION,
        "config": asdict(world.cfg),
        "biomes": [name for name, _ in BIOMES],
        "species": [{"key": s.key, "label": s.label, "kind": s.kind, "uses": list(s.uses),
                     "domestication_difficulty": s.difficulty, "food_value": s.food} for s in SPECIES],
        "minerals": list(MINERALS),
        "fields": {name: {"units": u, "meaning": m} for name, (u, m) in FIELDS.items()},
        "edge_notes": EDGE_NOTES,
        "grid": {"height": g.height, "width": g.width, "north_edge": g.lat0, "west_edge": g.lon0,
                 "degrees_per_cell": g.dpp, "km_per_cell_at_equator": g.km_per_px,
                 "whole_globe": g.wraps},
    }


def save_world(world, path):
    """Write a world (or an enhanced region) to `path` as a compressed .npz file of plain arrays."""
    arrays = {"lat": world.grid.lat, "lon": world.grid.lon, "plate": world.terrain.plate}
    for name in FIELDS:
        if name not in arrays:
            arrays[name] = getattr(world, name)
    np.savez_compressed(path, metadata=json.dumps(_metadata(world)), **arrays)


def load_world_data(path):
    """Read a file written by `save_world`. Returns (arrays, metadata)."""
    with np.load(path) as data:
        arrays = {name: data[name] for name in data.files if name != "metadata"}
        metadata = json.loads(str(data["metadata"]))
    return arrays, metadata


# ------------------------------------------------------------------ cache
def save_world_cache(world, path):
    """Snapshot the full World (floats stored as 32-bit to keep files small). Written to a
    temporary file and renamed into place, so another process loading the same world at
    the same time never reads a half-written file."""
    tmp = f"{path}.{os.getpid()}.tmp"
    with gzip.open(tmp, "wb", compresslevel=1) as f:
        pickle.dump(world, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def load_world_cache(path):
    """Load a snapshot made by `save_world_cache`. Only load files you made yourself:
    like any pickle, a snapshot can run code when loaded."""
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def load_or_generate(cfg, cache_dir="world_cache"):
    """Load the world for `cfg` from `cache_dir` if it was saved before, else generate and save it."""
    from . import __version__
    from .generator import generate_world

    key = json.dumps({"cfg": asdict(cfg), "v": __version__, "d": DATA_VERSION}, sort_keys=True)
    name = f"world_{cfg.seed}_{cfg.width}_{hashlib.sha1(key.encode()).hexdigest()[:12]}.pkl.gz"
    path = os.path.join(cache_dir, name)
    if os.path.exists(path):
        return load_world_cache(path)
    world = generate_world(cfg)
    os.makedirs(cache_dir, exist_ok=True)
    save_world_cache(world, path)
    # Return the saved copy, not the one just generated: the cache stores some fields at
    # lower precision, and a world should be the same whether it was cached or not.
    return load_world_cache(path)


# ------------------------------------------------------------------ docs
def fields_markdown():
    """DATA.md: the data contract, generated from FIELDS."""
    lines = [
        "# World data reference",
        "",
        f"Data version {DATA_VERSION}. Generated from `worldgen/export.py` "
        "(`python -m worldgen.export --docs > DATA.md`).",
        "",
        "Every field is an array with one value per map cell (rows north to south, columns",
        "west to east starting at 180W), available as `world.<name>` and in files written by",
        "`save_world`. Cells are about 50 km across at the equator on an 800 px world.",
        "",
        "| Field | Units | Meaning |",
        "|---|---|---|",
    ]
    lines += [f"| `{name}` | {units} | {meaning} |" for name, (units, meaning) in FIELDS.items()]
    lines += ["", "## Rivers on edges", "", EDGE_NOTES, "", "## Biomes", ""]
    lines += [f"{i}. {name}" for i, (name, _) in enumerate(BIOMES)]
    lines += ["", "## Wild species", "",
              "Index into the first axis of `species_suitability` and `species_native`.", "",
              "| # | Species | Kind | Uses | Domestication difficulty | Food value |", "|---|---|---|---|---|---|"]
    lines += [f"| {i} | {s.label} (`{s.key}`) | {s.kind} | {', '.join(s.uses)} | {s.difficulty} | {s.food} |"
              for i, s in enumerate(SPECIES)]
    lines += ["", "## Climate over time", "",
              "`worldgen.ClimateHistory(world).at(year)` gives the temperature change and rainfall factor per",
              "cell, sea level, droughts and volcanic cooling for any simulation year (year 0 = about",
              "10,000 BCE). See `worldgen/climate_history.py`."]
    lines += ["", "## Versions", "",
              "`data_version` changes whenever a field is added, removed or changes meaning, so a",
              "program can check it before trusting saved data."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__" and "--docs" in sys.argv:
    sys.stdout.write(fields_markdown())
