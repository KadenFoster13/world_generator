"""worldgen: procedural Earth-like worlds with plate tectonics, climate, lakes and rivers.

Quick start:

    from worldgen import WorldConfig, generate_world
    world = generate_world(WorldConfig(seed=42))
    world.elevation_m, world.biome, world.discharge_m3s   # numpy arrays, one value per map cell

    # More detail for part of the map (2-8x finer, matching the world):
    from worldgen import region_box, generate_region
    region = generate_region(world, region_box(world, north=20, south=0, west=10, east=50))

    # Reload instantly next time (generates and saves on the first call):
    from worldgen import load_or_generate
    world = load_or_generate(WorldConfig(seed=42))

Every field, with units, is listed in DATA.md (and in worldgen.export.FIELDS).

Open the interactive viewer with `python -m worldgen` (or the `worldgen` command once installed).
"""
from .config import DEFAULT, RESOLUTIONS, WorldConfig
from .export import (FIELDS, load_or_generate, load_world_cache, load_world_data, save_world,
                     save_world_cache)
from .generator import (BIOME_ID, BIOMES, DATA_VERSION, World, apply_climate, generate_terrain,
                        generate_world)
from .climate_history import ClimateHistory, ClimateState, sea_level_m
from .hydrology import cell_river_flow, edge_properties, river_depth_m, river_width_m
from .nature import MINERALS, SPECIES, SPECIES_KEYS
from .region import RegionBox, box_for_view, generate_region, region_box

__version__ = "1.5.0"

__all__ = [
    "DEFAULT", "RESOLUTIONS", "WorldConfig",
    "BIOMES", "BIOME_ID", "World", "generate_world", "generate_terrain", "apply_climate",
    "RegionBox", "region_box", "box_for_view", "generate_region",
    "save_world", "load_world_data", "save_world_cache", "load_world_cache", "load_or_generate",
    "FIELDS", "DATA_VERSION", "edge_properties", "cell_river_flow", "river_width_m", "river_depth_m",
    "ClimateHistory", "ClimateState", "sea_level_m", "SPECIES", "SPECIES_KEYS", "MINERALS",
]
