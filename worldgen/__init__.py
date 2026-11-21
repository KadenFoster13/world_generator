"""worldgen: procedural Earth-like worlds with plate tectonics, climate, lakes and rivers.

Quick start:

    from worldgen import WorldConfig, generate_world
    world = generate_world(WorldConfig(seed=42))
    world.elevation_m, world.biome, world.discharge_m3s   # numpy arrays, one value per map cell

    # More detail for part of the map (2-8x finer, matching the world):
    from worldgen import region_box, generate_region
    region = generate_region(world, region_box(world, north=20, south=0, west=10, east=50))

Open the interactive viewer with `python -m worldgen` (or the `worldgen` command once installed).
"""
from .config import DEFAULT, RESOLUTIONS, WorldConfig
from .export import load_world_data, save_world
from .generator import BIOME_ID, BIOMES, World, apply_climate, generate_terrain, generate_world
from .region import RegionBox, box_for_view, generate_region, region_box

__version__ = "0.1.0"

__all__ = [
    "DEFAULT", "RESOLUTIONS", "WorldConfig",
    "BIOMES", "BIOME_ID", "World", "generate_world", "generate_terrain", "apply_climate",
    "RegionBox", "region_box", "box_for_view", "generate_region",
    "save_world", "load_world_data",
]
