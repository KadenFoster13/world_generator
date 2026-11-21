"""Save a world's data to a file other programs can read.

The file is a NumPy .npz archive: one array per map layer, plus a JSON string with
the settings and what the values mean. Load it with `load_world_data` or directly:

    data = numpy.load("world.npz")
    data["elevation_m"], data["biome"], ...

A world can always be regenerated exactly from its settings (seed and config), so
for sharing a world the settings are enough; the file is for using the data
elsewhere without running the generator.
"""
import json
from dataclasses import asdict

import numpy as np

from .generator import BIOMES

LAYERS = {
    "elevation_m": "metres above (+) or below (-) sea level",
    "ocean": "bool: sea",
    "lake": "bool: lake surface",
    "salt_flat": "bool: dry lake bed",
    "temperature_c": "mean annual temperature, C",
    "precip_mm": "annual precipitation, mm",
    "biome": "index into metadata['biomes']",
    "discharge_m3s": "river flow through each cell, m3/s",
    "catchment_km2": "land area draining through each cell, km2",
    "receiver": "flat index (row * width + col) of the cell each cell drains into; -1 = none",
    "river": "bool: cell carries a river big enough to show when zoomed out",
    "stream": "bool: cell has enough catchment to count as a river at all",
    "lat": "latitude of each cell centre, degrees",
    "lon": "longitude of each cell centre, degrees",
}


def save_world(world, path):
    """Write a world (or an enhanced region) to `path` as a compressed .npz file."""
    arrays = {name: getattr(world, name, None) for name in LAYERS}
    arrays["lat"] = world.grid.lat
    arrays["lon"] = world.grid.lon
    arrays["plate"] = world.terrain.plate
    g = world.grid
    metadata = {
        "config": asdict(world.cfg),
        "biomes": [name for name, _ in BIOMES],
        "layers": {**LAYERS, "plate": "tectonic plate id"},
        "grid": {"height": g.height, "width": g.width, "north_edge": g.lat0, "west_edge": g.lon0,
                 "degrees_per_cell": g.dpp, "km_per_cell_at_equator": g.km_per_px,
                 "whole_globe": g.wraps},
    }
    np.savez_compressed(path, metadata=json.dumps(metadata), **arrays)


def load_world_data(path):
    """Read a file written by `save_world`. Returns (arrays, metadata)."""
    with np.load(path) as data:
        arrays = {name: data[name] for name in data.files if name != "metadata"}
        metadata = json.loads(str(data["metadata"]))
    return arrays, metadata
