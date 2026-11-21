# worldgen

Procedural Earth-like worlds for games and simulations. Each seed produces a whole
planet (an equirectangular map that wraps east–west) built from simple physics:

- **Plate tectonics**: crust pieces form 1–7 continents and are grouped into moving
  plates. Mountain chains rise where plates collide, rift valleys open where they pull
  apart, and island arcs, hotspot chains and coastal archipelagos dot the oceans.
- **Climate**: temperature from latitude and altitude; rain carried by prevailing
  winds, with rain shadows behind mountains and monsoons on subtropical east coasts.
- **Biomes** from temperature and rainfall (desert, savanna, rainforest, taiga, tundra…).
- **Water**: lakes and salt flats form in basins; rivers are routed downhill with
  discharge in m³/s, through lakes and out to sea.
- **Detail on demand**: the same world at 800, 1600 or 2400 px, and any area can be
  regenerated at 2–8× finer detail (down to ~3 km per cell) that matches the world map.

Everything uses real units (metres, °C, mm/yr, km, m³/s), and every setting lives in
[`worldgen/config.py`](worldgen/config.py).

## Install

Needs Python 3.10 or newer.

```
python -m venv venv
venv\Scripts\activate            (Windows)   or   source venv/bin/activate   (macOS/Linux)
pip install -e .                 (from this folder)
```

`-e` ("editable") means changes to the code take effect without reinstalling.

## The viewer

```
python -m worldgen                    (or just: worldgen)
python -m worldgen --seed 42 --width 1600
```

| Control | What it does |
|---|---|
| Prev / Next / Random, seed box | Browse seeds |
| Mouse wheel, `+` / `-` | Zoom around the cursor |
| Drag, arrow keys | Move around (the globe wraps east–west) |
| Reset view, `R` | Zoom back out |
| Resolution 800 / 1600 / 2400 | Same world, sharper (about 2 s / 9 s / 30 s per seed) |
| Enhance, `E` / Clear, `C` | Regenerate the zoomed-in area in more detail / remove it |
| Layer | Biomes, elevation, plates, temperature, precipitation |
| Options | Coastal archipelagos, lakes, rivers on/off |
| `S` | Save the current map as a PNG in `renders/` |
| `X` | Export the world's data (and any enhanced region) as `.npz` in `renders/` |
| Hover | Read position, biome, elevation, temperature, rainfall and river flow |

## Using it from code

```python
from worldgen import WorldConfig, generate_world, region_box, generate_region, save_world

world = generate_world(WorldConfig(seed=42))          # 800 x 400 cells, ~2 s
world.elevation_m      # metres, one value per cell (row 0 = north, column 0 = 180°W)
world.biome            # index into worldgen.BIOMES
world.temperature_c, world.precip_mm, world.ocean, world.lake
world.discharge_m3s    # river flow through each cell
world.receiver         # which cell each cell drains into (flat index)

region = generate_region(world, region_box(world, north=20, south=0, west=10, east=50))
save_world(world, "world.npz")                      # for use in other programs
```

The same seed and settings always give the same world, so sharing a seed (and any
changed settings) is enough to share a world.

## Tests

```
python -m unittest discover tests
```

## Project layout

| File | Contents |
|---|---|
| `worldgen/config.py` | Every setting, with comments |
| `worldgen/generator.py` | Plates, terrain, climate, lakes, biomes |
| `worldgen/hydrology.py` | Drainage, river discharge, river paths |
| `worldgen/region.py` | Enhanced regions |
| `worldgen/noise.py` | Noise on the sphere |
| `worldgen/render.py` | Map colours and layers |
| `worldgen/export.py` | Saving world data |
| `worldgen/viewer.py` | The interactive viewer |
