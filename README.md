# worldgen

Procedural Earth-like worlds for games and simulations. Each seed produces a whole
planet (an equirectangular map that wraps east–west) built from simple physics:

- **Plate tectonics**: crust pieces form 1–7 continents and are grouped into moving
  plates. Mountain chains rise where plates collide, rift valleys open where they pull
  apart, and island arcs, hotspot chains and coastal archipelagos dot the oceans.
- **Climate**: temperature from latitude and altitude; rain carried by prevailing
  winds, with rain shadows behind mountains and monsoons on subtropical east coasts.
  Ocean currents warm or chill coasts (mild western shores at high latitudes, coastal
  deserts along cold currents).
- **Biomes** from temperature and rainfall (desert, savanna, rainforest, taiga, tundra…),
  plus wetlands, mountain glaciers, ice sheets and permafrost.
- **Seasons**: warmest and coldest months, wet and dry seasons, and the growing season.
- **Water**: lakes and salt flats form in basins; rivers are routed downhill with
  discharge in m³/s, through lakes and out to sea, and carve their own valleys. Rivers run along the *edges* between
  cells, with width, depth, fords, navigability and rapids, so they can be borders.
- **Land use**: soil fertility, farmland, fresh water, timber, ore, salt, fishing,
  natural harbours, distance to the coast and the effort to cross each cell; floodplains
  (farmable on the yearly flood, even in desert) and land canals could irrigate.
- **Nature for history**: wild crops and animals with native homelands (not every
  continent gets horses or easy grains), copper, tin, iron, gold, silver, coal, oil and
  saltpeter deposits, malaria and tsetse zones, sailing winds and currents, ice-age land
  bridges, and a climate history over 12,000 years (`ClimateHistory`).
- **Detail on demand**: the same world at 800, 1600 or 2400 px, and any area can be
  regenerated at 2–8× finer detail (down to ~3 km per cell) that matches the world map,
  with meandering rivers and deltas.

Everything uses real units (metres, °C, mm/yr, km, m³/s), and every setting lives in
[`worldgen/config.py`](worldgen/config.py). Every output field is listed, with units, in
[DATA.md](DATA.md).

## Seed 0 is Earth

Seed 0 is the real planet, for playing on Earth and for checking a simulation against
Earth's geography. Its elevation comes from NOAA's ETOPO 2022 relief model instead of the
plate generator, and on land its temperature and rain are Earth's real 1981-2010 monthly
means instead of the generator's climate model (the model still covers the sea). Seasons
come from the real months too: the warmest and coldest month, wet and dry seasons, and
the months warm and wet enough to grow crops, so the monsoon and Mediterranean dry
summers come out right. Temperatures are corrected for height within each half-degree
cell. Earth's biomes follow from that climate, with one Earth-only rule: tundra only
where the warmest month stays below ~12 C, so cold-wintered steppe (Kazakhstan, Mongolia)
and taiga (Siberia) aren't classed as tundra. Everything else (sea level at 0 m, rivers,
lakes, soils, floodplains, ice-age coasts, disease) runs exactly as for any other seed.

- **Wild species** start from Earth's real homelands (wheat and barley in the Fertile
  Crescent, rice on the Yangtze and Ganges, teosinte in the Balsas valley, potatoes by
  Titicaca, horses on the Pontic-Caspian steppe...) and spread through suitable habitat
  as on any world. The Americas get no wild cattle, horses, sheep or goats, and Australia
  no sheep or goats. The table, with what each entry stands for, is in
  [`worldgen/earth_data.py`](worldgen/earth_data.py).
- **Minerals** are richest in Earth's known ore regions (tin in Cornwall, the Erzgebirge,
  Malaya, Yunnan, Bolivia; copper on Cyprus, at Rio Tinto, in the Andes and the
  copperbelt; silver at Laurion and Potosi...), from the same file. Elsewhere the usual
  geology-based fields apply, scaled down (iron stays mostly geological).
- **Lakes** come from a table of Earth's real lakes (the Great Lakes, Baikal, Victoria,
  Tanganyika, Titicaca, Ladoga, the Aral as it was before irrigation...), each grown from
  points along its length to its real area, at its real surface level, keeping near those
  points and out of ground above its surface (so it keeps its real shape): the
  elevation data records lake beds, not lake surfaces, so lakes can't be found from it. A
  few gorges too narrow for the grid (the Danube's Iron Gates, the Yangtze's Three Gorges,
  the Congo's cataracts, the St Lawrence) are cut back in, so the basins behind them drain
  instead of filling into false seas. Both tables are in `worldgen/earth_data.py`.
- Other widths are resampled from the stored 1600 x 800 grid; Enhance adds a little fine
  relief on top. The Plates layer shows a single plate.

Climate source: Willmott, C. J. and K. Matsuura, Terrestrial Air Temperature and
Precipitation: Monthly and Annual Time Series (V5.01), University of Delaware; the
1981-2010 long-term monthly means, data provided by the NOAA PSL, Boulder, Colorado, USA,
from their website at https://psl.noaa.gov. Stored as `worldgen/data/earth_climate.npz`
(0.5 degree grid, 3 MB), made by
[`scripts/make_earth_climate.py`](scripts/make_earth_climate.py), which names the files.

Elevation source: ETOPO 2022 15 arc-second global relief model (ice surface), NOAA
National Centers for Environmental Information, DOI
[10.25921/fd45-gt74](https://doi.org/10.25921/fd45-gt74). US-government data; NOAA's
ERDDAP server, where it was fetched, states it "may be used and redistributed for free".
It is stored as `worldgen/data/earth_elevation.npz` (int16 metres, 2 MB), made by
[`scripts/make_earth_elevation.py`](scripts/make_earth_elevation.py), which says how to
download the source again.

## Install

Needs Python 3.10 or newer.

```
python -m venv venv
venv\Scripts\activate            (Windows)   or   source venv/bin/activate   (macOS/Linux)
pip install -e .                 (from this folder)
```

`-e` ("editable") means changes to the code take effect without reinstalling.
For faster river routing (worth it at 1600/2400 px), install the optional extra:
`pip install -e ".[fast]"` (adds numba; results are identical either way).

## The viewer

```
python -m worldgen                    (or just: worldgen)
python -m worldgen --seed 42 --width 1600
python -m worldgen --seed 0           (Earth)
```

| Control | What it does |
|---|---|
| Prev / Next / Random, seed box | Browse seeds |
| Mouse wheel, `+` / `-` | Zoom around the cursor |
| Drag, arrow keys | Move around (the globe wraps east–west); rivers hide while you move and return when you stop, which keeps it smooth |
| Reset view, `R` | Zoom back out |
| Globe / Map, `O` | Switch to a globe view: drag to spin, scroll to zoom (hover still works) |
| Resolution 800 / 1600 / 2400 | Same world, sharper (about 2 s / 9 s / 30 s per seed) |
| Enhance, `E` / Clear, `C` | Regenerate the zoomed-in area in more detail (rivers keep the world map's courses) / remove it |
| Auto, `A` | Auto-Enhance on/off: once you stop moving, a zoomed-in view (about 3x or closer at 800 px) is enhanced at 2x by itself in 0.1-1 s; recent areas are remembered (setting remembered) |
| Layer | Biomes, elevation, plates, temperature, precipitation, growing season, farmland, irrigation (floodplain and canal-irrigable land), movement, rivers & harbours, wild species, minerals |
| Options | Coastal archipelagos, lakes, rivers on/off |
| `S` | Save the current map as a PNG in `renders/` |
| `X` | Export the world's data (and any enhanced region) as `.npz` in `renders/` |
| Hover | Read position, biome, elevation, temperatures, rainfall, growing season, farmland, harbour, movement and rivers |
| Favourite, `F` / Favourites >, `G` | Star the current seed / jump through starred seeds (saved in `~/.worldgen_favourites.json`) |
| Dark / Light, `D` | Switch the window to dark or light mode (remembered in `~/.worldgen_viewer.json`; starting default `DARK_MODE` in `config.py`) |

## Using it from code

```python
from worldgen import WorldConfig, generate_world, region_box, generate_region, save_world

world = generate_world(WorldConfig(seed=42))          # 800 x 400 cells, ~2 s
world.elevation_m      # metres, one value per cell (row 0 = north, column 0 = 180°W)
world.biome            # index into worldgen.BIOMES
world.temperature_c, world.precip_mm, world.ocean, world.lake
world.discharge_m3s    # river flow through each cell
world.receiver         # which cell each cell drains into (flat index)

world.growing_months, world.soil_fertility, world.arable, world.harbour, world.movement_cost

# Rivers on cell edges: "east" = the edge between a cell and its right-hand neighbour.
from worldgen import edge_properties
east = edge_properties(world, "east")   # flow, width_m, depth_m, fordable, navigable, rapids...

region = generate_region(world, region_box(world, north=20, south=0, west=10, east=50))
save_world(world, "world.npz")                      # plain arrays for use in other programs

from worldgen import load_or_generate
world = load_or_generate(WorldConfig(seed=42))      # cached: ~0.2 s after the first time
```

See [DATA.md](DATA.md) for every field (regenerate it with
`python -m worldgen.export --docs > DATA.md` after changing `worldgen/export.py`).

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
| `worldgen/hydrology.py` | Drainage, river discharge, rivers on cell edges |
| `worldgen/land.py` | Soil, farmland, water, resources, harbours, movement |
| `worldgen/region.py` | Enhanced regions |
| `worldgen/noise.py` | Noise on the sphere |
| `worldgen/render.py` | Map colours and layers |
| `worldgen/export.py` | The data contract (every field), export, cache |
| `worldgen/earth.py` | Seed 0: Earth's real elevation and the hooks for its tables |
| `worldgen/earth_data.py` | Earth's wild species homelands and ore regions |
| `worldgen/data/` | Earth's elevation and climate grids (see "Seed 0 is Earth") |
| `scripts/make_earth_elevation.py` | Rebuilds that grid from ETOPO 2022 |
| `scripts/make_earth_climate.py` | Rebuilds Earth's climate grid from the University of Delaware climatology |
| `worldgen/viewer.py` | The interactive viewer |
| `worldgen/_fast.py` | Optional compiled (numba) river routing |

## License

MIT: free to use, change and share, including commercially, as long as the
copyright notice in [LICENSE](LICENSE) is kept.
