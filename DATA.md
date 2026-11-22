# World data reference

Data version 1. Generated from `worldgen/export.py` (`python -m worldgen.export --docs > DATA.md`).

Every field is an array with one value per map cell (rows north to south, columns
west to east starting at 180W), available as `world.<name>` and in files written by
`save_world`. Cells are about 50 km across at the equator on an 800 px world.

| Field | Units | Meaning |
|---|---|---|
| `lat` | degrees | latitude of the cell centre (+ north) |
| `lon` | degrees | longitude of the cell centre (-180..180) |
| `elevation_m` | m | height above (+) or depth below (-) sea level |
| `ocean` | bool | sea |
| `slope_m_per_km` | m/km | steepness of the land (0 on water) |
| `plate` | id | tectonic plate the cell sits on |
| `temperature_c` | C | mean annual temperature |
| `warmest_month_c` | C | mean temperature of the warmest month |
| `coldest_month_c` | C | mean temperature of the coldest month |
| `precip_mm` | mm/yr | annual precipitation |
| `wettest_month_mm` | mm | rain in the wettest month |
| `driest_month_mm` | mm | rain in the driest month |
| `rain_seasonality` | 0..1 | 0 = rain all year, 1 = all rain in one season |
| `warm_months` | months | months warmer than 5 C |
| `growing_months` | months | months both warm (>5 C) and wet (>40 mm) enough for crops without irrigation |
| `biome` | index | index into metadata['biomes'] |
| `lake` | bool | lake surface |
| `salt_flat` | bool | dry lake bed |
| `lake_level_m` | m | water surface height where there's a lake (NaN elsewhere) |
| `discharge_m3s` | m3/s | water draining through the cell (rivers themselves are on edges) |
| `catchment_km2` | km2 | land area draining through the cell |
| `receiver` | flat index | cell this cell drains into (row * width + col); -1 = none |
| `river` | bool | a river big enough to show when zoomed out drains through this cell |
| `stream` | bool | enough catchment to count as a river at all |
| `river_flow_east` | m3/s | river along the edge to the east neighbour (0 = none) |
| `river_flow_south` | m3/s | river along the edge to the south neighbour (0 = none) |
| `river_dry_flow_east` | m3/s | dry-season flow of that river |
| `river_dry_flow_south` | m3/s | dry-season flow of that river |
| `river_slope_east` | m/km | how steeply that river falls (rapids above rapids_min_slope_m_per_km) |
| `river_slope_south` | m/km | how steeply that river falls |
| `soil_fertility` | 0..1 | natural soil quality (biome, floodplains, volcanic soil, slope) |
| `arable` | 0..1 | suitability for farming (fertility, slope, growing season, water) |
| `fresh_water` | 0..1 | access to fresh water (rivers, lakes, rainfall) |
| `timber` | 0..1 | usable wood |
| `ore` | 0..1 | likelihood of metal ores (mountain belts, active or ancient) |
| `salt` | 0..1 | salt from dry lake beds and coastal salt pans |
| `fish` | 0..1 | fishing from shallow coastal seas and lakes |
| `coastal` | bool | land cell next to the sea |
| `dist_to_coast_km` | km | distance to the coastline (land: to sea; sea: to land) |
| `harbour` | 0..1 | natural harbour quality: sheltered water and river mouths (coastal cells) |
| `wetland` | 0..1 | waterlogged ground (marsh, swamp, bog); cells above wetland_threshold get the Wetland biome |
| `permafrost` | 0..1 | permanently frozen ground (0 = none, 1 = deep continuous permafrost) |
| `movement_cost` | x | effort to cross the cell on foot, 1 = flat grassland (inf on water); river crossings are extra, see the river edge fields |

## Rivers on edges

River edges: rivers run along the edges between cells, so they can be borders and
must be crossed. Derived facts (see worldgen.hydrology.edge_properties):
width_m ~ 7 * sqrt(flow), depth_m ~ 0.27 * flow^0.39,
fordable = dry-season riffle depth (half the typical depth) < ford_max_depth_m,
navigable = depth >= navigable_min_depth_m and no rapids,
rapids = slope >= rapids_min_slope_m_per_km.

## Biomes

0. Deep ocean
1. Shallow sea
2. Sea ice
3. Ice sheet
4. Tundra
5. Boreal forest
6. Grassland
7. Temperate forest
8. Temperate rainforest
9. Desert
10. Scrubland
11. Savanna
12. Tropical dry forest
13. Tropical rainforest
14. Alpine
15. Lake
16. Salt flat
17. Glacier
18. Wetland

## Versions

`data_version` changes whenever a field is added, removed or changes meaning, so a
program can check it before trusting saved data.
