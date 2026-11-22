# World data reference

Data version 2. Generated from `worldgen/export.py` (`python -m worldgen.export --docs > DATA.md`).

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
| `species_suitability` | 0..1 (species, rows, cols) | how well each wild species in metadata['species'] could live in the cell |
| `species_native` | bool (species, rows, cols) | the species lives wild here at the start of history (year 0) |
| `copper` | 0..1 | copper deposits (mountain belts, volcanic arcs) |
| `tin` | 0..1 | tin deposits (rare; old and active mountain belts) |
| `iron` | 0..1 | iron ore (common; also bog iron in wetlands) |
| `gold` | 0..1 | gold deposits |
| `silver` | 0..1 | silver deposits |
| `coal` | 0..1 | coal (lowland basins beside mountain belts) |
| `oil` | 0..1 | oil (sedimentary basins and continental shelves; may be under shallow sea) |
| `saltpeter` | 0..1 | natural saltpeter (hot deserts); needed for gunpowder |
| `malaria` | 0..1 | malaria-type disease risk (hot, wet, low ground) |
| `tsetse` | 0..1 | tsetse-fly zone, deadly to cattle and horses (hot woodland savanna) |
| `wind_east` | m/s | prevailing wind over the sea, toward the east (negative = toward the west) |
| `wind_north` | m/s | prevailing wind over the sea, toward the north |
| `current_east` | m/s | surface ocean current, toward the east |
| `current_north` | m/s | surface ocean current, toward the north |
| `glacial_land` | bool | sea today but dry land at the height of the last ice age (land bridges) |
| `glacial_ice` | bool | under ice sheets at the height of the last ice age |
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

## Wild species

Index into the first axis of `species_suitability` and `species_native`.

| # | Species | Kind | Uses | Domestication difficulty | Food value |
|---|---|---|---|---|---|
| 0 | wild grain (wheat/barley-like) (`temperate_grain`) | crop | food, fodder | 1.0 | 1.0 |
| 1 | hardy grain (rye/oat-like) (`cold_grain`) | crop | food, fodder | 1.4 | 0.8 |
| 2 | wetland grain (rice-like) (`wet_grain`) | crop | food | 1.2 | 1.2 |
| 3 | dry grain (millet/sorghum-like) (`tropical_grain`) | crop | food, fodder | 1.1 | 0.8 |
| 4 | big grain, slow to tame (maize-like) (`slow_grain`) | crop | food, fodder | 3.0 | 1.3 |
| 5 | pulses (bean/lentil-like) (`legume`) | crop | food | 1.0 | 0.7 |
| 6 | highland tuber (potato-like) (`highland_tuber`) | crop | food | 1.3 | 1.1 |
| 7 | tropical root crop (yam/taro-like) (`tropical_tuber`) | crop | food | 1.2 | 1.0 |
| 8 | fruit and oil trees (olive/date/banana-like) (`tree_crop`) | crop | food, oil | 1.5 | 0.6 |
| 9 | fibre plant (flax/cotton-like) (`fibre`) | crop | fibre | 1.2 | 0.0 |
| 10 | wild cattle (`cattle`) | animal | meat, milk, traction, leather | 1.5 | 1.0 |
| 11 | wild sheep and goats (`sheep_goat`) | animal | meat, milk, wool | 1.0 | 0.8 |
| 12 | wild pig (`pig`) | animal | meat | 1.0 | 0.9 |
| 13 | wild horse (`horse`) | animal | riding, traction, war, milk | 2.0 | 0.5 |
| 14 | wild camel (`camel`) | animal | pack, riding, milk, war | 2.0 | 0.4 |
| 15 | wild highland camelid (llama-like) (`llama`) | animal | pack, wool, meat | 1.5 | 0.4 |
| 16 | wild fowl (chicken-like) (`fowl`) | animal | meat, eggs | 0.8 | 0.4 |

## Climate over time

`worldgen.ClimateHistory(world).at(year)` gives the temperature change and rainfall factor per
cell, sea level, droughts and volcanic cooling for any simulation year (year 0 = about
10,000 BCE). See `worldgen/climate_history.py`.

## Versions

`data_version` changes whenever a field is added, removed or changes meaning, so a
program can check it before trusting saved data.
