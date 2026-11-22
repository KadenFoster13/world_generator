"""Every tunable setting for the world generator lives here.

Change a default below and it applies everywhere: the generator, the viewer's
sliders (which start from these values), and any scripts.

Units: the map is an equirectangular projection of an Earth-sized sphere.
Distances are in km and areas in km2, so a seed looks the same at every map
resolution (800 px wide = about 50 km per pixel, 1600 = 25 km, 2400 = 17 km);
higher resolutions just add finer detail.
"""
from dataclasses import dataclass, field, replace


@dataclass
class WorldConfig:
    # ------------------------------------------------------------------ map
    seed: int = 1
    width: int = 800           # height is always width / 2; see RESOLUTIONS below

    # --------------------------------------------------------- landmasses
    continents: int = 0        # number of main landmasses; 0 = pick randomly per seed
    continents_min: int = 1    # range used when continents == 0
    continents_max: int = 7
    land_fraction: float = 0.30          # share of the globe above sea level (Earth ~0.29)
    coast_roughness: float = 0.38        # small-scale coastline noise (bays, inlets, offshore islands)
    coast_warp: float = 0.18             # large-scale coastline bending (peninsulas, gulfs)
    coast_noise_frequency: float = 2.5   # size of bays/peninsulas: higher = smaller, more of them
    continent_edge_blur_km: float = 400.0  # softens plate-shaped continent outlines
    polar_land_bias: float = 1.5         # 0 = continents anywhere, higher = fewer polar continents
    microcontinents: int = 3             # small continental fragments (like Madagascar or New Zealand)
    hotspots: int = 4                    # volcanic island chains in open ocean (like Hawaii)
    archipelagos: bool = True            # island chains with a few big islands off continental coasts
    archipelago_coast_gap_km: float = 300.0  # open sea kept between archipelago islands and the mainland
    island_warp_km: float = 75.0          # how much archipelago island shapes are bent (0 = plain ovals)
    archipelago_count: int = 5          # how many of those chains (when archipelagos is on)

    # ------------------------------------------------------ plate tectonics
    plates: int = 28                     # crust pieces; continents and plates are built from these
    tectonic_plates: int = 10            # pieces are grouped into this many moving plates
    plate_size_variation: float = 0.35   # 0 = equal-sized pieces, higher = a few big and many small
    plate_boundary_warp: float = 0.30    # how twisty plate boundaries (and so mountain chains) are
    mountain_height: float = 1.2         # uplift where plates collide on land
    mountain_width_km: float = 450.0     # half-width of a mountain chain's crest
    island_arc_height: float = 0.55      # uplift where plates collide at sea (island arcs)
    suture_factor: float = 0.55          # strength of old ranges where a continent's pieces welded together
    rift_depth: float = 0.30             # rift valleys where continents pull apart
    mid_ocean_ridge_height: float = 0.18 # undersea ridges where ocean plates pull apart
    chain_gap_frequency: float = 3.0     # breaks long chains into separate ranges
    ancient_mountains: float = 0.22      # worn-down inland ranges away from plate boundaries
    hill_roughness: float = 0.06         # small-scale hills everywhere on land

    # ---------------------------------------------------------------- water
    lakes: bool = True                   # fill basins in the land with lakes (or salt flats if dry)
    inland_sea_min_area: float = 0.0015  # enclosed seas smaller than this share of the globe become land
    lake_min_depth_m: float = 60.0       # basin must be at least this deep to hold a lake
    lake_min_area_km2: float = 15000.0   # ... and at least this big
    max_lake_area_km2: float = 225000.0  # biggest a single lake can get (the Caspian is ~370,000)
    max_lakes: int = 20                # only the biggest basins keep water; the rest drained long ago
    lake_full_precip_mm: float = 1100.0  # basins this wet fill to the brim; drier ones partly
                                         # (the dry ones get salt flats around a smaller salt lake)

    rivers: bool = True                  # draw rivers on the map (they're always computed)
    river_min_discharge_m3s: float = 600.0  # smallest river shown when zoomed out (m3/s)
    river_detail_min_m3s: float = 75.0   # smallest river shown when fully zoomed in
    river_min_catchment_km2: float = 20000.0  # land that must drain into a point before it's a river
    evaporation_per_degree_mm: float = 55.0 # yearly evaporation per degree C above 0 (drier = fewer rivers)
    erosion: float = 0.7                 # how deeply rivers carve their valleys (0 = none, and more lakes survive)
    valley_width_km: float = 40.0        # how wide carved valleys spread
    wetland_threshold: float = 0.38      # wetness score (0..1) at which land becomes wetland
    delta_min_flow_m3s: float = 5000.0   # rivers this big fan into deltas at the sea (enhanced regions)

    # -------------------------------------------------------------- climate
    equator_temp_c: float = 28.0         # mean annual temperature at sea level on the equator
    pole_temp_c: float = -27.0           # ... and at the poles
    lapse_rate_c_per_km: float = 6.5     # cooling with altitude
    max_elevation_m: float = 6500.0      # height of the tallest peak
    max_depth_m: float = 5500.0          # depth of the deepest ocean
    max_precip_mm: float = 2800.0        # rainfall on a wet equatorial coast
    moisture_reach_km: float = 7000.0    # how far rain-bearing winds reach inland; lower = drier interiors
    monsoon_strength: float = 0.45       # extra rain on subtropical east coasts (share of max_precip_mm)
    monsoon_reach_km: float = 1400.0     # how far inland that monsoon rain gets
    rain_shadow: float = 0.35            # moisture lost per km of climb over mountains
    seasonality: float = 1.0             # scales summer/winter temperature swings (0 = no seasons)
    ocean_currents: float = 1.0          # strength of warm/cold ocean currents (0 = none)
    glacier_summer_max_c: float = 1.0    # land whose warmest month stays below this (and snowy) is glacier

    # ------------------------------------------------- rivers as cell edges
    # Rivers run along the edges between cells, so they can form borders and be crossed.
    # Width and depth follow typical river shapes: width ~ 7 * sqrt(flow), depth ~ 0.27 * flow^0.39.
    ford_max_depth_m: float = 1.0        # fordable if dry-season shallows (riffles) are below this depth
    navigable_min_depth_m: float = 2.0   # navigable by river boats if the average depth is at least this
    rapids_min_slope_m_per_km: float = 1.5  # steeper river stretches have rapids (block boats)

    # --------------------------------------------------------------- viewer
    biome_blend: float = 1.0             # px on screen; softens borders between biomes (0 = hard edges)
    hillshade: float = 0.6               # 0 = flat colours, 1 = strong relief shading

    def copy(self, **changes) -> "WorldConfig":
        return replace(self, **changes)

    @property
    def height(self) -> int:
        return self.width // 2


DEFAULT = WorldConfig()


# ------------------------------------------------------------------ viewer controls
# Map widths offered in the viewer. Roughly: 800 ~2 s per seed, 1600 ~9 s, 2400 ~30 s.
RESOLUTIONS = [800, 1600, 2400]

# On/off switches shown in the viewer: (field name, label, stage).
# Stage "terrain" regenerates the whole world (~2 s); "climate" only redoes
# climate, water and colours (under a second); "display" just redraws.
TOGGLES = [
    ("archipelagos", "Coastal archipelagos", "terrain"),
    ("lakes", "Lakes", "climate"),
    ("rivers", "Rivers", "display"),
]

# Sliders shown in the viewer: (field name, label, min, max, step or None for smooth).
# Empty for now: every value above is the default. To get a slider back, add a line, e.g.
#   ("mountain_height", "Mountain height", 0.0, 2.0, None),       # in TERRAIN_SLIDERS
#   ("land_fraction", "Land fraction", 0.1, 0.7, None),           # in CLIMATE_SLIDERS
TERRAIN_SLIDERS = []
CLIMATE_SLIDERS = []
