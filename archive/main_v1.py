import numpy as np
import matplotlib.pyplot as plt
from opensimplex import OpenSimplex
from matplotlib.widgets import Slider
from scipy.ndimage import gaussian_filter
from matplotlib.patches import Patch
from matplotlib.widgets import RangeSlider


WIDTH = 800
HEIGHT = 400
SCALE = 40.0  # larger = smoother, more zoomed-out terrain

BIOME_COLOURS = {
    'Desert': [0.93, 0.79, 0.45],
    'Plains': [0.55, 0.75, 0.35],
    'Swamp': [0.35, 0.45, 0.30],
    'Savanna': [0.80, 0.70, 0.35],
    'Forest': [0.20, 0.50, 0.20],
    'Rainforest': [0.10, 0.35, 0.15],
    'Bare rock': [0.55, 0.50, 0.45],
    'Tundra': [0.65, 0.65, 0.60],
    'Snow': [0.95, 0.95, 0.97],
    'Ocean': [0.2, 0.4, 0.8],
}
simplex = OpenSimplex(seed=1)








#THIS SECTION CONTAINS ALL MAP GENERATION FUNCTIONS
def generate(sim, scale: float) -> np.ndarray: #DEFAULT NOISE GENERATION FUNCTION
    x_coords = np.arange(WIDTH) / scale
    y_coords = np.arange(HEIGHT) / scale
    return sim.noise2array(x_coords, y_coords)

def generate_terrain(sim, base_scale: float, octaves: int = 4, power: float = 1.5) -> np.ndarray: #TERRAIN SPECIFIC GENERATION FUNCTION
    x_coords = np.arange(WIDTH)
    y_coords = np.arange(HEIGHT)
    
    result = np.zeros((HEIGHT, WIDTH))
    amplitude = 1.0
    frequency = 1.0
    total_amplitude = 0.0
    
    for _ in range(octaves):
        layer = sim.noise2array(x_coords / (base_scale / frequency), y_coords / (base_scale / frequency))
        result += layer * amplitude
        total_amplitude += amplitude
        amplitude *= amplitude_slider.val
        frequency *= 2.0
    
    result = result / total_amplitude
    return np.sign(result) * np.abs(result) ** power

def add_archipelagos(elevation, sim, bump_field, sea_level, scale=8.0, strength=0.3):
    fine_detail = generate_terrain(sim, scale, octaves=2, power=1.0)

    island_threshold = 0.6 - bump_field * 0.5
    island_threshold = np.clip(island_threshold, 0.05, 0.9)

    is_ocean = elevation < sea_level
    becomes_island = is_ocean & (fine_detail > island_threshold)

    result = elevation.copy()
    result[becomes_island] = sea_level + strength
    return result

def generate_world(sim, warp_sim, base_scale, octaves, power, num_continents, core_radius, transition_width, seed):
    rng = np.random.default_rng(seed)
    centers_x = rng.uniform(0, WIDTH, num_continents)
    centers_y = rng.uniform(0, HEIGHT, num_continents)

    y_grid, x_grid = np.mgrid[0:HEIGHT, 0:WIDTH]
    min_dist = np.full((HEIGHT, WIDTH), np.inf)
    for cx, cy in zip(centers_x, centers_y):
        dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
        min_dist = np.minimum(min_dist, dist)

    warp = warp_sim.noise2array(np.arange(WIDTH) / 25.0, np.arange(HEIGHT) / 25.0)
    gate = compute_gate(min_dist, warp, core_radius, transition_width)

    terrain = generate_terrain(sim, base_scale, octaves, power=1.0)
    combined = terrain * gate - (1 - gate) * 0.8

    heightmap = np.sign(combined) * np.abs(combined) ** power
    return heightmap, gate

def continent_bump_field(centers_x, centers_y, spread: float, warp_sim, warp_strength: float = 15.0) -> np.ndarray:
    y_grid, x_grid = np.mgrid[0:HEIGHT, 0:WIDTH]
    
    warp_x = warp_sim.noise2array(np.arange(WIDTH) / 20.0, np.arange(HEIGHT) / 20.0) * warp_strength
    
    field = np.zeros((HEIGHT, WIDTH))
    for cx, cy in zip(centers_x, centers_y):
        warped_x_grid = x_grid + warp_x
        dist_sq = (warped_x_grid - cx) ** 2 + (y_grid - cy) ** 2
        field += np.exp(-dist_sq / (2 * spread ** 2))
    
    return field
    
def compute_gate(min_dist, warp, core_radius, transition_width):
    warped_dist = np.maximum(min_dist + warp * transition_width * 0.6, 0.0)
    gate = 1.0 - (warped_dist - core_radius) / transition_width
    return np.clip(gate, 0.0, 1.0)








#THIS SECTION COLOURS THE MAP AND ASSIGNS BIOMES
def elevation_to_colour(elevation: np.ndarray, moisture: np.ndarray, sea_level: float) -> np.ndarray: #NEEDS RENAME LATER
    height, width = elevation.shape
    image = np.zeros((height, width, 3))  # RGB image

    max_elevation = elevation.max()
    land_range = max_elevation - sea_level # creates cutoff ranges for low, mid and high moisture and elevation to be used in determining biome
    low_cutoff = sea_level + land_range * elevation_bands_slider.val[0] # percent value from range slider being directly inputed, however, map appears ok
    mid_cutoff = sea_level + land_range * elevation_bands_slider.val[1]
    dry_cutoff = moisture_bands_slider.val[0]
    wet_cutoff = moisture_bands_slider.val[1]

    ocean = elevation < sea_level # Assigns Biomes an elevation and moisture level
    desert = (elevation >= sea_level) & (elevation < low_cutoff) & (moisture <= dry_cutoff)
    plains = (elevation >= sea_level) & (elevation < low_cutoff) & (moisture > dry_cutoff) & (moisture < wet_cutoff)
    swamp = (elevation >= sea_level) & (elevation < low_cutoff) & (moisture >= wet_cutoff)
    savanna = (elevation >= low_cutoff) & (elevation < mid_cutoff) & (moisture <= dry_cutoff)
    forest = (elevation >= low_cutoff) & (elevation < mid_cutoff) & (moisture > dry_cutoff) & (moisture < wet_cutoff)
    rainforest = (elevation >= low_cutoff) & (elevation < mid_cutoff) & (moisture >= wet_cutoff)
    bare_rock = (elevation >= mid_cutoff) & (moisture <= dry_cutoff)
    tundra = (elevation >= mid_cutoff) & (moisture > dry_cutoff) & (moisture < wet_cutoff)
    snow = (elevation >= mid_cutoff) & (moisture >= wet_cutoff)

    #Colours the map according to biome location and colours
    image[ocean] = BIOME_COLOURS['Ocean']                # blue
    image[plains] = BIOME_COLOURS['Plains']              # grassy green
    image[desert] = BIOME_COLOURS['Desert']              # sandy tan
    image[swamp] = BIOME_COLOURS['Swamp']	            # murky olive-green
    image[savanna] = BIOME_COLOURS['Savanna']            # dry gold
    image[forest] = BIOME_COLOURS['Forest']	            # deep green
    image[rainforest] = BIOME_COLOURS['Rainforest']	    # dark jungle green
    image[bare_rock] = BIOME_COLOURS['Bare rock']	    # dull gray-brown
    image[tundra] = BIOME_COLOURS['Tundra']	            # pale gray-beige
    image[snow]	=BIOME_COLOURS['Snow']	                # near-white

    return image








#THIS SECTION CREATES AND ORGANIZES THE SLIDERS AT THE BOTTOM OF THE PROGRAM
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.45)  # leaves room for the slider

LEFT_COL_X = 0.10
RIGHT_COL_X = 0.60
COL_WIDTH = 0.30
SLIDER_HEIGHT = 0.03
ROW_SPACING = 0.05


def slider_position(col_x: float, row: int) -> tuple:
    bottom = 0.35 - row * ROW_SPACING
    return (col_x, bottom, COL_WIDTH, SLIDER_HEIGHT)

#Slider adjusters, each one sets the location and size of a specific slider, no 2 should be identical
scale_ax = plt.axes(slider_position(LEFT_COL_X, 0))
seed_ax = plt.axes(slider_position(LEFT_COL_X, 1))
sea_ax = plt.axes(slider_position(LEFT_COL_X, 2))
octaves_ax = plt.axes(slider_position(LEFT_COL_X, 3))
elevation_bands_ax = plt.axes(slider_position(RIGHT_COL_X, 0))
moisture_bands_ax = plt.axes(slider_position(RIGHT_COL_X, 1))
continents_ax = plt.axes(slider_position(RIGHT_COL_X, 2))
core_ax = plt.axes(slider_position(RIGHT_COL_X, 3))
transition_ax = plt.axes(slider_position(RIGHT_COL_X, 4))

#Creates the sliders and assigns their min, max, and starting values
scale_slider = Slider(scale_ax, 'Scale', valmin=5, valmax=100, valinit=SCALE)
seed_slider = Slider(seed_ax, 'Seed', valmin=1, valmax=100, valinit=1, valstep=16)
sea_slider = Slider(sea_ax, 'Sea Level', valmin=-0.5, valmax=0.5, valinit=0)
amplitude_slider = Slider(octaves_ax, 'Amplitude', valmin=0.1, valmax=1, valinit=0.85)
elevation_bands_slider = RangeSlider(elevation_bands_ax, 'Elevation bands', valmin=0.0, valmax=1.0, valinit=(0.9, 0.99))
moisture_bands_slider = RangeSlider(moisture_bands_ax, 'Moisture bands', valmin=-1, valmax=1.0, valinit=(-0.99, 0.99))
continents_slider = Slider(continents_ax, 'Continents', valmin=1, valmax=10, valinit=3, valstep=1)
core_radius_slider = Slider(core_ax, 'Core radius', valmin=1, valmax=300, valinit=95)
transition_width_slider = Slider(transition_ax, "Transition Width", valmin=1, valmax=300, valinit=35)







#sim, warp_sim, base_scale, octaves, power, num_continents, core_radius, transition_width, seed

#THIS SECTION DRAWS AND REDRAWS THE MAP AS NECCESARY AND ADDS THE BIOME LEGEND
moisture_simplex = OpenSimplex(seed=int(seed_slider.val) + 1000)
archipelago_simplex = OpenSimplex(seed=int(seed_slider.val) + 2000)
warp_simplex = OpenSimplex(seed=int(seed_slider.val) + 3000)

moisture_map =  generate(moisture_simplex,SCALE*2)
heightmap, gate = generate_world(simplex, warp_simplex, SCALE, 4, 0.2, int(continents_slider.val), core_radius_slider.val, transition_width_slider.val, seed=int(seed_slider.val))
heightmap = gaussian_filter(heightmap, sigma=2.0)
#heightmap = add_archipelagos(heightmap, archipelago_simplex, gate, sea_slider.val) #elevation, sim, bump_field, sea_level
image = ax.imshow(elevation_to_colour(heightmap, moisture_map, sea_level=0.0))
ax.set_title('Map')

legend_patches = [Patch(color=color, label=name) for name, color in BIOME_COLOURS.items()]
ax.legend(handles=legend_patches)

def redraw():
    new_heightmap, new_gate = generate_world(simplex, warp_simplex, scale_slider.val, 4, 0.2, int(continents_slider.val), core_radius_slider.val, transition_width_slider.val, seed=int(seed_slider.val))
    new_heightmap = gaussian_filter(new_heightmap, sigma=2.0)
    #new_heightmap = add_archipelagos(new_heightmap, archipelago_simplex, new_gate, sea_slider.val)
    new_moisture_map = generate(moisture_simplex, scale_slider.val*2)
    image.set_data(elevation_to_colour(new_heightmap, new_moisture_map, sea_level=sea_slider.val))
    fig.canvas.draw_idle()

def on_default_change(value): # change method called when only neccesary command is redraw()
    redraw()

def on_seed_change(value): # change method for seed slider
    global simplex
    global moisture_simplex
    simplex = OpenSimplex(seed=int(value))
    moisture_simplex = OpenSimplex(seed=int(value) + 1000)
    redraw()

scale_slider.on_changed(on_default_change)
seed_slider.on_changed(on_seed_change)
sea_slider.on_changed(on_default_change)
amplitude_slider.on_changed(on_default_change)
elevation_bands_slider.on_changed(on_default_change)
moisture_bands_slider.on_changed(on_default_change)
continents_slider.on_changed(on_default_change)
core_radius_slider.on_changed(on_default_change)
transition_width_slider.on_changed(on_default_change)

plt.show()