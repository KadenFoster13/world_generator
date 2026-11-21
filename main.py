"""Interactive world viewer.

Run:  python main.py            (starts at the seed in config.py)
      python main.py --seed 42

Controls
  Prev / Next / Random, or type a seed in the box
  Drag the map left/right, or use the arrow keys, to scroll around the globe (it wraps)
  Layer buttons on the right switch between biomes, elevation, plates, temperature, rain
  Options on the right turn features on/off (archipelagos, lakes, rivers); add more in config.py
  Press S to save the current map as a PNG in renders/
  Hover over the map to read the values under the cursor (bottom-right of the window)
"""
import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider, TextBox

from config import CLIMATE_SLIDERS, DEFAULT, TERRAIN_SLIDERS, TOGGLES
from render import LAYERS, RIVER_COLOUR, RIVER_LAYERS, render, river_segments, save_map
from worldgen import BIOMES, apply_climate, generate_terrain

# Free the arrow keys from matplotlib's default back/forward shortcuts.
for key in ("keymap.back", "keymap.forward"):
    plt.rcParams[key] = [k for k in plt.rcParams[key] if k not in ("left", "right")]

SLIDER_ROW_IN = 0.32    # vertical space per slider row, inches
DEBOUNCE_MS = 350
FIG_W = 15.0
MAP_W = 11.4            # inches; the map is always 2:1
MAP_H = MAP_W / 2


class Viewer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.layer = "Biomes"
        self.offset = 0          # horizontal scroll in pixels
        self.drag = None
        self.terrain = None
        self.world = None
        self.image = None
        self.pending = None      # "terrain" or "climate" while a change is waiting
        self.timings = ""

        # Figure height grows only if sliders are configured.
        rows = max(len(TERRAIN_SLIDERS), len(CLIMATE_SLIDERS))
        self.slider_h = rows * SLIDER_ROW_IN + 0.35 if rows else 0.0
        self.controls_y = 0.15 + self.slider_h            # inches from the bottom
        self.map_y = self.controls_y + 0.85
        self.fig_h = self.map_y + MAP_H + 0.4

        self.fig = plt.figure(figsize=(FIG_W, self.fig_h))
        self.fig.canvas.manager.set_window_title("World generator")
        self.ax = self._axes(0.4, self.map_y, MAP_W, MAP_H)
        self.ax.format_coord = self._describe

        self._build_controls()
        self._build_side_panel()

        self.timer = self.fig.canvas.new_timer(interval=DEBOUNCE_MS)
        self.timer.single_shot = True
        self.timer.add_callback(self._run_pending)

        c = self.fig.canvas
        c.mpl_connect("button_press_event", self._on_press)
        c.mpl_connect("button_release_event", self._on_release)
        c.mpl_connect("motion_notify_event", self._on_motion)
        c.mpl_connect("key_press_event", self._on_key)

        self.regenerate("terrain")

    # ------------------------------------------------------------- layout
    def _axes(self, x, y, w, h):
        """Add axes positioned in inches (from the bottom-left corner)."""
        return self.fig.add_axes([x / FIG_W, y / self.fig_h, w / FIG_W, h / self.fig_h])

    def _build_controls(self):
        y, bh = self.controls_y, 0.33
        self.btn_prev = Button(self._axes(0.9, y, 0.9, bh), "< Prev")
        self.seed_box = TextBox(self._axes(2.4, y, 1.2, bh), "Seed ", initial=str(self.cfg.seed))
        self.btn_next = Button(self._axes(3.75, y, 0.9, bh), "Next >")
        self.btn_rand = Button(self._axes(4.8, y, 1.0, bh), "Random")
        self.btn_prev.on_clicked(lambda _: self.set_seed(self.cfg.seed - 1))
        self.btn_next.on_clicked(lambda _: self.set_seed(self.cfg.seed + 1))
        self.btn_rand.on_clicked(lambda _: self.set_seed(int(np.random.randint(1, 1_000_000))))
        self.seed_box.on_submit(self._on_seed_text)

        self.sliders = {}
        if not self.slider_h:
            return
        self.btn_reset = Button(self._axes(5.95, y, 1.5, bh), "Reset sliders")
        self.btn_reset.on_clicked(lambda _: self._reset_sliders())
        top = self.controls_y - 0.3
        for col_x, specs, stage in ((2.1, TERRAIN_SLIDERS, "terrain"), (8.3, CLIMATE_SLIDERS, "climate")):
            if not specs:
                continue
            self.fig.text((col_x - 1.7) / FIG_W, top / self.fig_h,
                          "Terrain (regenerates)" if stage == "terrain" else "Climate & display (quick)",
                          fontsize=9, weight="bold")
            for row, (name, label, lo, hi, step) in enumerate(specs):
                ax = self._axes(col_x, top - 0.3 - row * SLIDER_ROW_IN, 3.3, 0.2)
                s = Slider(ax, label, lo, hi, valinit=getattr(self.cfg, name), valstep=step)
                s.label.set_fontsize(8)
                s.valtext.set_fontsize(8)
                s.on_changed(lambda v, n=name, st=stage: self._on_slider(n, v, st))
                self.sliders[name] = s

    def _build_side_panel(self):
        x = 12.2
        top = self.map_y + MAP_H
        legend_ax = self._axes(x, top - 2.9, 2.7, 2.9)
        legend_ax.set_axis_off()
        handles = [Patch(color=c, label=n) for n, c in BIOMES]
        handles.append(Line2D([], [], color=RIVER_COLOUR, linewidth=2, label="River"))
        legend_ax.legend(handles=handles,
                         loc="upper left", fontsize=8, frameon=False, labelspacing=0.3)

        y = top - 3.25
        layer_h = 0.3 * len(LAYERS)
        radio_ax = self._axes(x, y - layer_h, 2.3, layer_h)
        radio_ax.set_title("Layer", fontsize=11, loc="left", weight="bold")
        self.radio = RadioButtons(radio_ax, LAYERS, active=0,
                                  label_props={"fontsize": [11]}, radio_props={"s": [110]})
        self.radio.on_clicked(self._on_layer)

        if TOGGLES:
            y -= layer_h + 0.45
            opt_h = 0.42 * len(TOGGLES)
            opt_ax = self._axes(x, y - opt_h, 2.6, opt_h)
            opt_ax.set_title("Options", fontsize=11, loc="left", weight="bold")
            self.toggle_stage = {label: (name, stage) for name, label, stage in TOGGLES}
            self.checks = CheckButtons(opt_ax, [label for _, label, _ in TOGGLES],
                                       [bool(getattr(self.cfg, name)) for name, _, _ in TOGGLES],
                                       label_props={"fontsize": [12]},
                                       frame_props={"s": [260]}, check_props={"s": [260]})
            self.checks.on_clicked(self._on_toggle)

    # ---------------------------------------------------------- generation
    def regenerate(self, stage):
        self.ax.set_title("Generating...", fontsize=11)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        t0 = time.perf_counter()
        if stage == "terrain" or self.terrain is None:
            self.terrain = generate_terrain(self.cfg)
        t1 = time.perf_counter()
        self.world = apply_climate(self.terrain, self.cfg)
        t2 = time.perf_counter()
        self.timings = f"terrain {t1 - t0:.1f}s, climate {t2 - t1:.1f}s"
        self.redraw()

    def redraw(self):
        img = np.roll(render(self.world, self.layer), -self.offset, axis=1)
        if self.image is None:
            self.image = self.ax.imshow(img, interpolation="nearest")
            self.rivers = LineCollection([], colors=[RIVER_COLOUR], capstyle="round",
                                         joinstyle="round", zorder=3)
            self.ax.add_collection(self.rivers)
            self.ax.set_xlim(-0.5, self.cfg.width - 0.5)
            self.ax.set_ylim(self.cfg.height - 0.5, -0.5)
            self.ax.set_yticks(np.linspace(0, self.cfg.height, 7) - 0.5,
                               ["90N", "60N", "30N", "0", "30S", "60S", "90S"], fontsize=8)
        else:
            self.image.set_data(img)
        if self.cfg.rivers and self.layer in RIVER_LAYERS:
            segs, widths = river_segments(self.world, self.offset)
            self.rivers.set_segments(segs)
            self.rivers.set_linewidths(widths)
        else:
            self.rivers.set_segments([])
        self._update_lon_ticks()
        w = self.world
        self.ax.set_title(
            f"Seed {self.cfg.seed}   |   {w.major_landmasses} major landmasses, {w.islands} islands,"
            f" {w.lakes} lakes"
            f"   |   aimed for {w.terrain.target_continents} continents   |   {self.timings}",
            fontsize=10)
        self.fig.canvas.draw_idle()

    def _update_lon_ticks(self):
        w = self.cfg.width
        lons = np.arange(-180, 180, 60)
        xs = ((lons + 180) / 360 * w - self.offset) % w - 0.5
        labels = [f"{abs(l)}{'E' if l > 0 else 'W' if l < 0 else ''}" for l in lons]
        self.ax.set_xticks(xs, labels, fontsize=8)

    # -------------------------------------------------------------- events
    def set_seed(self, seed):
        self.cfg = self.cfg.copy(seed=int(seed))
        self.seed_box.set_val(str(self.cfg.seed))  # triggers _on_seed_text

    def _on_seed_text(self, text):
        try:
            seed = int(text)
        except ValueError:
            return
        if self.terrain is not None and seed == self.terrain.cfg.seed and self.pending is None:
            return
        self.cfg = self.cfg.copy(seed=seed)
        self.timer.stop()
        self.pending = None
        self.regenerate("terrain")

    def _on_toggle(self, label):
        name, stage = self.toggle_stage[label]
        self.cfg = self.cfg.copy(**{name: not getattr(self.cfg, name)})
        if stage == "display":  # nothing to regenerate
            self.redraw()
            return
        if self.pending != "terrain":
            self.pending = stage
        self.timer.stop()
        self.timer.start()

    def _on_slider(self, name, value, stage):
        if isinstance(getattr(DEFAULT, name), int):
            value = int(round(value))
        self.cfg = self.cfg.copy(**{name: value})
        # Wait until the slider stops moving, then do the cheapest update that covers all changes.
        if self.pending != "terrain":
            self.pending = stage
        self.timer.stop()
        self.timer.start()

    def _run_pending(self):
        stage, self.pending = self.pending, None
        if stage:
            self.regenerate(stage)

    def _reset_sliders(self):
        for name, s in self.sliders.items():
            s.set_val(getattr(DEFAULT, name))

    def _on_layer(self, label):
        self.layer = label
        self.redraw()

    def _on_press(self, event):
        if event.inaxes is self.ax and event.button == 1:
            self.drag = (event.x, self.offset)

    def _on_release(self, event):
        self.drag = None

    def _on_motion(self, event):
        if self.drag is None or event.x is None:
            return
        x0, off0 = self.drag
        bbox = self.ax.get_window_extent()
        px_per_cell = bbox.width / self.cfg.width
        new = int(round(off0 - (event.x - x0) / px_per_cell)) % self.cfg.width
        if new != self.offset:
            self.offset = new
            self.redraw()

    def _on_key(self, event):
        step = self.cfg.width // 12
        if event.key == "left":
            self.offset = (self.offset - step) % self.cfg.width
            self.redraw()
        elif event.key == "right":
            self.offset = (self.offset + step) % self.cfg.width
            self.redraw()
        elif event.key in ("s", "S"):
            os.makedirs("renders", exist_ok=True)
            path = os.path.join("renders", f"world_{self.cfg.seed}_{self.layer.lower()}.png")
            save_map(self.world, self.layer, path, self.offset, rivers=self.cfg.rivers)
            print(f"saved {path}")

    def _describe(self, x, y):
        if self.world is None:
            return ""
        w = self.world
        col = (int(round(x)) + self.offset) % self.cfg.width
        row = int(np.clip(round(y), 0, self.cfg.height - 1))
        lat, lon = w.terrain.lat[row, col], w.terrain.lon[row, col]
        return (f"{abs(lat):.0f}{'N' if lat >= 0 else 'S'} {abs(lon):.0f}{'E' if lon >= 0 else 'W'}  "
                f"{BIOMES[w.biome[row, col]][0]}  {w.elevation_m[row, col]:.0f} m  "
                f"{w.temperature_c[row, col]:.0f} C  {w.precip_mm[row, col]:.0f} mm/yr  "
                f"plate {w.terrain.plate[row, col]}"
                + (f"  river {w.discharge_m3s[row, col]:,.0f} m3/s" if w.river[row, col] else ""))


def main():
    parser = argparse.ArgumentParser(description="Interactive world generator")
    parser.add_argument("--seed", type=int, default=DEFAULT.seed)
    parser.add_argument("--width", type=int, default=DEFAULT.width, help="map width in px (height = width/2)")
    args = parser.parse_args()
    Viewer(DEFAULT.copy(seed=args.seed, width=args.width))
    plt.show()


if __name__ == "__main__":
    main()
