"""Interactive world viewer.

Run:  python -m worldgen                (starts at the seed in config.py)
      python -m worldgen --seed 42 --width 1600
      (or `worldgen` once installed, or `python main.py` from the project folder)

Controls
  Prev / Next / Random, or type a seed in the box
  Scroll the mouse wheel (or press + / -) to zoom in and out around the cursor
  Drag the map, or use the arrow keys, to move around; the globe wraps east-west
  Reset view (or press R) zooms back out
  Enhance (or press E) regenerates the area you're zoomed into with 2-8x finer detail;
    Clear (or C) removes it. The enhanced area matches the world map, just sharper.
  Resolution buttons regenerate the same world sharper (800 fast, 2400 slow)
  Layer buttons on the right switch between biomes, elevation, plates, temperature, rain
  Options on the right turn features on/off (archipelagos, lakes, rivers); add more in config.py
  Press S to save the current map as a PNG in renders/
  Press X to export the world's data (and any enhanced region) as .npz files in renders/
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

from .config import CLIMATE_SLIDERS, DEFAULT, RESOLUTIONS, TERRAIN_SLIDERS, TOGGLES
from .export import save_world
from .render import LAYERS, RIVER_COLOUR, RIVER_LAYERS, render, river_segments, save_map
from .region import MAX_REGION_PX, box_for_view, generate_region
from .generator import BIOMES, apply_climate, generate_terrain

# Free keys from matplotlib's default shortcuts that the viewer uses itself.
for key, taken in (("keymap.back", ("left",)), ("keymap.forward", ("right",)),
                   ("keymap.home", ("r",)), ("keymap.save", ("s",)),
                   ("keymap.zoom", ("o",)), ("keymap.pan", ("p",))):
    plt.rcParams[key] = [k for k in plt.rcParams[key] if k not in taken]

SLIDER_ROW_IN = 0.32    # vertical space per slider row, inches
DEBOUNCE_MS = 350
FIG_W = 15.0
MAP_W = 11.4            # inches; the map is always 2:1
MAP_H = MAP_W / 2
MAX_ZOOM = 32
ZOOM_STEP = 1.35
TICK_STEPS = [60, 30, 20, 10, 5, 2, 1, 0.5, 0.25]   # degrees between grid labels
ACTIVE_COLOUR, IDLE_COLOUR = "#9cc3e6", "0.85"


class Viewer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.layer = "Biomes"
        self.offset = 0          # horizontal scroll in map pixels (the map wraps)
        self.view = None         # visible (x0, x1, y0, y1) in display coordinates
        self.drag = None
        self.terrain = None
        self.world = None
        self.image = None
        self.pending = None      # "terrain" or "climate" while a change is waiting
        self.timings = ""
        self.region = None       # enhanced region (a World), drawn over the map
        self.region_image = None
        self._cache = {}         # rendered layers: (id(world), layer) -> image

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
        c.mpl_connect("scroll_event", self._on_scroll)

        self.regenerate("terrain")

    # ------------------------------------------------------------- layout
    def _axes(self, x, y, w, h):
        """Add axes positioned in inches (from the bottom-left corner)."""
        return self.fig.add_axes([x / FIG_W, y / self.fig_h, w / FIG_W, h / self.fig_h])

    def _build_controls(self):
        y, bh = self.controls_y, 0.33
        self.btn_prev = Button(self._axes(0.9, y, 0.8, bh), "< Prev")
        self.seed_box = TextBox(self._axes(2.25, y, 0.95, bh), "Seed ", initial=str(self.cfg.seed))
        self.btn_next = Button(self._axes(3.3, y, 0.8, bh), "Next >")
        self.btn_rand = Button(self._axes(4.2, y, 0.9, bh), "Random")
        self.btn_view = Button(self._axes(5.25, y, 1.1, bh), "Reset view")
        self.btn_prev.on_clicked(lambda _: self.set_seed(self.cfg.seed - 1))
        self.btn_next.on_clicked(lambda _: self.set_seed(self.cfg.seed + 1))
        self.btn_rand.on_clicked(lambda _: self.set_seed(int(np.random.randint(1, 1_000_000))))
        self.btn_view.on_clicked(lambda _: self.reset_view())
        self.seed_box.on_submit(self._on_seed_text)

        self.fig.text(6.5 / FIG_W, (y + 0.1) / self.fig_h, "Resolution", fontsize=10)
        self.res_buttons = {}
        for i, width in enumerate(RESOLUTIONS):
            b = Button(self._axes(7.35 + i * 0.7, y, 0.65, bh), str(width))
            b.on_clicked(lambda _, wd=width: self.set_resolution(wd))
            self.res_buttons[width] = b
        self._highlight_resolution()
        self.btn_enhance = Button(self._axes(9.6, y, 1.0, bh), "Enhance", color=ACTIVE_COLOUR)
        self.btn_clear = Button(self._axes(10.7, y, 0.8, bh), "Clear")
        self.btn_enhance.on_clicked(lambda _: self.enhance())
        self.btn_clear.on_clicked(lambda _: self.clear_region())

        self.sliders = {}
        if not self.slider_h:
            return
        self.btn_reset = Button(self._axes(10.95, y, 1.0, bh), "Reset sliders")
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

    def _highlight_resolution(self):
        for width, b in self.res_buttons.items():
            colour = ACTIVE_COLOUR if width == self.cfg.width else IDLE_COLOUR
            b.color = colour
            b.ax.set_facecolor(colour)

    # ---------------------------------------------------------- generation
    def regenerate(self, stage):
        slow = "  (high resolution, this takes a while)" if self.cfg.width > 1000 and stage == "terrain" else ""
        self.ax.set_title(f"Generating...{slow}", fontsize=11)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.clear_region(redraw=False)  # it belonged to the old world
        t0 = time.perf_counter()
        if stage == "terrain" or self.terrain is None:
            self.terrain = generate_terrain(self.cfg)
        t1 = time.perf_counter()
        self.world = apply_climate(self.terrain, self.cfg)
        t2 = time.perf_counter()
        self.timings = f"terrain {t1 - t0:.1f}s, climate {t2 - t1:.1f}s"
        self.redraw()

    def _rendered(self, world, layer):
        key = (id(world), layer)
        if key not in self._cache:
            if len(self._cache) > 12:
                self._cache.clear()
            self._cache[key] = render(world, layer)
        return self._cache[key]

    def redraw(self):
        img = np.roll(self._rendered(self.world, self.layer), -self.offset, axis=1)
        if self.image is not None and self.image.get_array().shape[:2] != img.shape[:2]:
            self.image.remove()  # resolution changed
            self.image = None
        if self.image is None:
            self.image = self.ax.imshow(img, interpolation="nearest", zorder=1)
            if not hasattr(self, "rivers"):
                self.rivers = LineCollection([], colors=[RIVER_COLOUR], capstyle="round",
                                             joinstyle="round", zorder=3)
                self.ax.add_collection(self.rivers)
            if self.view is None:
                self.view = self._full_view()
        else:
            self.image.set_data(img)
        self._apply_view()
        w = self.world
        zoom = self.zoom
        zoom_text = f"   |   zoom {zoom:.1f}x" if zoom > 1.05 else ""
        if self.region is not None:
            zoom_text += (f"   |   enhanced x{self.region.box.factor}"
                          f" ({self.region.grid.km_per_px:.0f} km/px)")
        self.ax.set_title(
            f"Seed {self.cfg.seed}   |   {w.major_landmasses} major landmasses, {w.islands} islands,"
            f" {w.lakes} lakes   |   {self.cfg.width} px{zoom_text}   |   {self.timings}",
            fontsize=10)
        self.fig.canvas.draw_idle()

    # ---------------------------------------------------------------- view
    @property
    def zoom(self):
        if self.view is None:
            return 1.0
        return self.cfg.width / (self.view[1] - self.view[0])

    def _full_view(self):
        return (-0.5, self.cfg.width - 0.5, -0.5, self.cfg.height - 0.5)

    def reset_view(self):
        self.view = self._full_view()
        self.redraw()

    def _apply_view(self):
        x0, x1, y0, y1 = self.view
        self.ax.set_xlim(x0, x1)
        self.ax.set_ylim(y1, y0)  # rows grow downward
        extent = self._region_extent()
        show_rivers = self.cfg.rivers and self.layer in RIVER_LAYERS
        if show_rivers:
            segs, widths = river_segments(self.world, self.offset, self.zoom, xlim=(x0, x1))
            if extent is not None and len(segs):  # the enhanced region draws its own rivers
                left, right, bottom, top = extent
                mid = segs.mean(axis=1)
                inside = (mid[:, 0] > left) & (mid[:, 0] < right) & (mid[:, 1] > top) & (mid[:, 1] < bottom)
                segs, widths = segs[~inside], widths[~inside]
            self.rivers.set_segments(segs)
            self.rivers.set_linewidths(widths)
        else:
            self.rivers.set_segments([])
        self._draw_region(extent, show_rivers)
        self._update_ticks()

    def _set_view(self, x0, x1, y0, y1):
        """Store a new view, scrolling the wrapped map so the view stays on it."""
        w, h = self.cfg.width, self.cfg.height
        span_x, span_y = x1 - x0, y1 - y0
        # Horizontal: shift the map instead of letting the view run off its edge.
        k = 0
        if x0 < -0.5:
            k = int(np.floor(x0 + 0.5))
        elif x1 > w - 0.5:
            k = int(np.ceil(x1 - (w - 0.5)))
        self.offset = (self.offset + k) % w
        x0 -= k
        x1 = x0 + span_x
        # Vertical: no wrapping over the poles, so clamp.
        y0 = min(max(y0, -0.5), h - 0.5 - span_y)
        self.view = (x0, x1, y0, y0 + span_y)

    def zoom_at(self, factor, cx=None, cy=None):
        """Zoom by `factor` (>1 = closer) keeping the map point under (cx, cy) in place."""
        x0, x1, y0, y1 = self.view
        cx = (x0 + x1) / 2 if cx is None else cx
        cy = (y0 + y1) / 2 if cy is None else cy
        w = self.cfg.width
        new_span = np.clip((x1 - x0) / factor, w / MAX_ZOOM, w)
        f = new_span / (x1 - x0)
        nx0 = cx - (cx - x0) * f
        ny0 = cy - (cy - y0) * f
        self._set_view(nx0, nx0 + new_span, ny0, ny0 + new_span / 2)
        self.redraw()

    def _update_ticks(self):
        w, h = self.cfg.width, self.cfg.height
        x0, x1, y0, y1 = self.view
        span = (x1 - x0) / w * 360
        step = next((s for s in reversed(TICK_STEPS) if span / s <= 7), 60)
        lons = np.arange(-180, 180, step)
        xs = ((lons + 180) / 360 * w - 0.5 - self.offset) % w
        self.ax.set_xticks(xs, [_fmt_deg(l, "E", "W") for l in lons], fontsize=8)
        lats = np.arange(-90, 90 + step / 2, step)
        ys = (90 - lats) / 180 * h - 0.5
        self.ax.set_yticks(ys, [_fmt_deg(l, "N", "S") for l in lats], fontsize=8)
        self.ax.set_xlim(x0, x1)
        self.ax.set_ylim(y1, y0)

    # ------------------------------------------------------------- enhance
    def enhance(self):
        x0, x1, y0, y1 = self.view
        box = box_for_view(self.world, x0 + self.offset, x1 + self.offset, y0, y1)
        if box is None:
            most = MAX_REGION_PX // 2
            self.ax.set_title(f"Zoom in further to enhance (at most {most} map pixels across;"
                              f" now {x1 - x0:.0f})", fontsize=11)
            self.fig.canvas.draw_idle()
            return
        self.ax.set_title(f"Enhancing this area at {box.factor}x detail...", fontsize=11)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        t0 = time.perf_counter()
        self.clear_region(redraw=False)
        self.region = generate_region(self.world, box)
        self.timings = f"enhanced in {time.perf_counter() - t0:.1f}s"
        self.redraw()

    def clear_region(self, redraw=True):
        if self.region is None:
            return
        self.region = None
        if self.region_image is not None:
            self.region_image.remove()
            self.region_image = None
        self.region_rivers.set_segments([])
        if redraw:
            self.redraw()

    def _region_extent(self):
        """Where the enhanced region sits in display coordinates: (left, right, bottom, top)."""
        if self.region is None:
            return None
        box = self.region.box
        w = self.cfg.width
        x0, x1, _, _ = self.view
        left = box.col0 - 0.5 - self.offset
        # Pick the copy of the region (the map wraps) nearest the view.
        while left + box.cols < x0:
            left += w
        while left > x1:
            left -= w
        top = box.row0 - 0.5
        return left, left + box.cols, top + box.rows, top

    def _draw_region(self, extent, show_rivers):
        if not hasattr(self, "region_rivers"):
            self.region_rivers = LineCollection([], colors=[RIVER_COLOUR], capstyle="round",
                                                joinstyle="round", zorder=4)
            self.ax.add_collection(self.region_rivers)
        if extent is None:
            self.region_rivers.set_segments([])
            return
        img = self._rendered(self.region, self.layer)
        if self.region_image is None:
            self.region_image = self.ax.imshow(img, extent=extent, interpolation="nearest", zorder=2)
        else:
            self.region_image.set_data(img)
            self.region_image.set_extent(extent)
        if show_rivers:
            left, _, _, top = extent
            key = (id(self.region), round(self.zoom, 4))
            if getattr(self, "_region_segs_key", None) != key:  # only changes with zoom
                f = self.region.box.factor
                segs, widths = river_segments(self.region, zoom=self.zoom)
                self._region_segs = ((segs + 0.5) / f, widths)
                self._region_segs_key = key
            rel, widths = self._region_segs
            self.region_rivers.set_segments(rel + [left, top])
            self.region_rivers.set_linewidths(widths)
        else:
            self.region_rivers.set_segments([])

    # -------------------------------------------------------------- events
    def set_seed(self, seed):
        self.cfg = self.cfg.copy(seed=int(seed))
        self.seed_box.set_val(str(self.cfg.seed))  # triggers _on_seed_text

    def set_resolution(self, width):
        if width == self.cfg.width:
            return
        scale = width / self.cfg.width
        # Keep looking at the same place on the globe.
        x0, x1, y0, y1 = self.view
        self.view = ((x0 + 0.5) * scale - 0.5, (x1 + 0.5) * scale - 0.5,
                     (y0 + 0.5) * scale - 0.5, (y1 + 0.5) * scale - 0.5)
        self.offset = int(round(self.offset * scale)) % width
        self.cfg = self.cfg.copy(width=width)
        self._highlight_resolution()
        self.timer.stop()
        self.pending = None
        self.regenerate("terrain")

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

    def _on_scroll(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            return
        factor = ZOOM_STEP if event.button == "up" else 1 / ZOOM_STEP
        self.zoom_at(factor, event.xdata, event.ydata)

    def _on_press(self, event):
        if event.inaxes is self.ax and event.button == 1:
            self.drag = (event.x, event.y, self.offset, self.view)

    def _on_release(self, event):
        self.drag = None

    def _on_motion(self, event):
        if self.drag is None or event.x is None:
            return
        x_start, y_start, off0, view0 = self.drag
        x0, x1, y0, y1 = view0
        px_per_cell = self.ax.get_window_extent().width / (x1 - x0)
        dx = (event.x - x_start) / px_per_cell
        dy = (event.y - y_start) / px_per_cell   # screen y points up, rows point down
        self.offset = off0
        self._set_view(x0 - dx, x1 - dx, y0 + dy, y1 + dy)
        self.redraw()

    def _on_key(self, event):
        x0, x1, y0, y1 = self.view
        step = (x1 - x0) / 6
        if event.key in ("left", "right"):
            d = -step if event.key == "left" else step
            self._set_view(x0 + d, x1 + d, y0, y1)
            self.redraw()
        elif event.key in ("up", "down"):
            d = -step if event.key == "up" else step
            self._set_view(x0, x1, y0 + d, y1 + d)
            self.redraw()
        elif event.key in ("+", "="):
            self.zoom_at(ZOOM_STEP)
        elif event.key == "-":
            self.zoom_at(1 / ZOOM_STEP)
        elif event.key in ("r", "R"):
            self.reset_view()
        elif event.key in ("e", "E"):
            self.enhance()
        elif event.key in ("c", "C"):
            self.clear_region()
        elif event.key in ("s", "S"):
            os.makedirs("renders", exist_ok=True)
            path = os.path.join("renders", f"world_{self.cfg.seed}_{self.cfg.width}_{self.layer.lower()}.png")
            save_map(self.world, self.layer, path, self.offset, rivers=self.cfg.rivers)
            print(f"saved {path}")
        elif event.key in ("x", "X"):
            os.makedirs("renders", exist_ok=True)
            path = os.path.join("renders", f"world_{self.cfg.seed}_{self.cfg.width}.npz")
            save_world(self.world, path)
            print(f"saved {path}")
            if self.region is not None:
                b = self.region.box
                path = os.path.join("renders", f"region_{self.cfg.seed}_{self.cfg.width}"
                                               f"_r{b.row0}_c{b.col0}_x{b.factor}.npz")
                save_world(self.region, path)
                print(f"saved {path}")

    def _describe(self, x, y):
        if self.world is None:
            return ""
        w = self.world
        col = (int(round(x)) + self.offset) % self.cfg.width
        row = int(np.clip(round(y), 0, self.cfg.height - 1))
        extent = self._region_extent()
        if extent is not None:
            left, right, bottom, top = extent
            if left <= x < right and top <= y < bottom:
                f = self.region.box.factor
                w = self.region
                h_r, w_r = w.elevation_m.shape
                col = int(np.clip((x - left) * f, 0, w_r - 1))
                row = int(np.clip((y - top) * f, 0, h_r - 1))
        lat, lon = w.terrain.lat[row, col], w.terrain.lon[row, col]
        q = w.discharge_m3s[row, col]
        on_river = w.stream[row, col] and q >= self.cfg.river_detail_min_m3s
        return (f"{_fmt_deg(lat, 'N', 'S', 1)} {_fmt_deg(lon, 'E', 'W', 1)}  "
                f"{BIOMES[w.biome[row, col]][0]}  {w.elevation_m[row, col]:.0f} m  "
                f"{w.temperature_c[row, col]:.0f} C  {w.precip_mm[row, col]:.0f} mm/yr  "
                f"plate {w.terrain.plate[row, col]}"
                + (f"  river {q:,.0f} m3/s" if on_river else ""))


def _fmt_deg(value, pos, neg, decimals=None):
    if decimals is None:
        decimals = 0 if float(value).is_integer() else 2 if abs(value * 4) % 1 else 1
    text = f"{abs(value):.{decimals}f}".rstrip("0").rstrip(".") if decimals else f"{abs(value):.0f}"
    return text + (pos if value > 0 else neg if value < 0 else "")


def main():
    parser = argparse.ArgumentParser(description="Interactive world generator")
    parser.add_argument("--seed", type=int, default=DEFAULT.seed)
    parser.add_argument("--width", type=int, default=DEFAULT.width,
                        help=f"map width in px, e.g. {RESOLUTIONS} (height = width/2)")
    args = parser.parse_args()
    Viewer(DEFAULT.copy(seed=args.seed, width=args.width))
    plt.show()


if __name__ == "__main__":
    main()
