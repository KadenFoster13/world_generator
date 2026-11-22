"""Interactive world viewer.

Run:  python -m worldgen                (starts at the seed in config.py)
      python -m worldgen --seed 42 --width 1600
      (or `worldgen` once installed, or `python main.py` from the project folder)

Controls
  Prev / Next / Random, or type a seed in the box
  Scroll the mouse wheel (or press + / -) to zoom in and out around the cursor
  Drag the map, or use the arrow keys, to move around; the globe wraps east-west
  Reset view (or press R) zooms back out
  Globe (or press O) switches between the flat map and a globe: drag to spin it, scroll to zoom
  Enhance (or press E) regenerates the area you're zoomed into with 2-8x finer detail;
    Clear (or C) removes it. The enhanced area matches the world map, just sharper.
  Auto (or A) turns on auto-Enhance: once you stop moving, a zoomed-in view is enhanced at
    2x by itself (recent areas are remembered, so going back is instant). Remembered.
  Resolution buttons regenerate the same world sharper (800 fast, 2400 slow)
  Layer buttons on the right switch between biomes, elevation, plates, temperature, rain
  Options on the right turn features on/off (archipelagos, lakes, rivers); add more in config.py
  Favourite (or F) stars the current seed; Favourites > (or G) jumps through starred seeds
    (kept in ~/.worldgen_favourites.json)
  Dark / Light (or D) switches the window between dark and light mode (remembered)
  Press S to save the current map as a PNG in renders/
  Press X to export the world's data (and any enhanced region) as .npz files in renders/
  Hover over the map to read the values under the cursor (bottom-right of the window)
"""
import argparse
import json
import os
import time
from dataclasses import fields

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider, TextBox

from .config import CLIMATE_SLIDERS, DARK_MODE, DEFAULT, RESOLUTIONS, TERRAIN_SLIDERS, TOGGLES
from .export import save_world
from .render import (GLOBE_BACKGROUND, LAYERS, RIVER_COLOUR, RIVER_LAYERS, globe_rivers, globe_view,
                     render, river_segments, save_map)
from .region import MAX_REGION_PX, RegionBox, box_for_view, generate_region
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
FAVOURITES_FILE = os.path.join(os.path.expanduser("~"), ".worldgen_favourites.json")
PREFERENCES_FILE = os.path.join(os.path.expanduser("~"), ".worldgen_viewer.json")
# Window colours: background, text, buttons, buttons under the mouse, highlighted buttons, lines.
THEMES = {
    "light": {"bg": "white", "fg": "black", "button": "0.85", "hover": "0.95",
              "active": "#9cc3e6", "edge": "0.25"},
    "dark": {"bg": "#1d1f23", "fg": "#e4e6ea", "button": "#363a41", "hover": "#474c55",
             "active": "#2e6596", "edge": "#9aa0a8"},
}
SMOOTH_FROM_ZOOM = 3     # blend map pixels when zoomed in at least this far
GLOBE_PX = 600           # height of the globe picture (it's twice as wide, like the map area)
GLOBE_MAX_ZOOM = 8
FRAME_MS = 8             # gather mouse moves for this long, then draw the latest
SETTLE_MS = 220          # after the last drag or scroll step, redraw in full (rivers, auto-Enhance)
AUTO_ENHANCE_FACTOR = 2  # auto-Enhance detail (quick: ~0.1-0.5 s)
AUTO_ENHANCE_MAX_CELLS = 260   # ... only for views at most this many map cells across
AUTO_ENHANCE_MARGIN = 0.25     # ... enhancing a bit beyond the view, so small moves don't redo it
AUTO_ENHANCE_CACHE = 8   # enhanced areas remembered


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
        self.globe = False       # showing the globe instead of the flat map
        self.globe_lon, self.globe_lat, self.globe_zoom = 0.0, 20.0, 1.0
        self.globe_image = None
        self.region_image = None
        self._cache = {}         # rendered layers: (id(world), layer) -> image
        self.dark = bool(_load_preferences().get("dark", DARK_MODE))
        self.auto_enhance = bool(_load_preferences().get("auto_enhance", False))
        self._regions = []                # recently enhanced areas: (world id, region)

        # Figure height grows only if sliders are configured.
        rows = max(len(TERRAIN_SLIDERS), len(CLIMATE_SLIDERS))
        self.slider_h = rows * SLIDER_ROW_IN + 0.35 if rows else 0.0
        self.controls_y = 0.15 + self.slider_h            # inches from the bottom
        self.map_y = self.controls_y + 0.85
        self.fig_h = self.map_y + MAP_H + 0.4

        self.font_scale = 1.0    # grows and shrinks with the window
        self.fig = plt.figure(figsize=(FIG_W, self.fig_h))
        self.fig.canvas.manager.set_window_title("World generator")
        self.ax = self._axes(0.4, self.map_y, MAP_W, MAP_H)
        self.ax.format_coord = self._describe

        self._build_controls()
        self._build_side_panel()

        self.timer = self.fig.canvas.new_timer(interval=DEBOUNCE_MS)
        self.timer.single_shot = True
        self.timer.add_callback(self._run_pending)
        # While dragging or scrolling: rivers hidden, only the map redrawn (fast); in full
        # once the view settles.
        self.moving = False
        self._background = None          # the window without the map, for fast redraws
        self.settle_timer = self.fig.canvas.new_timer(interval=SETTLE_MS)
        self.settle_timer.single_shot = True
        self.settle_timer.add_callback(self._settle)
        # Mouse moves and wheel steps arrive faster than frames can be drawn: remember only
        # where the mouse is now (and the total zoom asked for), and draw that once.
        self._drag_to = None
        self._zoom_wanted = None          # (factor, x, y) still to apply
        self.frame_timer = self.fig.canvas.new_timer(interval=FRAME_MS)
        self.frame_timer.single_shot = True
        self.frame_timer.add_callback(self._next_frame)
        self._frame_due = False
        self._blit_box = None

        c = self.fig.canvas
        c.mpl_connect("button_press_event", self._on_press)
        c.mpl_connect("button_release_event", self._on_release)
        c.mpl_connect("motion_notify_event", self._on_motion)
        c.mpl_connect("key_press_event", self._on_key)
        c.mpl_connect("scroll_event", self._on_scroll)
        c.mpl_connect("resize_event", self._on_resize)
        self._capture_text_sizes()
        self.apply_theme()

        self.regenerate("terrain")

    # ------------------------------------------------------------- layout
    def _axes(self, x, y, w, h):
        """Add axes positioned in inches (from the bottom-left corner)."""
        return self.fig.add_axes([x / FIG_W, y / self.fig_h, w / FIG_W, h / self.fig_h])

    def _build_controls(self):
        y, bh = self.controls_y, 0.33
        self.btn_prev = Button(self._axes(0.9, y, 0.75, bh), "< Prev")
        self.seed_box = TextBox(self._axes(2.2, y, 0.85, bh), "Seed ", initial=str(self.cfg.seed))
        self.btn_next = Button(self._axes(3.15, y, 0.75, bh), "Next >")
        self.btn_rand = Button(self._axes(4.0, y, 0.85, bh), "Random")
        self.btn_view = Button(self._axes(4.95, y, 1.0, bh), "Reset view")
        self.btn_globe = Button(self._axes(6.05, y, 0.85, bh), "Globe")
        self.btn_globe.on_clicked(lambda _: self.toggle_globe())
        self.btn_prev.on_clicked(lambda _: self.set_seed(self.cfg.seed - 1))
        self.btn_next.on_clicked(lambda _: self.set_seed(self.cfg.seed + 1))
        self.btn_rand.on_clicked(lambda _: self.set_seed(int(np.random.randint(1, 1_000_000))))
        self.btn_view.on_clicked(lambda _: self.reset_view())
        self.seed_box.on_submit(self._on_seed_text)

        self.fig.text(7.05 / FIG_W, (y + 0.1) / self.fig_h, "Resolution", fontsize=10)
        self.res_buttons = {}
        for i, width in enumerate(RESOLUTIONS):
            b = Button(self._axes(7.9 + i * 0.65, y, 0.6, bh), str(width))
            b.on_clicked(lambda _, wd=width: self.set_resolution(wd))
            self.res_buttons[width] = b
        self._highlight_resolution()
        self.btn_enhance = Button(self._axes(9.95, y, 0.8, bh), "Enhance")
        self.btn_auto = Button(self._axes(10.8, y, 0.62, bh), "Auto")
        self.btn_auto.on_clicked(lambda _: self.toggle_auto_enhance())
        self.btn_clear = Button(self._axes(11.47, y, 0.55, bh), "Clear")
        self.btn_enhance.on_clicked(lambda _: self.enhance())
        self.btn_clear.on_clicked(lambda _: self.clear_region())

        # Favourite seeds (under the side panel)
        self.favourites = _load_favourites()
        self.btn_fav = Button(self._axes(12.1, y, 1.25, bh), "Favourite")
        self.btn_next_fav = Button(self._axes(13.45, y, 1.4, bh), "Favourites >")
        self.btn_fav.on_clicked(lambda _: self.toggle_favourite())
        self.btn_next_fav.on_clicked(lambda _: self.next_favourite())

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
        self.btn_theme = Button(self._axes(13.75, top + 0.06, 1.0, 0.28), "Light" if self.dark else "Dark")
        self.btn_theme.on_clicked(lambda _: self.toggle_dark())
        legend_ax = self._axes(x - 0.1, top - 1.75, 2.8, 1.75)
        legend_ax.set_axis_off()
        handles = [Patch(color=c, label=n) for n, c in BIOMES]
        handles.append(Line2D([], [], color=RIVER_COLOUR, linewidth=2, label="River"))
        legend_ax.legend(handles=handles, ncols=2, loc="upper left", fontsize=7, frameon=False,
                         labelspacing=0.25, columnspacing=0.6, handlelength=1.2, borderaxespad=0)

        y = top - 2.05
        layer_h = 0.235 * len(LAYERS)
        radio_ax = self._axes(x, y - layer_h, 2.4, layer_h)
        radio_ax.set_title("Layer", fontsize=11, loc="left", weight="bold")
        self.radio = RadioButtons(radio_ax, LAYERS, active=0,
                                  label_props={"fontsize": [9.5]}, radio_props={"s": [80]})
        self.radio.on_clicked(self._on_layer)

        if TOGGLES:
            y -= layer_h + 0.4
            opt_h = 0.38 * len(TOGGLES)
            opt_ax = self._axes(x, y - opt_h, 2.6, opt_h)
            opt_ax.set_title("Options", fontsize=11, loc="left", weight="bold")
            self.toggle_stage = {label: (name, stage) for name, label, stage in TOGGLES}
            self.checks = CheckButtons(opt_ax, [label for _, label, _ in TOGGLES],
                                       [bool(getattr(self.cfg, name)) for name, _, _ in TOGGLES],
                                       label_props={"fontsize": [12]},
                                       frame_props={"s": [260]}, check_props={"s": [260]})
            self.checks.on_clicked(self._on_toggle)

    def _highlight_resolution(self):
        t = self.theme
        for width, b in self.res_buttons.items():
            colour = t["active"] if width == self.cfg.width else t["button"]
            b.color = colour
            b.ax.set_facecolor(colour)

    # ------------------------------------------------------------- theme
    @property
    def theme(self):
        return THEMES["dark" if self.dark else "light"]

    def toggle_dark(self):
        self.dark = not self.dark
        self.btn_theme.label.set_text("Light" if self.dark else "Dark")
        _save_preferences({**_load_preferences(), "dark": self.dark})
        self.apply_theme()
        if self.world is not None:
            self.redraw()

    def apply_theme(self):
        """Colour the window, text and controls for light or dark mode (the map itself is unchanged)."""
        from matplotlib.text import Text

        t = self.theme
        self.fig.patch.set_facecolor(t["bg"])
        for text in self.fig.findobj(Text):
            text.set_color(t["fg"])
        for ax in self.fig.axes:
            ax.set_facecolor(t["bg"])
            ax.tick_params(colors=t["fg"])
            for spine in ax.spines.values():
                spine.set_edgecolor(t["edge"])
        for widget in vars(self).values():
            if isinstance(widget, (Button, TextBox)):
                widget.color, widget.hovercolor = t["button"], t["hover"]
                widget.ax.set_facecolor(t["button"])
        self.btn_enhance.color = t["active"]
        self.btn_enhance.ax.set_facecolor(t["active"])
        auto = t["active"] if self.auto_enhance else t["button"]
        self.btn_auto.color = auto
        self.btn_auto.ax.set_facecolor(auto)
        self._background = None
        self._blit_box = None
        self._highlight_resolution()
        if hasattr(self, "radio"):
            self.radio.set_radio_props({"edgecolor": t["fg"], "facecolor": t["fg"]})
        if hasattr(self, "checks"):
            self.checks.set_frame_props({"edgecolor": t["fg"]})
            self.checks.set_check_props({"facecolor": t["fg"]})
        for s in self.sliders.values():
            s.ax.set_facecolor(t["button"])
        self.fig.canvas.draw_idle()

    # ---------------------------------------------------------- generation
    def regenerate(self, stage):
        slow = "  (high resolution, this takes a while)" if self.cfg.width > 1000 and stage == "terrain" else ""
        self.ax.set_title(f"Generating...{slow}", fontsize=self.fs(11))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.clear_region(redraw=False)  # it belonged to the old world
        self._regions = []
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

    def _rendered8(self, world, layer, shrink=1):
        """The same picture as 8-bit colour (converted once; quick to draw), optionally
        shrunk by `shrink` along each side (averaging blocks of pixels)."""
        key = (id(world), layer, "u8", shrink)
        if key not in self._cache:
            img = np.clip(self._rendered(world, layer), 0, 1)
            if shrink > 1:
                h, w = img.shape[:2]
                img = img.reshape(h // shrink, shrink, w // shrink, shrink, -1).mean(axis=(1, 3))
            self._cache[key] = (img * 255 + 0.5).astype(np.uint8)
        return self._cache[key]

    def redraw(self, fast=False):
        """Draw the view. fast=True (while moving) redraws only the map, if the window allows it."""
        if self.globe:
            self._draw_globe()
        else:
            self._draw_map()
        self._draw_title()
        if fast and self._blit():
            return
        self._background = None
        self._blit_box = None
        self.fig.canvas.draw_idle()

    def _blit(self):
        """Redraw just the map over a saved picture of the rest of the window."""
        c = self.fig.canvas
        if not getattr(c, "supports_blit", False):
            return False
        try:
            if self._background is None:
                self.ax.set_visible(False)
                c.draw()
                self._background = c.copy_from_bbox(self.fig.bbox)
                self.ax.set_visible(True)
            c.restore_region(self._background)
            renderer = c.get_renderer()
            self.ax.draw(renderer)
            if self._blit_box is None:   # the map with its title and labels
                from matplotlib.transforms import Bbox
                box = Bbox.union([self.ax.get_tightbbox(renderer), self.ax.bbox]).padded(6)
                self._blit_box = Bbox.intersection(box, self.fig.bbox) or self.fig.bbox
            c.blit(self._blit_box)
            return True
        except Exception:          # a backend that can't: fall back to a full redraw
            self.ax.set_visible(True)
            self._background = None
            return False

    def _keep_moving(self):
        """Note a drag or scroll step; the full redraw waits until the view settles."""
        self.moving = True
        self.settle_timer.stop()
        self.settle_timer.start()

    def _settle(self):
        self.settle_timer.stop()
        if self._zoom_wanted is not None:     # a wheel step not drawn yet
            self.frame_timer.stop()
            self._next_frame()
        if not self.moving:
            return
        self.moving = False
        if self.auto_enhance and not self.globe and self._auto_enhance():
            return               # it redrew
        self.redraw()

    def _draw_map(self):
        if self.globe_image is not None:
            self.globe_image.set_visible(False)
            self.globe_rivers.set_segments([])
        if self.view is None:
            self.view = self._full_view()
        img, extent = self._visible_part(self.world, self.layer)
        if self.image is None:
            self.image = self.ax.imshow(img, interpolation="nearest", zorder=1, extent=extent)
            if not hasattr(self, "rivers"):
                self.rivers = LineCollection([], colors=[RIVER_COLOUR], capstyle="round",
                                             joinstyle="round", zorder=3)
                self.ax.add_collection(self.rivers)
            if self.view is None:
                self.view = self._full_view()
        else:
            self.image.set_data(img)
            self.image.set_extent(extent)
        self.image.set_visible(True)
        self.ax.set_axis_on()
        self.ax.set_facecolor(self.theme["bg"])
        self._apply_view()

    def _visible_part(self, world, layer):
        """Just the part of the (wrapped) map in view, plus a small margin, as 8-bit colour.
        When the view holds more map pixels than the screen has (zoomed out at high
        resolution), a half-size copy is used: the same on screen, much quicker to draw."""
        h, w = world.elevation_m.shape
        x0, x1, y0, y1 = self.view
        screen_px = max(self.ax.bbox.width, 1)
        f = 1
        while f < 4 and (x1 - x0) / (2 * f) >= screen_px and w % (2 * f) == 0 and h % (2 * f) == 0:
            f *= 2
        full = self._rendered8(world, layer, f)
        m = 2 * f
        c0 = int(np.floor(x0 + 0.5)) - m
        c1 = int(np.ceil(x1 + 0.5)) + m
        c0 -= (c0 + self.offset) % f                 # line up with the half-size pixels
        c1 += (-(c1 + self.offset)) % f
        c0, c1 = max(c0, -(self.offset % f)), min(c1, w + f)   # at most once round the globe
        r0 = max((int(np.floor(y0 + 0.5)) - m) // f * f, 0)
        r1 = min(-(-(int(np.ceil(y1 + 0.5)) + m) // f) * f, h)
        hw = w // f
        a = ((c0 + self.offset) % w) // f
        n = max(-(-(c1 - c0) // f), 1)
        rows = slice(r0 // f, r1 // f)
        if a + n <= hw:                               # no wrap: a view of the map, no copying
            img = full[rows, a:a + n]
        else:                                         # crosses the map's edge
            img = np.concatenate([full[rows, a:], full[rows, :a + n - hw]], axis=1)
        c1 = c0 + n * f
        return img, (c0 - 0.5, c1 - 0.5, r1 - 0.5, r0 - 0.5)

    def _draw_globe(self):
        # Half resolution while being dragged, so spinning stays smooth.
        g = GLOBE_PX // 2 if self.drag is not None else GLOBE_PX
        gw = 2 * g
        img, _, _, _ = globe_view(self._rendered(self.world, self.layer), self.globe_lon, self.globe_lat,
                                  g, self.globe_zoom, gw)
        if self.globe_image is None:
            self.globe_image = self.ax.imshow(img, interpolation="bilinear", zorder=5)
            self.globe_rivers = LineCollection([], colors=[RIVER_COLOUR], capstyle="round",
                                               joinstyle="round", zorder=6)
            self.ax.add_collection(self.globe_rivers)
        else:
            self.globe_image.set_data(img)
        self.globe_image.set_extent((-0.5, gw - 0.5, g - 0.5, -0.5))
        self.globe_image.set_visible(True)
        # Hide the flat map and everything drawn on it.
        for artist in (self.image, self.region_image):
            if artist is not None:
                artist.set_visible(False)
        self.rivers.set_segments([])
        if hasattr(self, "region_rivers"):
            self.region_rivers.set_segments([])
        if self.cfg.rivers and self.layer in RIVER_LAYERS and self.drag is None:  # hidden while spinning
            segs, widths = globe_rivers(self.world, self.globe_lon, self.globe_lat, g, self.globe_zoom, gw)
            self.globe_rivers.set_segments(segs)
            self.globe_rivers.set_linewidths(widths)
        else:
            self.globe_rivers.set_segments([])
        self.ax.set_xlim(-0.5, gw - 0.5)
        self.ax.set_ylim(g - 0.5, -0.5)
        self.ax.set_facecolor(GLOBE_BACKGROUND)
        self.ax.set_axis_off()

    def toggle_globe(self):
        self.globe = not self.globe
        self.btn_globe.label.set_text("Map" if self.globe else "Globe")
        if self.globe:
            # Start the globe looking at the middle of the current map view.
            x0, x1, y0, y1 = self.view
            col = ((x0 + x1) / 2 + self.offset) % self.cfg.width
            self.globe_lon = (col + 0.5) / self.cfg.width * 360 - 180
            self.globe_lat = float(np.clip(90 - ((y0 + y1) / 2 + 0.5) / self.cfg.height * 180, -60, 60))
            self.globe_zoom = float(np.clip(self.zoom / 2, 1, GLOBE_MAX_ZOOM))
        self.redraw()

    def _globe_drag_scale(self):
        """Degrees of rotation per screen pixel of drag."""
        r_screen = self.ax.get_window_extent().height / 2 * 0.95 * self.globe_zoom
        return np.degrees(1 / max(r_screen, 1))

    def _draw_title(self):
        w = self.world
        zoom = self.globe_zoom if self.globe else self.zoom
        zoom_text = f"   |   zoom {zoom:.1f}x" if zoom > 1.05 else ""
        if self.globe:
            zoom_text = "   |   globe" + zoom_text
        if self.region is not None:
            zoom_text += (f"   |   enhanced x{self.region.box.factor}"
                          f" ({self.region.grid.km_per_px:.0f} km/px)")
        star = "\u2605 " if self._is_favourite() else ""
        title = (f"{star}Seed {self.cfg.seed}   |   {w.major_landmasses} major landmasses, {w.islands} islands,"
                 f" {w.lakes} lakes   |   {self.cfg.width} px{zoom_text}   |   {self.timings}")
        changed = _changed_settings(self.cfg)
        if changed:
            title += "\nChanged from config.py: " + ", ".join(changed)
        self.ax.set_title(title, fontsize=self.fs(10))
        self.btn_fav.label.set_text("\u2605 Unfavourite" if star else "\u2606 Favourite")

    # ---------------------------------------------------------------- view
    @property
    def zoom(self):
        if self.view is None:
            return 1.0
        return self.cfg.width / (self.view[1] - self.view[0])

    def _full_view(self):
        return (-0.5, self.cfg.width - 0.5, -0.5, self.cfg.height - 0.5)

    def reset_view(self):
        if self.globe:
            self.globe_zoom = 1.0
        else:
            self.view = self._full_view()
        self.redraw()

    def _apply_view(self):
        x0, x1, y0, y1 = self.view
        self.ax.set_xlim(x0, x1)
        self.ax.set_ylim(y1, y0)  # rows grow downward
        # Zoomed in, blend the map's pixels instead of showing hard-edged blocks.
        smooth = "bilinear" if self.zoom >= SMOOTH_FROM_ZOOM and not self.moving else "nearest"
        self.image.set_interpolation(smooth)
        if self.region_image is not None:
            region_zoom = self.zoom / self.region.box.factor
            self.region_image.set_interpolation("bilinear" if region_zoom >= SMOOTH_FROM_ZOOM else "nearest")
        extent = self._region_extent()
        show_rivers = self.cfg.rivers and self.layer in RIVER_LAYERS and not self.moving
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
        if self.moving:              # labels catch up when the view settles
            self.ax.set_xlim(x0, x1)
            self.ax.set_ylim(y1, y0)
        else:
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
        self.redraw(fast=self.moving)

    def _update_ticks(self):
        w, h = self.cfg.width, self.cfg.height
        x0, x1, y0, y1 = self.view
        span = (x1 - x0) / w * 360
        step = next((s for s in reversed(TICK_STEPS) if span / s <= 7), 60)
        lons = np.arange(-180, 180, step)
        xs = ((lons + 180) / 360 * w - 0.5 - self.offset) % w
        self.ax.set_xticks(xs, [_fmt_deg(l, "E", "W") for l in lons], fontsize=self.fs(8))
        lats = np.arange(-90, 90 + step / 2, step)
        ys = (90 - lats) / 180 * h - 0.5
        self.ax.set_yticks(ys, [_fmt_deg(l, "N", "S") for l in lats], fontsize=self.fs(8))
        self.ax.set_xlim(x0, x1)
        self.ax.set_ylim(y1, y0)

    # ------------------------------------------------------------- enhance
    def enhance(self):
        if self.globe:
            self.ax.set_title("Enhance works on the flat map: press Map (or O) first", fontsize=self.fs(11))
            self.fig.canvas.draw_idle()
            return
        x0, x1, y0, y1 = self.view
        box = box_for_view(self.world, x0 + self.offset, x1 + self.offset, y0, y1)
        if box is None:
            most = MAX_REGION_PX // 2
            self.ax.set_title(f"Zoom in further to enhance (at most {most} map pixels across;"
                              f" now {x1 - x0:.0f})", fontsize=self.fs(11))
            self.fig.canvas.draw_idle()
            return
        self.ax.set_title(f"Enhancing this area at {box.factor}x detail...", fontsize=self.fs(11))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        t0 = time.perf_counter()
        self.clear_region(redraw=False)
        self.region = generate_region(self.world, box)
        self.timings = f"enhanced in {time.perf_counter() - t0:.1f}s"
        self.redraw()

    def toggle_auto_enhance(self):
        self.auto_enhance = not self.auto_enhance
        _save_preferences({**_load_preferences(), "auto_enhance": self.auto_enhance})
        self.apply_theme()
        if self.auto_enhance and not self.globe and self._auto_enhance():
            return
        self.redraw()

    def _auto_enhance(self):
        """Enhance the view at 2x if it is zoomed in enough and not enhanced already.
        Returns True if it drew."""
        x0, x1, y0, y1 = self.view
        if x1 - x0 > AUTO_ENHANCE_MAX_CELLS:
            return False
        if self.region is not None and self._covers(self.region.box):
            return False
        for wid, region in self._regions:          # enhanced here before?
            if wid == id(self.world) and self._covers(region.box):
                self.clear_region(redraw=False)
                self.region = region
                self.redraw()
                return True
        mx, my = (x1 - x0) * AUTO_ENHANCE_MARGIN, (y1 - y0) * AUTO_ENHANCE_MARGIN
        box = (box_for_view(self.world, x0 - mx + self.offset, x1 + mx + self.offset, y0 - my, y1 + my)
               or box_for_view(self.world, x0 + self.offset, x1 + self.offset, y0, y1))
        if box is None:
            return False
        box = RegionBox(box.row0, box.col0, box.rows, box.cols, AUTO_ENHANCE_FACTOR)
        self.ax.set_title(f"Enhancing this area at {box.factor}x detail...", fontsize=self.fs(11))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        t0 = time.perf_counter()
        self.clear_region(redraw=False)
        self.region = generate_region(self.world, box)
        self._regions = [(w, r) for w, r in self._regions if w == id(self.world)][-(AUTO_ENHANCE_CACHE - 1):]
        self._regions.append((id(self.world), self.region))
        self.timings = f"auto-enhanced in {time.perf_counter() - t0:.1f}s"
        self.redraw()
        return True

    def _covers(self, box):
        """Does a region box contain the whole current view?"""
        x0, x1, y0, y1 = self.view
        w = self.cfg.width
        left = (box.col0 - 0.5 - self.offset - x0) % w + x0   # nearest copy at or right of x0
        if left > x0 + 1e-6:
            left -= w
        top = box.row0 - 0.5
        return left <= x0 + 1e-6 and left + box.cols >= x1 - 1e-6 and top <= y0 + 1e-6 and top + box.rows >= y1 - 1e-6

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
            self.region_image.set_visible(True)
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

    # ------------------------------------------------------- window scaling
    def fs(self, size):
        """A font size scaled to the current window size."""
        return size * self.font_scale

    def _capture_text_sizes(self):
        """Remember the designed size of every fixed piece of text and every radio/check marker."""
        from matplotlib.text import Text

        self._texts = [(t, t.get_fontsize()) for t in self.fig.findobj(Text)
                       if t.axes is not self.ax and t.get_text()]
        self._markers = []
        for widget in (getattr(self, "radio", None), getattr(self, "checks", None)):
            for name in ("_buttons", "_frames", "_checks"):
                coll = getattr(widget, name, None) if widget is not None else None
                if coll is not None and hasattr(coll, "get_sizes"):
                    self._markers.append((coll, np.array(coll.get_sizes(), float)))

    def _on_resize(self, event):
        w_in, h_in = self.fig.get_size_inches()
        scale = max(0.5, min(w_in / FIG_W, h_in / self.fig_h))
        if abs(scale - self.font_scale) < 0.01:
            return
        self.font_scale = scale
        for text, size in self._texts:
            text.set_fontsize(size * scale)
        for coll, sizes in self._markers:
            coll.set_sizes(sizes * scale ** 2)  # marker sizes are areas
        if self.world is not None:
            self.redraw()

    # ---------------------------------------------------------- favourites
    def _is_favourite(self):
        return any(f["seed"] == self.cfg.seed for f in self.favourites)

    def toggle_favourite(self):
        if self._is_favourite():
            self.favourites = [f for f in self.favourites if f["seed"] != self.cfg.seed]
        else:
            self.favourites.append({"seed": self.cfg.seed, "width": self.cfg.width})
        _save_favourites(self.favourites)
        self.redraw()

    def next_favourite(self):
        if not self.favourites:
            self.ax.set_title("No favourites yet: press Favourite (or F) to star a seed", fontsize=self.fs(11))
            self.fig.canvas.draw_idle()
            return
        seeds = [f["seed"] for f in self.favourites]
        later = [i for i, sd in enumerate(seeds) if sd == self.cfg.seed]
        i = (later[0] + 1) % len(seeds) if later else 0
        self.set_seed(seeds[i])

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
        if self.globe:
            self.globe_zoom = float(np.clip(self.globe_zoom * factor, 1, GLOBE_MAX_ZOOM))
            self.redraw()
            return
        self._keep_moving()
        old = self._zoom_wanted[0] if self._zoom_wanted else 1.0
        self._zoom_wanted = (old * factor, event.xdata, event.ydata)
        self._schedule_frame()

    def _on_press(self, event):
        if event.inaxes is self.ax and event.button == 1:
            self.drag = (event.x, event.y, self.offset, self.view, self.globe_lon, self.globe_lat)

    def _on_release(self, event):
        was_dragging = self.drag is not None
        self.drag = None
        if was_dragging and self.globe:
            self.redraw()  # sharpen the globe again
        elif was_dragging and self.moving:
            self.frame_timer.stop()
            self._next_frame()  # the last position
            self._settle()      # rivers back, auto-Enhance

    def _on_motion(self, event):
        if self.drag is None or event.x is None:
            return
        x_start, y_start, off0, view0, lon0, lat0 = self.drag
        if self.globe:  # spin the globe: dragging moves the surface under the cursor
            deg = self._globe_drag_scale()
            self.globe_lon = (lon0 - (event.x - x_start) * deg + 180) % 360 - 180
            self.globe_lat = float(np.clip(lat0 - (event.y - y_start) * deg, -89, 89))
            self.redraw()
            return
        x0, x1, y0, y1 = view0
        px_per_cell = self.ax.get_window_extent().width / (x1 - x0)
        dx = (event.x - x_start) / px_per_cell
        dy = (event.y - y_start) / px_per_cell   # screen y points up, rows point down
        self._drag_to = (event.x, event.y)
        self.moving = True
        self._schedule_frame()

    def _schedule_frame(self):
        if not self._frame_due:
            self._frame_due = True
            self.frame_timer.start()

    def _next_frame(self):
        """Draw the newest drag position and the zoom gathered since the last frame."""
        self._frame_due = False
        drew = False
        if self._zoom_wanted is not None:
            factor, cx, cy = self._zoom_wanted
            self._zoom_wanted = None
            self.zoom_at(factor, cx, cy)
            drew = True
        if self._drag_to is not None and self.drag is not None and not self.globe:
            x, y = self._drag_to
            self._drag_to = None
            x_start, y_start, off0, view0, _, _ = self.drag
            x0, x1, y0, y1 = view0
            px_per_cell = self.ax.get_window_extent().width / (x1 - x0)
            dx = (x - x_start) / px_per_cell
            dy = (y - y_start) / px_per_cell   # screen y points up, rows point down
            self.offset = off0
            self._set_view(x0 - dx, x1 - dx, y0 + dy, y1 + dy)
            self.redraw(fast=True)
            drew = True
        return drew

    def _on_key(self, event):
        x0, x1, y0, y1 = self.view
        step = (x1 - x0) / 6
        if event.key in ("o", "O"):
            self.toggle_globe()
            return
        if self.globe and event.key in ("left", "right", "up", "down", "+", "=", "-"):
            turn = 15 / self.globe_zoom
            if event.key in ("left", "right"):
                self.globe_lon = (self.globe_lon + (-turn if event.key == "left" else turn) + 180) % 360 - 180
            elif event.key in ("up", "down"):
                self.globe_lat = float(np.clip(self.globe_lat + (turn if event.key == "up" else -turn), -89, 89))
            else:
                factor = ZOOM_STEP if event.key in ("+", "=") else 1 / ZOOM_STEP
                self.globe_zoom = float(np.clip(self.globe_zoom * factor, 1, GLOBE_MAX_ZOOM))
            self.redraw()
            return
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
        elif event.key in ("f", "F"):
            self.toggle_favourite()
        elif event.key in ("a", "A"):
            self.toggle_auto_enhance()
        elif event.key in ("d", "D"):
            self.toggle_dark()
        elif event.key in ("g", "G"):
            self.next_favourite()
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
        if self.globe:
            g = GLOBE_PX
            r = g / 2 * 0.95 * self.globe_zoom
            gx, gy = (x + 0.5 - g) / r, -(y + 0.5 - g / 2) / r   # picture is 2g wide, g tall
            if gx * gx + gy * gy > 1:
                return "space"
            gz = np.sqrt(1 - gx * gx - gy * gy)
            phi0 = np.radians(self.globe_lat)
            lat = np.degrees(np.arcsin(np.clip(gz * np.sin(phi0) + gy * np.cos(phi0), -1, 1)))
            lon = (self.globe_lon + np.degrees(np.arctan2(gx, gz * np.cos(phi0) - gy * np.sin(phi0))) + 180) % 360 - 180
            x = (lon + 180) / 360 * self.cfg.width - 0.5 - self.offset
            y = (90 - lat) / 180 * self.cfg.height - 0.5
        col = (int(round(x)) + self.offset) % self.cfg.width
        row = int(np.clip(round(y), 0, self.cfg.height - 1))
        extent = None if self.globe else self._region_extent()
        if extent is not None:
            left, right, bottom, top = extent
            if left <= x < right and top <= y < bottom:
                f = self.region.box.factor
                w = self.region
                h_r, w_r = w.elevation_m.shape
                col = int(np.clip((x - left) * f, 0, w_r - 1))
                row = int(np.clip((y - top) * f, 0, h_r - 1))
        lat, lon = w.terrain.lat[row, col], w.terrain.lon[row, col]
        rc = (row, col)
        text = (f"{_fmt_deg(lat, 'N', 'S', 1)} {_fmt_deg(lon, 'E', 'W', 1)}  "
                f"{BIOMES[w.biome[rc]][0]}  {w.elevation_m[rc]:.0f} m  "
                f"{w.temperature_c[rc]:.0f} C ({w.coldest_month_c[rc]:.0f} to {w.warmest_month_c[rc]:.0f})  "
                f"{w.precip_mm[rc]:.0f} mm/yr  growing {w.growing_months[rc]:.0f} mo")
        if not (w.ocean[rc] or w.lake[rc]):
            text += (f"  farm {w.arable[rc]:.2f}  move {w.movement_cost[rc]:.1f}"
                     + (f"  harbour {w.harbour[rc]:.2f}" if w.coastal[rc] else ""))
            from .nature import MINERALS, SPECIES
            minerals = [m for m in MINERALS if getattr(w, m)[rc] > 0.3]
            wild = [s.key.replace("_", " ") for i, s in enumerate(SPECIES) if w.species_native[i][rc]]
            if minerals:
                text += "  ores: " + ", ".join(minerals)
            if wild:
                text += f"  wild: {len(wild)} ({', '.join(wild[:4])}{'...' if len(wild) > 4 else ''})"
        flow, notes = self._edge_info(w)
        if flow[rc] > 0:
            tags = [name for name, mask in notes.items() if mask[rc]]
            text += f"  river {flow[rc]:,.0f} m3/s" + (f" ({', '.join(tags)})" if tags else "")
        return text

    def _edge_info(self, world):
        """Per cell: biggest river on its edges, and whether any of those edges is a ford,
        navigable, or has rapids (cached per world)."""
        key = id(world)
        if getattr(self, "_edge_key", None) != key:
            from .hydrology import cell_river_flow, edge_properties

            notes = {}
            for tag in ("fordable", "navigable", "rapids"):
                e = edge_properties(world, "east")[tag]
                s_ = edge_properties(world, "south")[tag]
                west = np.roll(e, 1, axis=1) if world.grid.wraps else np.pad(e[:, :-1], ((0, 0), (1, 0)))
                north = np.pad(s_[:-1], ((1, 0), (0, 0)))
                notes[tag] = e | west | s_ | north
            self._edge_data = (cell_river_flow(world), notes)
            self._edge_key = key
        return self._edge_data


def _load_favourites():
    try:
        with open(FAVOURITES_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return [d for d in data if isinstance(d, dict) and isinstance(d.get("seed"), int)]
    except (OSError, ValueError):
        return []


def _save_favourites(favourites):
    try:
        with open(FAVOURITES_FILE, "w", encoding="utf-8") as f:
            json.dump(favourites, f, indent=1)
    except OSError as e:
        print(f"couldn't save favourites: {e}")


def _load_preferences():
    try:
        with open(PREFERENCES_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_preferences(prefs):
    try:
        with open(PREFERENCES_FILE, "w", encoding="utf-8") as f:
            json.dump(prefs, f, indent=1)
    except OSError as e:
        print(f"couldn't save viewer preferences: {e}")


def _changed_settings(cfg):
    """Settings that differ from config.py's defaults (seed and resolution aside)."""
    out = []
    for f in fields(cfg):
        if f.name in ("seed", "width"):
            continue
        value, default = getattr(cfg, f.name), getattr(DEFAULT, f.name)
        if value != default:
            out.append(f"{f.name} {'on' if value is True else 'off' if value is False else value}")
    return out


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
