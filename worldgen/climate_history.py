"""How a world's climate changes over the span of history.

Times are in simulation years: year 0 corresponds to about 10,000 BCE (the end of the
last ice age), year 10,000 to about 1 CE and year 12,000 to about 2000 CE.

The broad shape follows Earth's Holocene: a cold start, abrupt warming, a warm peak
around 6000-4000 BCE, slow cooling, and medieval-warm / little-ice-age swings. On top
of that, each world gets its own regional swings, megadroughts and volcanic winters,
all deterministic from the world seed. Warming from industry is not included: that is
the history simulator's job.

    history = ClimateHistory(world)
    state = history.at(5000)      # temperature offset and rainfall factor per cell
"""
from dataclasses import dataclass

import numpy as np

from .noise import Perlin3, fbm

# Global mean temperature change from the world's present-day climate, C.
GLOBAL_TEMPERATURE = (
    (-12000, -5.0), (-1500, -4.0), (0, -2.0), (250, -1.8), (350, -0.6), (2000, -0.2),
    (4000, 0.4), (6000, 0.4), (8000, 0.1), (10000, 0.05), (10900, 0.15), (11250, 0.1),
    (11400, -0.15), (11600, -0.35), (11850, -0.3), (11950, 0.0), (10 ** 6, 0.0),
)
# Sea level relative to today, m (ice sheets melting after the ice age).
SEA_LEVEL = (
    (-12000, -120), (-3000, -95), (0, -55), (1000, -45), (2000, -35), (3000, -25),
    (4000, -15), (5000, -6), (6000, -2), (7000, 0), (10 ** 6, 0),
)
PATTERNS = 4             # independent regional patterns of warm/cold and wet/dry
DROUGHTS_PER_1000_YEARS = 3.0
VOLCANIC_WINTERS_PER_1000_YEARS = 8.0


def sea_level_m(year):
    """Sea level relative to today in simulation year `year`."""
    years, levels = zip(*SEA_LEVEL)
    return float(np.interp(year, years, levels))


@dataclass
class ClimateState:
    year: float
    global_temperature_c: float   # change in global mean temperature
    temperature_offset_c: np.ndarray  # change per cell, add to world.temperature_c
    precip_factor: np.ndarray     # multiply world.precip_mm by this
    sea_level_m: float
    droughts: list                # (start, end, lat, lon, radius_km, severity) active this year
    volcanic: float               # cooling from volcanic eruptions this year, C


class ClimateHistory:
    """The climate timeline for one world (works on a whole-globe world or a region)."""

    def __init__(self, world, start=-12000, end=20000):
        grid = world.grid
        seed = world.cfg.seed
        rng = np.random.default_rng((seed, 9001))
        xyz = grid.xyz
        self.lat = grid.lat
        sin_lat = np.sin(np.radians(np.abs(self.lat)))
        self.polar_amplification = (1 + 1.5 * sin_lat ** 2).astype(np.float32)

        def pattern(k, freq):
            f = fbm(Perlin3(seed * 7919 + 900001 + k * 104729), xyz, freq, 3)
            return ((f - f.mean()) / max(float(f.std()), 1e-6)).astype(np.float32)

        self.temp_patterns = [pattern(k, rng.uniform(1.2, 3.0)) for k in range(PATTERNS)]
        self.rain_patterns = [pattern(k + 50, rng.uniform(1.2, 3.0)) for k in range(PATTERNS)]
        # Each pattern rises and falls with a few overlapping cycles of 40-800 years.
        self.cycles = [[(np.exp(rng.uniform(np.log(40), np.log(800))), rng.uniform(0.2, 1.0),
                         rng.uniform(0, 2 * np.pi)) for _ in range(4)] for _ in range(2 * PATTERNS)]
        self.global_cycles = [(np.exp(rng.uniform(np.log(60), np.log(1000))), rng.uniform(0.05, 0.15),
                               rng.uniform(0, 2 * np.pi)) for _ in range(3)]

        span = end - start
        land = np.flatnonzero((~world.ocean).ravel())
        self.droughts = []
        for _ in range(rng.poisson(DROUGHTS_PER_1000_YEARS * span / 1000)):
            year = rng.uniform(start, end)
            cell = land[rng.integers(len(land))] if len(land) else 0
            self.droughts.append((year, year + rng.uniform(20, 300), float(grid.lat.ravel()[cell]),
                                  float(grid.lon.ravel()[cell]), rng.uniform(800, 2500), rng.uniform(0.2, 0.5)))
        self.droughts.sort()
        self.volcanic = sorted((rng.uniform(start, end), rng.uniform(2, 6), rng.uniform(0.3, 1.5))
                               for _ in range(rng.poisson(VOLCANIC_WINTERS_PER_1000_YEARS * span / 1000)))
        self._xyz = np.stack(xyz, axis=-1)

    # --------------------------------------------------------------- queries
    def global_temperature(self, year):
        years, temps = zip(*GLOBAL_TEMPERATURE)
        t = float(np.interp(year, years, temps))
        t += sum(a * np.sin(2 * np.pi * year / p + ph) for p, a, ph in self.global_cycles)
        return t - self.volcanic_cooling(year)

    def volcanic_cooling(self, year):
        cooling = 0.0
        for start, duration, strength in self.volcanic:
            if start <= year < start + duration:
                cooling = max(cooling, strength * (1 - (year - start) / duration))
        return cooling

    def _weight(self, k, year):
        return sum(a * np.sin(2 * np.pi * year / p + ph) for p, a, ph in self.cycles[k]) / 2

    def at(self, year):
        """The climate in simulation year `year`."""
        g = self.global_temperature(year)
        temp = g * self.polar_amplification
        rain_log = np.zeros_like(temp) + 0.03 * g     # a warmer world is a little wetter
        for k in range(PATTERNS):
            temp = temp + 0.5 * self._weight(k, year) * self.temp_patterns[k]
            rain_log = rain_log + 0.12 * self._weight(k + PATTERNS, year) * self.rain_patterns[k]
        precip = np.exp(rain_log)
        active = [d for d in self.droughts if d[0] <= year < d[1]]
        for start, end, lat, lon, radius_km, severity in active:
            la, lo = np.radians(lat), np.radians(lon)
            centre = np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
            dist_km = np.linalg.norm(self._xyz - centre, axis=-1) * 6371.0
            ramp = min(1.0, (year - start) / 15, (end - year) / 15)   # builds up and fades
            precip = precip * (1 - severity * ramp * np.exp(-(dist_km / radius_km) ** 2))
        return ClimateState(year=year, global_temperature_c=g, temperature_offset_c=temp.astype(np.float32),
                            precip_factor=precip.astype(np.float32), sea_level_m=sea_level_m(year),
                            droughts=active, volcanic=self.volcanic_cooling(year))
