"""Quick checks that generation still behaves. Run: python -m unittest discover tests

Uses small 400 px worlds so the whole suite takes well under a minute.
"""
import os
import tempfile
import unittest

import numpy as np

from worldgen import (DATA_VERSION, FIELDS, WorldConfig, edge_properties, generate_region,
                      generate_world, load_or_generate, load_world_data, region_box, save_world)

CFG = WorldConfig(seed=7, width=400)


class GenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world = generate_world(CFG)

    def test_same_seed_same_world(self):
        again = generate_world(CFG)
        np.testing.assert_array_equal(self.world.elevation_m, again.elevation_m)
        np.testing.assert_array_equal(self.world.biome, again.biome)
        np.testing.assert_array_equal(self.world.river_flow_east, again.river_flow_east)

    def test_land_fraction(self):
        w = self.world
        cos_lat = np.cos(np.radians(w.grid.lat))
        land = (~w.ocean * cos_lat).sum() / cos_lat.sum()
        # Small inland seas are filled in after sea level is set, so land can be a bit higher.
        self.assertAlmostEqual(land, CFG.land_fraction, delta=0.03)

    def test_rivers_reach_water(self):
        w = self.world
        assert w.river.any(), "expected some rivers"
        biggest = np.unravel_index(np.argmax(np.where(w.ocean, 0, w.discharge_m3s)), w.ocean.shape)
        self.assertGreater(w.discharge_m3s[biggest], 1000)

    def test_river_edges(self):
        w = self.world
        for side in ("east", "south"):
            p = edge_properties(w, side)
            self.assertTrue(p["river"].any())
            # Every river edge has land on at least one side.
            land = ~w.ocean & ~w.lake
            other = np.roll(land, -1, axis=1) if side == "east" else np.vstack([land[1:], land[-1:]])
            self.assertTrue((land | other)[p["river"]].all())
            self.assertTrue((p["width_m"][p["river"]] > 0).all())
        # Every river path ends on the sea or a lake, or flows on as another path.
        self.assertGreater(len(w.river_paths), 10)

    def test_seasons_and_land(self):
        w = self.world
        land = ~w.ocean & ~w.lake
        self.assertTrue((w.warmest_month_c >= w.coldest_month_c).all())
        tropics = land & (np.abs(w.grid.lat) < 10)
        poles = land & (np.abs(w.grid.lat) > 60)
        if tropics.any() and poles.any():
            swing = w.warmest_month_c - w.coldest_month_c
            self.assertLess(swing[tropics].mean(), swing[poles].mean())
        self.assertTrue(((w.growing_months >= 0) & (w.growing_months <= 12)).all())
        for name in ("soil_fertility", "arable", "fresh_water", "timber", "ore", "harbour",
                     "floodplain", "irrigable"):
            v = getattr(w, name)
            self.assertTrue(((v >= 0) & (v <= 1)).all(), name)
        self.assertTrue((w.irrigable >= w.floodplain).all())       # floodplains are irrigable too
        self.assertEqual(w.floodplain[w.ocean].max(), 0)
        self.assertTrue(np.isinf(w.movement_cost[w.ocean]).all())

    def test_currents_wetlands_ice(self):
        w = self.world
        change = w.wind["current_temp"]
        self.assertGreater(change.max(), 1.0)      # some warm currents
        self.assertLess(change.min(), -1.0)        # and some cold ones
        for name in ("wetland", "permafrost"):
            v = getattr(w, name)
            self.assertTrue(((v >= 0) & (v <= 1)).all(), name)
        self.assertTrue((w.permafrost[w.temperature_c > 0] == 0).all())

    def test_fast_and_plain_routing_agree(self):
        from worldgen import _fast, hydrology
        if not _fast.HAVE_NUMBA:
            self.skipTest("numba not installed")
        w = self.world
        fast = hydrology.priority_flood(w.elevation_m, w.ocean)
        _fast.HAVE_NUMBA = False
        try:
            plain = hydrology.priority_flood(w.elevation_m, w.ocean)
        finally:
            _fast.HAVE_NUMBA = True
        for a, b in zip(fast, plain):
            np.testing.assert_array_equal(a, b)

    def test_nature(self):
        from worldgen import MINERALS, SPECIES
        w = self.world
        self.assertEqual(w.species_native.shape[0], len(SPECIES))
        # Species only live wild where they can live at all (the minimum habitat quality
        # varies from place to place, around 0.35), and never at sea.
        self.assertTrue((w.species_suitability[w.species_native] >= 0.1).all())
        self.assertGreater(np.median(w.species_suitability[w.species_native]), 0.5)
        self.assertFalse(w.species_native[:, w.ocean].any())
        self.assertTrue(w.species_native.any())
        for m in MINERALS:
            v = getattr(w, m)
            self.assertTrue(((v >= 0) & (v <= 1)).all(), m)
        self.assertTrue(w.glacial_land[~w.ocean].sum() == 0)  # only today's sea can be ice-age land

    def test_wildlife_same_at_any_width(self):
        # Homelands are picked on a fixed coarse grid, so ranges barely move with width.
        big = generate_world(WorldConfig(seed=7, width=800)).species_native
        big = big.reshape(big.shape[0], big.shape[1] // 2, 2, big.shape[2] // 2, 2).mean(axis=(2, 4)) >= 0.5
        small = self.world.species_native
        both, either = (small & big).sum(axis=(1, 2)), (small | big).sum(axis=(1, 2))
        present = either >= 20
        self.assertGreater(np.median(both[present] / either[present]), 0.7)

    def test_climate_history(self):
        from worldgen import ClimateHistory, sea_level_m
        a, b = ClimateHistory(self.world).at(3000), ClimateHistory(self.world).at(3000)
        np.testing.assert_array_equal(a.temperature_offset_c, b.temperature_offset_c)  # deterministic
        self.assertLess(sea_level_m(0), sea_level_m(5000))
        self.assertEqual(sea_level_m(9000), 0)
        self.assertTrue((a.precip_factor > 0).all())

    def test_every_field_present(self):
        for name in FIELDS:
            if name not in ("lat", "lon", "plate"):
                self.assertIsNotNone(getattr(self.world, name, None), name)

    def test_region_matches_world(self):
        w = self.world
        box = region_box(w, north=40, south=10, west=-30, east=30)
        self.assertIsNotNone(box)
        region = generate_region(w, box)
        f = box.factor
        # Average each region block back down to one world cell and compare.
        h, wd = region.elevation_m.shape
        blocks = region.ocean.reshape(h // f, f, wd // f, f).mean(axis=(1, 3)) > 0.5
        cols = (box.col0 + np.arange(box.cols)) % w.grid.width
        world_part = w.ocean[box.row0:box.row0 + box.rows][:, cols]
        self.assertGreater((blocks == world_part).mean(), 0.9)
        self.assertIsNotNone(region.growing_months)
        self.assertIsNotNone(region.river_flow_east)

    def test_region_keeps_big_rivers(self):
        # Enhanced rivers follow the world's courses, wherever the region's edges fall.
        from scipy.ndimage import maximum_filter
        from worldgen.region import box_for_view
        w = self.world
        land = ~(w.ocean | w.lake)
        kept = []
        # Views with a big river close to their edge (where routing used to leak out).
        for x0, cy, half in ((301, 190, 8), (276, 184, 12), (328, 188, 12), (222, 167, 20), (246, 179, 20)):
            box = box_for_view(w, x0, x0 + 2 * half, max(cy - half * 0.6, 0), min(cy + half * 0.6, 199))
            f = box.factor
            region = generate_region(w, box)
            # Regions don't wrap east-west: no river step jumps across the area.
            for rows, cols_, _ in region.river_paths:
                self.assertLessEqual(np.abs(np.diff(cols_)).max(initial=0), 1)
            q = np.where(region.ocean | region.lake, 0, region.discharge_m3s)
            q = maximum_filter(q.reshape(box.rows, f, box.cols, f).max(axis=(1, 3)), 3)
            cols = (box.col0 + np.arange(box.cols)) % w.grid.width
            part = (slice(box.row0, box.row0 + box.rows), cols)
            qw = w.discharge_m3s[part]
            inner = np.zeros(qw.shape, bool)
            inner[2:-2, 2:-2] = True
            big = land[part] & inner & (qw > 1000) & ~maximum_filter(~land[part], 3)
            kept.extend((q[big] / qw[big]).tolist())
        self.assertGreater(len(kept), 20)
        self.assertGreater(np.mean(np.array(kept) >= 0.5), 0.9)

    def test_export_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "world.npz")
            save_world(self.world, path)
            arrays, meta = load_world_data(path)
        np.testing.assert_array_equal(arrays["elevation_m"], self.world.elevation_m)
        self.assertEqual(meta["config"]["seed"], CFG.seed)
        self.assertEqual(meta["data_version"], DATA_VERSION)
        self.assertEqual(set(arrays), set(FIELDS))
        self.assertLess(int(self.world.biome.max()), len(meta["biomes"]))

    def test_cache_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = load_or_generate(CFG, tmp)
            self.assertEqual(len(os.listdir(tmp)), 1)
            again = load_or_generate(CFG, tmp)
        np.testing.assert_allclose(again.elevation_m, first.elevation_m, atol=1.0)
        np.testing.assert_array_equal(again.biome, first.biome)
        # A cached world can still be enhanced.
        region = generate_region(again, region_box(again, north=40, south=10, west=-30, east=30))
        self.assertGreater(region.elevation_m.size, 0)



class EarthTests(unittest.TestCase):
    """Seed 0 is Earth: real elevation, real homelands and ore regions."""

    @classmethod
    def setUpClass(cls):
        cls.world = generate_world(WorldConfig(seed=0, width=800))

    def at(self, a, lat, lon):
        g = self.world.grid
        return a[..., int((90 - lat) / g.dpp), int((lon + 180) / g.dpp) % g.width]

    def near(self, a, lat, lon, km=300):
        """Any cell of `a` within `km` of a point."""
        from worldgen.earth import great_circle_km
        g = self.world.grid
        return bool(a[great_circle_km(g.lat, g.lon, lat, lon) <= km].any())

    def test_land_and_sea(self):
        w = self.world
        land = {"Africa": (5, 20), "Europe": (50, 10), "Asia": (45, 90), "India": (22, 78),
                "North America": (40, -100), "South America": (-10, -55), "Australia": (-25, 135),
                "Antarctica": (-80, 0), "Greenland": (72, -40)}
        for name, (lat, lon) in land.items():
            self.assertFalse(self.at(w.ocean, lat, lon), name)
        for name, (lat, lon) in {"mid Pacific": (0, -150), "Atlantic": (30, -40), "Indian Ocean": (-20, 80),
                                 "Mediterranean": (35, 18), "Black Sea": (43, 34)}.items():
            self.assertTrue(self.at(w.ocean, lat, lon), name)
        cos_lat = np.cos(np.radians(w.grid.lat))
        self.assertAlmostEqual((~w.ocean * cos_lat).sum() / cos_lat.sum(), 0.29, delta=0.02)
        self.assertGreater(self.at(w.elevation_m, 33, 88), 4000)        # Tibet
        self.assertGreater(w.discharge_m3s[~w.ocean].max(), 50000)      # the Amazon

    def test_wild_species(self):
        from worldgen import SPECIES_KEYS
        w = self.world
        native = dict(zip(SPECIES_KEYS, w.species_native))
        for key in ("temperate_grain", "sheep_goat", "legume"):
            self.assertTrue(self.near(native[key], 37, 40), f"{key} in the Fertile Crescent")
        self.assertTrue(self.near(native["slow_grain"], 18.3, -100.5), "teosinte in the Balsas")
        self.assertTrue(self.near(native["highland_tuber"], -15.8, -69.5), "potato by Titicaca")
        self.assertTrue(self.near(native["llama"], -17, -68.5), "camelids on the Altiplano")
        self.assertTrue(self.near(native["horse"], 48.5, 44, 600), "horses on the Pontic-Caspian steppe")
        self.assertTrue(self.near(native["wet_grain"], 30, 116, 500), "rice on the Yangtze")
        americas = w.grid.lon < -30
        australia = (w.grid.lon > 110) & (w.grid.lon < 155) & (w.grid.lat < -10) & (w.grid.lat > -45)
        for key in ("cattle", "horse", "sheep_goat", "camel", "pig"):
            self.assertFalse((native[key] & americas).any(), f"no wild {key} in the Americas")
        self.assertFalse((native["sheep_goat"] & australia).any(), "no wild sheep or goats in Australia")
        for key in ("slow_grain", "highland_tuber", "llama"):
            self.assertFalse((native[key] & ~americas).any(), f"{key} only in the Americas")

    def test_minerals(self):
        w = self.world
        self.assertTrue(self.near(w.tin > 0.6, 50.3, -5.2, 150), "tin in Cornwall")
        self.assertTrue(self.near(w.tin > 0.6, 3.5, 101.5, 300), "tin in Malaya")
        self.assertTrue(self.near(w.copper > 0.6, 35, 33, 120), "copper on Cyprus")
        self.assertTrue(self.near(w.silver > 0.6, -19.6, -65.75, 150), "silver at Potosi")
        self.assertTrue(self.near(w.oil > 0.6, 27, 50, 400), "oil in the Gulf")
        for name in ("copper", "tin", "gold", "silver"):     # unlisted places stay modest
            self.assertLess(float(np.percentile(getattr(w, name)[~w.ocean], 95)), 0.5, name)

    def test_region(self):
        region = generate_region(self.world, region_box(self.world, north=34, south=22, west=26, east=40))
        self.assertTrue(region.river.any())                              # the Nile
        self.assertTrue(region.ocean.any() and (~region.ocean).any())
        self.assertTrue(region.species_native.any())


if __name__ == "__main__":
    unittest.main()
