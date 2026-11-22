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
        for name in ("soil_fertility", "arable", "fresh_water", "timber", "ore", "harbour"):
            v = getattr(w, name)
            self.assertTrue(((v >= 0) & (v <= 1)).all(), name)
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


if __name__ == "__main__":
    unittest.main()
