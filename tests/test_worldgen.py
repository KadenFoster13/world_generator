"""Quick checks that generation still behaves. Run: python -m unittest discover tests

Uses small 400 px worlds so the whole suite takes well under a minute.
"""
import os
import tempfile
import unittest

import numpy as np

from worldgen import (WorldConfig, generate_region, generate_world, load_world_data,
                      region_box, save_world)

CFG = WorldConfig(seed=7, width=400)


class GenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world = generate_world(CFG)

    def test_same_seed_same_world(self):
        again = generate_world(CFG)
        np.testing.assert_array_equal(self.world.elevation_m, again.elevation_m)
        np.testing.assert_array_equal(self.world.biome, again.biome)

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
        agreement = (blocks == world_part).mean()
        self.assertGreater(agreement, 0.9)

    def test_export_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "world.npz")
            save_world(self.world, path)
            arrays, meta = load_world_data(path)
        np.testing.assert_array_equal(arrays["elevation_m"], self.world.elevation_m)
        self.assertEqual(meta["config"]["seed"], CFG.seed)
        self.assertLess(int(self.world.biome.max()), len(meta["biomes"]))


if __name__ == "__main__":
    unittest.main()
