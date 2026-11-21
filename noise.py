"""Vectorised 3D Perlin noise sampled on a sphere.

Sampling noise on the surface of a 3D sphere (instead of a flat 2D plane)
makes the map wrap seamlessly east-west, and avoids a seam or pinching
artefacts at the poles.
"""
import numpy as np

_GRADIENTS = np.array([
    [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
    [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
    [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1],
    [1, 1, 0], [0, -1, 1], [-1, 1, 0], [0, -1, -1],
], dtype=np.float32)
_GX, _GY, _GZ = (_GRADIENTS[:, i].copy() for i in range(3))


def _fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


class Perlin3:
    """Classic improved Perlin noise in 3D, evaluated on arbitrary point arrays."""

    def __init__(self, seed: int):
        rng = np.random.default_rng(seed)
        p = rng.permutation(256)
        self.perm = np.concatenate([p, p]).astype(np.int32)
        self.offset = rng.uniform(-500, 500, 3)

    def __call__(self, x, y, z):
        # 32-bit floats: twice as fast as 64-bit and far more precise than terrain needs.
        x = (x + self.offset[0]).astype(np.float32)
        y = (y + self.offset[1]).astype(np.float32)
        z = (z + self.offset[2]).astype(np.float32)
        xi0 = np.floor(x)
        yi0 = np.floor(y)
        zi0 = np.floor(z)
        xf, yf, zf = x - xi0, y - yi0, z - zi0
        xi = xi0.astype(np.int32) & 255
        yi = yi0.astype(np.int32) & 255
        zi = zi0.astype(np.int32) & 255
        u, v, w = _fade(xf), _fade(yf), _fade(zf)

        perm = self.perm
        a = perm[xi] + yi
        aa = perm[a] + zi
        ab = perm[a + 1] + zi
        b = perm[xi + 1] + yi
        ba = perm[b] + zi
        bb = perm[b + 1] + zi

        def grad(h, gx, gy, gz):
            h = h & 15
            return _GX[h] * gx + _GY[h] * gy + _GZ[h] * gz

        x1 = (grad(perm[aa], xf, yf, zf) * (1 - u) + grad(perm[ba], xf - 1, yf, zf) * u)
        x2 = (grad(perm[ab], xf, yf - 1, zf) * (1 - u) + grad(perm[bb], xf - 1, yf - 1, zf) * u)
        y1 = x1 * (1 - v) + x2 * v
        x1 = (grad(perm[aa + 1], xf, yf, zf - 1) * (1 - u) + grad(perm[ba + 1], xf - 1, yf, zf - 1) * u)
        x2 = (grad(perm[ab + 1], xf, yf - 1, zf - 1) * (1 - u) + grad(perm[bb + 1], xf - 1, yf - 1, zf - 1) * u)
        y2 = x1 * (1 - v) + x2 * v
        return y1 * (1 - w) + y2 * w


class SphereXYZ(tuple):
    """(x, y, z) unit-sphere coordinates of a lat/lon pixel grid, plus what noise needs to know.

    wraps   True for a whole-globe map (east and west edges meet), False for a region
    px_rad  angular size of one pixel, in radians
    """

    def __new__(cls, x, y, z, wraps=True, px_rad=None):
        obj = super().__new__(cls, (x, y, z))
        obj.wraps = wraps
        obj.px_rad = px_rad if px_rad is not None else 2 * np.pi / x.shape[1]
        obj._coarse = {}
        return obj

    def coarse(self, factor):
        """Coordinates of factor x factor blocks (block centres line up with the full grid)."""
        if factor not in self._coarse:
            h, w = self[0].shape
            blocks = [c.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3)) for c in self]
            norm = np.sqrt(blocks[0] ** 2 + blocks[1] ** 2 + blocks[2] ** 2)
            self._coarse[factor] = tuple(b / norm for b in blocks)
        return self._coarse[factor]


def _as_grid(xyz):
    return xyz if isinstance(xyz, SphereXYZ) else SphereXYZ(*xyz)


def _upsample(a, factor, height, width, wraps=True):
    """Bilinear upsample; wraps east-west on whole-globe maps, clamps at the edges otherwise."""
    if factor == 1:
        return a
    p = np.pad(a, ((1, 1), (0, 0)), mode="edge")
    p = np.pad(p, ((0, 0), (1, 1)), mode="wrap" if wraps else "edge")
    ys = (np.arange(height) + 0.5) / factor + 0.5
    xs = (np.arange(width) + 0.5) / factor + 0.5
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    ty, tx = (ys - y0)[:, None], (xs - x0)[None, :]
    rows = p[y0] * (1 - ty) + p[y0 + 1] * ty
    return rows[:, x0] * (1 - tx) + rows[:, x0 + 1] * tx


def _factor_for(frequency, grid):
    """How much an octave can be downsampled: low frequencies are smooth, so need fewer samples."""
    height, width = grid[0].shape
    px_per_cell = 1.0 / (frequency * grid.px_rad)
    f = 1
    while f < 32 and px_per_cell / (f * 2) >= 5 and width % (f * 2) == 0 and height % (f * 2) == 0:
        f *= 2
    return f


def fbm(noise: Perlin3, xyz, frequency: float, octaves: int = 5,
        persistence: float = 0.5, lacunarity: float = 2.0, extra: int = 0) -> np.ndarray:
    """Fractal Brownian motion: layered noise, roughly in [-1, 1].

    `extra` adds finer octaves on top for high-resolution maps. They add detail
    without changing the overall scale, so a seed looks the same at any resolution.
    """
    xyz = _as_grid(xyz)
    height, width = xyz[0].shape
    layers = {}
    amp, freq, norm = 1.0, frequency, 0.0
    for octave in range(octaves + extra):
        shift = octave * 31.7  # de-align octaves so they don't all cross 0 at the same spot
        f = _factor_for(freq, xyz)
        x, y, z = xyz if f == 1 else xyz.coarse(f)
        layer = amp * noise(x * freq + shift, y * freq + shift, z * freq + shift)
        layers[f] = layers.get(f, 0) + layer
        if octave < octaves:
            norm += amp
        amp *= persistence
        freq *= lacunarity
    total = sum(_upsample(a, f, height, width, xyz.wraps) for f, a in layers.items())
    return total / norm * 1.6  # Perlin rarely reaches +-1, stretch it a bit


def ridged(noise: Perlin3, xyz, frequency: float, octaves: int = 5,
           persistence: float = 0.5, lacunarity: float = 2.0, extra: int = 0) -> np.ndarray:
    """Ridged multifractal noise in [0, 1]: sharp crests, like mountain ridges.

    Each octave is weighted by the one before (so detail gathers on the crests).
    Coarse octaves are computed at low resolution and upsampled as the octaves get finer.
    """
    xyz = _as_grid(xyz)
    height, width = xyz[0].shape
    amp, freq, norm = 1.0, frequency, 0.0
    total = weight = None
    current = None  # downsample factor the running arrays are stored at
    for octave in range(octaves + extra):
        shift = octave * 17.3
        f = _factor_for(freq, xyz)
        x, y, z = xyz if f == 1 else xyz.coarse(f)
        if total is None:
            total, weight, current = np.zeros_like(x), np.ones_like(x), f
        elif f != current:  # move the running sums up to the finer grid
            total = _upsample_between(total, current, f, height, width, xyz.wraps)
            weight = _upsample_between(weight, current, f, height, width, xyz.wraps)
            current = f
        n = 1.0 - np.abs(noise(x * freq + shift, y * freq + shift, z * freq + shift) * 1.6)
        n = np.clip(n, 0, 1) ** 2
        n *= weight
        weight = np.clip(n * 1.5, 0, 1)
        total += amp * n
        if octave < octaves:
            norm += amp
        amp *= persistence
        freq *= lacunarity
    return _upsample(total, current, height, width, xyz.wraps) / norm


def _upsample_between(a, from_factor, to_factor, height, width, wraps):
    """Resample an array stored at 1/from_factor resolution to 1/to_factor resolution."""
    full = _upsample(a, from_factor, height, width, wraps)
    if to_factor == 1:
        return full
    return _coarse_sample(full, to_factor)


def _coarse_sample(a, factor):
    """Average blocks of factor x factor pixels."""
    h, w = a.shape
    return a.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))
