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
], dtype=np.float64)
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
        x = x + self.offset[0]
        y = y + self.offset[1]
        z = z + self.offset[2]
        xi0 = np.floor(x)
        yi0 = np.floor(y)
        zi0 = np.floor(z)
        xf, yf, zf = x - xi0, y - yi0, z - zi0
        xi = xi0.astype(np.int64) & 255
        yi = yi0.astype(np.int64) & 255
        zi = zi0.astype(np.int64) & 255
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


_coarse_cache = {}


def _coarse_xyz(height, width, factor):
    """Sphere coordinates at 1/factor resolution (pixel centres line up with the full grid)."""
    key = (height, width, factor)
    if key not in _coarse_cache:
        w, h = width // factor, height // factor
        lon = (np.arange(w) + 0.5) / w * 2 * np.pi - np.pi
        lat = np.pi / 2 - (np.arange(h) + 0.5) / h * np.pi
        lon, lat = np.meshgrid(lon, lat)
        _coarse_cache[key] = (np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat))
    return _coarse_cache[key]


def _upsample(a, factor, height, width):
    """Bilinear upsample that wraps east-west and clamps at the poles."""
    if factor == 1:
        return a
    p = np.pad(a, ((1, 1), (0, 0)), mode="edge")
    p = np.pad(p, ((0, 0), (1, 1)), mode="wrap")
    ys = (np.arange(height) + 0.5) / factor + 0.5
    xs = (np.arange(width) + 0.5) / factor + 0.5
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    ty, tx = (ys - y0)[:, None], (xs - x0)[None, :]
    rows = p[y0] * (1 - ty) + p[y0 + 1] * ty
    return rows[:, x0] * (1 - tx) + rows[:, x0 + 1] * tx


def _factor_for(frequency, width, height):
    """How much an octave can be downsampled: low frequencies are smooth, so need fewer samples."""
    px_per_cell = width / (2 * np.pi * frequency)
    f = 1
    while f < 8 and px_per_cell / (f * 2) >= 5 and width % (f * 2) == 0 and height % (f * 2) == 0:
        f *= 2
    return f


def fbm(noise: Perlin3, xyz, frequency: float, octaves: int = 5,
        persistence: float = 0.5, lacunarity: float = 2.0) -> np.ndarray:
    """Fractal Brownian motion: layered noise, roughly in [-1, 1]."""
    height, width = xyz[0].shape
    layers = {}
    amp, freq, norm = 1.0, frequency, 0.0
    for octave in range(octaves):
        shift = octave * 31.7  # de-align octaves so they don't all cross 0 at the same spot
        f = _factor_for(freq, width, height)
        x, y, z = xyz if f == 1 else _coarse_xyz(height, width, f)
        layer = amp * noise(x * freq + shift, y * freq + shift, z * freq + shift)
        layers[f] = layers.get(f, 0) + layer
        norm += amp
        amp *= persistence
        freq *= lacunarity
    total = sum(_upsample(a, f, height, width) for f, a in layers.items())
    return total / norm * 1.6  # Perlin rarely reaches +-1, stretch it a bit


def ridged(noise: Perlin3, xyz, frequency: float, octaves: int = 5,
           persistence: float = 0.5, lacunarity: float = 2.0) -> np.ndarray:
    """Ridged multifractal noise in [0, 1]: sharp crests, like mountain ridges."""
    x, y, z = xyz
    total = np.zeros_like(x)
    amp, freq, norm = 1.0, frequency, 0.0
    weight = np.ones_like(x)
    for octave in range(octaves):
        shift = octave * 17.3
        n = 1.0 - np.abs(noise(x * freq + shift, y * freq + shift, z * freq + shift) * 1.6)
        n = np.clip(n, 0, 1) ** 2
        n *= weight
        weight = np.clip(n * 1.5, 0, 1)
        total += amp * n
        norm += amp
        amp *= persistence
        freq *= lacunarity
    return total / norm
