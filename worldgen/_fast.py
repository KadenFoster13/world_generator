"""Compiled versions of the slow pure-Python loops (drainage and flow accumulation).

Used automatically when numba is installed (`pip install worldgen[fast]` or
`pip install numba`); otherwise hydrology.py runs the plain Python versions, which give
identical results, just slower. The first run after installing compiles them (a few
seconds); the compiled code is cached for later runs.
"""
import heapq

import numpy as np

try:
    import numba
except ImportError:  # optional dependency
    numba = None

HAVE_NUMBA = numba is not None

if HAVE_NUMBA:
    @numba.njit(cache=True)
    def flood_kernel(z, seeds, visited, h, w, wraps=True):
        """Priority flood; see hydrology.priority_flood. Same order and ties as the Python version."""
        n = h * w
        filled = z.copy()
        receiver = np.full(n, -1, np.int64)
        order = np.empty(n, np.int64)
        count = 0
        first = seeds[0]
        heap = [(z[first], z[first], first)]
        for k in range(1, len(seeds)):
            s = seeds[k]
            heap.append((z[s], z[s], s))
        heapq.heapify(heap)
        dys = (-1, -1, -1, 0, 0, 1, 1, 1)
        dxs = (-1, 0, 1, -1, 1, -1, 0, 1)
        while len(heap) > 0:
            fc, _, c = heapq.heappop(heap)
            y = c // w
            x = c - y * w
            for k in range(8):
                yy = y + dys[k]
                if yy < 0 or yy >= h:
                    continue
                xx = x + dxs[k]
                if not wraps and (xx < 0 or xx >= w):
                    continue
                j = yy * w + xx % w
                if visited[j]:
                    continue
                visited[j] = 1
                receiver[j] = c
                fj = z[j] if z[j] > fc else fc
                filled[j] = fj
                order[count] = j
                count += 1
                heapq.heappush(heap, (fj, z[j], j))
        return filled, receiver, order[:count]

    @numba.njit(cache=True)
    def accumulate_kernel(flow, receiver, order, ok):
        """Sum flow downstream; see hydrology.accumulate."""
        for i in range(len(order) - 1, -1, -1):
            c = order[i]
            r = receiver[c]
            if r >= 0 and ok[c]:
                flow[r] += flow[c]
        return flow
