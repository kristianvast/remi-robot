"""Horizontal ray lookup derived only from the occupancy image and closures."""

from functools import lru_cache
from math import tau
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from sim.macsim.world import Floor, floor

STRIDE = 5
ANGLES = 144
MAX_RAY = 7.0


class RangeMap:
    """Quantize starts to 10 cm/2.5°; retain 2 cm occupancy in the ray march."""

    def __init__(self, drop_furniture: bool = False, image: Path | None = None):
        # image: what the car believes (e.g. a merged obstacle map); the simulated world stays the scan.
        self.floor = f = floor() if image is None else Floor(image)
        self.h, self.w = f.image.shape
        clearance = f.clearance
        if drop_furniture:
            # Erase interior occupied blobs under 1 m2 (the furniture) from this map only,
            # so the simulated world contains obstacles the localiser does not know about.
            occupied = (~f.free).astype(np.uint8)
            n, labels, stats, _ = cv2.connectedComponentsWithStats(occupied, connectivity=8)
            x, y, w, h, area = stats.T
            inner = (x > 0) & (y > 0) & (x + w < self.w) & (y + h < self.h)
            small = np.flatnonzero(inner & (area * f.res**2 < 1.0))
            free = f.free | np.isin(labels, small[small > 0])
            clearance = (distance_transform_edt(free) * f.res - f.res / 2).astype(np.float32)
        self.clearance = clearance
        self.owner = np.full(((self.h+4)//5, (self.w+4)//5), -1, np.int32)
        grid = (clearance > 0)[2::STRIDE, 2::STRIDE]
        rr, cc = np.nonzero(grid)
        self.owner[rr, cc] = np.arange(len(rr))
        self.table = np.empty((len(rr), ANGLES), np.uint16)
        az = np.arange(ANGLES)*(tau/ANGLES)
        cos, sin = np.cos(az).astype(np.float32), np.sin(az).astype(np.float32)
        for i in range(0, len(rr), 512):
            rc = rr[i:i+512]*STRIDE+2.5
            cc0 = cc[i:i+512]*STRIDE+2.5
            ray = np.zeros((len(rc), ANGLES), np.float32)
            ir, ia = np.indices(ray.shape)
            ir, ia = ir.ravel(), ia.ravel()
            d = np.zeros(len(ir), np.float32)
            while len(d):
                r = (rc[ir] - d*sin[ia]/f.res).astype(int)
                c = (cc0[ir] + d*cos[ia]/f.res).astype(int)
                ahead = clearance[np.clip(r, 0, self.h-1), np.clip(c, 0, self.w-1)]
                live = (ahead > 0) & (d < MAX_RAY) & (r >= 0) & (c >= 0) & (r < self.h) & (c < self.w)
                ray[ir[~live], ia[~live]] = np.minimum(d[~live], MAX_RAY)
                ir, ia = ir[live], ia[live]
                d = d[live] + np.maximum(.012, ahead[live] - .015)
            self.table[i:i+len(rc)] = np.round(ray*1000)

    def ranges(self, x, y, a) -> np.ndarray:
        r, c = self.floor.cells(x, y)
        r, c = np.clip(r, 0, self.h-1), np.clip(c, 0, self.w-1)
        key = self.owner[r//STRIDE, c//STRIDE]
        angle = np.rint(a*(ANGLES/tau)).astype(np.int32) % ANGLES
        return np.where(key[:, None] >= 0, self.table[key[:, None], angle]*.001, .06)


@lru_cache(maxsize=4)
def range_map(drop_furniture: bool = False, image: Path | None = None) -> RangeMap:
    return RangeMap(drop_furniture, image)
