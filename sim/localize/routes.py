"""Build deterministic clearance-aware evaluation routes from the floor plan."""

from dataclasses import dataclass
from math import atan2, degrees

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from sim.macsim.world import floor, scene_meta


@dataclass(frozen=True, slots=True)
class Route:
    index: int
    start_area: str
    goal_area: str
    points: np.ndarray
    yaw: float
    length: float


class Planner:
    def __init__(self):
        self.floor = f = floor()
        stride = 5
        clearance = f.clearance[2::stride, 2::stride]
        rr, cc = np.where(clearance >= .205)
        keys = np.full(clearance.shape, -1, np.int32)
        keys[rr, cc] = np.arange(len(rr))
        self.xy = np.column_stack(f.xy(rr*stride+2, cc*stride+2))
        self.tree = cKDTree(self.xy)
        u, v, weights = [], [], []
        for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            r, c = rr+dr, cc+dc
            ok = (r >= 0) & (r < keys.shape[0]) & (c >= 0) & (c < keys.shape[1])
            ii = np.flatnonzero(ok)
            jj = keys[r[ok], c[ok]]
            sel = jj >= 0
            ii, jj = ii[sel], jj[sel]
            mid = (self.xy[ii]+self.xy[jj])*.5
            sel = f.valid(*mid.T, .194)
            ii, jj = ii[sel], jj[sel]
            u.extend(ii)
            v.extend(jj)
            low = np.minimum(clearance[rr[ii], cc[ii]], clearance[rr[jj], cc[jj]])
            weights.extend(np.hypot(dr, dc)*.1*(1 + .22*np.exp(-np.maximum(low-.20, 0)/.16)))
        graph = coo_matrix((weights, (u, v)), shape=(len(rr), len(rr))).tocsr()
        meta = scene_meta()
        goals = meta["goals"]
        if isinstance(goals, dict):
            goals = list(goals.values())
        self.g = self.tree.query([[g["x"], g["y"]] for g in goals])[1]
        self.dist, self.parents = dijkstra(graph, directed=False, indices=self.g, return_predecessors=True)
        self.node_labels = f.labels[2+rr*stride, 2+cc*stride]

    def make(self, index: int, node: int, goal: int, yaw: float | None) -> Route:
        ids = [node]
        while ids[-1] != self.g[goal]:
            ids.append(self.parents[goal, ids[-1]])
        xy = self.xy[ids]
        ds = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        length = float(ds.sum())
        end = np.r_[0, np.cumsum(ds)]
        sampled = np.minimum(np.r_[np.arange(0, length, .11), length], length)
        points = np.column_stack([np.interp(sampled, end, v) for v in xy.T])
        if yaw is None:
            delta = points[2]-points[0]
            yaw = atan2(delta[1], delta[0])
        return Route(index, self.floor.area(*points[0]), self.floor.area(*points[-1]), points, yaw, length)

    def routes(self, global_starts: bool) -> list[Route]:
        rng = np.random.default_rng(602911 if global_starts else 240112)
        output = []
        for label in range(1, len(self.floor.rooms)+1):
            nodes = np.flatnonzero(self.node_labels == label)
            for node in rng.permutation(nodes):
                # Reject routes that are too short after geometric reconstruction,
                # and make sure tracking samples travel between named areas.
                choices = np.flatnonzero((self.dist[:, node] >= 11.6) & (self.dist[:, node] < 30) &
                                         (self.node_labels[self.g] != label))
                for g in rng.permutation(choices):
                    yaw = float(rng.uniform(-np.pi, np.pi)) if global_starts else None
                    route = self.make(len(output), int(node), int(g), yaw)
                    if 10.25 <= route.length <= 29.75:
                        output.append(route)
                        break
                if len(output) == label*(5 if global_starts else 3):
                    break
            assert len(output) == label*(5 if global_starts else 3), label
        return output


class Driver:
    """Ground-truth pure pursuit for repeatable excitation, not a navigation demo."""

    def __init__(self, route: Route):
        self.points = route.points
        self.index = 0
        self.done = False

    def command(self, pose) -> tuple[int, int]:
        pos = np.array(pose[:2])
        points = self.points
        if self.index >= len(points)-5 and np.linalg.norm(points[-1]-pos) < .13:
            self.done = True
            return 0, 0
        for _ in range(4):
            if self.index < len(points)-1 and np.linalg.norm(points[self.index]-pos) < .15:
                self.index += 1
        target = points[min(len(points)-1, self.index+1)] - pos
        error = (atan2(target[1], target[0])-pose[2]+np.pi) % (2*np.pi)-np.pi
        deg = degrees(error)
        if abs(deg) > 48:
            turn = 62 if deg > 0 else -62
            return -turn, turn
        base = int(np.interp(np.linalg.norm(target), [.05, .18, .24], [40, 81, 107]))
        turn = max(-49, min(49, round(1.25*deg)))
        return base-turn, base+turn
