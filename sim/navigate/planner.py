"""Grid planner on the floor map: clearance margin, cost-to-go field per goal, obstacle marks."""

from math import atan2, hypot

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from sim.localize.range_map import range_map

STRIDE = 3            # 6 cm planning cells on the 2 cm floor map
MARGIN = .16          # hard clearance (car radius 0.135 m + 2.5 cm)
SOFT = .15            # extra cost decays over this distance beyond the margin
MARK_RADIUS = .20     # cells within this of a sensed obstacle are closed
MARK_LIFE = 60.       # seconds a sensed obstacle stays in the plan


class Planner:
    """Plan on the same map variant the localiser uses (drop_furniture for the mismatch test)."""

    def __init__(self, drop_furniture: bool = False, map_image=None):
        m = range_map(drop_furniture, map_image)
        self.floor = f = m.floor
        clearance = m.clearance[1::STRIDE, 1::STRIDE]
        self.shape = clearance.shape
        rr, cc = np.where(clearance >= MARGIN)
        self.key = np.full(self.shape, -1, np.int32)
        self.key[rr, cc] = np.arange(len(rr))
        self.rc = np.column_stack((rr, cc))
        self.xy = np.column_stack(f.xy(rr*STRIDE+1, cc*STRIDE+1))
        self.tree = cKDTree(self.xy)
        cost = 1 + 3*np.exp(-(clearance[rr, cc]-MARGIN)/SOFT)
        u, v, w = [], [], []
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            r, c = rr+dr, cc+dc
            ok = (r >= 0) & (r < self.shape[0]) & (c >= 0) & (c < self.shape[1])
            i = np.flatnonzero(ok)
            j = self.key[r[ok], c[ok]]
            i, j = i[j >= 0], j[j >= 0]
            u.append(i)
            v.append(j)
            w.append(hypot(dr, dc)*STRIDE*f.res*(cost[i]+cost[j])/2)
        self.u, self.v, self.w = map(np.concatenate, (u, v, w))
        self.marks: list[tuple[float, float, float]] = []   # x, y, expiry time
        self.goal = None
        self.field = self.next = None

    def node(self, x: float, y: float) -> int | None:
        """Nearest traversable cell within 0.6 m (an estimate may sit inside the margin)."""
        d, i = self.tree.query((x, y), distance_upper_bound=.6)
        return None if np.isinf(d) else int(i)

    def blocked_nodes(self, now: float) -> np.ndarray:
        self.marks = [m for m in self.marks if m[2] > now]
        if not self.marks:
            return np.zeros(0, int)
        hits = self.tree.query_ball_point(np.array([m[:2] for m in self.marks]), MARK_RADIUS)
        return np.unique(np.concatenate([np.asarray(h, int) for h in hits]))

    def plan(self, goal: tuple[float, float], now: float = 0.) -> bool:
        """Cost-to-go field to the goal. Returns False when the goal is unreachable."""
        self.goal = goal
        g = self.node(*goal)
        if g is None:
            return False
        keep = np.ones(len(self.u), bool)
        blocked = self.blocked_nodes(now)
        blocked = blocked[blocked != g]
        if len(blocked):
            closed = np.zeros(len(self.xy), bool)
            closed[blocked] = True
            keep = ~(closed[self.u] | closed[self.v])
        n = len(self.xy)
        graph = coo_matrix((self.w[keep], (self.u[keep], self.v[keep])), shape=(n, n)).tocsr()
        self.field, pred = dijkstra(graph, directed=False, indices=g, return_predecessors=True)
        self.next = pred
        self.goal_node = g
        return True

    def cost_to_go(self, x: float, y: float) -> float:
        i = self.node(x, y)
        return np.inf if i is None else float(self.field[i]) + hypot(*(self.xy[i]-(x, y)))

    def path(self, x: float, y: float, max_m: float = 40.) -> np.ndarray:
        """Cells from the estimate toward the goal (follows the field), as (n, 2) metres."""
        i = self.node(x, y)
        if i is None or np.isinf(self.field[i]):
            return np.zeros((0, 2))
        ids = [i]
        step = STRIDE*self.floor.res
        while ids[-1] != self.goal_node and len(ids)*step < max_m:
            ids.append(self.next[ids[-1]])
        if ids[-1] == self.goal_node:
            return np.vstack((self.xy[ids], self.goal))
        return self.xy[ids]

    def waypoint(self, pose, lookahead: float = .45) -> tuple[float, float, np.ndarray]:
        """(heading error rad, straight distance to goal m, path) for the estimated pose."""
        x, y, yaw = pose
        path = self.path(x, y, max_m=3.)
        if len(path) == 0:
            return np.nan, np.inf, path
        d = np.hypot(*(path-(x, y)).T)
        ahead = np.flatnonzero(d >= lookahead)
        target = path[ahead[0]] if len(ahead) else path[-1]
        err = (atan2(target[1]-y, target[0]-x)-yaw+np.pi) % (2*np.pi)-np.pi
        return err, hypot(self.goal[0]-x, self.goal[1]-y), path

    def line_of_sight(self, pose, spacing: float = .22, count: int = 12,
                      corridor: float = .10) -> tuple[float, float, np.ndarray]:
        """(heading error rad, distance to the target m, path) with the learned driver's training command.

        As train/driver_env.py: points every 0.22 m along the path; the target is the farthest of the
        first 12 whose run stays within 10 cm of the line through the third, at least the third.
        """
        x, y, yaw = pose
        path = self.path(x, y, max_m=3.)
        if len(path) == 0:
            return np.nan, np.inf, path
        along = np.concatenate(([0.], np.cumsum(np.hypot(*np.diff(path, axis=0).T))))
        s = np.minimum(np.arange(1, count + 1) * spacing, along[-1])
        rel = np.column_stack((np.interp(s, along, path[:, 0]), np.interp(s, along, path[:, 1]))) - (x, y)
        unit = rel[2] / max(hypot(*rel[2]), 1e-6)
        cross = np.abs(rel[:, 0] * unit[1] - rel[:, 1] * unit[0])
        k = int(np.clip(np.cumprod(cross < corridor).sum() - 1, 2, count - 1))
        err = (atan2(rel[k, 1], rel[k, 0])-yaw+np.pi) % (2*np.pi)-np.pi
        return err, hypot(*rel[k]), path

    def mark(self, points: np.ndarray, now: float) -> None:
        """Add sensed obstacle points (n, 2) in map metres; they close cells at the next plan()."""
        self.marks.extend((float(px), float(py), now+MARK_LIFE) for px, py in points)

    def clear_marks(self) -> None:
        self.marks.clear()
