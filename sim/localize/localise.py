"""Sensor-only Monte Carlo localization in floor-map coordinates."""

from dataclasses import dataclass
from math import cos, exp, pi, radians, sin, tau

import numpy as np

from sim.macsim.depth import FX, PAN_X, Z_FAR, Z_NEAR
from sim.macsim.drive import Drive
from sim.macsim.world import Pose

from .range_map import MAX_RAY, range_map

DT = .1
PITCH = radians(20)
# A few pixels above the nominal horizon (row 105.2) to avoid floor returns.
COLS = np.arange(7, 155, 9)
AXIS = cos(PITCH) - sin(PITCH)*(60-101.5)/FX
LATERAL = (COLS+.5-80)/FX
FAN = np.arctan2(-LATERAL, AXIS)
FACTOR = np.hypot(AXIS, LATERAL)
CONE = np.deg2rad(np.arange(-15, 16, 5))
LOG_SPAN = np.log(Z_FAR/Z_NEAR)
STABLE_STEPS = 50  # an estimate that jumped > 0.5 m regains full confidence over 5 s
BAD_FIT_MAX = 24
WALL_BAD_MAX = 24
EARN_FIT = 6  # certainty is earned when bad_fit <= this (chosen on dev seeds, .omo/waves/p3/confidence.md)
SETTLE_S = 5.  # or earned with any fit once the cloud held >= 0.5 this long without seeing through a wall
BEYOND_LOG = .3  # a view reaching ~35 % beyond the map's wall (more than depth's scale error) sees through it


def wrap(angle):
    return (angle+pi) % tau-pi


@dataclass(frozen=True, slots=True)
class Estimate:
    """Two separate questions: "do I know where I am" (confidence: the particle cloud, the
    see-through-walls check, and a fit good enough to earn it once) and "does what I sense match the
    map here" (map_fit). A tracked estimate next to furniture the map draws differently keeps high
    confidence with a low map_fit."""
    pose: Pose
    confidence: float  # position certainty: heuristic cloud concentration, not a calibrated probability
    look_around: float | None  # absolute pan target in CCW radians; None = no request
    map_fit: float = 1.  # 1 = the sensors fit the map at the estimate, 0 = persistently bad fit
    spread_m: float = 0.  # weighted RMS distance of the estimate's mode particles from the estimate
    spread_deg: float = 0.  # weighted RMS heading deviation of the same particles
    mass: float = 1.  # particle weight in the estimate's mode (low = several places compete)
    structure_fit: float = 1.  # 1 = the sensors never see beyond the map's walls at the estimate


class Localizer:
    """Supply one update every 100 ms, with the PWM sent over that interval.

    Gyro is integrated relative CCW yaw in radians. Range is integer cm or None.
    Depth is encoded (120,160) or None; passes 100 ms behind gyro and head by
    default. The current measured/settled head angle is CCW radians. Never
    pass simulator truth except the deliberately known-pose reset prior.
    """

    def __init__(self, *, seed: int = 0, particles: int = 1800, camera_lag: bool = True,
                 drop_furniture: bool = False, map_image=None):
        self.map = range_map(drop_furniture, map_image)
        # Walls and large blobs only: unmapped or moved furniture shortens views but never lengthens them.
        self.walls = range_map(True, map_image)
        self.floor = self.map.floor
        self.rng = np.random.default_rng(seed)
        self.n = particles
        self.camera_lag = camera_lag
        self.reset()

    def reset(self, *, pose: Pose | None = None, room: str | None = None) -> None:
        """Initialize globally, in an area by id/name, or near a supplied pose."""
        if pose is not None and room is not None:
            raise ValueError("Specify one starting prior")
        valid = self.map.clearance[::4, ::4] >= .145
        self.global_rc = np.column_stack(np.where(valid))*4
        if room is not None:
            matches = [i for i, r in enumerate(self.floor.rooms, 1) if room.casefold() in
                       (r["id"].casefold(), r["name"].casefold())]
            if len(matches) != 1:
                raise ValueError(f"Unknown or ambiguous room: {room}")
            rc = self.global_rc
            self.prior_rc = rc[self.floor.labels[tuple(rc.T)] == matches[0]]
        else:
            self.prior_rc = self.global_rc
        self.p = self._uniform(self.n if pose is not None else 7*self.n, self.prior_rc)
        if pose is not None:
            self.p[:, :3] = pose + self.rng.normal(size=(self.n, 3))*[.025, .025, .025]
        self.weights = np.full(len(self.p), 1/len(self.p))
        self.previous_heading = self.previous_head = 0.
        self.previous_delta = np.zeros((len(self.p), 3))
        self.drive = Drive()
        self.gyro_is_set = False
        self.i = 0
        self.sensor_age = 0
        self.scan = 0
        self.bad_fit = 0
        self.wall_bad = 0
        self.certain = False
        self.settled_s = 0.
        self.last_pose = pose
        self.start_pose = pose
        self.last_xy = None if pose is None else np.array(pose[:2], float)
        self.stable = STABLE_STEPS if pose is not None else 0

    def _valid(self, x, y, radius: float = .10):
        r, c = self.floor.cells(x, y)
        h, w = self.map.clearance.shape
        inside = (r >= 0) & (r < h) & (c >= 0) & (c < w)
        return inside & (self.map.clearance[np.clip(r, 0, h - 1), np.clip(c, 0, w - 1)] >= radius)

    def _uniform(self, n, rc) -> np.ndarray:
        r, c = rc[self.rng.integers(len(rc), size=n)].T
        x, y = self.floor.xy(r, c)
        return np.column_stack((x, y, self.rng.uniform(-pi, pi, n),
                                self.rng.normal(1, .14, n), self.rng.uniform(-.27, .27, n),
                                self.rng.normal(0, .002, n)))

    def _motion(self, command, gyro):
        raw = wrap(gyro-self.previous_heading) if self.gyro_is_set else 0.
        self.previous_heading = gyro
        self.gyro_is_set = True
        ds = da = 0.
        for _ in range(10):
            v, w = self.drive.tick(command)
            ds += .01*v
            da += .01*w
        yaw = da if abs(raw) >= radians(24.3) and abs(da) > abs(raw) else raw
        p, rng = self.p, self.rng
        n = len(p)
        theta = yaw + DT*p[:, 5] + rng.normal(0, .003+abs(yaw)*.025, n)
        dist = ds*p[:, 3] + rng.normal(0, .004+abs(ds)*.12, n)
        a = p[:, 2] + theta/2
        dx, dy = dist*np.cos(a), dist*np.sin(a)
        valid = self._valid(p[:, 0]+dx, p[:, 1]+dy)
        dx, dy = dx*valid, dy*valid
        p[:, 0] += dx
        p[:, 1] += dy
        p[:, 2] = wrap(p[:, 2] + theta)
        p[:, 3] = np.clip(p[:, 3] + rng.normal(0, .004, n), .60, 1.42)
        p[:, 5] = np.clip(p[:, 5] + rng.normal(0, .00012, n), -.008, .008)
        self.previous_delta = np.column_stack((dx, dy, theta))

    def _predict(self, poses, head, angles, offset, rays=None):
        x, y, yaw = poses.T
        a = yaw + head
        return (rays or self.map).ranges(x + PAN_X*np.cos(yaw) + offset*np.cos(a),
                               y + PAN_X*np.sin(yaw) + offset*np.sin(a),
                               a[:, None] + angles)

    def _likelihood(self, cm, depth, head) -> tuple[np.ndarray, bool]:
        p = self.p
        score = np.zeros(len(p))
        informative = False
        # Decimate repeated pictures; scale/shape and scene errors are correlated.
        if depth is not None and self.i % 3 == 0:
            fan = np.median(depth[99:104, COLS], axis=0)
            if np.isfinite(fan).all() and np.ptp(depth[94:108]) >= .035 and np.count_nonzero(fan > .035) >= 5:
                obs = np.log(Z_NEAR) + (1-fan)*LOG_SPAN + np.log(FACTOR)
                old = p[:, :3] - self.previous_delta if self.camera_lag else p[:, :3]
                pan = self.previous_head if self.camera_lag else head
                predicted = self._predict(old, pan, FAN, -.0028)
                mu = np.log(np.maximum(predicted, .15))
                diff = obs[None, :] - mu
                valid = (fan > .025) & (fan < .99)
                residual = np.median(diff[:, valid], axis=1) if valid.sum() >= 4 else p[:, 4]
                correction = np.clip(residual - p[:, 4], -.25, .25)
                aligned = diff - (p[:, 4] + .65*correction)[:, None]
                aligned[:, fan <= .025] = np.maximum(aligned[:, fan <= .025], 0)
                aligned[:, fan >= .99] = np.minimum(aligned[:, fan >= .99], 0)
                # Clip impact of mismatching furniture and reject entire
                # frames with poor agreement under every plausible hypothesis.
                loss = np.mean(np.minimum((aligned/(.20+.055/np.maximum(predicted, .2)))**2, 5), axis=1)
                cost = loss + .35*np.minimum(((residual-p[:, 4])/.25)**2, 5)
                if np.min(cost) < 2.4:
                    score -= np.minimum(4, 3.7*cost)
                    p[:, 4] = np.clip(p[:, 4]+.055*correction*(cost < 1.9), -.43, .43)
                    informative = True
        if cm is not None and self.i % 3 == 0:
            sonar = np.minimum(1.5, self._predict(p[:, :3], head, CONE, .022).min(axis=1))
            score -= np.minimum(1.7, .65*((sonar-cm*.01)/.13)**2)
            informative = True
        return score, informative

    def _sees_through_walls(self, cm, depth, head) -> bool | None:
        """Does the view at the last estimate reach beyond the walls of the furniture-free map?
        None = nothing to judge. Furniture the map lacks or draws wrongly only shortens views, so a
        view through a wall says the estimate is in the wrong place, not that the map is approximate."""
        if self.last_pose is None or self.i % 3:
            return None
        pose = np.array([self.last_pose])
        bias = float(np.dot(self.weights, self.p[:, 4]))
        beyond = None
        if depth is not None:
            fan = np.median(depth[99:104, COLS], axis=0)
            valid = np.isfinite(fan) & (fan > .025) & (fan < .99)
            if valid.sum() >= 4:
                obs = np.log(Z_NEAR) + (1-fan)*LOG_SPAN + np.log(FACTOR)
                walls = self._predict(pose, self.previous_head if self.camera_lag else head, FAN, -.0028,
                                      self.walls)[0]
                far = valid & (walls < .9*MAX_RAY) & (obs - bias > np.log(np.maximum(walls, .15)) + BEYOND_LOG)
                beyond = bool(far.sum() >= max(2, .3*valid.sum()))
        if cm is not None:
            walls = min(1.5, float(self._predict(pose, head, CONE, .022, self.walls).min()))
            sonar = walls < 1.2 and cm*.01 > walls + .3
            beyond = sonar or bool(beyond)
        return beyond

    def _estimate(self) -> tuple[Pose, float, float, float]:
        p, w = self.p, self.weights
        # Choose the densest occupied SE(2) mode, not a mean across rooms.
        groups = np.column_stack((np.floor(p[:, :2]/.48), np.floor((p[:, 2]+pi)/.45))).astype(int)
        keys = (groups[:, 0]+64)*4096 + (groups[:, 1]+64)*32 + groups[:, 2]
        _, inverse = np.unique(keys, return_inverse=True)
        mode = np.argmax(np.bincount(inverse, weights=w))
        center = np.average(p[inverse == mode, :2], axis=0, weights=w[inverse == mode])
        angle = np.angle(np.sum(w[inverse == mode] * np.exp(1j*p[inverse == mode, 2])))
        near = (np.sum((p[:, :2]-center)**2, axis=1) < .48**2) & (abs(wrap(p[:, 2]-angle)) < .55)
        mass = float(w[near].sum())
        xy = np.average(p[near, :2], axis=0, weights=w[near])
        az = np.angle(np.sum(w[near]*np.exp(1j*p[near, 2])))
        wn = w[near] / mass
        spread_m = float(np.sqrt(np.dot(wn, np.sum((p[near, :2] - xy)**2, axis=1))))
        spread_deg = float(np.degrees(np.sqrt(np.dot(wn, wrap(p[near, 2] - az)**2))))
        if not self._valid(*xy):
            xy = center
        jumped = self.last_xy is None or np.hypot(*(xy - self.last_xy)) > .5
        self.stable = 0 if jumped else min(STABLE_STEPS, self.stable + 1)
        self.last_xy = np.asarray(xy, float)
        return (float(xy[0]), float(xy[1]), float(az)), mass, spread_m, spread_deg

    def update(self, command: tuple[int, int] | None, gyro: float, range: int | None,
               depth: np.ndarray | None, head_angle: float) -> Estimate:
        self.i += 1
        self._motion(command, gyro)
        through = self._sees_through_walls(range, depth, head_angle)
        if through is not None:
            self.wall_bad = min(WALL_BAD_MAX, max(0, self.wall_bad + (3 if through else -1)))
        score, informative = self._likelihood(range, depth, head_angle)
        self.previous_head = head_angle
        if informative:
            self.sensor_age = 0
            self.weights *= np.exp(score - score.max())
            self.weights /= self.weights.sum()
            # Fit of the measurement under the updated posterior: high when the estimate
            # explains what the sensors see, low after a kidnap or a map mismatch.
            fit = np.dot(self.weights, score)
            self.bad_fit = min(BAD_FIT_MAX, max(0, self.bad_fit + (3 if fit < -3.7 else -2)))
        else:
            self.sensor_age += 1
        pose, mass, spread_m, spread_deg = self._estimate()
        self.last_pose = pose
        held = mass * exp(-max(0, self.sensor_age-10)/30) * self.stable / STABLE_STEPS
        # Certainty: one compact mode holding the weight, not recently jumped, recently updated, and
        # not seeing through the map's walls. A fresh convergence must also fit the map (a wrong mode
        # often fits badly); once earned, a bad fit alone (furniture the map draws wrongly) does not
        # revoke it. How well the sensors fit the full map is map_fit, reported apart.
        cloud = held * exp(-max(0., spread_m-.15)/.15) * (1 - self.wall_bad/WALL_BAD_MAX)
        self.settled_s = self.settled_s + DT if cloud >= .5 and self.wall_bad == 0 else 0.
        earned = self.bad_fit <= EARN_FIT or self.settled_s >= SETTLE_S
        self.certain = cloud >= .5 and (self.certain or earned)
        confidence = cloud if self.certain else cloud * exp(-self.bad_fit/24)
        # The filter keeps exploring (injection, wider jitter, head scan) while either is in doubt.
        explore = held * exp(-self.bad_fit/24)
        if informative and 1/np.dot(self.weights,self.weights) < .72*len(self.p):
            n = self.n if self.i >= 23 else len(self.p)
            # Rejuvenate any mode, but retain a small exploration budget for
            # cold start / kidnapping instead of permanently deleting a room.
            k = int(n*(.025 if self.i > 25 and (explore < .50 or self.bad_fit > 8) else 0))
            cdf = np.cumsum(self.weights)
            cdf[-1] = 1.
            count = n-k
            ind = np.searchsorted(cdf, (np.arange(count)+self.rng.random())/count)
            carry = self.p[ind].copy()
            xy = carry[:, :2] + self.rng.normal(0, .022 if explore > .65 else .042, (count, 2))
            carry[:, :2] = np.where(self._valid(*xy.T)[:, None], xy, carry[:, :2])
            carry[:, 2] = wrap(carry[:, 2]+self.rng.normal(0, .016 if explore > .65 else .031, count))
            pool = self.prior_rc if self.i < 45 else self.global_rc
            self.p = np.vstack((carry, self._uniform(k, pool))) if k else carry
            self.weights = np.full(n, 1/n)
        look = None
        if explore < .72:
            if self.i % 10 == 1:
                self.scan += 1
            look = radians((0, 68, -68, 0, -42, 42)[self.scan % 6])
        return Estimate(pose, float(confidence), look, 1 - self.bad_fit/BAD_FIT_MAX, spread_m, spread_deg, mass,
                        1 - self.wall_bad/WALL_BAD_MAX)
