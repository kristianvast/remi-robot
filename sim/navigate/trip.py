"""One navigator trip in macsim: depth corruptions on, car response randomised, truth only recorded."""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from sim.macsim.simulator import Simulator

from .navigator import DT, Navigator, Status


@dataclass
class Kidnap:
    after_m: float                      # true distance driven before the car is carried away
    candidates: np.ndarray              # (n, 2) free spots; one 3-8 m from the car is chosen
    seed: int
    done_t: float | None = None
    to: tuple[float, float, float] | None = None

    def choose(self, x: float, y: float) -> tuple[float, float, float] | None:
        rng = np.random.default_rng(self.seed)
        d = np.hypot(*(self.candidates - (x, y)).T)
        ok = np.flatnonzero((d >= 3) & (d <= 8))
        if not len(ok):
            return None
        cx, cy = self.candidates[rng.choice(ok)]
        return float(cx), float(cy), float(rng.uniform(-np.pi, np.pi))


@dataclass
class Trip:
    t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    truth: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    est: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    conf: np.ndarray = field(default_factory=lambda: np.zeros(0))
    map_fit: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # per step: mode mass, stable steps, sensor age, spread m, bad_fit, wall_bad (certainty's ingredients)
    parts: np.ndarray = field(default_factory=lambda: np.zeros((0, 6)))
    state: list[str] = field(default_factory=list)
    contacts: int = 0
    driven_m: float = 0.
    final: Status | None = None
    planned: np.ndarray | None = None
    nav_stats: dict = field(default_factory=dict)
    depth: list = field(default_factory=list)


def run_trip(start: tuple[float, float, float], goal, start_room: str | None, *, seed: int,
             driver: str = "hand", drop_furniture: bool = False, car_turn: bool = True,
             kidnap: Kidnap | None = None, map_image=None,
             time_limit: float = 300., keep_depth_every: int = 0,
             on_status: Callable[[Status], None] | None = None) -> Trip:
    sim = Simulator(seed=seed, randomize=True, corrupt=True)
    sim.reset(start)
    nav = Navigator(goal, start_room, driver=driver, car_turn=car_turn, drop_furniture=drop_furniture,
                    map_image=map_image, time_limit=time_limit, seed=seed + 1)
    trip = Trip()
    rows, states = [], []
    obs = sim.observe()
    bumped_before = False
    last = np.array(start[:2])
    i = 0
    while True:
        left, right, head, status = nav.step(obs)
        rows.append((status.t, *sim.pose, *status.pose, status.confidence, status.map_fit,
                     nav.est.mass, nav.loc.stable, nav.loc.sensor_age, nav.est.spread_m, nav.loc.bad_fit,
                     nav.loc.wall_bad))
        states.append(status.state)
        if keep_depth_every and i % keep_depth_every == 0:
            trip.depth.append(obs.depth)
        if on_status:
            on_status(status)
        if status.state in ("arrived", "failed"):
            break
        sim.set_head(head)
        sim.step(left, right, dt=DT)
        if sim.bumped and not bumped_before:
            trip.contacts += 1
        bumped_before = sim.bumped
        pos = np.array(sim.pose[:2])
        trip.driven_m += float(np.linalg.norm(pos - last))
        last = pos
        if kidnap and kidnap.done_t is None and trip.driven_m >= kidnap.after_m:
            kidnap.to = kidnap.choose(*sim.pose[:2])
            if kidnap.to is not None:
                sim.pose = kidnap.to
                last = np.array(kidnap.to[:2])
            kidnap.done_t = status.t
        obs = sim.observe()
        i += 1
    a = np.array(rows)
    trip.t, trip.truth, trip.est, trip.conf, trip.map_fit = a[:, 0], a[:, 1:4], a[:, 4:7], a[:, 7], a[:, 8]
    trip.parts = a[:, 9:15]
    trip.state = states
    trip.final = status
    trip.planned = nav.planned_path
    trip.nav_stats = dict(recoveries=nav.recoveries, replans=nav.replans,
                          localise_s=round(nav.localise_s, 1), first_confident_s=nav.first_confident,
                          map_wrong_cells=len(nav.map_wrong),
                          veto_frac=round(nav.driver.vetoes / max(nav.driver.steps, 1), 3)
                          if hasattr(nav.driver, "vetoes") else "")
    return trip
