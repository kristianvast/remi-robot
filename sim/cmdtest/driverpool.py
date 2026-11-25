"""Drivers hosted in worker processes, so drivers whose act() holds the GIL (torch on CUDA) run in parallel.

Each worker process loads the same driver (api.load_driver) once per car slot it serves and keeps that instance's
state; car threads call act()/reset() over the worker's pipe (one lock per worker). The CarObs and DriveCommand a
worker receives are the exact objects AuditedDriver checked, pickled; the PWM comes back unchanged. Only the place
where act() executes changes.
"""

import multiprocessing as mp
import threading


def _serve(conn, name, kw):
    from sim.cmdtest.api import load_driver

    drivers = {}
    while (msg := conn.recv()) is not None:
        op, slot, args = msg
        if op == "new":
            drivers[slot] = load_driver(name, **kw)
            d = drivers[slot]
            conn.send((getattr(d, "name", name), getattr(d, "command_style", "line_of_sight")))
        elif op == "reset":
            drivers[slot].reset(*args)
            conn.send(None)
        else:
            d = drivers[slot]
            out = d.act(*args)
            conn.send((tuple(int(v) for v in out), float(getattr(d, "front", float("inf")))))


class RemoteDriver:
    def __init__(self, conn, lock, slot):
        self.conn, self.lock, self.slot = conn, lock, slot
        self.name, self.command_style = self._call("new")
        self.front = float("inf")

    def _call(self, op, *args):
        with self.lock:
            self.conn.send((op, self.slot, args))
            return self.conn.recv()

    def reset(self, seed=None):
        self._call("reset", seed)

    def act(self, obs, cmd):
        pwm, self.front = self._call("act", obs, cmd)
        return pwm


class DriverPool:
    def __init__(self, name, workers, kw=None):
        ctx = mp.get_context("spawn")
        self.name = name
        self.links = []
        for _ in range(workers):
            parent, child = ctx.Pipe()
            ctx.Process(target=_serve, args=(child, name, kw or {}), daemon=True).start()
            self.links.append((parent, threading.Lock()))

    def driver(self, slot):
        conn, lock = self.links[slot % len(self.links)]
        return RemoteDriver(conn, lock, slot)

    def close(self):
        for conn, lock in self.links:
            with lock:
                conn.send(None)
