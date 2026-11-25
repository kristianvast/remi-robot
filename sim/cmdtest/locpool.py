"""sim/localize's Localizer in one worker process per car slot, so the cars' particle filters run in parallel.

The navigator builds `Localizer(...)` itself; install() replaces the name in sim.navigate.navigator with a factory
that returns a proxy bound to the calling car thread's worker (bind()). The worker runs the unchanged Localizer;
reset() and update() go over a pipe, and `.map` (read-only, used by the navigator's obstacle marking) is a local
copy of the same cached range map. Same inputs, same code, same seed: only where it runs changes.
"""

import multiprocessing as mp
import threading

_local = threading.local()


def _serve(conn):
    from sim.localize.localise import Localizer

    loc = None
    while (msg := conn.recv()) is not None:
        op, kw = msg
        if op == "new":
            loc = Localizer(**kw)
            conn.send(None)
        else:
            conn.send(getattr(loc, op)(**kw))


class LocalizerProxy:
    def __init__(self, conn, **kw):
        from sim.localize.range_map import range_map

        self.conn = conn
        self.map = range_map(kw.get("drop_furniture", False), kw.get("map_image"))
        self.floor = self.map.floor
        self._call("new", **kw)

    def _call(self, op, **kw):
        self.conn.send((op, kw))
        return self.conn.recv()

    def reset(self, *, pose=None, room=None):
        return self._call("reset", pose=pose, room=room)

    def update(self, command, gyro, range, depth, head_angle):
        return self._call("update", command=command, gyro=gyro, range=range, depth=depth, head_angle=head_angle)


class Pool:
    def __init__(self, slots):
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for _ in range(slots):
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=_serve, args=(child,), daemon=True)
            proc.start()
            self.conns.append(parent)
            self.procs.append(proc)

    def install(self):
        import sim.navigate.navigator as navigator

        def factory(**kw):
            conn = getattr(_local, "conn", None)
            if conn is None:
                from sim.localize.localise import Localizer

                return Localizer(**kw)
            return LocalizerProxy(conn, **kw)

        navigator.Localizer = factory

    def bind(self, slot):
        _local.conn = self.conns[slot]

    def close(self):
        for conn in self.conns:
            conn.send(None)
        for proc in self.procs:
            proc.join(timeout=5)
