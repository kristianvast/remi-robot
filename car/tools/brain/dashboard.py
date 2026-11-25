"""Browser control pane for the car agent: live view of camera, sensors and agent, plus manual control.

agent.py starts it on http://localhost:8765. The car takes a single control connection, so the pane
shares the agent's link: every sensor read and command goes through Car and its lock.
"""

import asyncio
import json
import logging
from pathlib import Path

from aiohttp import WSMsgType, web

import mapping
from vision import camera_mirrored
from voice import GOOGLE_VOICES

log = logging.getLogger("car-dashboard")

PAGE = Path(__file__).with_name("dashboard") / "index.html"

# Measured from the CAD model (sim/urdf/elegoo_v4.urdf) and the firmware; drawn to scale by the pane.
GEOMETRY = {
    "body_length_cm": 23.2,
    "body_width_cm": 13.2,
    "nose_x_cm": 11.5,
    "wheel_x_cm": 4.1,
    "wheel_y_cm": 5.8,
    "wheel_radius_cm": 3.33,
    "wheel_width_cm": 2.6,
    "sonar_x_cm": 11.0,
    "sonar_height_cm": 10.2,
    "sonar_cone_deg": 15,
    "sonar_max_cm": 150,
    "camera_x_cm": 8.3,
    "camera_height_cm": 12.6,
    "camera_hfov_deg": 52,
    "camera_vfov_deg": 40,
    "camera_tilt_deg": 10,  # estimate: the camera points up and misses the floor close ahead
    "line_x_cm": 9.2,
    "line_threshold": 950,
}


class Dashboard:
    def __init__(self, telemetry, pilot, actions, stream_url, port=8765):
        self.t = telemetry
        self.pilot = pilot
        self.actions = actions
        self.stream_url = stream_url
        self.port = port
        self._clients = set()
        self._runner = None
        self._tasks = []

    async def start(self):
        app = web.Application()
        app.router.add_get("/", self._page)
        app.router.add_static("/static/", PAGE.parent)
        app.router.add_get("/frame.jpg", self._frame)
        app.router.add_get("/map.png", self._file(mapping.MAP_IMAGE))
        app.router.add_get("/car.glb", self._file(mapping.CAR_MODEL))
        app.router.add_get("/ws", self._ws)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        await web.TCPSite(self._runner, "127.0.0.1", self.port, reuse_address=True).start()
        self._tasks = [asyncio.create_task(self.pilot.run()), asyncio.create_task(self._broadcast())]
        log.info("control pane: http://localhost:%d", self.port)

    async def close(self):
        for task in self._tasks:
            task.cancel()
        for ws in list(self._clients):
            await ws.close()
        if self._runner:
            await self._runner.cleanup()

    async def _page(self, _request):
        return web.FileResponse(PAGE, headers={"Cache-Control": "no-store"})

    @staticmethod
    def _file(path):
        async def serve(_request):
            return web.FileResponse(path, headers={"Cache-Control": "max-age=3600"})
        return serve

    async def _frame(self, _request):
        jpeg = self.t.frame["jpeg"]
        if jpeg is None:
            return web.Response(status=404, text="no frame yet")
        return web.Response(body=jpeg, content_type="image/jpeg", headers={"Cache-Control": "no-store"})

    async def _broadcast(self):
        sent = -1
        while True:
            await asyncio.sleep(0.1)
            if self._clients and self.t.version != sent:
                sent = self.t.version
                payload = json.dumps(self.t.snapshot())
                for ws in list(self._clients):
                    try:
                        await ws.send_str(payload)
                    except ConnectionError:
                        self._clients.discard(ws)

    async def _ws(self, request):
        ws = web.WebSocketResponse(heartbeat=5)
        await ws.prepare(request)
        self._clients.add(ws)
        await ws.send_str(json.dumps({
            "t": "hello", "name": "Remi", "geometry": GEOMETRY, "stream_url": self.stream_url,
            "stream_mirrored": camera_mirrored(self.stream_url), "speed": self.pilot.speed,
            "car_model": "/car.glb", "map": self.t.map.hello(), "voices": GOOGLE_VOICES,
        }))
        await ws.send_str(json.dumps(self.t.snapshot()))
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    await self._command(json.loads(msg.data))
        finally:
            self._clients.discard(ws)
            if not self._clients:
                self.pilot.release_all()
        return ws

    async def _command(self, cmd):
        kind = cmd.get("t")
        if kind == "drive":
            self.pilot.set_drive(cmd.get("dir"))
        elif kind == "pan":
            self.pilot.set_pan(int(cmd.get("dir", 0)))
        elif kind == "hb":
            self.pilot.heartbeat()
        elif kind == "center":
            await self.pilot.center()
        elif kind == "scan":
            asyncio.create_task(self.pilot.scan())
        elif kind == "estop":
            self.actions.set_autonomy(False)
            await self.pilot.estop()
            await self.actions.estop()
        elif kind == "speed":
            self.pilot.speed = max(60, min(255, int(cmd.get("value", self.pilot.speed))))
        elif kind == "autonomy":
            self.actions.set_autonomy(bool(cmd.get("on")))
            self.t.notice("Agent driving enabled" if self.t.agent["autonomy"] else "Agent driving paused")
        elif kind == "initiative":
            self.actions.set_initiative(bool(cmd.get("on")))
            self.t.notice("Initiative enabled" if self.t.agent["initiative"] else "Initiative off")
        elif kind == "mic":
            self.actions.set_mic(bool(cmd.get("on")))
        elif kind == "voice" and cmd.get("name") in GOOGLE_VOICES and self.t.agent.get("voice"):
            self.actions.set_voice(cmd["name"])
        elif kind == "say" and cmd.get("text", "").strip():
            self.actions.say(cmd["text"].strip())
        elif kind == "place":
            await self.actions.place(float(cmd["x"]), float(cmd["y"]), float(cmd.get("yaw", 0.0)))
        self.t.touch()
