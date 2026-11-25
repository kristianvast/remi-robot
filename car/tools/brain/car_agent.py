# noqa: SIZE_OK
# Over 250 lines on purpose: the tool surface the model sees, 18 function tools whose docstrings are the
# model-facing descriptions and must stay word for word.

import asyncio
import base64
import time
from typing import Literal

from livekit.agents import Agent, function_tool
from livekit.agents.llm import ImageContent

import navigation
import places
from car import _back, _explore, _forward, _look, _scan, _turn
from car_camera import SETTLE_S
from find_object import ObjectModel, find_object
from follow import follow_person
from patrol import patrol
from persona import instructions
from tasks import TaskManager
from trips import go_to, set_home
from voice import LOCATION, MODEL, PROJECT

REPEAT_S = 3.0  # Gemini Live re-issues the move it just made; an identical move this soon is dropped once


class CarAgent(Agent):
    def __init__(self, car, camera, memory):
        super().__init__(instructions=instructions(memory))
        self.car = car
        self.camera = camera
        self.memory = memory
        self.telemetry = car.telemetry
        self.odometry = car.telemetry.odometry
        self.jobs = TaskManager(car, camera.vision, self._speak_task, self._greet)
        self.object_model = ObjectModel(PROJECT, LOCATION, MODEL)
        self._last_move = None  # (label, monotonic time it finished); cleared by every new user turn

    def new_user_turn(self):
        """The repeat guard only catches moves the model re-issues by itself, never ones the person asked for."""
        self._last_move = None

    async def on_enter(self):
        # On speech start, not the final transcript: Gemini Live often moves before the transcript is final.
        self.session.on("user_state_changed", lambda ev: ev.new_state == "speaking" and self.new_user_turn())

    def _greet(self):
        self.session.generate_reply(instructions=(
            "Someone just showed up in your camera view. Greet them warmly in one short line. If your memory "
            "says who they are, use their name; if not, ask who they are."))

    async def _speak_task(self, text: str) -> bool:
        if self.session.tts:
            speech = self.session.say(text, allow_interruptions=True)
        else:  # Gemini Live has no separate voice to read text aloud, so the model says it
            speech = self.session.generate_reply(
                instructions=f"Say exactly this to the person, word for word, and call no tools: {text}",
                allow_interruptions=True,
            )
        try:
            await speech.wait_for_playout()
        except asyncio.CancelledError:
            speech.interrupt()
            raise
        return not speech.interrupted

    async def llm_node(self, chat_ctx, tools, model_settings):
        # Every model call, including the one right after a move, sees the camera as it is now.
        # The picture goes into a copy of the context, so old frames never pile up in the history.
        jpeg = await self.camera.snapshot()
        chat_ctx = chat_ctx.copy()
        if jpeg:
            image = ImageContent(image="data:image/jpeg;base64," + base64.b64encode(jpeg).decode())
            chat_ctx.add_message(role="user", content=["[Your camera view right now]", image])
        else:
            chat_ctx.add_message(role="user", content=["[Your camera is not answering right now; you cannot see.]"])
        async for chunk in Agent.default.llm_node(self, chat_ctx, tools, model_settings):
            yield chunk

    async def _drive(self, label, fn):
        if not self.telemetry.agent["autonomy"]:
            return self._refuse(label, "Not moved: the human has paused my driving from the control pane.")
        if self.car.human_active():
            return self._refuse(label, "Not moved: the human is driving me right now. Wait and watch the camera.")
        last, self._last_move = self._last_move, None
        if last and last[0] == label and time.monotonic() - last[1] < REPEAT_S:
            return self._refuse(label, "Skipped: that exact move just finished. Look at the camera view; if you still want it, call it again.")
        result = await self.jobs.dispatch(label, fn, background=False)
        await asyncio.sleep(SETTLE_S)
        self.camera.moved()
        await self.camera.after_move()
        self._last_move = (label, time.monotonic())
        return result

    def _refuse(self, label, reason):
        self.telemetry.tool_end(self.telemetry.tool_start(label), reason, ok=False)
        return reason

    @function_tool()
    async def drive_forward(self, centimeters: int) -> str:
        """Drive straight forward, smoothly. Slows down and stops early if something is in the way.

        Args:
            centimeters: distance to drive, 5 to 1000.
        """
        cm = max(5, min(1000, centimeters))
        return await self._drive(f"Drive forward {cm} cm", lambda r: _forward(r, cm))

    @function_tool()
    async def drive_back(self, centimeters: int) -> str:
        """Back up straight. There is no distance sensor behind; the stall check still stops you at a wall.

        Args:
            centimeters: distance to back up, 5 to 200.
        """
        cm = max(5, min(200, centimeters))
        return await self._drive(f"Back up {cm} cm", lambda r: _back(r, cm))

    @function_tool()
    async def turn(self, direction: Literal["left", "right"], degrees: int) -> str:
        """Spin the whole car in place.

        Args:
            direction: which way to spin, from the car's point of view.
            degrees: how far to spin, 5 to 360.
        """
        deg = max(5, min(360, abs(degrees))) * (1 if direction == "left" else -1)
        return await self._drive(f"Turn {direction} {abs(deg)}°", lambda r: _turn(r, deg))

    @function_tool()
    async def explore(self, seconds: int) -> str:
        """Roam without a specific goal, avoiding obstacles.

        Returns at once while the car keeps driving. You are told when it ends.

        Args:
            seconds: how long to explore, 10 to 600.
        """
        secs = max(10, min(600, seconds))
        return await self.jobs.dispatch(f"exploring for {secs} seconds", lambda r: _explore(r, secs))

    @function_tool()
    async def look_direction(self, direction: Literal["left", "right", "forward"], degrees: int = 45) -> str:
        """Pan only the head (camera and distance sensor); the car stays still.

        Args:
            direction: where to point the head, from the car's point of view.
            degrees: how far from straight ahead, 10 to 80. Ignored for forward.
        """
        deg = 0 if direction == "forward" else max(10, min(80, abs(degrees))) * (1 if direction == "left" else -1)
        label = "Look forward" if deg == 0 else f"Look {direction} {abs(deg)}°"
        result = await self.jobs.dispatch(label, lambda r: _look(r, deg), background=False)
        await asyncio.sleep(SETTLE_S)
        self.camera.moved()
        await self.camera.after_move()
        return result

    @function_tool()
    async def scan_distances(self) -> str:
        """Sweep the head left to right and measure free space in each direction, in cm."""
        return await self.jobs.dispatch("Scan distances", _scan, background=False)

    @function_tool()
    async def follow_person(self, distance_cm: int = 85) -> str:
        """Follow a visible person, keeping roughly distance_cm away, and report if lost.

        Args:
            distance_cm: desired separation, 40 to 200 cm.
        """
        cm = max(40, min(200, distance_cm))
        return await self.jobs.dispatch("following a person", lambda r: follow_person(r, self.jobs.vision, cm))

    @function_tool()
    async def patrol(self, seconds: int = 30) -> str:
        """Roam, save pictures, and report when done.

        Args:
            seconds: desired duration, 10 to 600 seconds.
        """
        secs = max(10, min(600, seconds))
        return await self.jobs.dispatch(f"a {secs} second patrol", lambda r: patrol(r, self.jobs.vision, secs))

    @function_tool()
    async def find_object(self, name: str) -> str:
        """Search for a named visible object, then approach safely and report.

        Args:
            name: concise visual name of a physical object, not a named place.
        """
        name = " ".join(name.split())[:100]
        if not name:
            return "Tell me what object to look for."
        loop = asyncio.get_running_loop()
        return await self.jobs.dispatch(
            f"searching for {name}",
            lambda r: find_object(r, self.jobs.vision, lambda jpeg: self.object_model.locate(name, jpeg), name, loop),
        )

    @function_tool()
    async def go_to(self, place: str, start_room: str = "") -> str:
        """Drive by yourself to a named place in the house, such as the kitchen, or home.

        Returns at once while the car drives. You are told when it ends.

        Args:
            place: the place the person named, for example "kitchen" or "home".
            start_room: the room you are in now, only if the person said so and it disagrees with your map
                position. Leave empty to start from your map position (or home, or where your last trip ended).
        """
        try:
            trip = await asyncio.to_thread(navigation.plan_trip, place, start_room or None, self.jobs.where,
                                           self.odometry.pose())
        except ValueError as e:  # UnknownPlace lists the known places; others: the map cannot reach the goal
            return str(e) if isinstance(e, places.UnknownPlace) else f"I can't go there: {e}."
        if (job := self.jobs.active) and job.label == trip.doing and not job.cancel.is_set():
            # Gemini Live re-issues the running trip's call (seen for "where are you?"); restarting would lose it.
            return f"Already {trip.doing}; not restarted. {self.jobs.where.describe()}"
        result = await self.jobs.dispatch(trip.doing, lambda r: go_to(r, self.jobs.vision, trip, self.jobs.where))
        return f"{result} I assume I start from {trip.start}." if result.startswith("Started") else result

    @function_tool()
    async def where_am_i(self) -> str:
        """Where you are on the building map: room, position, heading, how sure, and the way home. Does not move."""
        fix = self.jobs.where.fix
        if fix and fix.live:
            return f"{self.jobs.where.describe()} {self.odometry.describe()}"
        return self.odometry.describe()

    @function_tool()
    async def directions_to(self, place: str) -> str:
        """Straight-line distance and which way to turn to face a room or home, from your map position.

        Args:
            place: a room name such as "kitchen", or "home".
        """
        if " ".join(place.casefold().split()) in ("home", "the home"):
            if not (home := self.odometry.home):
                return "Home isn't set: the crew sets it by clicking my start spot on the dashboard map."
            return self.odometry.directions(home["x"], home["y"], "home")
        if not (spot := self.telemetry.map.place(place)):
            return f"No room called {place!r} on my map. Rooms: {', '.join(p['name'] for p in self.telemetry.map.places)}."
        return self.odometry.directions(spot["x"], spot["y"], f"the {spot['name']}")

    @function_tool()
    async def remember_person(self, name: str, note: str) -> str:
        """Save someone to long-term memory, or add a note about them.

        Args:
            name: their name as they say it.
            note: something worth remembering: what they're building, how they look, a joke, a preference.
        """
        result = await asyncio.to_thread(self.memory.remember_person, name, note)
        self.telemetry.memory = self.memory.counts()
        return result

    @function_tool()
    async def remember(self, note: str) -> str:
        """Save something that happened, or a fact about the place, to long-term memory.

        Args:
            note: what to remember, in a sentence.
        """
        result = await asyncio.to_thread(self.memory.remember, note)
        self.telemetry.memory = self.memory.counts()
        return result

    @function_tool()
    async def recall(self, about: str) -> str:
        """Search long-term memory: people, past events and old conversations.

        Args:
            about: a name or a few keywords.
        """
        return await asyncio.to_thread(self.memory.recall, about)

    @function_tool()
    async def set_home(self, room: str = "") -> str:
        """Remember the spot you are on as home, when the owner says this is home.

        Unless you just arrived here with go_to, you look around and turn once on the spot to find where you are.

        Args:
            room: the room you are in, if the owner said it. Needed unless you just arrived here with go_to.
        """
        if pose := self.jobs.where.start_pose() or self.odometry.pose():
            try:
                saved = await asyncio.to_thread(places.save_home, pose)
            except ValueError as e:
                return f"I did not save home: {e}."
            self.odometry.home = {"x": saved.x, "y": saved.y, "yaw": saved.yaw}
            return f"Saved: this spot is home now, in {places.room_name(pose)}."
        if not room:
            return "I'm not sure exactly where I am. Tell me which room I'm in, and I'll look around to find my spot."
        try:
            name, _, area = places.find_place(room)
        except places.UnknownPlace as e:
            return str(e)
        if name == "home":
            return "Tell me which room this is, for example the kitchen."
        return await self.jobs.dispatch(
            "finding my spot for home", lambda r: set_home(r, self.jobs.vision, name, area, self.jobs.where),
            background=False,
        )

    @function_tool()
    async def task_status(self) -> str:
        """Check the current or most recent task without ending it."""
        return self.jobs.status()

    @function_tool()
    async def stop(self) -> str:
        """Stop all motion immediately."""
        return await self.jobs.stop("I was told to stop")
