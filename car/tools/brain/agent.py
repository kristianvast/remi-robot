"""Remi, the talking, seeing robot car: LiveKit Agents driving the car with Gemini Live (or an
STT -> LLM -> TTS pipeline), with a long-term memory (memory.py) and a dead-reckoned spot on the building map
(mapping.py) that the control pane shows and the owner sets by clicking the start on the map.

Run on the Mac (mic + speakers):  .venv/bin/python agent.py console
Type instead of talking:          .venv/bin/python agent.py console --text
Control pane (see + drive):       http://localhost:8765 while the agent runs
Without the car:                  .venv/bin/python fake_car.py --port 18100 --camera-port 18101
                                  CAR_HOST=127.0.0.1 CAR_PORT=18100 \
                                  CAR_CAMERA_URL=http://127.0.0.1:18101/capture \
                                  CAR_STREAM_URL=http://127.0.0.1:18101/stream .venv/bin/python agent.py console --text

Google Cloud: Vertex AI in project GOOGLE_CLOUD_PROJECT (default your-gcp-project) via gcloud
application-default credentials.

CAR_VOICE=gemini-live (default): one realtime model (GEMINI_LIVE_MODEL, default gemini-3.8-live,
in GEMINI_LIVE_LOCATION, default us-central1) hears the mic, watches the car camera (one frame a
second, plus a fresh frame right after every move) and speaks with a Google voice (GEMINI_VOICE,
default Sadachbia). Fastest replies.

CAR_VOICE=elevenlabs: ElevenLabs speech-to-text hears the mic, Gemini (GEMINI_MODEL in
GEMINI_LOCATION) reads the words plus a fresh car camera picture on every turn and picks the moves,
and ElevenLabs speaks the reply (ELEVEN_VOICE_ID, ELEVEN_MODEL). The ElevenLabs key comes from
ELEVENLABS_API_KEY / ELEVEN_API_KEY, else from Secret Manager
(gcloud secrets versions access latest --secret=ELEVENLABS_API_KEY); it is kept in memory only.

find_object always asks GEMINI_MODEL in GEMINI_LOCATION where the object is.

Following a person and patrol sightings need the person detector (3.7 MB, once per venv):
  mkdir -p .venv/share/elegoo && curl -fsSL -o .venv/share/elegoo/yolox_nano.onnx \
    https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx

drive.py steers each move from the Mac with the gyro and range sensor. The firmware stops the
wheels when commands stop arriving. The only safety stops are the crash ones: forward moves stop when
the ultrasonic sensor sees something closer than STOP_CM, a StallWatch (stall.py) stops wheels that spin without
the car moving, and every move stops when the floor sensors say the car is lifted or at an edge.
"""

import asyncio
import os

from livekit import agents
from livekit.agents import AgentServer

import behaviors
import car as car_link
import car_agent
import car_camera
import places
import voice
from dashboard import Dashboard
from memory import Memory
from pilot import Pilot
from robot import DEFAULT_HOST
from telemetry import Telemetry

HOST = os.environ.get("CAR_HOST", DEFAULT_HOST)
PORT = int(os.environ.get("CAR_PORT", "100"))
CAMERA_URL = os.environ.get("CAR_CAMERA_URL", f"http://{HOST}/capture")
STREAM_URL = os.environ.get("CAR_STREAM_URL", f"http://{HOST}:81/stream")
DASHBOARD_PORT = int(os.environ.get("DASHBOARD_PORT", "8765"))
ECHO_GUARD = os.environ.get("CAR_ECHO_GUARD", "1") != "0"  # mic off while speaking; 0 with headphones to allow barge-in
ECHO_TAIL_S = 0.6  # room echo after the last word
INITIATIVE = os.environ.get("CAR_INITIATIVE", "1") == "1"
MOTION_TOOLS = {"drive_forward", "drive_back", "turn", "explore", "follow_person", "patrol", "find_object", "go_to", "set_home"}


server = AgentServer()


class PaneActions:
    def __init__(self, session, telemetry, jobs):
        self.session = session
        self.telemetry = telemetry
        self.jobs = jobs

    async def estop(self):
        await self.session.interrupt(force=True)

    def set_autonomy(self, on):
        self.telemetry.agent["autonomy"] = on
        if not on:
            self.jobs.cancel("the human paused autonomy")

    def set_initiative(self, on):
        self.telemetry.agent["initiative"] = on
        if not on:
            self.jobs.cancel("initiative was switched off", initiative_only=True)

    def set_mic(self, on):
        self.session.input.set_audio_enabled(on)
        self.telemetry.agent["mic"] = on
        self.telemetry.touch()

    def set_voice(self, name):
        """Gemini Live reconnects with the new voice; the conversation so far carries over."""
        self.session.llm.update_options(voice=name)
        self.telemetry.agent["voice"] = name
        self.telemetry.notice(f"Voice set to {name}")

    def say(self, text):
        self.session.current_agent.new_user_turn()
        self.session.generate_reply(user_input=text)

    async def place(self, x, y, yaw):
        """The owner clicked the car's start spot on the map: track from there, and that spot is home."""
        odometry = self.telemetry.odometry
        odometry.place(x, y, yaw)
        try:
            saved = await asyncio.to_thread(places.save_home, (x, y, yaw))
        except ValueError as e:
            self.telemetry.notice(f"Start set, but not saved as home: {e}", "warn")
            return
        odometry.home = {"x": saved.x, "y": saved.y, "yaw": saved.yaw}
        room = self.telemetry.map.room_at(x, y)
        self.telemetry.notice(f"Start and home set{f' in the {room}' if room else ''}")


def _follow_session(session, telemetry, memory):
    unmute = None

    def mic_back():
        if telemetry.agent.get("mic", True):  # the owner's dashboard mute wins
            session.input.set_audio_enabled(True)

    def agent_state(ev):
        nonlocal unmute
        telemetry.agent["state"] = ev.new_state
        telemetry.touch()
        if ECHO_GUARD:
            # The Mac's speaker reaches its mic: without this the model hears itself and acts on it.
            if unmute:
                unmute.cancel()
                unmute = None
            if ev.new_state == "speaking":
                session.input.set_audio_enabled(False)
            elif ev.old_state == "speaking":
                unmute = asyncio.get_running_loop().call_later(ECHO_TAIL_S, mic_back)

    def user_state(ev):
        telemetry.agent["user_state"] = ev.new_state
        telemetry.touch()

    def heard(ev):
        if ev.transcript.strip():
            telemetry.say("user", ev.transcript, ev.is_final)

    def item(ev):
        role, text = getattr(ev.item, "role", None), getattr(ev.item, "text_content", None)
        if role in ("user", "assistant") and text:
            telemetry.say("user" if role == "user" else "agent", text)
            memory.log("crew" if role == "user" else "Remi", text)

    def tools_done(ev):
        # Remi's diary of what he did, so tomorrow's session can bring it up
        for call, out in ev.zipped():
            if call.name in MOTION_TOOLS and out.output:
                memory.remember(f"{call.name}({call.arguments}) -> {out.output[:200]}")
        telemetry.memory = memory.counts()

    session.on("agent_state_changed", agent_state)
    session.on("user_state_changed", user_state)
    session.on("user_input_transcribed", heard)
    session.on("conversation_item_added", item)
    session.on("function_tools_executed", tools_done)


@server.rtc_session()
async def entrypoint(ctx: agents.JobContext):
    telemetry = Telemetry(car_link.STOP_CM)
    telemetry.agent["initiative"] = INITIATIVE
    telemetry.agent["voice"] = voice.LIVE_VOICE if voice.CAR_VOICE == "gemini-live" else None  # None: ElevenLabs speaks
    car = car_link.Car(HOST, PORT, telemetry)
    camera = car_camera.CarCamera(CAMERA_URL, STREAM_URL, telemetry)
    memory = Memory()
    telemetry.memory = memory.counts()
    session = voice.make_session()
    _follow_session(session, telemetry, memory)
    agent = car_agent.CarAgent(car, camera, memory)
    await session.start(room=ctx.room, agent=agent)
    if voice.CAR_VOICE == "gemini-live":
        session.input.video = car_camera.LiveVideo(camera)
    camera.start()
    agent.jobs.start()
    pane = Dashboard(telemetry, Pilot(car, telemetry, car_link.MANUAL_SPEED, behaviors.scan),
                     PaneActions(session, telemetry, agent.jobs), None, DASHBOARD_PORT)
    await pane.start()

    async def shutdown():
        await pane.close()
        await agent.jobs.close()
        camera.close()
        await asyncio.to_thread(car.close)

    ctx.add_shutdown_callback(shutdown)


if __name__ == "__main__":
    agents.cli.run_app(server)
