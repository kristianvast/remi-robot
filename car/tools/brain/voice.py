"""LiveKit voice pipelines and provider settings."""

import functools
import os
import subprocess

from google.genai import types as genai
from livekit.agents import AgentSession
from livekit.plugins import elevenlabs, google, silero

PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "your-gcp-project")
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
LOCATION = os.environ.get("GEMINI_LOCATION", "global")  # gemini-3.8-flash is not served in us-central1 for this project
CAR_VOICE = os.environ.get("CAR_VOICE", "gemini-live")
LIVE_MODEL = os.environ.get("GEMINI_LIVE_MODEL", "gemini-3.8-live")
LIVE_LOCATION = os.environ.get("GEMINI_LIVE_LOCATION", "us-central1")  # Live is not served on "global"
LIVE_VOICE = os.environ.get("GEMINI_VOICE", "Sadachbia")  # Google AI Studio's "lively" voice
# Every Gemini Live voice (livekit google realtime api_proto.Voice) with Google's one-word style for it
GOOGLE_VOICES = {
    "Achernar": "Soft", "Achird": "Friendly", "Algenib": "Gravelly", "Algieba": "Smooth",
    "Alnilam": "Firm", "Aoede": "Breezy", "Autonoe": "Bright", "Callirrhoe": "Easy-going",
    "Charon": "Informative", "Despina": "Smooth", "Enceladus": "Breathy", "Erinome": "Clear",
    "Fenrir": "Excitable", "Gacrux": "Mature", "Iapetus": "Clear", "Kore": "Firm",
    "Laomedeia": "Upbeat", "Leda": "Youthful", "Orus": "Firm", "Pulcherrima": "Forward",
    "Puck": "Upbeat", "Rasalgethi": "Informative", "Sadachbia": "Lively", "Sadaltager": "Knowledgeable",
    "Schedar": "Even", "Sulafat": "Warm", "Umbriel": "Easy-going", "Vindemiatrix": "Gentle",
    "Zephyr": "Bright", "Zubenelgenubi": "Casual",
}
ELEVEN_VOICE_ID = os.environ.get("ELEVEN_VOICE_ID", "cgSgspJ2msm6clMCkdW9")  # premade "Jessica - Playful, Bright, Warm"
ELEVEN_MODEL = os.environ.get("ELEVEN_MODEL", "eleven_flash_v2_5")  # ElevenLabs' ultra-low-latency model
ENDPOINT_MIN_S = 0.9  # silence before a turn counts as finished; shorter cut the speaker off mid-sentence


@functools.cache
def _eleven_key():
    key = os.environ.get("ELEVENLABS_API_KEY") or os.environ.get("ELEVEN_API_KEY")
    if key:
        return key
    cmd = ["gcloud", "secrets", "versions", "access", "latest", "--secret=ELEVENLABS_API_KEY", f"--project={PROJECT}"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=True)
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"Cannot read the ElevenLabs key from Secret Manager: {e.stderr.strip()}") from e
    except (OSError, subprocess.TimeoutExpired) as e:
        raise SystemExit(f"Cannot run gcloud to read the ElevenLabs key: {e}") from e
    return out.stdout.strip()


def make_tts():
    return elevenlabs.TTS(voice_id=ELEVEN_VOICE_ID, model=ELEVEN_MODEL, api_key=_eleven_key())


def make_llm():
    return google.LLM(
        vertexai=True,
        project=PROJECT,
        location=LOCATION,
        model=MODEL,
        thinking_config={"thinking_level": "low"},  # less thinking, sooner first word; "minimal" is refused by this model
    )


def make_session():
    if CAR_VOICE == "gemini-live":
        return AgentSession(
            llm=google.realtime.RealtimeModel(
                vertexai=True,
                project=PROJECT,
                location=LIVE_LOCATION,
                model=LIVE_MODEL,
                voice=LIVE_VOICE,
                language="en-US",
                # 3.8 Live defaults to NON_BLOCKING tools and then re-issued the same move up to 6 times
                tool_behavior=genai.Behavior.BLOCKING,
                realtime_input_config=genai.RealtimeInputConfig(
                    automatic_activity_detection=genai.AutomaticActivityDetection(silence_duration_ms=int(ENDPOINT_MIN_S * 1000)),
                ),
                # Without compression Vertex ends an audio+video session after about 2 minutes; a small window keeps replies quick
                context_window_compression=genai.ContextWindowCompressionConfig(
                    trigger_tokens=32000, sliding_window=genai.SlidingWindow(target_tokens=16000),
                ),
                session_resumption=genai.SessionResumptionConfig(),  # reconnects every ~10 min keep the conversation
            ),
            video_sampler=None,  # LiveVideo paces the frames itself and must not lose the one after a move
        )
    if CAR_VOICE == "elevenlabs":
        return AgentSession(
            stt=elevenlabs.STT(api_key=_eleven_key(), language_code="en", use_realtime=True),
            min_endpointing_delay=ENDPOINT_MIN_S,
            llm=make_llm(),
            tts=make_tts(),
            vad=silero.VAD.load(),
        )
    raise SystemExit(f"CAR_VOICE must be gemini-live or elevenlabs, not {CAR_VOICE!r}")
