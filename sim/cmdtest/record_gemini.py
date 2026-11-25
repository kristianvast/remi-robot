"""Record the tool calls Gemini makes for typed requests, with Remi's exact model, prompt and tools.

Run once on the Mac with the brain venv (needs Google application-default credentials; no car):
    cd car/tools/brain
    .venv/bin/python ../../../sim/cmdtest/record_gemini.py            # writes sim/cmdtest/gemini_script.json
    .venv/bin/python ../../../sim/cmdtest/record_gemini.py --dry-run  # print the declarations only

The Live connect config is built by livekit's own RealtimeSession._build_connect_config from the RealtimeModel
that voice.make_session() creates, with persona.INSTRUCTIONS and every @function_tool of car_agent.CarAgent, so model,
system prompt, tool declarations and BLOCKING behaviour are byte-for-byte what the car agent sends. Each request
gets a fresh session (the car starts every trial from home). Text goes in with send_client_content; every tool
call is answered at once with a plausible stub result in car_agent.py's wording and recorded. Audio is discarded.
The resulting script is fixed: the harness replays it and never calls Gemini.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
BRAIN = HERE.parents[1] / "car" / "tools" / "brain"
sys.path.insert(0, str(BRAIN))

import car_agent  # noqa: E402
import persona  # noqa: E402
import voice  # noqa: E402
from google import genai  # noqa: E402
from google.genai import types  # noqa: E402
from livekit.agents import llm  # noqa: E402
from livekit.plugins.google.realtime.realtime_api import RealtimeSession  # noqa: E402

OUT = HERE / "gemini_script.json"
TURN_TIMEOUT_S = 40.0

# (id, kind, turns). A turn is text, or (text, seconds into the previous turn's first motion call) for a
# request spoken while the car is still moving ("stop" mid-move). kind is the command type for the summary.
REQUESTS = [
    ("goto_kitchen", "go_to", ["Go to the kitchen."]),
    ("goto_dining", "go_to", ["Can you drive to the dining room?"]),
    ("goto_office", "go_to", ["Please go to the office."]),
    ("goto_west_corridor", "go_to", ["Drive to the west corridor."]),
    ("goto_meeting", "go_to", ["Go to the meeting room please."]),
    ("goto_middle_corridor", "go_to", ["Head over to the middle corridor."]),
    ("goto_lounge", "go_to", ["Take yourself to the lounge."]),
    ("goto_back_hall", "go_to", ["Go to the back hall."]),
    ("goto_stair_hall", "go_to", ["Drive over to the stair hall."]),
    ("goto_store_room", "go_to", ["Go to the store room."]),
    ("go_home", "go_to", ["Go home."]),
    ("goto_then_home", "go_to", ["Go to the middle corridor.", "Now come back home."]),
    ("fwd_1m", "drive_forward", ["Drive forward one metre."]),
    ("fwd_50cm", "drive_forward", ["Move forward 50 centimetres."]),
    ("fwd_bit", "drive_forward", ["Go forward a bit."]),
    ("fwd_2m", "drive_forward", ["Drive straight ahead two metres."]),
    ("fwd_little", "drive_forward", ["Inch forward a little."]),
    ("fwd_20cm", "drive_forward", ["Drive forward 20 cm."]),
    ("back_30cm", "drive_back", ["Back up 30 centimetres."]),
    ("back_bit", "drive_back", ["Reverse a bit."]),
    ("back_10cm", "drive_back", ["Drive backwards 10 cm."]),
    ("turn_left_90", "turn", ["Turn left 90 degrees."]),
    ("turn_right_45", "turn", ["Turn right 45 degrees."]),
    ("turn_around", "turn", ["Turn around."]),
    ("turn_left_bit", "turn", ["Turn left a little."]),
    ("turn_right_30", "turn", ["Turn 30 degrees to the right."]),
    ("look_left", "look_direction", ["Look to your left."]),
    ("explore_30", "explore", ["Explore for 30 seconds."]),
    ("explore_wander", "explore", ["Wander around for 20 seconds."]),
    ("patrol_60", "patrol", ["Patrol for a minute."]),
    ("patrol_30", "patrol", ["Do a 30 second patrol."]),
    ("stop_fwd", "stop", ["Drive forward three metres.", ("Stop!", 1.5)]),
    ("stop_explore", "stop", ["Explore for a minute.", ("Stop.", 6.0)]),
    ("stop_goto", "stop", ["Go to the office.", ("Stop right there.", 10.0)]),
    ("two_turn_drive", "two_step", ["Turn around and drive a metre."]),
    ("two_fwd_turn", "two_step", ["Drive forward half a metre and then turn left."]),
    ("two_right_fwd", "two_step", ["Turn right 90 degrees, then go forward one metre."]),
    ("two_back_turn", "two_step", ["Back up a bit and turn left 45 degrees."]),
    ("no_fwd_1m", "drive_forward", ["Kjør fram en meter."]),
    ("no_turn_left", "turn", ["Snu til venstre nitti grader."]),
    ("no_goto_kitchen", "go_to", ["Gå til kjøkkenet."]),
    ("no_back", "drive_back", ["Rygg litt."]),
    ("no_stop", "stop", ["Kjør fremover to meter.", ("Stopp!", 1.5)]),
]


def stub(name, args):
    a = args or {}
    match name:
        case "drive_forward":
            cm = max(5, min(300, int(a.get("centimeters", 50))))
            return f"Drove forward about {cm} cm. Clear distance ahead is 140 cm."
        case "drive_back":
            return f"Backed up about {max(5, min(40, int(a.get('centimeters', 20))))} cm."
        case "turn":
            return f"Turned {max(5, min(180, abs(int(a.get('degrees', 90)))))} degrees {a.get('direction', 'left')}."
        case "look_direction":
            d = a.get("direction", "forward")
            return "Head facing forward." if d == "forward" else f"Head turned {a.get('degrees', 45)} degrees {d}."
        case "explore":
            return f"Started exploring for {a.get('seconds', 30)} seconds."
        case "patrol":
            return f"Started a {a.get('seconds', 30)} second patrol."
        case "go_to":
            place = a.get("place", "")
            doing = "going home" if place.strip().lower() == "home" else f"going to the {place}"
            return f"Started {doing}. I assume I start from home."
        case "stop":
            return "Stopped: I was told to stop."
        case "where_am_i":
            return "I think I'm in the dining room (about -3.7, 3.8 on my map, confidence 90%)."
        case "task_status":
            return "Nothing is running."
        case "scan_distances":
            return "Free distance by head angle (+ is left): -80 deg: 60 cm, -40 deg: 120 cm, +0 deg: 140 cm, +40 deg: 90 cm, +80 deg: 50 cm"
        case "set_home":
            return "Saved: this spot is home now, in the dining room."
        case "find_object" | "follow_person":
            return f"Started {name.replace('_', ' ')}."
    return "Done."


def connect_config():
    session = voice.make_session()
    rt = session.llm
    rt._opts.instructions = persona.INSTRUCTIONS
    car_agent = object.__new__(car_agent.CarAgent)  # tool schemas only; no car, camera or task manager
    tools = llm.ToolContext(llm.find_function_tools(car_agent))
    shell = object.__new__(RealtimeSession)  # never connects; only its config builder is used
    shell._opts, shell._tools, shell._replay_tool_history, shell._session_resumption_handle = rt._opts, tools, False, None
    return rt._opts, shell._build_connect_config()


async def run_turn(live, text, calls, log):
    await live.send_client_content(turns=types.Content(role="user", parts=[types.Part(text=text)]), turn_complete=True)
    deadline = time.monotonic() + TURN_TIMEOUT_S
    words = []
    while time.monotonic() < deadline:
        try:
            async for msg in live.receive():
                if msg.tool_call:
                    responses = []
                    for fc in msg.tool_call.function_calls or []:
                        args = dict(fc.args or {})
                        result = stub(fc.name, args)
                        calls.append({"name": fc.name, "args": args, "stub_result": result})
                        log(f"    call {fc.name}({json.dumps(args, ensure_ascii=False)})")
                        responses.append(types.FunctionResponse(id=fc.id, name=fc.name, response={"output": result}))
                    await live.send_tool_response(function_responses=responses)
                sc = msg.server_content
                if sc and sc.output_transcription and sc.output_transcription.text:
                    words.append(sc.output_transcription.text)
                if sc and sc.turn_complete:
                    return "".join(words).strip()
        except asyncio.TimeoutError:
            break
    return "".join(words).strip() + " [turn timed out]"


async def record(only):
    opts, conf = connect_config()
    client = genai.Client(vertexai=True, project=voice.PROJECT, location=voice.LIVE_LOCATION)
    conf.output_audio_transcription = types.AudioTranscriptionConfig()
    entries = []
    for rid, kind, turns in REQUESTS:
        if only and rid not in only:
            continue
        print(f"{rid}:", flush=True)
        record_turns = []
        async with client.aio.live.connect(model=opts.model, config=conf) as live:
            for turn in turns:
                text, after_s = (turn, None) if isinstance(turn, str) else turn
                print(f"  > {text}", flush=True)
                calls = []
                reply = await asyncio.wait_for(run_turn(live, text, calls, print), TURN_TIMEOUT_S + 5)
                record_turns.append({"text": text, "after_s": after_s, "calls": calls, "reply": reply})
        entries.append({"id": rid, "kind": kind, "turns": record_turns})
    return opts, conf, entries


TWO_STEP = {"two_turn_drive": ["turn", "drive_forward"], "two_fwd_turn": ["drive_forward", "turn"],
            "two_right_fwd": ["turn", "drive_forward"], "two_back_turn": ["drive_back", "turn"]}


def expected_calls(rid, kind, turns):
    if rid in TWO_STEP:
        return TWO_STEP[rid]
    if kind == "stop":
        return {"stop_fwd": "drive_forward", "stop_explore": "explore", "stop_goto": "go_to",
                "no_stop": "drive_forward"}[rid].split() + ["stop"]
    return [kind] * len(turns)


def from_log(path):
    """Rebuild the recorded requests from record_gemini's stdout (the run that wrote it, replies not kept)."""
    turns_by_id = {rid: (kind, turns) for rid, kind, turns in REQUESTS}
    entries, current = [], None
    for line in path.read_text().splitlines():
        if line.endswith(":") and line[:-1] in turns_by_id:
            kind, turns = turns_by_id[line[:-1]]
            current = {"id": line[:-1], "kind": kind, "turns": []}
            entries.append(current)
        elif line.startswith("  > ") and current is not None:
            spec = turns_by_id[current["id"]][1][len(current["turns"])]
            after_s = None if isinstance(spec, str) else spec[1]
            current["turns"].append({"text": line[4:], "after_s": after_s, "calls": [], "reply": None})
        elif line.startswith("    call ") and current is not None:
            name, _, rest = line[9:].partition("(")
            args = json.loads(rest[:-1])
            current["turns"][-1]["calls"].append({"name": name, "args": args, "stub_result": stub(name, args)})
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print model, prompt length and tool declarations")
    ap.add_argument("--only", nargs="*", help="request ids (a test run; does not overwrite the full script)")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--from-log", type=Path, help="rebuild the script from a recording run's stdout; no Gemini call")
    a = ap.parse_args()
    if a.dry_run:
        opts, conf = connect_config()
        print("model", opts.model, "location", voice.LIVE_LOCATION, "modalities", conf.response_modalities)
        print("instructions chars", len(conf.system_instruction.parts[0].text))
        for tool in conf.tools:
            for fd in tool.function_declarations:
                print(json.dumps(fd.model_dump(exclude_none=True, mode="json"), ensure_ascii=False))
        return
    began = datetime.now(timezone.utc)
    if a.from_log:
        async def config():
            return connect_config()
        (opts, conf), entries = asyncio.run(config()), from_log(a.from_log)
        began = datetime.fromtimestamp(a.from_log.stat().st_mtime, timezone.utc)
    else:
        opts, conf, entries = asyncio.run(record(set(a.only or ())))
    for e in entries:
        e["expected_tools"] = expected_calls(e["id"], e["kind"], e["turns"])
        e["gemini_matches_request"] = [c["name"] for t in e["turns"] for c in t["calls"]] == e["expected_tools"]
    script = {
        "recorded_utc": began.isoformat(timespec="seconds"),
        "recorded_how": "rebuilt from the recording run's log" if a.from_log else "live",
        "model": opts.model,
        "location": voice.LIVE_LOCATION,
        "project": voice.PROJECT,
        "source": "car/tools/brain/ (voice.make_session, persona.INSTRUCTIONS, car_agent.CarAgent tools)",
        "tool_declarations": [fd.model_dump(exclude_none=True, mode="json")
                              for tool in conf.tools for fd in tool.function_declarations],
        "note": "Text in, no camera frames, no car; every tool answered at once with a stub. Fixed: replayed, never regenerated.",
        "requests": entries,
    }
    out = a.out if not a.only else a.out.with_name("gemini_script_test.json")
    out.write_text(json.dumps(script, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {out} ({len(entries)} requests, {sum(len(t['calls']) for e in entries for t in e['turns'])} calls)")


if __name__ == "__main__":
    os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
    main()
