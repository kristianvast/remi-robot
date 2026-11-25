"""Remi's long-term memory: the people he has met, what happened, and every conversation, kept on disk.

remi_memory/memory.json holds people (name -> notes) and a diary of events; remi_memory/conversations/
holds one plain-text log per day. The digest of both goes into Remi's instructions when a session starts,
and recall() searches all of it. The folder is git-ignored: it is about real people.
"""

import json
import threading
from datetime import datetime
from pathlib import Path

DIR = Path(__file__).with_name("remi_memory")
FILE = DIR / "memory.json"
LOGS = DIR / "conversations"
DIGEST_EVENTS = 30
RECALL_HITS = 12


def _now():
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")


class Memory:
    def __init__(self, folder: Path = DIR):
        self.file = folder / FILE.name
        self.logs = folder / LOGS.name
        self._lock = threading.Lock()
        self.data = json.loads(self.file.read_text()) if self.file.exists() else {"people": {}, "events": []}

    def _save(self):
        self.file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1, ensure_ascii=False))
        tmp.replace(self.file)

    def counts(self):
        return {"people": len(self.data["people"]), "events": len(self.data["events"])}

    def remember_person(self, name: str, note: str) -> str:
        name = " ".join(name.split())[:60]
        if not name:
            return "Who? I need a name."
        with self._lock:
            person = self.data["people"].setdefault(name.casefold(), {"name": name, "first_met": _now(), "notes": []})
            person["last_seen"] = _now()
            if note.strip():
                person["notes"].append({"t": _now(), "text": note.strip()[:400]})
            self._save()
        return f"Got it, {person['name']} is in my memory ({len(person['notes'])} notes)."

    def remember(self, text: str) -> str:
        if not text.strip():
            return "Nothing to remember."
        with self._lock:
            self.data["events"].append({"t": _now(), "text": text.strip()[:400]})
            self._save()
        return "Saved to memory."

    def log(self, role: str, text: str):
        stamp = datetime.now().astimezone()
        self.logs.mkdir(parents=True, exist_ok=True)
        with (self.logs / f"{stamp:%Y-%m-%d}.log").open("a") as f:
            f.write(f"{stamp:%H:%M:%S} {role}: {' '.join(text.split())}\n")

    def recall(self, query: str) -> str:
        words = [w for w in query.casefold().split() if len(w) > 2]
        if not words:
            return self.digest()
        hits = []
        for person in self.data["people"].values():
            text = person["name"] + ": " + "; ".join(n["text"] for n in person["notes"])
            if any(w in text.casefold() for w in words):
                hits.append(f"[person] {text}")
        hits += [f"[{e['t']}] {e['text']}" for e in self.data["events"] if any(w in e["text"].casefold() for w in words)]
        for log in sorted(self.logs.glob("*.log"), reverse=True) if self.logs.exists() else ():
            for line in reversed(log.read_text().splitlines()):
                if len(hits) >= RECALL_HITS:
                    break
                if any(w in line.casefold() for w in words):
                    hits.append(f"[{log.stem}] {line}")
        return "\n".join(hits[:RECALL_HITS]) if hits else f"Nothing in my memory about {query!r}."

    def digest(self) -> str:
        people = [
            f"- {p['name']} (met {p['first_met']}): " + ("; ".join(n["text"] for n in p["notes"][-6:]) or "no notes yet")
            for p in self.data["people"].values()
        ]
        events = [f"- {e['t']}: {e['text']}" for e in self.data["events"][-DIGEST_EVENTS:]]
        return (
            "People you know:\n" + ("\n".join(people) or "- nobody yet")
            + "\n\nWhat happened lately (oldest first):\n" + ("\n".join(events) or "- nothing yet, this is day one")
        )
