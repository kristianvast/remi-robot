"""Named destinations and saved home on the navigation map."""

import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

# The project root (four folders up) holds sim/, whose packages import each other as `sim.<package>`.
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from sim.macsim.world import floor
from sim.navigate.places import PLACES, area_name, places


class UnknownPlace(ValueError):
    """The name matches no place in places.yaml; the message lists the known ones."""


@dataclass(frozen=True, slots=True)
class Home:
    x: float
    y: float
    yaw: float
    area: str
    placeholder: bool


def home() -> Home:
    h = yaml.safe_load(PLACES.read_text())["home"]
    return Home(float(h["x"]), float(h["y"]), float(h["yaw"]), str(h["area"]), bool(h.get("placeholder")))


def save_home(pose: tuple[float, float, float]) -> Home:
    """Rewrite places.yaml's home line (the comments and the areas stay as they are)."""
    x, y, yaw = pose
    area = floor().area(x, y)
    if not area:
        raise ValueError("that position is outside every room on my map")
    line = f"home: {{x: {x:.2f}, y: {y:.2f}, yaw: {yaw:.4f}, area: {area}, placeholder: false}}"
    text, n = re.subn(r"^home:.*$", line, PLACES.read_text(), count=1, flags=re.MULTILINE)
    if n != 1:
        raise ValueError(f"no home line in {PLACES}")
    tmp = PLACES.with_suffix(".yaml.tmp")
    tmp.write_text(text)
    tmp.replace(PLACES)
    return home()


def place_names() -> list[str]:
    areas = yaml.safe_load(PLACES.read_text())["areas"].values()
    return [spec["name"] for spec in areas] + ["home"]


def find_place(name: str) -> tuple[str, tuple[float, float], str]:
    """Case-insensitive place name -> (name, goal x/y, area id)."""
    key = " ".join(name.casefold().split()).removeprefix("the ")
    if key == "home":
        h = home()
        return "home", (h.x, h.y), h.area
    if (point := places().get(key)) is not None:
        x, y, area = point
        return key, (x, y), area
    raise UnknownPlace(f"I don't know a place called {name.strip()!r}. I know: {', '.join(place_names())}.")


def room_name(pose: tuple[float, float, float]) -> str:
    area = floor().area(pose[0], pose[1]) if np.all(np.isfinite(pose[:2])) else ""
    return f"the {area_name(area)}" if area else "a spot I can't place on my map"
