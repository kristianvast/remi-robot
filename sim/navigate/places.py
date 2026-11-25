"""Human place names (places.yaml, provisional) to goal points."""

from functools import lru_cache
from pathlib import Path

import yaml

PLACES = Path(__file__).with_name("places.yaml")


@lru_cache(maxsize=1)
def places() -> dict[str, tuple[float, float, str]]:
    """Every accepted name (lower case) -> (x, y, area id)."""
    data = yaml.safe_load(PLACES.read_text())
    out = {}
    for area, spec in data["areas"].items():
        for i, g in spec["goals"].items():
            point = (float(g["x"]), float(g["y"]), area)
            out[f"{spec['name']} {i}"] = out[f"{area}_goal_{i}"] = point
            if i == spec["default"]:
                out[spec["name"]] = out[area] = point
    return out


def area_name(area: str) -> str:
    return yaml.safe_load(PLACES.read_text())["areas"][area]["name"]


def resolve(goal: str | tuple[float, float]) -> tuple[float, float, str]:
    if not isinstance(goal, str):
        return float(goal[0]), float(goal[1]), ""
    key = goal.strip().casefold()
    if key not in places():
        raise ValueError(f"Unknown place {goal!r}; known: {', '.join(sorted(places()))}")
    return places()[key]
