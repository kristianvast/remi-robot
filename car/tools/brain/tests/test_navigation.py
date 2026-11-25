"""Named navigation places are resolved through the shipped map."""

import pytest

from places import UnknownPlace, find_place, room_name


def test_named_place_resolves_to_room_and_unknown_place_raises() -> None:
    place = find_place("  THE Kitchen  ")

    assert isinstance(place, tuple)
    name, (x, y), area = place
    assert name == "kitchen"
    assert area == "area_01"
    assert room_name((x, y, 0.0)) == "the kitchen"

    with pytest.raises(UnknownPlace):
        find_place("an unlisted moon crater")
