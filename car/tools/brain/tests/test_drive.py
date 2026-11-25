"""Robot wire and motion regressions against the loopback stand-in."""

from typing import Final

import pytest
from conftest import FakeCar

from drive import CM_PER_S, CREEP, CRUISE, NO_FLOOR, TURN_RETRY, Driver
from robot import MAX_RANGE_CM, Robot

SIM_CRUISE_CM_PER_S: Final = 55.0  # Stand-in physics, independent of the car's calibration.


def test_robot_decodes_readings_and_stop_acknowledges(robot: Robot, fake_car: FakeCar) -> None:
    fake_car.state({"pose": (60, 200, -180)})

    heading, grounded = robot.heading()
    distance = robot.distance()
    wheel_heading, wheel_grounded = robot.wheels(95, 95)

    assert isinstance(heading, float) and heading == pytest.approx(-180, abs=0.1)
    assert grounded is True
    assert type(distance) is int and 1 <= distance <= MAX_RANGE_CM
    assert distance == pytest.approx(55, abs=1)
    assert isinstance(wheel_heading, float)
    assert wheel_heading == pytest.approx(-180, abs=0.1)
    assert wheel_grounded is True
    assert fake_car.state().wheels == (95, 95)

    # Latched motion cannot coast to a watchdog stop and mask an ineffective stop().
    robot.drive("forward", 100)
    assert fake_car.state().wheels == (100, 100)
    assert robot.stop() is None  # Returns only after receiving the command reply.
    assert fake_car.state().wheels == (0, 0)


def test_forward_travels_requested_distance_and_brakes_for_wall(robot: Robot, fake_car: FakeCar) -> None:
    driver = Driver(robot)
    origin = fake_car.state({"pose": (150, 200, 0)})

    requested = 35
    _, reason = driver.forward(requested, stop_cm=30)
    moved = fake_car.state()

    assert reason is None
    # The driver integrates the calibrated speed, but the stand-in uses its own
    # speed line. Scale to simulator cm, then allow two wheel/reply ticks (~0.13 s).
    expected_cm = requested * SIM_CRUISE_CM_PER_S / CM_PER_S
    assert moved.x - origin.x == pytest.approx(expected_cm, abs=SIM_CRUISE_CM_PER_S * 0.13)
    assert moved.y == pytest.approx(origin.y, abs=1)
    assert moved.wheels == (0, 0)

    wall_start = fake_car.state({"pose": (60, 200, 180)})
    stop_cm = 30
    _, reason = driver.forward(100, stop_cm=stop_cm)
    stopped = fake_car.state()

    assert reason is not None and "obstacle" in reason
    assert stopped.x < wall_start.x
    # The west wall is x=0; sonar reads about 5 cm less than centre clearance.
    # Near the threshold, three wheel ticks plus the sonar and stop exchanges
    # take about 0.3 s, at creep speed (allow 2 cm extra for rounding/deceleration).
    threshold_x = stop_cm + 5
    braking_cm = (CM_PER_S * CREEP / CRUISE) * 0.3 + 2
    assert threshold_x - braking_cm <= stopped.x <= threshold_x + 1
    assert stopped.wheels == (0, 0)


def test_turn_reaches_requested_heading_within_retry_tolerance(robot: Robot, fake_car: FakeCar) -> None:
    origin = fake_car.state({"pose": (150, 200, 12)})
    degrees = 90

    turned, reason = Driver(robot).turn(degrees)

    settled = fake_car.state()
    assert reason is None
    # turn() reports what it really turned. Its correcting pass is the last one and can itself coast past
    # TURN_RETRY (seen: 97.1 for 90 on the stand-in), so the landing bound is two retry widths.
    assert turned == pytest.approx(settled.heading - origin.heading, abs=1)
    assert turned == pytest.approx(degrees, abs=2 * TURN_RETRY)
    assert settled.wheels == (0, 0)


def test_forward_on_lifted_car_returns_no_floor(robot: Robot, fake_car: FakeCar) -> None:
    fake_car.state({"lifted": True})

    travelled, reason = Driver(robot).forward(25, stop_cm=15)

    assert reason == NO_FLOOR
    assert travelled == 0
    assert fake_car.state().wheels == (0, 0)
