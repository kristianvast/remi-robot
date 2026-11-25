import time

CENTER = 90


def scan(robot, angles=range(10, 171, 20), settle=0.3):
    readings = []
    for angle in angles:
        robot.head(angle)
        time.sleep(settle)
        readings.append((angle, robot.distance()))
    robot.head(CENTER)
    return readings


def turn(robot, degrees, speed=150, ms_per_deg=6.0):
    if abs(degrees) < 10:
        return
    robot.drive("left" if degrees > 0 else "right", speed)
    time.sleep(abs(degrees) * ms_per_deg / 1000)
    robot.stop()


def face_forward(robot, settle=0.4):
    robot.head(CENTER)
    time.sleep(settle)


def patrol(robot, seconds, clear_cm=30, speed=100, log=print):
    face_forward(robot)
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        d = robot.distance()
        if d > clear_cm:
            robot.drive("forward", speed)
            time.sleep(0.1)
            continue
        robot.stop()
        readings = scan(robot)
        angle, best = max(readings, key=lambda r: r[1])
        log(f"blocked at {d} cm; scan {readings}")
        if best <= clear_cm:
            log("boxed in -> turning around")
            turn(robot, 180)
        else:
            log(f"clearest {best} cm at head {angle} deg -> turning {angle - CENTER} deg")
            turn(robot, angle - CENTER)
    robot.stop()


def follow(robot, seconds, target_cm=20, band_cm=8, lost_cm=80, log=print):
    face_forward(robot)
    deadline = time.monotonic() + seconds
    state = None
    while time.monotonic() < deadline:
        d = robot.distance()
        if d >= lost_cm or abs(d - target_cm) <= band_cm:
            new, action = "hold", robot.stop
        elif d > target_cm:
            new, action = "forward", lambda: robot.drive("forward", 90)
        else:
            new, action = "back", lambda: robot.drive("back", 90)
        if new != state:
            log(f"{d} cm -> {new}")
            state = new
        action()
        time.sleep(0.1)
    robot.stop()
