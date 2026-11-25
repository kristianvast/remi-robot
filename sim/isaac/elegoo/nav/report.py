"""Turn pulled building_run.py outputs into sim/isaac/views/ (Mac side, no Isaac).

  sim/.venv/bin/python sim/isaac/elegoo/nav/report.py <pulled out/ folder>

Writes <name>_rgb.jpg, <name>_depth.png (coloured, 0-5 m), <name>_depth_mm.png (16-bit mm),
index.html, contact_sheet.jpg, drive_rgb.mp4, drive_path.png and drive_areas.json.
"""

import base64
import csv
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.measure import points_in_poly

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SCAN_VIEWS = ROOT / "scans" / "merged" / "views"
MERGED = ROOT / "scans" / "merged"
VIEWS = HERE.parents[1] / "views"
DEPTH_MAX_M = 5.0
TURBO = np.array([[48, 18, 59], [70, 107, 227], [40, 187, 236], [49, 242, 153], [162, 252, 60],
                  [237, 208, 58], [251, 128, 34], [208, 47, 4], [122, 4, 3]], dtype=float)


def colour_depth(mm: np.ndarray) -> Image.Image:
    metres = mm.astype(float) / 1000
    t = np.clip(metres / DEPTH_MAX_M, 0, 1) * (len(TURBO) - 1)
    lo = np.floor(t).astype(int).clip(0, len(TURBO) - 2)
    rgb = TURBO[lo] + (TURBO[lo + 1] - TURBO[lo]) * (t - lo)[..., None]
    rgb[mm == 0] = 0
    return Image.fromarray(rgb.astype(np.uint8))


def thumb_uri(img: Image.Image, width: int = 360) -> str:
    img = img.convert("RGB").resize((width, int(img.height * width / img.width)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def export_views(pulled: Path) -> list[dict]:
    poses = json.loads((SCAN_VIEWS / "poses_rendered.json").read_text())["poses"]
    logs = {}
    for log in (pulled / "views").glob("views_log_*.json"):
        logs.update({entry["file"]: entry for entry in json.loads(log.read_text())})
    done = []
    for pose in poses:
        stem = Path(pose["file"]).with_suffix("")
        src = pulled / "views" / stem.parent
        rgb, mm = src / f"{stem.name}_rgb.jpg", src / f"{stem.name}_depth_mm.png"
        if not rgb.exists():
            continue
        dst = VIEWS / ("scan_drive" if pose["kind"] == "drive" else "")
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy(rgb, dst / rgb.name)
        shutil.copy(mm, dst / mm.name)
        depth = np.array(Image.open(mm))
        colour_depth(depth).save(dst / f"{stem.name}_depth.png")
        done.append({**pose, "name": stem.name, "dir": dst, **logs.get(pose["file"], {})})
    return done


def gallery(done: list[dict]) -> None:
    def card(view: dict) -> str:
        raw = Image.open(SCAN_VIEWS / view["file"])
        rgb = Image.open(view["dir"] / f"{view['name']}_rgb.jpg")
        depth = Image.open(view["dir"] / f"{view['name']}_depth.png")
        cam = view.get("camera_xyz", [float("nan")] * 3)
        caption = (f"{view['name']} &middot; x {view['x']:.2f} y {view['y']:.2f} yaw {view['yaw']:.0f}&deg; "
                   f"pitch {view['pitch']:.0f}&deg; &middot; sim camera z {cam[2]:.3f} m, scan camera z "
                   f"{view['z']:.3f} m &middot; sonar {view.get('range_cm', '?')} cm")
        imgs = "".join(f'<figure><img src="{thumb_uri(i)}"><figcaption>{label}</figcaption></figure>'
                       for i, label in ((raw, "raw scan (Blender)"), (rgb, "Isaac Sim RGB"),
                                        (depth, f"Isaac Sim depth 0-{DEPTH_MAX_M:.0f} m")))
        return f'<div class="card"><div class="row">{imgs}</div><p>{caption}</p></div>'

    sections = []
    areas = sorted({v["room"] for v in done if v.get("room")})
    for area in areas:
        cards = "".join(card(v) for v in done if v.get("room") == area)
        sections.append(f"<h2>{area}</h2>{cards}")
    drive = [v for v in done if v["kind"] == "drive"]
    if drive:
        cards = "".join(card(v) for v in drive[::20])
        sections.append(f"<h2>scan drive path (every 20th of {len(drive)} frames)</h2>{cards}")
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>ELEGOO car camera in Isaac Sim</title>
<style>body{{font:14px system-ui,sans-serif;margin:24px;background:#f4f3f0;color:#222}}
h2{{margin-top:32px;border-bottom:1px solid #ccc}}.card{{background:#fff;padding:10px;margin:12px 0;border-radius:6px}}
.row{{display:flex;gap:8px;flex-wrap:wrap}}figure{{margin:0}}img{{width:360px;display:block}}
figcaption,p{{font-size:12px;color:#555;margin:4px 0}}</style></head><body>
<h1>ELEGOO V4 car camera in the scanned building: raw scan vs Isaac Sim 6.2 RTX</h1>
<p>Same poses as scans/merged/views/poses_rendered.json. 640x480, hFOV 65&deg;, pitch as listed (20&deg; up for
room views, 0&deg; for level views). Isaac Sim pictures use the car's own camera on the car body placed at the pose
and settled on the collider, with the generated walls and ceilings on. Depth is distance to the image plane,
black = no hit within 20 m.</p>{''.join(sections)}</body></html>"""
    (VIEWS / "index.html").write_text(html)


def contact_sheet(done: list[dict]) -> None:
    rooms = [v for v in done if v["kind"] == "room"]
    w, h, gap, per_row = 248, 186, 6, 3
    cell_w = 3 * w + 2 * gap
    width = per_row * cell_w + (per_row + 1) * 16
    rows = (len(rooms) + per_row - 1) // per_row
    sheet = Image.new("RGB", (width, rows * (h + 22) + 10), (30, 30, 30))
    draw = ImageDraw.Draw(sheet)
    for k, view in enumerate(rooms):
        x0 = 16 + (k % per_row) * (cell_w + 16)
        y0 = 6 + (k // per_row) * (h + 22)
        for j, img in enumerate((Image.open(SCAN_VIEWS / view["file"]),
                                 Image.open(view["dir"] / f"{view['name']}_rgb.jpg"),
                                 Image.open(view["dir"] / f"{view['name']}_depth.png"))):
            sheet.paste(img.convert("RGB").resize((w, h)), (x0 + j * (w + gap), y0 + 16))
        draw.text((x0, y0 + 2), f"{view['name']}   raw scan | Isaac RGB | Isaac depth 0-5 m", fill=(230, 230, 230))
    assert sheet.width <= 2400
    sheet.save(VIEWS / "contact_sheet.jpg", quality=88)


def drive_outputs(pulled: Path) -> None:
    frames = pulled / "drive_frames"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{1 / 0.07:.4f}", "-i",
                    str(frames / "%05d.jpg"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
                    str(VIEWS / "drive_rgb.mp4")], check=True)
    plan = json.loads((HERE / "building_plan.json").read_text())["drive"]
    with (pulled / "drive_log.csv").open() as file:
        log = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(file)]
    result = json.loads((pulled / "drive_result.json").read_text())
    rooms = [(r["id"], np.array(r["polygon"])) for r in json.loads((MERGED / "rooms_next.json").read_text())]
    sequence = []
    for row in log:
        area = next((name for name, poly in rooms if points_in_poly([[row["x"], row["y"]]], poly)[0]), None)
        if area and (not sequence or sequence[-1]["area"] != area):
            sequence.append({"area": area, "t": row["t"], "x": row["x"], "y": row["y"]})
    travelled = float(sum(np.hypot(b["x"] - a["x"], b["y"] - a["y"]) for a, b in zip(log, log[1:])))
    (VIEWS / "drive_areas.json").write_text(json.dumps(
        {"area_sequence": sequence, "travelled_m": round(travelled, 2), **result}, indent=1))

    grid = np.array(Image.open(MERGED / "floor_map.png"))
    colour = np.full((*grid.shape, 3), 150, np.uint8)
    colour[grid == 254] = (250, 250, 250)
    colour[grid == 0] = (20, 20, 20)
    to_px = lambda x, y: ((x + 11.49) / 0.02, (11.17 - y) / 0.02)
    pts = np.array([to_px(r["x"], r["y"]) for r in log] + [to_px(*w) for w in plan["waypoints"]])
    lo, hi = pts.min(0) - 60, pts.max(0) + 60
    scale = 3
    img = Image.fromarray(colour).crop((*lo.astype(int), *hi.astype(int)))
    img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=30)
    shift = lambda p: ((p[0] - int(lo[0])) * scale, (p[1] - int(lo[1])) * scale)
    for x0, y0, x1, y1 in json.loads((HERE / "building_plan.json").read_text())["closure"]:
        draw.line([shift(to_px(x0, y0)), shift(to_px(x1, y1))], fill=(230, 0, 230), width=5)
    draw.line([shift(to_px(*w)) for w in plan["waypoints"]], fill=(60, 110, 230), width=3)
    draw.line([shift(to_px(r["x"], r["y"])) for r in log], fill=(220, 30, 30), width=3)
    for name, poly in rooms:
        px = [shift(to_px(*p)) for p in poly]
        cx, cy = np.mean(px, axis=0)
        draw.text((cx, cy), name, fill=(0, 140, 0), font=font)
    for mark, colour_ in ((log[0], (0, 170, 0)), (log[-1], (200, 0, 200))):
        x, y = shift(to_px(mark["x"], mark["y"]))
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), outline=colour_, width=3)
    for stuck in result["stuck"]:
        x, y = shift(to_px(stuck["x"], stuck["y"]))
        draw.rectangle((x - 9, y - 9, x + 9, y + 9), outline=(255, 140, 0), width=3)
    draw.text((10, 10), f"blue: planned {plan['length_m']} m   red: true path {travelled:.1f} m   green ring: start   "
                        f"magenta ring: end   orange box: stuck ({len(result['stuck'])})   magenta line: closure "
                        f"colliders   reached={result['reached']}", fill=(0, 0, 0), font=font)
    img.save(VIEWS / "drive_path.png")


def main() -> None:
    pulled = Path(sys.argv[1])
    VIEWS.mkdir(parents=True, exist_ok=True)
    if (pulled / "views").exists():
        done = export_views(pulled)
        gallery(done)
        contact_sheet(done)
        print("VIEWS", len(done))
    if (pulled / "drive_log.csv").exists():
        drive_outputs(pulled)
        print("DRIVE", (VIEWS / "drive_areas.json").read_text()[:600])


if __name__ == "__main__":
    main()
