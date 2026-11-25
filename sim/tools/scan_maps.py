"""Occupancy and annotated top-down exports from the retained scan geometry."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
import trimesh

from scan_raster import (FLOOR_BAND, Grid, disk, floor_faces, floor_mask, footprint, raster,
                         small_holes)
from scan_rooms import write_rooms


def annotated(grid: Grid, masks: tuple[np.ndarray, ...]) -> Image.Image:
    """Assemble a north-up figure without resampling the metric raster."""
    floor, mid, high, a, b = masks
    colors = ((190, 221, 189), (210, 38, 38), (145, 166, 222),
              (15, 112, 112), (155, 51, 122))
    data = np.full((*floor.shape, 3), 255, np.uint8)
    data[high] = colors[2]
    data[floor] = colors[0]
    data[mid] = colors[1]
    rows, cols = np.indices(floor.shape, sparse=True)
    for i, (outline, color) in enumerate(zip((a, b), colors[3:], strict=True)):
        stroke = outline & ~ndimage.binary_erosion(outline, iterations=2)
        dash = (rows + cols + 5 * i) % 14 < 10
        data[stroke & dash] = color
    height = 92
    width, map_height = grid.size
    canvas = Image.new("RGB", (width, map_height + height), "white")
    canvas.paste(Image.fromarray(data), (0, height))
    draw = ImageDraw.Draw(canvas)
    title = ImageFont.load_default(size=26)
    label = ImageFont.load_default(size=16)
    ink = (35, 43, 54)
    draw.text((24, 12), "Merged floor scan", font=title, fill=ink)
    draw.text((width - 210, 17), "Metric plan  |  +Y up", font=label, fill=ink)
    legend = ("Floor", "4-20 cm", "Above 20 cm", "102526 outline", "103223 outline")
    x = 24
    for text, color in zip(legend, colors, strict=True):
        draw.rounded_rectangle((x, 59, x + 18, 71), radius=2, fill=color)
        draw.text((x + 25, 55), text, font=label, fill=ink)
        x += int(draw.textlength(text, font=label)) + 50
    x, y = width - 105, 63
    length = round(1 / grid.resolution)
    draw.line((x, y, x + length, y), fill=ink, width=3)
    for endpoint in (x, x + length):
        draw.line((endpoint, y - 5, endpoint, y + 5), fill=ink, width=2)
    draw.text((x + length / 2, y - 21), "1 m", font=label, fill=ink, anchor="mt")
    draw.line((0, height - 1, width, height - 1), fill=(215, 221, 229))
    return canvas


def write_maps(out: Path, grid: Grid, a: trimesh.Trimesh, b: trimesh.Trimesh,
               added: trimesh.Trimesh) -> None:
    """Rasterize occupied height bands exactly and do not infer free unknown space."""
    triangles = np.concatenate((a.triangles, added.triangles))
    obstacles = np.concatenate([m.triangles[~floor_faces(m, FLOOR_BAND)] for m in (a, added)])
    occupied = raster(grid, obstacles, (.03, .16))
    ground = floor_mask(grid, a, FLOOR_BAND) | floor_mask(grid, added, FLOOR_BAND)
    # Bridge only enclosed pinholes (up to 9 cells), never open scan boundaries.
    free = small_holes(ground, 9) & ~occupied
    state = np.full(occupied.shape, 205, np.uint8)
    state[free] = 254
    state[occupied] = 0
    Image.fromarray(state).save(out / "floor_map.png")
    x, y = grid.origin
    yaml = ("image: floor_map.png\nresolution: 0.02\n"
            f"origin: [{x:.2f}, {y:.2f}, 0.0]\n"
            "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\nmode: trinary\n")
    (out / "floor_map.yaml").write_text(yaml, encoding="ascii")
    print("Occupancy (width, height), origin, occupied/free/unknown counts:",
          grid.size, grid.origin, occupied.sum(), free.sum(), (state == 205).sum(), flush=True)
    lower, upper = raster(grid, triangles, (.04, .20)), raster(grid, triangles, (.20, 3.0))
    visual_floor = floor_mask(grid, a, (-.12, .04)) | floor_mask(grid, added, (-.12, .04))
    visual = annotated(grid, (visual_floor, lower, upper, footprint(grid, a), footprint(grid, b)))
    visual.save(out / "merged_topdown.png")
    write_rooms(out, grid, free, occupied, (a, added))
