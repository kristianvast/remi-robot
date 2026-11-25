import json, sys
import numpy as np
import trimesh
from PIL import Image, ImageDraw

R = sys.argv[1]
sc = trimesh.load(f"{R}/scans/merged/building.glb")
allp, allc = [], []
for name, g in sc.geometry.items():
    T = sc.graph.get(sc.graph.nodes_geometry[[sc.graph[n][1] for n in sc.graph.nodes_geometry].index(name)])[0]
    tex = np.asarray(g.visual.material.baseColorTexture.convert("RGB"))
    th, tw = tex.shape[:2]
    pts, fi = trimesh.sample.sample_surface(g, 4_000_000)
    bary = trimesh.triangles.points_to_barycentric(g.triangles[fi], pts)
    uv = (g.visual.uv[g.faces[fi]] * bary[:, :, None]).sum(1)
    px = np.clip((uv[:, 0] % 1) * (tw - 1), 0, tw - 1).astype(int)
    py = np.clip((1 - uv[:, 1] % 1) * (th - 1), 0, th - 1).astype(int)
    allp.append(trimesh.transform_points(pts, T))
    allc.append(tex[py, px])
    print(name, len(pts), T[:3, :3].round(2).tolist())
pts, col = np.concatenate(allp), np.concatenate(allc)
print("z range", np.percentile(pts[:, 2], [1, 99]))
np.savez_compressed(f"{R}/scans/film/work/scan_colour_samples.npz", pts=pts.astype(np.float32), col=col)
keep = pts[:, 2] < 0.35
pts, col = pts[keep], col[keep]
res, ox, oy, W, H = 0.02, -11.50, -11.76, 1428, 1147
c = ((pts[:, 0] - ox) / res).astype(int)
r = (H - (pts[:, 1] - oy) / res).astype(int)
ok = (c >= 0) & (c < W) & (r >= 0) & (r < H)
order = np.argsort(pts[ok, 2])
img = np.full((H, W, 3), 255, np.uint8)
img[r[ok][order], c[ok][order]] = col[ok][order]
im = Image.fromarray(img)
d = ImageDraw.Draw(im)
for room in json.load(open(f"{R}/scans/merged/rooms.json")):
    x, y = room["center"]["x"], room["center"]["y"]
    d.text(((x - ox) / res, H - (y - oy) / res), room["id"][5:], fill=(0, 0, 0))
im.save(f"{R}/scans/film/work/colour_topdown.png")
