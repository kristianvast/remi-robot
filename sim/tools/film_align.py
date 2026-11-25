"""Align the film's COLMAP model to the metric building scan and draw the coverage map.

Usage (sim venv, project root):
  sim/.venv/bin/python -B sim/tools/film_align.py align  <colmap_txt_model_dir> <tag>   (once per SfM model)
  sim/.venv/bin/python -B sim/tools/film_align.py merge  <tag> [<tag> ...]
  sim/.venv/bin/python -B sim/tools/film_align.py coverage

align writes scans/film/work/align_<tag>.json (x_scan = s * R @ x_sfm + t) and cams_<tag>.json;
merge writes scans/film/splat/transform.json (one transform per SfM model) and
scans/film/work/film_cams_metric.json. Method: gravity from the normal of the camera-path plane
(a level walk) when the path is planar, else the mean camera up vector; floor plane by RANSAC below
the cameras; scale is searched so the camera is 1.15-1.75 m above that floor (the film was shot at
standing height; free-scale fitting shrinks the cloud into furniture clutter), a brute-force
yaw x scale search with FFT translation correlation of SfM points at 0.8-1.8 m against the scan's
vertical surfaces at 0.8-1.6 m plus a floor-colour channel (SfM floor points vs building.glb floor
texture, classed as red / green / wood / grey by hue and saturation), then trimmed rigid ICP in 3D against 2 M surface samples of
building.glb at each candidate's scale. Candidates are ranked by inlier fraction (5 cm) times the
fraction of camera-to-point sight lines that do not pass through scan surfaces above 0.25 m
(the last 0.3 m before each point is ignored).

coverage writes scans/film/coverage.png and scans/film/work/covered.npy: floor-map cells within
5 m of a registered camera, inside its horizontal field of view, with a clear line of sight on a
raster of scan surfaces between 1.0 and 1.8 m (walls, tall furniture).
"""
import json
import math
import os
import sys

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FILM = os.path.join(ROOT, "scans", "film")
WORK = os.path.join(FILM, "work")
MERGED = os.path.join(ROOT, "scans", "merged")
ASSUMED_CAM_HEIGHT = 1.45
CELL = 0.05


def qvec2rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                     [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                     [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def read_model(d):
    cams = {}
    for line in open(os.path.join(d, "cameras.txt")):
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        cams[int(p[0])] = dict(model=p[1], w=int(p[2]), h=int(p[3]), params=[float(v) for v in p[4:]])
    imgs = []
    lines = [l for l in open(os.path.join(d, "images.txt")) if not l.startswith("#")]
    for i in range(0, len(lines), 2):
        p = lines[i].split()
        if len(p) < 10:
            continue
        Rw = qvec2rot([float(v) for v in p[1:5]])
        t = np.array([float(v) for v in p[5:8]])
        imgs.append(dict(id=int(p[0]), name=p[9], cam=int(p[8]), R=Rw, t=t, C=-Rw.T @ t))
    imgs.sort(key=lambda x: x["name"])
    xyz, rgb, err, obs = [], [], [], []
    for line in open(os.path.join(d, "points3D.txt")):
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        xyz.append([float(v) for v in p[1:4]])
        rgb.append([int(v) for v in p[4:7]])
        err.append(float(p[7]))
        obs.append(int(p[8]))
    read_model.first_obs = np.array(obs)
    return cams, imgs, np.array(xyz), np.array(rgb), np.array(err)


def rot_a_to_b(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


def floor_plane(P, C, g, rng):
    hc = C @ g
    cand = P[(P @ g) < np.median(hc)]
    ext = np.ptp(C, axis=0).max()
    thr = 0.004 * ext
    best = []
    for _ in range(4000):
        a, b, c = cand[rng.choice(len(cand), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-12:
            continue
        n /= np.linalg.norm(n)
        if n @ g < 0:
            n = -n
        if n @ g < 0.97:
            continue
        d = (cand - a) @ n
        inl = int((np.abs(d) < thr).sum())
        best.append((inl, float(a @ n), n))
    best.sort(key=lambda x: -x[0])
    top = [b for b in best[:200] if b[0] >= 0.5 * best[0][0]]
    inl, off, n = min(top, key=lambda b: b[1])
    pts = cand[np.abs(cand @ n - off) < thr]
    centroid = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - centroid)
    n = vt[2] if vt[2] @ g > 0 else -vt[2]
    return n, centroid, len(pts)


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    Rm = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / xs.var(0).sum()
    return s, Rm, mu_d - s * Rm @ mu_s


def rigid_refit(s, src, dst):
    xs = s * src
    mu_s, mu_d = xs.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((dst - mu_d).T @ (xs - mu_s))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    Rm = U @ D @ Vt
    return Rm, mu_d - Rm @ mu_s


def floor_class(rgb):
    x = rgb.astype(np.float32) / 255.0
    mx, mn = x.max(1), x.min(1)
    sat = (mx - mn) / np.maximum(mx, 1e-3)
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    d = np.maximum(mx - mn, 1e-3)
    hue = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    cls = np.full(len(x), 3)
    cls[(sat > 0.35) & ((hue < 12) | (hue > 320))] = 0
    cls[(sat > 0.15) & (hue > 75) & (hue < 210)] = 1
    cls[(sat > 0.30) & (hue >= 12) & (hue <= 50)] = 2
    cls[(sat >= 0.15) & (cls == 3)] = -1
    return cls


def raster(xy, lo, shape):
    ij = np.floor((xy - lo) / CELL).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < shape[0]) & (ij[:, 1] < shape[1])
    img = np.zeros(shape, np.float32)
    np.add.at(img, (ij[ok, 0], ij[ok, 1]), 1.0)
    return img


def occupancy(spts):
    spts = spts[spts[:, 2] > 0.25]
    lo = spts.min(0) - 0.5
    shape = tuple(np.ceil((spts.max(0) + 0.5 - lo) / CELL).astype(int))
    occ = np.zeros(shape, bool)
    ijk = np.floor((spts - lo) / CELL).astype(int)
    occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True

    def blocked_fraction(cams_xyz, pts_xyz, margin=0.30):
        v = pts_xyz - cams_xyz
        L = np.linalg.norm(v, axis=1)
        hit = np.zeros(len(L), bool)
        for f in np.arange(0.0, 1.0, 0.01):
            q = cams_xyz + v * f
            ok = (L * f > 0.10) & (L * (1 - f) > margin)
            ijk = np.floor((q - lo) / CELL).astype(int)
            inb = ok & np.all((ijk >= 0) & (ijk < shape), axis=1)
            hit[inb] |= occ[ijk[inb, 0], ijk[inb, 1], ijk[inb, 2]]
        return float(hit.mean())
    return blocked_fraction


def cmd_align(model_dir, tag):
    rng = np.random.default_rng(0)
    cams, imgs, P, rgb, err = read_model(model_dir)
    first_obs = read_model.first_obs[err < 2.0]
    P = P[err < 2.0]
    C = np.array([im["C"] for im in imgs])
    cam_of_id = {im["id"]: im["C"] for im in imgs}
    ups = np.array([-im["R"][1] for im in imgs])
    g = ups.mean(0)
    g /= np.linalg.norm(g)
    _, sig, vt = np.linalg.svd(C - C.mean(0))
    planarity = float(sig[2] / sig[0])
    if planarity < 0.08:
        g = vt[2] if vt[2] @ g > 0 else -vt[2]
    n, p0, n_floor = floor_plane(P, C, g, rng)
    h_sfm = float(np.median((C - p0) @ n))
    s0 = ASSUMED_CAM_HEIGHT / h_sfm
    Rg = rot_a_to_b(n, np.array([0.0, 0.0, 1.0]))
    print("ALIGN %s: %d cams, %d points, path planarity %.3f, floor inliers %d, gravity-vs-floor %.1f deg, "
          "cam height %.4f sfm units" % (tag, len(C), len(P), planarity, n_floor,
                                          math.degrees(math.acos(min(1, g @ n))), h_sfm))

    S = np.load(os.path.join(WORK, "scan_samples.npz"))
    spts, snrm = S["pts"].astype(np.float64), S["nrm"]
    wall = (np.abs(snrm[:, 2]) < 0.3) & (spts[:, 2] > 0.8) & (spts[:, 2] < 1.6)
    lo = spts[:, :2].min(0) - 3.0
    hi = spts[:, :2].max(0) + 3.0
    shape = tuple(np.ceil((hi - lo) / CELL).astype(int))
    target = raster(spts[wall, :2], lo, shape)
    target = ndimage.gaussian_filter(np.minimum(target, 3.0), 2.0)
    F_t = np.fft.rfft2(target)
    SC = np.load(os.path.join(WORK, "scan_colour_samples.npz"))
    floor_s = SC["pts"][:, 2] < 0.10
    s_cls = floor_class(SC["col"][floor_s])
    F_col = []
    for k in range(4):
        tk = raster(SC["pts"][floor_s][s_cls == k, :2].astype(np.float64), lo, shape)
        F_col.append(np.fft.rfft2(ndimage.gaussian_filter(np.minimum(tk, 3.0), 3.0)))
    p_cls = floor_class(rgb[err < 2.0])

    Pl = (P - p0) @ Rg.T
    Cl = (C - p0) @ Rg.T
    floor_known = known_floor_mask()
    results = []
    for f in np.arange(1.15, 1.76, 0.025) / ASSUMED_CAM_HEIGHT:
        s = s0 * f
        Q = Pl * s
        bm = (Q[:, 2] > 0.8) & (Q[:, 2] < 1.8)
        band = Q[bm]
        fm_ = (Q[:, 2] < 0.15) & (Q[:, 2] > -0.15)
        if f == np.arange(1.15, 1.76, 0.025)[0] / ASSUMED_CAM_HEIGHT:
            print("ALIGN floor-colour classes red/green/wood/grey/other:",
                  [int((p_cls[fm_] == k).sum()) for k in (0, 1, 2, 3, -1)])
        for yaw in np.arange(0, 360, 1.0):
            c, si = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
            rot2 = np.array([[c, si], [-si, c]])
            ctr = (band[:, :2] @ rot2).mean(0)
            off = -ctr + (hi - lo) / 2 + lo
            src = raster(band[:, :2] @ rot2 + off, lo, shape)
            spec = F_t * np.conj(np.fft.rfft2(src)) / len(band)
            fxy = Q[fm_, :2] @ rot2 + off
            nf = max(1, int((p_cls[fm_] >= 0).sum()))
            for k in range(4):
                sk = raster(fxy[p_cls[fm_] == k], lo, shape)
                if sk.any():
                    spec = spec + F_col[k] * np.conj(np.fft.rfft2(sk)) / nf
            corr = np.fft.irfft2(spec, s=shape)
            k = int(np.argmax(corr))
            di, dj = np.unravel_index(k, shape)
            di = di - shape[0] if di > shape[0] // 2 else di
            dj = dj - shape[1] if dj > shape[1] // 2 else dj
            t2 = off + np.array([di, dj]) * CELL
            results.append((float(corr.flat[k]), f, yaw, t2))
    results.sort(key=lambda r: -r[0])
    print("ALIGN top coarse candidates (score, scale factor, yaw):",
          [(round(r[0], 4), round(r[1], 2), r[2]) for r in results[:5]])

    tree = cKDTree(spts)
    blocked_fraction = occupancy(spts)
    sub = rng.choice(len(P), min(4000, len(P)), replace=False)
    Csub = np.array([cam_of_id[i] for i in first_obs[sub]])
    best = None
    seen = []
    for score, f, yaw, t2 in results:
        if any(abs(f - f2) < 0.05 and abs((yaw - y2 + 180) % 360 - 180) < 8 for f2, y2 in seen):
            continue
        seen.append((f, yaw))
        if len(seen) > 16:
            break
        c, si = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        Rz = np.array([[c, -si, 0], [si, c, 0], [0, 0, 1.0]])
        s = s0 * f
        R_tot = Rz @ Rg
        t_tot = np.array([t2[0], t2[1], 0.0]) - s * R_tot @ p0
        for thr in (0.40, 0.25, 0.15, 0.10, 0.08, 0.08, 0.08):
            X = s * P @ R_tot.T + t_tot
            d, j = tree.query(X, distance_upper_bound=thr)
            m = np.isfinite(d)
            if m.sum() < 100:
                break
            R_tot, t_tot = rigid_refit(s, P[m], spts[j[m]])
        X = s * P @ R_tot.T + t_tot
        d, _ = tree.query(X)
        inl = d < 0.10
        Cm = s * C @ R_tot.T + t_tot
        cam_ok = float(np.mean([floor_known(x, y) for x, y in Cm[:, :2]]))
        blocked = blocked_fraction(s * Csub @ R_tot.T + t_tot, X[sub])
        quality = float((d < 0.05).mean()) * (1 - blocked) * cam_ok
        print("ALIGN cand H=%.2f yaw=%5.1f coarse=%.4f -> inl5=%.3f inl10=%.3f blocked=%.3f cams-in=%.2f q=%.4f"
              % (f * ASSUMED_CAM_HEIGHT, yaw, score, (d < 0.05).mean(), inl.mean(), blocked, cam_ok, quality))
        if best is None or quality > best[0]:
            best = (quality, s, R_tot, t_tot, d, cam_ok)
    _, s, R_tot, t_tot, d, cam_ok = best
    inl = d < 0.10
    tilt = math.degrees(math.acos(min(1.0, (R_tot @ n)[2])))
    Cm = C * s @ R_tot.T + t_tot
    heights = Cm[:, 2]
    res = dict(
        convention="x_scan = scale * R @ x_sfm + t (metres, scan frame +Z up, floor Z=0)",
        scale=float(s), R=R_tot.tolist(), t=t_tot.tolist(),
        matrix_4x4=np.vstack([np.hstack([s * R_tot, t_tot[:, None]]), [0, 0, 0, 1]]).tolist(),
        residual_cm=dict(
            median_inlier=round(float(np.median(d[inl])) * 100, 2),
            rms_inlier=round(float(np.sqrt(np.mean(d[inl] ** 2))) * 100, 2),
            inlier_threshold_cm=10, inlier_fraction=round(float(inl.mean()), 3),
            median_all_points=round(float(np.median(d)) * 100, 2), n_points=int(len(d)),
            measured_as="nearest of 2M area-weighted building.glb surface samples to each SfM point (err<2px)"),
        floor_tilt_after_icp_deg=round(tilt, 2), path_planarity=round(planarity, 4),
        frames=[im["name"] for im in imgs],
        camera_height_m=dict(median=round(float(np.median(heights)), 3),
                             p5=round(float(np.percentile(heights, 5)), 3),
                             p95=round(float(np.percentile(heights, 95)), 3)),
        cams_inside_known_floor=round(cam_ok, 3),
        seed_scale_from_assumed_height=dict(assumed_m=ASSUMED_CAM_HEIGHT, sfm_units=h_sfm))
    with open(os.path.join(WORK, "align_%s.json" % tag), "w") as fh:
        json.dump(res, fh, indent=1)
    cam = cams[imgs[0]["cam"]]
    out = []
    for im in imgs:
        Rc = R_tot @ im["R"].T
        out.append(dict(name=im["name"], C=(s * R_tot @ im["C"] + t_tot).tolist(), R_c2w=Rc.tolist()))
    with open(os.path.join(WORK, "cams_%s.json" % tag), "w") as fh:
        json.dump(dict(camera=cam, cams=out), fh)
    print("ALIGN result", json.dumps({k: res[k] for k in ("scale", "residual_cm", "floor_tilt_after_icp_deg",
                                                         "camera_height_m", "cams_inside_known_floor")}))


def cmd_merge(tags):
    models, cams, camera = {}, [], None
    for tag in tags:
        models[tag] = json.load(open(os.path.join(WORK, "align_%s.json" % tag)))
        c = json.load(open(os.path.join(WORK, "cams_%s.json" % tag)))
        camera = camera or c["camera"]
        cams += [dict(x, model=tag) for x in c["cams"]]
    cams.sort(key=lambda x: x["name"])
    joins = []
    for a, b in zip(cams, cams[1:]):
        if a["model"] != b["model"]:
            gap = int(b["name"][6:10]) - int(a["name"][6:10])
            dist = float(np.linalg.norm(np.array(a["C"]) - np.array(b["C"])))
            joins.append(dict(from_frame=a["name"], to_frame=b["name"], frame_gap=gap, distance_m=round(dist, 2)))
    os.makedirs(os.path.join(FILM, "splat"), exist_ok=True)
    with open(os.path.join(FILM, "splat", "transform.json"), "w") as fh:
        json.dump(dict(note="one similarity per SfM model; x_scan = scale * R @ x_sfm + t", models=models,
                       joins_between_models=joins), fh, indent=1)
    with open(os.path.join(WORK, "film_cams_metric.json"), "w") as fh:
        json.dump(dict(camera=camera, cams=cams), fh)
    print("MERGE %d cams from %s; joins %s" % (len(cams), tags, joins))


class FloorMap:
    def __init__(self):
        from PIL import Image
        meta = {}
        for line in open(os.path.join(MERGED, "floor_map.yaml")):
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip()
        self.res = float(meta["resolution"])
        self.ox, self.oy = [float(v) for v in meta["origin"].strip("[]").split(",")[:2]]
        self.img = np.array(Image.open(os.path.join(MERGED, meta["image"])))
        self.h, self.w = self.img.shape

    def to_px(self, x, y):
        return (np.asarray(x) - self.ox) / self.res, self.h - (np.asarray(y) - self.oy) / self.res

    def to_world(self, col, row):
        return self.ox + (np.asarray(col) + 0.5) * self.res, self.oy + (self.h - np.asarray(row) - 0.5) * self.res


def known_floor_mask():
    fm = FloorMap()
    known = ndimage.binary_closing(fm.img != 205, iterations=10)

    def inside(x, y):
        c, r = fm.to_px(x, y)
        c, r = int(c), int(r)
        return bool(0 <= r < fm.h and 0 <= c < fm.w and known[r, c])
    return inside


def cmd_coverage():
    from PIL import Image, ImageDraw, ImageFont
    fm = FloorMap()
    data = json.load(open(os.path.join(WORK, "film_cams_metric.json")))
    cam = data["camera"]
    fx = cam["params"][0]
    hfov = 2 * math.atan(cam["w"] / (2 * fx))
    S = np.load(os.path.join(WORK, "scan_samples.npz"))
    spts = S["pts"]
    tall = spts[(spts[:, 2] > 1.0) & (spts[:, 2] < 1.8)]
    c, r = fm.to_px(tall[:, 0], tall[:, 1])
    block = np.zeros((fm.h, fm.w), bool)
    ok = (r >= 0) & (r < fm.h) & (c >= 0) & (c < fm.w)
    block[r[ok].astype(int), c[ok].astype(int)] = True
    block = ndimage.binary_dilation(block, iterations=1)
    seen = np.zeros((fm.h, fm.w), np.uint16)
    rng_px = int(5.0 / fm.res)
    for cm in data["cams"]:
        C = np.array(cm["C"])
        fwd = np.array(cm["R_c2w"])[:, 2]
        yaw = math.atan2(fwd[1], fwd[0])
        c0, r0 = fm.to_px(C[0], C[1])
        visited = np.zeros((fm.h, fm.w), bool)
        for a in np.linspace(yaw - hfov / 2, yaw + hfov / 2, 120):
            dc, dr = math.cos(a), -math.sin(a)
            for k in range(1, rng_px):
                cc, rr = int(c0 + dc * k), int(r0 + dr * k)
                if not (0 <= cc < fm.w and 0 <= rr < fm.h) or block[rr, cc]:
                    break
                visited[rr, cc] = True
        seen += visited
    covered = ndimage.binary_closing(seen >= 2, iterations=3) & (fm.img != 205)
    np.save(os.path.join(WORK, "covered.npy"), covered)
    img = Image.fromarray(fm.img).convert("RGB")
    over = np.array(img)
    shade = covered
    over[shade] = (0.55 * over[shade] + 0.45 * np.array([60, 200, 90])).astype(np.uint8)
    img = Image.fromarray(over)
    d = ImageDraw.Draw(img, "RGBA")
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    except OSError:
        font = ImageFont.load_default()
    rooms = json.load(open(os.path.join(MERGED, "rooms.json")))
    room_cov = {}
    for room in rooms:
        mimg = Image.new("L", (fm.w, fm.h), 0)
        ImageDraw.Draw(mimg).polygon([tuple(map(float, fm.to_px(x, y))) for x, y in room["polygon"]], fill=1)
        mm = np.array(mimg, bool) & (fm.img == 254)
        frac = float((covered & mm).sum() / max(1, mm.sum()))
        room_cov[room["id"]] = round(frac, 2)
        d.polygon([tuple(map(float, fm.to_px(x, y))) for x, y in room["polygon"]], outline=(40, 40, 200, 255), width=2)
        cx, cy = fm.to_px(room["center"]["x"], room["center"]["y"])
        d.text((float(cx) - 20, float(cy) - 10), "%s %d%%" % (room["id"].replace("area_", "A"), round(frac * 100)),
               fill=(20, 20, 160, 255), font=font)
    path = [tuple(map(float, fm.to_px(cm["C"][0], cm["C"][1]))) for cm in data["cams"]]
    d.line(path, fill=(230, 30, 30, 255), width=3)
    for i, (x, y) in enumerate(path):
        if i % 25 == 0:
            d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(230, 30, 30, 255))
            d.text((x + 6, y - 8), data["cams"][i]["name"][6:10], fill=(150, 0, 0, 255), font=font)
    d.text((10, 10), "red: film camera path (labels = frame no.)   green: floor area the film saw (>=2 frames, <=5 m, "
           "line of sight)   blue: rooms with % of free floor covered", fill=(0, 0, 0, 255), font=font)
    known = fm.img != 205
    rr, cc = np.nonzero(known)
    box = (max(cc.min() - 40, 0), 0, min(cc.max() + 40, fm.w), min(rr.max() + 40, fm.h))
    img.crop(box).save(os.path.join(FILM, "coverage.png"))
    with open(os.path.join(WORK, "room_coverage.json"), "w") as fh:
        json.dump(room_cov, fh, indent=1)
    print("COVERAGE film hfov %.1f deg, covered %.1f m2, per room %s" % (
        math.degrees(hfov), covered.sum() * fm.res ** 2, room_cov))


if __name__ == "__main__":
    if sys.argv[1] == "align":
        cmd_align(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "merge":
        cmd_merge(sys.argv[2:])
    elif sys.argv[1] == "coverage":
        cmd_coverage()
