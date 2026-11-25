"""Sheets and ground-truth-free accuracy checks over probe.py dumps.

usage: analyze.py   (reads results/depth/<config>/*.npy;
                     writes depth_sheet.jpg, depth_sheet_film.jpg, results/metrics.json)
"""
import glob
import json
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DEP = os.path.join(HERE, "results", "depth")
CAR_DIR = os.path.join(ROOT, "patrol-photos", "bench")
FILM_DIR = os.path.join(HERE, "data", "film640")
NOISE_DIR = os.path.join(HERE, "data", "noise")
SIM_DIR = os.path.join(ROOT, "sim", "isaac", "elegoo", "proof")

SHOW = ["da2rel_518_coreml_all", "da2met_518_coreml_all", "da2met_336_coreml_all",
        "dp2s_640_mps_bf16", "dp2s_320_mps_bf16"]
KIND = {c: ("inv" if c.startswith("da2rel") else "metric") for c in SHOW}
LABEL = {"da2rel_518_coreml_all": "DA2-S rel 518", "da2met_518_coreml_all": "DA2-S metric 518",
         "da2met_336_coreml_all": "DA2-S metric 336", "dp2s_640_mps_bf16": "DepthPro2-S 640",
         "dp2s_320_mps_bf16": "DepthPro2-S 320"}
CAR = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(CAR_DIR + "/*.jpg"))
FILM_SHEET = [f"frame_{i:04d}" for i in range(0, 392, 40)]
SEQ = [f"frame_{i:04d}" for i in range(200, 224)]
SIM = ["center_rgb", "left_rgb", "right_rgb"]
TW, TH = 400, 300


def load(cfg, stem, size=(160, 120)):
    a = np.load(os.path.join(DEP, cfg, stem + ".npy")).astype(np.float32)
    a = np.where(np.isfinite(a), a, 1e3 if KIND[cfg] == "metric" else 0)
    return cv2.resize(a, size, interpolation=cv2.INTER_AREA) if size else a


def inv(cfg, a):
    return 1.0 / np.clip(a, 0.05, 1e3) if KIND[cfg] == "metric" else a


def viz(cfg, a):
    if KIND[cfg] == "metric":
        t = (np.log(8.0) - np.log(np.clip(a, 0.2, 8.0))) / (np.log(8.0) - np.log(0.2))
    else:
        lo, hi = np.percentile(a, [2, 98])
        t = np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)
    return cv2.cvtColor((t * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def label(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(img, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def rgb_path(stem):
    for d in (CAR_DIR, FILM_DIR):
        p = os.path.join(d, stem + ".jpg")
        if os.path.exists(p):
            return p
    return os.path.join(SIM_DIR, stem + ".png")


def sheet(stems, out, title, extra=None):
    rows = []
    head = np.zeros((40, TW * (len(SHOW) + 1), 3), np.uint8)
    cv2.putText(head, title, (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    rows.append(head)
    for s in stems:
        tiles = [label(cv2.resize(cv2.imread(rgb_path(s)), (TW, TH), interpolation=cv2.INTER_AREA), s)]
        for c in SHOW:
            tiles.append(label(cv2.resize(viz(c, load(c, s, None)), (TW, TH), interpolation=cv2.INTER_NEAREST),
                               LABEL[c]))
        rows.append(np.hstack(tiles))
        if extra and s in extra:
            rows.append(extra[s])
    cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])


def sim_truth_row(stem):
    d = cv2.imread(os.path.join(SIM_DIR, stem.replace("_rgb", "_depth.png")))
    t = cv2.resize(d, (TW, TH), interpolation=cv2.INTER_NEAREST)
    pad = np.zeros((TH, TW * len(SHOW), 3), np.uint8)
    cv2.putText(pad, "<- simulator exact depth (log preview, black = no hit / sky)", (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return np.hstack([label(t, stem.replace("_rgb", "_depth (sim truth)")), pad])


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a.ravel())), np.argsort(np.argsort(b.ravel()))
    return float(np.corrcoef(ra, rb)[0, 1])


def ordinal(a, b, rng, n=20000, thr=np.log(1.05)):
    i, j = rng.integers(0, a.size, n), rng.integers(0, a.size, n)
    da = np.log(a.ravel()[i] + 1e-6) - np.log(a.ravel()[j] + 1e-6)
    db = np.log(b.ravel()[i] + 1e-6) - np.log(b.ravel()[j] + 1e-6)
    m = np.abs(da) > thr
    return float(np.mean(np.sign(da[m]) == np.sign(db[m])))


def consistency(stems):
    rng = np.random.default_rng(0)
    out = {}
    for x in range(len(SHOW)):
        for y in range(x + 1, len(SHOW)):
            a, b = SHOW[x], SHOW[y]
            rs = [spearman(inv(a, load(a, s)), inv(b, load(b, s))) for s in stems]
            os_ = [ordinal(inv(a, load(a, s)), inv(b, load(b, s)), rng) for s in stems]
            out[f"{a} vs {b}"] = {"spearman": round(float(np.mean(rs)), 3), "ordinal": round(float(np.mean(os_)), 3)}
    return out


def flicker():
    out = {}
    for c in SHOW:
        med = np.array([np.median(load(c, s)[30:90, 40:120]) for s in SEQ])
        out[c] = {"central_median_step_pct_p50": round(float(np.median(np.abs(np.diff(np.log(med))))) * 100, 1)}
    return out


def noise():
    out = {}
    for c in SHOW:
        rel, glob_ = [], []
        for s in CAR:
            stack = np.stack([inv(c, load(c, os.path.splitext(os.path.basename(p))[0]))
                              for p in sorted(glob.glob(f"{NOISE_DIR}/{s}_n*.jpg"))])
            rel.append(np.median(stack.std(0) / (stack.mean(0) + 1e-6)))
            glob_.append(np.std(np.log(np.median(stack.reshape(len(stack), -1), 1))))
        out[c] = {"pixel_rel_std_pct": round(float(np.median(rel)) * 100, 2),
                  "global_scale_std_pct": round(float(np.median(glob_)) * 100, 2)}
    return out


def edge_width():
    out = {}
    for c in SHOW:
        widths = []
        for s in SIM:
            d = cv2.imread(os.path.join(SIM_DIR, s.replace("_rgb", "_depth.png")), 0).astype(np.float32)
            p = cv2.resize(inv(c, load(c, s, None)), (640, 480), interpolation=cv2.INTER_LINEAR)
            gx = np.abs(np.diff(d, axis=1))
            for y in range(150, 400, 10):
                for x in np.where(gx[y] > 25)[0]:
                    if 25 < x < 615:
                        prof = p[y, x - 20:x + 21]
                        lo, hi = prof[:5].mean(), prof[-5:].mean()
                        if abs(hi - lo) < 0.05 * (abs(hi) + abs(lo)) + 1e-9:
                            continue
                        t = (prof - lo) / (hi - lo)
                        i10, i90 = np.argmax(t > 0.1), np.argmax(t > 0.9)
                        if i90 > i10:
                            widths.append(i90 - i10)
        out[c] = {"edge_10_90_px_at_640": float(np.median(widths)) if widths else None, "n_edges": len(widths)}
    return out


def blown(stems):
    out = {}
    for c in SHOW:
        r = []
        for s in stems:
            rgb = cv2.resize(cv2.imread(rgb_path(s)), (160, 120), interpolation=cv2.INTER_AREA)
            m = rgb.min(2) >= 245
            if m.sum() < 20:
                continue
            d = load(c, s)
            r.append(float(np.median(inv(c, d)[m]) / np.median(inv(c, d))))
        out[c] = {"blown_inv_depth_vs_image_median": round(float(np.median(r)), 2) if r else None, "n_frames": len(r)}
    return out


if __name__ == "__main__":
    sheet(CAR, os.path.join(HERE, "depth_sheet.jpg"),
          "CAR PHOTOS (patrol-photos/bench, NOT rotated). Bright = near. Metric: log 0.2-8 m fixed; rel: per-image")
    sheet(FILM_SHEET + SIM, os.path.join(HERE, "depth_sheet_film.jpg"),
          "FILM FRAMES - NOT FROM THE CAR (iPhone, standing height, 48.7 deg HFOV) + simulator RGB. Bright = near",
          extra={s: sim_truth_row(s) for s in SIM})
    film_all = sorted(set(FILM_SHEET + SEQ))
    metrics = {"consistency_car": consistency(CAR), "consistency_film": consistency(film_all),
               "flicker_seq": flicker(), "noise_static": noise(),
               "edge_width_sim": edge_width(), "blown_out": blown(CAR + film_all)}
    json.dump(metrics, open(os.path.join(HERE, "results", "metrics.json"), "w"), indent=1)
    print(json.dumps(metrics, indent=1))
