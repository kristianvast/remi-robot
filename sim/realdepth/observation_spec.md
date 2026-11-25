# Depth observation spec (simulator and real car)

Version 1, 2026-11-24. Both sides must produce exactly this array. The Mac side is
implemented in `sim/realdepth/realdepth.py` (`observe(rgb)`); its `encode(z)` is the
reference for step 3 below and the simulator should copy it verbatim.

## 1. Camera model (both sides)

| Item | Value |
|---|---|
| Output size | 160 x 120 (W x H), one channel |
| Projection | pinhole, fx = fy = 125.58 px, cx = 80.0, cy = 60.0 (pixel-centre convention: pixel (u, v) centre at u + 0.5) |
| Field of view | 65.0 deg horizontal, 51.1 deg vertical (same as fx = 502.3 px at 640 x 480) |
| Mount | 0.126 m above floor, ~0.08 m ahead of base centre, pitch +20 deg (up), roll 0 |
| Quantity | z = distance along the optical axis in metres (NOT ray length) |

Simulator: render z directly at 160 x 120 with these intrinsics. Per-batch camera
randomisation from `calibration.yaml` stays; keep HFOV within 60-70 deg.

Real car: the car's true intrinsics are not calibrated yet (see section 6). Until they
are, `realdepth.py` assumes the values above and does no undistortion or crop: the full
640 x 480 frame is resized to 320 x 240. Once calibrated, undistort and resample the
640 x 480 frame to the pinhole above before inference; do not change the observation.

## 2. Real-car pipeline (reference: realdepth.py)

1. JPEG 640 x 480 from the car -> RGB uint8. Image is used as delivered: the bench
   photos are already upright; do NOT rotate them 180 deg (that makes them upside down).
2. Resize to 320 x 240 with area interpolation.
3. Apple Depth Pro 2-S (`apple/DepthPro2-S`), PyTorch MPS, bfloat16, focal length
   given as f_px = 251.15 (502.3 x 320/640), focal estimator off. Output z (m), inf = sky.
4. 2 x 2 block mean of inverse depth (1/z, z clipped >= 0.15 m) -> 160 x 120, back to z.
   (Averaging inverse depth biases mixed pixels toward the nearer surface: deliberate.)
5. Encode (section 3).

Fallback if the Depth Pro 2 licence is unacceptable: Depth Anything V2 Metric-Indoor-Small
(Apache-2.0) as Core ML at 336 x 252 (`models/coreml/da2_metric_336x252.mlpackage`),
its metres output resized to 160 x 120 the same way. Same observation, same corruptions,
but use the "DA2" magnitudes in section 4 where they differ.

## 3. Encoding (both sides, identical code)

```
Z_NEAR, Z_FAR = 0.15, 5.0
z   = clip(z, Z_NEAR, Z_FAR)            # inf / NaN / no hit -> Z_FAR before clipping
obs = 1 - log(z / Z_NEAR) / log(Z_FAR / Z_NEAR)
```

* dtype float32, shape (120, 160) (add a leading channel axis in the policy if needed).
* 1.0 = 0.15 m or nearer, 0.0 = 5 m or farther, sky, no hit, or invalid.
  Reference points: 0.3 m -> 0.80, 0.5 m -> 0.66, 1 m -> 0.46, 2 m -> 0.26, 3 m -> 0.15.
* Metric, not per-image normalised: the policy may rely on absolute distance, so the
  scale errors below must be simulated. Never min-max normalise per frame.
* Log spacing gives every distance the same relative resolution (one 8-bit step is ~1.4 %
  of depth) and keeps 0.3-3 m, the range that matters for avoidance, in 0.15-0.80.

## 4. Corruptions the simulator must apply to its perfect z

Apply in this order, on z in metres at 160 x 120, before encoding. "Measured" values
come from runs in `sim/realdepth/results/` (Depth Pro 2-S 320 unless stated); "estimated"
values are judgement from the depth sheets and should be widened if the policy is brittle.

| # | Corruption | How | Magnitude |
|---|---|---|---|
| 1 | Global scale error, per episode | z *= s, log s ~ U(log 0.75, log 1.33) | Measured: object heights off by -9 to +5 % (DP2-S 320, n = 3), -31 to +29 % over all metric models; the car's focal length is uncertain by up to ~18 % (estimator says fx 594 vs assumed 502) and DP2 scale is proportional to it. DA2: same range. |
| 2 | Per-frame scale jitter | z *= exp(e_t), e_t = 0.5 e_(t-1) + N(0, 0.087) (stationary sd 0.10) | Measured: sd of global scale over 12 noisy re-encodes of the same photo 9.6 % (DA2-336: 8.4 %); median frame-to-frame change on a film sequence 6.1 % (includes real motion). |
| 3 | Low-frequency shape error | z *= exp(f), f = Gaussian-blurred white noise, blur sigma 20 px, rescaled to sd 0.10 | Estimated from cross-model rank agreement (Spearman 0.85-0.92 between models on the same frames: they disagree on slopes, not edges). |
| 4 | Close textureless surfaces flattened | if > 60 % of pixels have z < 0.6 m: with p = 0.5 replace z by median(z) * exp(N(0, 0.3)) over the whole frame, plus corruption 3 | Observed on car photo 5 and film frame 160 (close wall / cabinet): metric outputs became a nearly uniform plane. |
| 5 | Edge blur / bleeding | blur 1/z with Gaussian sigma ~ U(0.3, 1.0) px; then with p = 0.5 dilate near objects 1 px (grey-dilate 1/z with 3 x 3) | Measured 10-90 % edge width on the sim proof frames: 3 px at 640 x 480 = 0.75 px here (DA2-336: 5 px = 1.25 px). Real frames are softer, hence the range. |
| 6 | Thin structures lost | morphological opening of 1/z with 2 x 2 kernel (removes near structures <= ~1.5 px wide), applied with p = 0.7; structures 2-3 px wide keep 50-100 % of their contrast | Observed: mop handle (~5 px at 640 = 1.2 px here) faint or missing; chair legs (~12 px at 640 = 3 px here) kept by DP2, smeared by DA2-336. Cables and table edges seen edge-on are likely lost. |
| 7 | Pixel noise | z *= exp(n), n = N(0, 0.06) blurred with sigma 1.5 px and rescaled to sd 0.06 | Measured: per-pixel sd of depth over noisy re-encodes 11.5 % total, minus 9.6 % global part = ~6 % local. |
| 8 | Windows, mirrors, lights | on walls in view, with p = 0.3 per episode cut 1-2 rectangles (20-60 px wide, upper image half) and set z to "far": z = 1.8 x median(z) (DP2) or 1.2 x median(z) (DA2) | Measured on saturated pixels (8 photos/frames with blown areas): DP2 inverse depth there 0.53-0.56 x the image median (reads ~1.8 x farther); DA2 0.78-0.82 x. Mirrors: unobserved, treat like windows. |
| 9 | Synthetic upper walls / ceiling | pixels whose world point is above 1.2 m (generated geometry in the scan) get an extra per-episode z *= U(0.7, 1.4) | Estimated: the scan has real walls only to 1-1.5 m; real ceilings in the car photos were reconstructed as smooth far planes with scale unknown. |
| 10 | Whole-frame failure | with p = 0.03 per step (bursts of 1-3 steps) replace the whole frame by one of: all far (z = inf, obs 0), all near uniform (z = U(0.3, 0.8) m), or a smooth random field | Measured: 1 of 8 car photos (motion blur + blown ceiling light) failed completely: DP2 returned infinite depth everywhere, DA2 metric a uniform ~0.5 m wall. No image statistic separated it from good frames, so the real side cannot flag it; the policy must tolerate it (ultrasonic range remains in the observation). The 3 % rate is a guess for a moving car. |
| 11 | Latency | observation lags the action by one 100 ms step | Mac side: 11-14 ms; the Wi-Fi JPEG transfer is unmeasured (see section 6). |

Corruptions 1, 2, 7 and 10 are the ones a policy trained on clean depth would most
likely fail on: an all-far frame looks like open space.

## 5. What not to simulate

* Per-image min-max normalisation (the real side never does it).
* Holes / NaNs: the model outputs a value everywhere; only sky gives inf (-> obs 0).
* Colour or texture effects: the policy sees depth only.

## 6. Open items that change numbers here

* Car camera intrinsics and distortion: calibrate with a checkerboard (the bench photos
  show clear barrel distortion; HFOV estimates range 57-65 deg). This sets fx in step 3
  and removes most of corruption 1's focal-length share.
* Depth at the real 12.6 cm viewpoint: no car frame from floor level exists yet; all
  magnitudes come from bench photos (camera on a workbench) and standing-height film.
* Wi-Fi JPEG latency and frame drops from the ESP32 camera.
