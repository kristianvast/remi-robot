import { $, el, rad } from "./display-math.js";
import { geometry, telemetry } from "./socket.js";

let profileDist = null;
export function drawProfile(stopCm = 15) {
  if (!geometry) return;
  const svg = $("profile");
  svg.innerHTML = "";
  const Z = 2;
  const y = (z) => -z * Z;
  const half = rad(geometry.sonar_cone_deg / 2);
  const sx = geometry.sonar_x_cm, sz = geometry.sonar_height_cm;
  const floorHit = sx + sz / Math.tan(half);
  const xEnd = 98;
  const lowerAt = (x) => Math.max(0, sz - (x - sx) * Math.tan(half));
  const stopX = sx + stopCm;
  const minH = lowerAt(stopX);

  el("line", { x1: -14, y1: 0, x2: 100, y2: 0, stroke: "var(--muted)", "stroke-width": 1, "vector-effect": "non-scaling-stroke" }, svg);
  for (let x = 0; x <= 90; x += 10) {
    el("line", { x1: x, y1: 0, x2: x, y2: 1.5, stroke: "var(--faint)", "vector-effect": "non-scaling-stroke" }, svg);
    if (x % 20 === 0 && x > 0) {
      el("text", { x, y: 4.3, "font-size": 2.7, fill: "var(--faint)", "text-anchor": "middle", "font-family": "var(--mono)" }, svg).textContent = `${x}`;
    }
  }

  el("path", { d: `M ${geometry.nose_x_cm} 0 L ${geometry.nose_x_cm} ${y(lowerAt(geometry.nose_x_cm))} L ${floorHit} 0 Z`, fill: "rgba(255,56,80,0.16)", stroke: "none" }, svg);
  const upperEnd = sz + (xEnd - sx) * Math.tan(half);
  el("path", { d: `M ${sx} ${y(sz)} L ${xEnd} ${y(upperEnd)} L ${xEnd} 0 L ${floorHit} 0 Z`, fill: "rgba(38,232,255,0.14)" }, svg);
  el("line", { x1: sx, y1: y(sz), x2: xEnd, y2: y(upperEnd), stroke: "rgba(38,232,255,0.68)", "vector-effect": "non-scaling-stroke" }, svg);
  el("line", { x1: sx, y1: y(sz), x2: floorHit, y2: 0, stroke: "rgba(38,232,255,0.68)", "vector-effect": "non-scaling-stroke" }, svg);

  const cx = geometry.camera_x_cm, cz = geometry.camera_height_cm, tilt = rad(geometry.camera_tilt_deg), hv = rad(geometry.camera_vfov_deg / 2);
  const camLow = tilt - hv;
  const camLowEnd = camLow < 0 ? { x: cx + cz / Math.tan(-camLow), z: 0 } : { x: xEnd, z: cz + (xEnd - cx) * Math.tan(camLow) };
  const camHighZ = Math.min(22, cz + (xEnd - cx) * Math.tan(tilt + hv));
  el("line", { x1: cx, y1: y(cz), x2: Math.min(camLowEnd.x, xEnd), y2: y(camLowEnd.z), stroke: "var(--agent)", "stroke-dasharray": "3 2", "vector-effect": "non-scaling-stroke" }, svg);
  el("line", { x1: cx, y1: y(cz), x2: xEnd, y2: y(camHighZ), stroke: "var(--agent)", "stroke-dasharray": "3 2", "vector-effect": "non-scaling-stroke" }, svg);

  el("line", { x1: stopX, y1: 0, x2: stopX, y2: y(minH), stroke: "var(--warn)", "stroke-width": 1.5, "vector-effect": "non-scaling-stroke" }, svg);
  el("text", { x: stopX + 1.4, y: y(minH / 2) + 1, "font-size": 2.9, fill: "var(--warn)", "font-family": "var(--mono)" }, svg).textContent = `${minH.toFixed(1)}cm`;
  el("text", { x: (geometry.nose_x_cm + floorHit) / 2 + 10, y: -1.4, "font-size": 2.8, fill: "var(--danger)", "text-anchor": "middle", "font-family": "var(--mono)" }, svg).textContent = "BLIND";
  el("text", { x: 80, y: y(sz + (80 - sx) * Math.tan(half)) + 5, "font-size": 2.8, fill: "var(--accent)", "text-anchor": "middle", "font-family": "var(--mono)" }, svg).textContent = "SONAR";

  const car = el("g", {}, svg);
  el("rect", { x: -geometry.body_length_cm / 2, y: y(6.7), width: geometry.body_length_cm, height: (6.7 - 1.8) * Z, rx: 1, fill: "var(--surface-3)", stroke: "var(--accent)", "vector-effect": "non-scaling-stroke" }, car);
  el("rect", { x: -9, y: y(8.4), width: 18, height: 1.2 * Z, fill: "var(--surface-2)", stroke: "var(--muted)", "vector-effect": "non-scaling-stroke" }, car);
  el("rect", { x: 7.6, y: y(13.4), width: 1.6, height: (13.4 - 7.9) * Z, fill: "var(--muted)" }, car);
  el("rect", { x: sx - 1.2, y: y(sz + 1.3), width: 2.4, height: 2.6 * Z, rx: 0.5, fill: "var(--accent)" }, car);
  el("rect", { x: cx - 1, y: y(cz + 0.9), width: 2, height: 1.8 * Z, rx: 0.4, fill: "var(--agent)" }, car);
  for (const wx of [-geometry.wheel_x_cm, geometry.wheel_x_cm]) {
    el("ellipse", { cx: wx, cy: y(geometry.wheel_radius_cm), rx: geometry.wheel_radius_cm, ry: geometry.wheel_radius_cm * Z, fill: "var(--faint)" }, car);
  }
  el("circle", { cx: geometry.line_x_cm, cy: y(0.4), r: 0.85, fill: "var(--ok)" }, car);

  profileDist = el("g", {}, svg);
  $("blind-note").innerHTML = `Floor hit <b>${Math.round(floorHit - sx)} cm</b> ahead · at <b>${stopCm} cm</b> stop ring, objects below <b>${minH.toFixed(1)} cm</b> sit in sonar blind zone.`;
}

export function renderProfileDistance() {
  if (!profileDist || !telemetry || !geometry) return;
  profileDist.innerHTML = "";
  const d = telemetry.sensors.distance;
  const stopCm = telemetry.stop_cm ?? 15;
  const centered = Math.abs(telemetry.sensors.head - 90) <= 10;
  if (d == null || d >= geometry.sonar_max_cm || !centered) return;
  const x = geometry.sonar_x_cm + d;
  if (x > 98) return;
  const half = rad(geometry.sonar_cone_deg / 2);
  const lo = Math.max(0, geometry.sonar_height_cm - d * Math.tan(half));
  const hi = geometry.sonar_height_cm + d * Math.tan(half);
  const col = d < stopCm ? "var(--danger)" : d < stopCm * 2 ? "var(--warn)" : "var(--accent)";
  el("line", { x1: x, y1: -lo * 2, x2: x, y2: -hi * 2, stroke: col, "stroke-width": 2.8, "stroke-linecap": "round", "vector-effect": "non-scaling-stroke" }, profileDist);
}
