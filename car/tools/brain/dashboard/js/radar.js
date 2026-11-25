import { $, el, rad } from "./display-math.js";
import { geometry, telemetry } from "./socket.js";

const sonarOrigin = () => ({ x: 0, y: -geometry.sonar_x_cm });
const polar = (headDeg, r) => {
  const th = rad(headDeg - 90);
  const o = sonarOrigin();
  return { x: o.x - Math.sin(th) * r, y: o.y - Math.cos(th) * r };
};
let radarDyn = null;
let radarSweepLine = null;
let radarSweepAngle = 0;

export function drawRadarStatic(stopCm = 15) {
  if (!geometry) return;
  const svg = $("radar");
  svg.innerHTML = "";
  const o = sonarOrigin();
  const grid = el("g", { fill: "none", stroke: "rgba(38,232,255,0.16)", "stroke-width": 1, "vector-effect": "non-scaling-stroke" }, svg);
  for (const r of [50, 100, 150]) {
    el("path", { d: `M ${o.x - r} ${o.y} A ${r} ${r} 0 0 1 ${o.x + r} ${o.y}` }, grid);
    el("text", { x: o.x + 3, y: o.y - r + 10, fill: "var(--faint)", stroke: "none", "font-size": 9, "font-family": "var(--mono)" }, svg).textContent = `${r}`;
  }
  for (const a of [10, 50, 90, 130, 170]) {
    const p = polar(a, 150);
    el("line", { x1: o.x, y1: o.y, x2: p.x, y2: p.y, stroke: "rgba(38,232,255,0.11)", "stroke-dasharray": "2 4", "vector-effect": "non-scaling-stroke" }, svg);
  }
  el("path", {
    d: `M ${o.x - stopCm} ${o.y} A ${stopCm} ${stopCm} 0 0 1 ${o.x + stopCm} ${o.y}`,
    fill: "rgba(255, 56, 80, 0.08)",
    stroke: "var(--warn)",
    "stroke-width": 1.5,
    "stroke-dasharray": "4 3",
    "vector-effect": "non-scaling-stroke",
  }, svg);

  radarDyn = el("g", {}, svg);
  radarSweepLine = el("line", {
    x1: o.x, y1: o.y, x2: o.x, y2: o.y - 150,
    stroke: "rgba(38, 232, 255, 0.34)",
    "stroke-width": 1.2,
    "vector-effect": "non-scaling-stroke",
  }, svg);

  const L = geometry.body_length_cm, W = geometry.body_width_cm;
  const car = el("g", {}, svg);
  for (const sx of [-1, 1]) {
    for (const sy of [-1, 1]) {
      el("rect", {
        x: sx * (geometry.wheel_y_cm + 1.3) - geometry.wheel_width_cm / 2,
        y: -sy * geometry.wheel_x_cm - geometry.wheel_radius_cm,
        width: geometry.wheel_width_cm,
        height: geometry.wheel_radius_cm * 2,
        rx: 1,
        fill: "rgba(38,232,255,0.35)",
      }, car);
    }
  }
  el("rect", {
    x: -W / 2 + 2.6, y: -L / 2, width: W - 5.2, height: L, rx: 2,
    fill: "rgba(11, 25, 41, 0.95)", stroke: "var(--accent)", "stroke-width": 1.1, "vector-effect": "non-scaling-stroke",
  }, car);
  el("text", { x: 0, y: L / 2 + 12, fill: "var(--faint)", "font-size": 8.5, "text-anchor": "middle", "font-family": "var(--mono)" }, svg).textContent = `STOP ${stopCm} CM`;
}

export function renderRadar() {
  if (!radarDyn || !telemetry || !geometry) return;
  radarDyn.innerHTML = "";
  const o = sonarOrigin();
  const head = telemetry.sensors.head;
  const half = geometry.sonar_cone_deg / 2;
  const max = geometry.sonar_max_cm;
  const stopCm = telemetry.stop_cm ?? 15;

  const pts = (telemetry.sonar || []).filter((p) => p[1] < max).slice().sort((a, b) => a[0] - b[0]);
  if (pts.length >= 3) {
    const d = [
      `M ${o.x} ${o.y}`,
      ...pts.map((p) => { const q = polar(p[0], p[1]); return `L ${q.x.toFixed(1)} ${q.y.toFixed(1)}`; }),
      "Z",
    ].join(" ");
    el("path", { d, fill: "rgba(38,232,255,0.08)", stroke: "rgba(38,232,255,0.34)", "stroke-width": 1, "vector-effect": "non-scaling-stroke" }, radarDyn);
  }

  const wedge = (r, fill) => {
    const a = polar(head - half, r), b = polar(head + half, r);
    el("path", { d: `M ${o.x} ${o.y} L ${a.x} ${a.y} A ${r} ${r} 0 0 0 ${b.x} ${b.y} Z`, fill }, radarDyn);
  };
  wedge(max, "rgba(38,232,255,0.07)");
  const dist = telemetry.sensors.distance;
  if (dist != null) {
    const r = Math.min(dist, max);
    const col = dist < stopCm ? "var(--danger)" : dist < stopCm * 2 ? "var(--warn)" : "var(--accent)";
    const fillCol = dist < stopCm ? "rgba(255,56,80,0.24)" : dist < stopCm * 2 ? "rgba(255,184,56,0.22)" : "rgba(38,232,255,0.24)";
    wedge(r, fillCol);
    if (dist < max) {
      const a = polar(head - half, r), b = polar(head + half, r);
      el("path", { d: `M ${a.x} ${a.y} A ${r} ${r} 0 0 0 ${b.x} ${b.y}`, fill: "none", stroke: col, "stroke-width": 2.8, "stroke-linecap": "round", "vector-effect": "non-scaling-stroke" }, radarDyn);
    }
  }
  for (const [h, cm, age] of (telemetry.sonar || [])) {
    if (cm >= max) continue;
    const q = polar(h, cm);
    const col = cm < stopCm ? "var(--danger)" : cm < stopCm * 2 ? "var(--warn)" : "var(--accent)";
    el("circle", { cx: q.x, cy: q.y, r: 2.5, fill: col, opacity: Math.max(0.15, 1 - age / 8) }, radarDyn);
  }
  const tip = polar(head, 16);
  el("line", { x1: o.x, y1: o.y, x2: tip.x, y2: tip.y, stroke: "#fff", "stroke-width": 1.8, "stroke-linecap": "round", "vector-effect": "non-scaling-stroke" }, radarDyn);
}

export function stepRadar(dt) {
  if (radarSweepLine && geometry) {
    radarSweepAngle = (radarSweepAngle + dt * 110) % 160;
    const degSweep = 10 + radarSweepAngle;
    const p = polar(degSweep, 150);
    radarSweepLine.setAttribute("x2", p.x.toFixed(1));
    radarSweepLine.setAttribute("y2", p.y.toFixed(1));
  }
}
