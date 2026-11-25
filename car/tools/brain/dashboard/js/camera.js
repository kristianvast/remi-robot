import { $, el } from "./display-math.js";
import { geometry, telemetry } from "./socket.js";
import { toast } from "./comms.js";

let camSource = "agent";
let streamUrl = null;
let streamMirrored = false;
let streamFailed = false;
let lastFrameId = 0;

export function configureCamera(msg) {
  streamUrl = msg.stream_url;
  streamMirrored = !!msg.stream_mirrored;
  $("src-live").parentElement.hidden = !streamUrl;
}

export function bindCamera() {
  $("src-live").addEventListener("click", () => setCamSource("live"));
  $("src-agent").addEventListener("click", () => setCamSource("agent"));
}

export function setCamSource(src = camSource) {
  camSource = src;
  $("src-live").setAttribute("aria-pressed", String(src === "live"));
  $("src-agent").setAttribute("aria-pressed", String(src === "agent"));
  const img = $("cam-img");
  if (src === "live" && streamUrl && !streamFailed) {
    img.onerror = () => {
      streamFailed = true;
      toast("Live stream unavailable, showing Remi's frames", "warn");
      setCamSource("agent");
    };
    img.onload = () => { img.hidden = false; $("cam-empty").hidden = true; };
    img.src = streamUrl;
    img.style.transform = streamMirrored ? "scaleX(-1)" : "";
  } else {
    img.style.transform = "";
    img.onerror = null;
    img.onload = () => { img.hidden = false; $("cam-empty").hidden = true; };
    if (lastFrameId) img.src = `/frame.jpg?id=${lastFrameId}`;
  }
}

export function renderCameraFrame(state) {
  if (state.frame && state.frame.id && state.frame.id !== lastFrameId) {
    lastFrameId = state.frame.id;
    if (camSource === "agent") $("cam-img").src = `/frame.jpg?id=${lastFrameId}`;
  }
  const age = state.frame && state.frame.t ? state.now - state.frame.t : null;
  const camOk = age != null && age < 4;
  $("cam-dot").className = "dot " + (camOk ? "ok" : age == null ? "" : "bad");
  $("cam-txt").textContent = age == null ? "waiting" : camOk ? (camSource === "live" && !streamFailed ? "live" : `${age.toFixed(1)}s`) : "no frames";
}

export function renderHud() {
  const svg = $("hud-svg");
  svg.innerHTML = "";
  if (!geometry || !telemetry) return;
  const head = telemetry.sensors.head - 90;
  const stopCm = telemetry.stop_cm ?? 15;

  // Corner optical frame ticks
  const frameCol = "rgba(38, 232, 255, 0.45)";
  const corners = [
    "M 12 24 L 12 12 L 24 12",
    "M 376 12 L 388 12 L 388 24",
    "M 12 276 L 12 288 L 24 288",
    "M 376 288 L 388 288 L 388 276",
  ];
  for (const d of corners) {
    el("path", { d, fill: "none", stroke: frameCol, "stroke-width": 1.2, "vector-effect": "non-scaling-stroke" }, svg);
  }

  // Head compass along top
  const cx = 200, w = 154, top = 20;
  el("line", { x1: cx - w / 2, y1: top, x2: cx + w / 2, y2: top, stroke: "rgba(38,232,255,0.45)", "stroke-width": 1, "vector-effect": "non-scaling-stroke" }, svg);
  for (const a of [-80, -40, 0, 40, 80]) {
    const x = cx - (a / 80) * (w / 2);
    el("line", { x1: x, y1: top - (a === 0 ? 6 : 3), x2: x, y2: top + (a === 0 ? 6 : 3), stroke: "rgba(220,246,255,0.72)", "vector-effect": "non-scaling-stroke" }, svg);
  }
  const hx = cx - (head / 80) * (w / 2);
  el("path", { d: `M ${hx} ${top + 4} l -5 7 h 10 z`, fill: "var(--accent)" }, svg);
  const ht = el("text", { x: cx, y: top + 24, fill: "rgba(220,246,255,0.92)", "font-size": 10, "font-family": "var(--mono)", "text-anchor": "middle", style: "paint-order:stroke;stroke:rgba(0,0,0,0.75);stroke-width:3px" }, svg);
  ht.textContent = head === 0 ? "HEAD 0° CTR" : `HEAD ${Math.abs(head)}° ${head > 0 ? "L" : "R"}`;
  el("text", { x: cx - w / 2 - 6, y: top + 3.5, fill: "rgba(220,246,255,0.8)", "font-size": 9.5, "font-family": "var(--mono)", "text-anchor": "end" }, svg).textContent = "L";
  el("text", { x: cx + w / 2 + 6, y: top + 3.5, fill: "rgba(220,246,255,0.8)", "font-size": 9.5, "font-family": "var(--mono)" }, svg).textContent = "R";

  // Sonar reticle inside camera
  const fx = 200;
  const fy = 150 + (geometry.camera_tilt_deg / geometry.camera_vfov_deg) * 300;
  const rx = (geometry.sonar_cone_deg / geometry.camera_hfov_deg) * 400 / 2;
  const ry = (geometry.sonar_cone_deg / geometry.camera_vfov_deg) * 300 / 2;
  const d = telemetry.sensors.distance;
  const danger = d != null && d < stopCm;
  const warn = d != null && !danger && d < stopCm * 2;
  const color = danger ? "var(--danger)" : warn ? "var(--warn)" : "rgba(38,232,255,0.92)";
  el("ellipse", { cx: fx, cy: fy, rx, ry, fill: danger ? "rgba(255,56,80,0.16)" : "rgba(38,232,255,0.06)", stroke: color, "stroke-width": 1.4, "stroke-dasharray": "5 3", "vector-effect": "non-scaling-stroke" }, svg);
  el("line", { x1: fx - 7, y1: fy, x2: fx + 7, y2: fy, stroke: color, "vector-effect": "non-scaling-stroke" }, svg);
  el("line", { x1: fx, y1: fy - 7, x2: fx, y2: fy + 7, stroke: color, "vector-effect": "non-scaling-stroke" }, svg);

  const read = $("hud-read");
  read.style.left = "50%";
  read.style.top = `${((fy - ry) / 300) * 100}%`;
  read.style.transform = "translate(-50%, calc(-100% - 6px))";
  $("hud-dist").textContent = d == null ? "--" : d >= geometry.sonar_max_cm ? "CLEAR" : `${d} cm`;
  $("hud-dist").style.color = danger ? "var(--danger)" : warn ? "var(--warn)" : "#e8fbff";
}
