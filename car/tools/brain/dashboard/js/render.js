import { $, compass, deg } from "./display-math.js";
import { geometry } from "./socket.js";
import { drawRadarStatic, renderRadar } from "./radar.js";
import { drawProfile, renderProfileDistance } from "./profile.js";
import { renderCameraFrame, renderHud } from "./camera.js";
import { updateMapPrompt } from "./map-input.js";
import { renderConvo, renderTools, toast } from "./comms.js";

let lastDrawnStopCm = null;
const shownToasts = new Set();

export function drawSensorGeometry(stopCm) {
  lastDrawnStopCm = stopCm;
  drawRadarStatic(stopCm);
  drawProfile(stopCm);
}

export function render(state) {
  const s = state.sensors || {};
  const stopCm = state.stop_cm ?? 15;
  $("stop-ring-lbl").textContent = stopCm;

  if (geometry && lastDrawnStopCm !== stopCm) {
    drawSensorGeometry(stopCm);
  }

  // Link status
  const link = state.link ? state.link.state : "connecting";
  $("link-dot").className = "dot " + (link === "online" ? "ok" : link === "offline" ? "bad" : "warn");
  $("link-txt").textContent = link === "online"
    ? `online${state.link.rtt_ms != null ? ` · ${state.link.rtt_ms}ms` : ""}`
    : link;
  $("chip-link").title = (state.link && state.link.error) || "";

  // Remi agent status
  const ag = state.agent || {};
  const st = ag.state || "initializing";
  const labels = { initializing: "starting", idle: "idle", listening: "listening", thinking: "thinking", speaking: "speaking" };
  $("agent-txt").textContent = labels[st] || st;
  $("agent-dot").className = "dot " + (st === "speaking" ? "agent pulse" : st === "thinking" ? "warn pulse" : st === "listening" ? "ok" : "");
  $("agent-sub").textContent = ag.user_state === "speaking" ? "YOU SPEAKING" : st === "speaking" ? "REMI SPEAKING" : "";

  $("t-auto").setAttribute("aria-checked", String(!!ag.autonomy));
  $("t-initiative").setAttribute("aria-checked", String(!!ag.initiative));
  $("t-mic").setAttribute("aria-checked", String(!!ag.mic));
  $("t-mic-lbl").textContent = ag.mic ? "Mic on" : "Mic muted";
  const voice = $("voice");
  voice.disabled = !ag.voice;  // null when ElevenLabs speaks instead of Gemini Live
  voice.title = ag.voice ? "" : "Only for CAR_VOICE=gemini-live";
  if (ag.voice && document.activeElement !== voice) voice.value = ag.voice;

  // Memory Core
  if (state.memory) {
    $("mem-txt").textContent = `${state.memory.people ?? 0}P · ${state.memory.events ?? 0}E`;
  }

  renderCameraFrame(state);

  // Sensor metrics
  const maxCm = geometry ? geometry.sonar_max_cm : 150;
  const distTxt = s.distance == null ? "--" : s.distance >= maxCm ? "CLEAR" : `${s.distance} cm`;
  $("m-dist").textContent = distTxt;
  $("m-dist").style.color = s.distance != null && s.distance < stopCm ? "var(--danger)" : s.distance != null && s.distance < stopCm * 2 ? "var(--warn)" : "var(--accent)";

  const head = (s.head ?? 90) - 90;
  const headTxt = head === 0 ? "0° CTR" : `${Math.abs(head)}° ${head > 0 ? "L" : "R"}`;
  $("m-head").textContent = headTxt;
  $("m-rtt").textContent = state.link && state.link.rtt_ms != null ? `${state.link.rtt_ms} ms` : "--";

  // 3D viewport HUD pills
  $("c3-head").textContent = headTxt;
  $("c3-sonar").textContent = distTxt;
  $("c3-sonar").className = s.distance != null && s.distance < stopCm ? "danger" : s.distance != null && s.distance < stopCm * 2 ? "warn" : "";
  const vCms = state.pose && typeof state.pose.v_cms === "number" ? state.pose.v_cms : 0;
  $("c3-vel").textContent = `${vCms.toFixed(1)} cm/s`;

  // Motion status
  const m = state.motion || {};
  const verb = { forward: "forward", back: "backing up", left: "spinning left", right: "spinning right" }[m.dir];
  const driver = m.by === "human" ? "You" : "Remi";
  $("who").innerHTML = m.dir ? `<b>${driver}</b> // ${verb}` : "Standby // no driver";
  $("c3-dir").textContent = m.dir ? `${driver.toUpperCase()} ${m.dir.toUpperCase()}` : "IDLE";
  const who = $("hud-who");
  who.hidden = !m.dir;
  if (m.dir) who.textContent = `${driver}: ${verb}`;

  // Floor IR sensors
  if (geometry && s.line) {
    const thr = geometry.line_threshold, top = 1023;
    const lifted = s.line.every((v) => v > thr);
    [["f-l", "f-lv", "L"], ["f-m", "f-mv", "M"], ["f-r", "f-rv", "R"]].forEach(([f, v, n], i) => {
      const val = s.line[i];
      $(f).style.height = `${Math.min(100, (val / top) * 100)}%`;
      $(f).classList.toggle("off", val > thr);
      $(v).textContent = `${n} ${val}`;
    });
    document.querySelectorAll(".meter .thr").forEach((t) => (t.style.bottom = `${(thr / top) * 100}%`));
    $("f-state").textContent = lifted ? "NO FLOOR" : "GROUNDED";
    $("f-state").style.color = lifted ? "var(--danger)" : "var(--ok)";
  }

  // Pose & Map Readouts
  updateMapPrompt();
  const pose = state.pose;
  if (pose && pose.placed) {
    $("p-room").textContent = (pose.room || "CORRIDOR / OPEN").toUpperCase();
    $("p-xy").textContent = `${pose.x.toFixed(2)}, ${pose.y.toFixed(2)}`;
    const hdgDeg = ((Math.round(deg(pose.yaw)) % 360) + 360) % 360;
    $("p-hdg").textContent = `${String(hdgDeg).padStart(3, "0")}° ${compass(pose.yaw)}`;
    $("p-dist").textContent = `${(pose.travelled_m ?? 0).toFixed(1)}m ±${(pose.drift_m ?? 0).toFixed(2)}m`;
    $("map-source").textContent = `POSE: ${(pose.source || "dead-reckoning").toUpperCase()}`;

    if (pose.home) {
      const dx = pose.home.x - pose.x;
      const dy = pose.home.y - pose.y;
      const distHome = Math.hypot(dx, dy);
      const bearingRad = Math.atan2(dy, dx);
      const bDeg = ((Math.round(deg(bearingRad)) % 360) + 360) % 360;
      $("p-home").textContent = `${distHome.toFixed(1)}m · ${String(bDeg).padStart(3, "0")}° ${compass(bearingRad)}`;
    } else {
      $("p-home").textContent = "NOT SET";
    }
  } else {
    $("p-room").textContent = "UNPLACED";
    $("p-xy").textContent = "-- , --";
    $("p-hdg").textContent = "--";
    $("p-home").textContent = pose && pose.home ? "AWAITING FIX" : "NOT SET";
    $("p-dist").textContent = pose ? `${(pose.travelled_m ?? 0).toFixed(1)}m / --` : "--";
    $("map-source").textContent = "POSE: UNPLACED";
  }

  renderConvo(state.transcript || []);
  renderTools(state.tools || []);
  for (const n of (state.notices || [])) {
    if (!shownToasts.has(n.id)) {
      shownToasts.add(n.id);
      if (state.now - n.t < 5) toast(n.text, n.level);
    }
  }

  renderRadar();
  renderProfileDistance();
  renderHud();
}
