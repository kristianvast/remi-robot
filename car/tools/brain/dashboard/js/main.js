import { $ } from "./display-math.js";
import { connect, geometry, telemetry } from "./socket.js";
import { bindControls, releaseAll } from "./controls.js";
import { bindCamera, configureCamera, setCamSource } from "./camera.js";
import { drawSensorGeometry, render } from "./render.js";
import { configureMap } from "./map-image.js";
import { bindMapCoordinates } from "./map-state.js";
import { bindMapInput } from "./map-input.js";
import { drawMapFrame } from "./map-draw.js";
import { stepRadar } from "./radar.js";
import { init3D, step3D } from "./car3d.js";
import { loadCarModel } from "./car-model.js";

window.__remi = window.__remi || {};
window.__remi.carLoaded = false;

function tickClock() {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  $("hud-clock").textContent = `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}
tickClock();
setInterval(tickClock, 1000);

function onHello(msg) {
  configureCamera(msg);
  if (typeof msg.speed === "number") {
    $("speed").value = msg.speed;
    $("speed-v").textContent = msg.speed;
  }
  if (geometry && geometry.line_threshold != null) {
    $("f-thr-lbl").textContent = `THR ${geometry.line_threshold}`;
  }
  const stopCm = (telemetry && telemetry.stop_cm != null) ? telemetry.stop_cm : 15;
  drawSensorGeometry(stopCm);
  setCamSource();

  if (msg.voices) {
    $("voice").replaceChildren(...Object.entries(msg.voices).map(([name, style]) => new Option(`${name} · ${style}`, name)));
  }
  if (msg.map) configureMap(msg.map);
  loadCarModel(msg.car_model || "/car.glb");
}

bindControls();
bindCamera();
bindMapCoordinates();
bindMapInput();

let lastRaf = performance.now();
function animate(now) {
  const dt = Math.min(0.1, (now - lastRaf) / 1000);
  lastRaf = now;

  step3D(dt, now);
  drawMapFrame(now);

  stepRadar(dt);

  requestAnimationFrame(animate);
}
requestAnimationFrame(animate);

init3D();
connect({ onHello, render, releaseAll });
