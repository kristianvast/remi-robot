import * as THREE from "three";
import { $, clamp, rad } from "./display-math.js";
import { geometry, telemetry } from "./socket.js";

const carStage = $("car3d-stage");
const carCanvas = $("car3d-canvas");
export const car3d = {
  ok: false,
  renderer: null,
  scene: null,
  camera: null,
  root: null,
  wheels: { fl: null, fr: null, rl: null, rr: null },
  sensorPan: null,
  sensorPanBaseZ: 0,
  sonarCone: null,
  sonarMat: null,
  ringOuter: null,
  orbitAz: 0.62,
  orbitEl: 0.38,
  orbitDist: 0.56,
  fitRadius: 0.155,
  targetY: 0.064,
  dragging: false,
  lastX: 0,
  lastY: 0,
  lastInteract: 0,
  modelRequested: false,
};

export function show3dFallback(msg) {
  $("car3d-fallback").textContent = msg;
  $("car3d-fallback").hidden = false;
  $("car3d-status").textContent = "3D OFFLINE";
}

export function init3D() {
  try {
    const renderer = new THREE.WebGLRenderer({ canvas: carCanvas, antialias: true, alpha: true, powerPreference: "high-performance" });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor(0x000000, 0);
    car3d.renderer = renderer;

    const scene = new THREE.Scene();
    car3d.scene = scene;

    const camera = new THREE.PerspectiveCamera(36, 1.8, 0.02, 10);
    car3d.camera = camera;

    scene.add(new THREE.AmbientLight(0x286888, 1.1));
    const keyLight = new THREE.DirectionalLight(0x6cf2ff, 2.2);
    keyLight.position.set(1.2, 1.8, 1.0);
    scene.add(keyLight);
    const rimLight = new THREE.DirectionalLight(0x1b78ff, 1.4);
    rimLight.position.set(-1.4, 0.9, -1.2);
    scene.add(rimLight);

    // Ground holographic polar grid ring
    const gridGroup = new THREE.Group();
    const ringMat = new THREE.LineBasicMaterial({ color: 0x26e8ff, transparent: true, opacity: 0.34, blending: THREE.AdditiveBlending });
    for (const r of [0.11, 0.18, 0.25]) {
      const pts = [];
      for (let i = 0; i <= 72; i++) {
        const a = (i / 72) * Math.PI * 2;
        pts.push(new THREE.Vector3(Math.cos(a) * r, 0.001, Math.sin(a) * r));
      }
      gridGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), ringMat));
    }
    const spokePts = [];
    for (let i = 0; i < 12; i++) {
      const a = (i / 12) * Math.PI * 2;
      spokePts.push(new THREE.Vector3(Math.cos(a) * 0.06, 0.001, Math.sin(a) * 0.06));
      spokePts.push(new THREE.Vector3(Math.cos(a) * 0.25, 0.001, Math.sin(a) * 0.25));
    }
    gridGroup.add(new THREE.LineSegments(
      new THREE.BufferGeometry().setFromPoints(spokePts),
      new THREE.LineBasicMaterial({ color: 0x26e8ff, transparent: true, opacity: 0.15, blending: THREE.AdditiveBlending })
    ));

    // Rotating outer dashed tick ring
    const tickPts = [];
    for (let i = 0; i < 36; i++) {
      const a = (i / 36) * Math.PI * 2;
      const r0 = 0.265, r1 = i % 3 === 0 ? 0.285 : 0.274;
      tickPts.push(new THREE.Vector3(Math.cos(a) * r0, 0.001, Math.sin(a) * r0));
      tickPts.push(new THREE.Vector3(Math.cos(a) * r1, 0.001, Math.sin(a) * r1));
    }
    const ringOuter = new THREE.LineSegments(
      new THREE.BufferGeometry().setFromPoints(tickPts),
      new THREE.LineBasicMaterial({ color: 0x26e8ff, transparent: true, opacity: 0.5, blending: THREE.AdditiveBlending })
    );
    gridGroup.add(ringOuter);
    car3d.ringOuter = ringOuter;

    // Subtle ground glow disc
    const disc = new THREE.Mesh(
      new THREE.CircleGeometry(0.25, 48),
      new THREE.MeshBasicMaterial({ color: 0x0a3850, transparent: true, opacity: 0.22, side: THREE.DoubleSide, depthWrite: false })
    );
    disc.rotation.x = -Math.PI / 2;
    gridGroup.add(disc);
    scene.add(gridGroup);

    // Interactive orbit drag
    carCanvas.addEventListener("pointerdown", (e) => {
      car3d.dragging = true;
      car3d.lastX = e.clientX;
      car3d.lastY = e.clientY;
      car3d.lastInteract = performance.now();
      try { carCanvas.setPointerCapture(e.pointerId); } catch {}
    });
    carCanvas.addEventListener("pointermove", (e) => {
      if (!car3d.dragging) return;
      const dx = e.clientX - car3d.lastX;
      const dy = e.clientY - car3d.lastY;
      car3d.lastX = e.clientX;
      car3d.lastY = e.clientY;
      car3d.orbitAz -= dx * 0.01;
      car3d.orbitEl = clamp(car3d.orbitEl + dy * 0.008, 0.14, 1.28);
      car3d.lastInteract = performance.now();
    });
    const endOrbit = () => { car3d.dragging = false; };
    carCanvas.addEventListener("pointerup", endOrbit);
    carCanvas.addEventListener("pointercancel", endOrbit);

    car3d.ok = true;
  } catch (err) {
    show3dFallback(`WebGL unavailable: ${err && err.message ? err.message : "hardware context blocked"}`);
  }
}

export function step3D(dt, nowMs) {
  if (!car3d.ok || !car3d.renderer) return;
  const r = carStage.getBoundingClientRect();
  if (r.width < 4 || r.height < 4) return;
  const w = Math.floor(r.width);
  const h = Math.floor(r.height);
  const needResize = carCanvas.clientWidth !== w || carCanvas.clientHeight !== h || car3d.lastW !== w || car3d.lastH !== h;
  if (needResize) {
    car3d.lastW = w;
    car3d.lastH = h;
    car3d.renderer.setSize(w, h, false);
    car3d.camera.aspect = w / h;
    car3d.camera.updateProjectionMatrix();
  }

  // Slow continuous orbit when user isn't actively dragging
  if (!car3d.dragging && nowMs - car3d.lastInteract > 2200) {
    car3d.orbitAz += dt * 0.24;
  }
  if (car3d.ringOuter) car3d.ringOuter.rotation.y -= dt * 0.14;

  const vFov = rad(car3d.camera.fov);
  const hFov = 2 * Math.atan(Math.tan(vFov / 2) * car3d.camera.aspect);
  const fitR = car3d.fitRadius || 0.155;
  const minDistV = (fitR / Math.sin(vFov / 2)) * 1.16;
  const minDistH = (fitR / Math.sin(hFov / 2)) * 1.24;
  const d = Math.max(car3d.orbitDist, minDistV, minDistH);
  const el = car3d.orbitEl;
  const az = car3d.orbitAz;
  const targetY = car3d.targetY ?? 0.064;
  car3d.camera.position.set(
    Math.cos(el) * Math.cos(az) * d,
    targetY + Math.sin(el) * d,
    Math.cos(el) * Math.sin(az) * d
  );
  car3d.camera.lookAt(0.01, targetY, 0);

  if (telemetry) {
    // Pan head node about local Z by (head - 90) deg (+ = left)
    if (car3d.sensorPan && telemetry.sensors) {
      const targetZ = car3d.sensorPanBaseZ + rad((telemetry.sensors.head ?? 90) - 90);
      car3d.sensorPan.rotation.z += (targetZ - car3d.sensorPan.rotation.z) * Math.min(1, dt * 12);
    }

    // Wheel spin about local Y by motion.dir or pose.v_cms
    const dir = telemetry.motion ? telemetry.motion.dir : null;
    const vCms = telemetry.pose && typeof telemetry.pose.v_cms === "number" ? telemetry.pose.v_cms : 0;
    const wheelRadiusM = (geometry && geometry.wheel_radius_cm ? geometry.wheel_radius_cm : 3.33) / 100;
    let lRate = 0, rRate = 0;
    if (dir === "forward") {
      const wRad = Math.max(4.5, Math.abs(vCms) / 100 / wheelRadiusM);
      lRate = wRad; rRate = wRad;
    } else if (dir === "back") {
      const wRad = Math.max(4.5, Math.abs(vCms) / 100 / wheelRadiusM);
      lRate = -wRad; rRate = -wRad;
    } else if (dir === "left") {
      lRate = -6.0; rRate = 6.0;
    } else if (dir === "right") {
      lRate = 6.0; rRate = -6.0;
    } else if (Math.abs(vCms) > 0.5) {
      const wRad = (vCms / 100) / wheelRadiusM;
      lRate = wRad; rRate = wRad;
    }
    if (car3d.wheels.fl) car3d.wheels.fl.rotation.y += lRate * dt;
    if (car3d.wheels.rl) car3d.wheels.rl.rotation.y += lRate * dt;
    if (car3d.wheels.fr) car3d.wheels.fr.rotation.y += rRate * dt;
    if (car3d.wheels.rr) car3d.wheels.rr.rotation.y += rRate * dt;

    // Sonar cone length + warning colors
    if (car3d.sonarCone && car3d.sonarMat && telemetry.sensors) {
      const dist = telemetry.sensors.distance;
      const stopCm = telemetry.stop_cm ?? 15;
      const cm = dist == null ? 45 : clamp(dist, 4, 150);
      const lenM = cm / 100;
      const halfAngle = rad(((geometry && geometry.sonar_cone_deg) || 15) / 2);
      const radM = Math.max(0.015, lenM * Math.tan(halfAngle));
      car3d.sonarCone.scale.set(lenM, radM, radM);

      if (dist != null && dist < stopCm) {
        car3d.sonarMat.color.setHex(0xff3850);
        car3d.sonarMat.opacity = 0.42;
      } else if (dist != null && dist < 2 * stopCm) {
        car3d.sonarMat.color.setHex(0xffb838);
        car3d.sonarMat.opacity = 0.34;
      } else {
        car3d.sonarMat.color.setHex(0x26e8ff);
        car3d.sonarMat.opacity = dist == null ? 0.14 : 0.24;
      }
    }
  }

  car3d.renderer.render(car3d.scene, car3d.camera);
}
