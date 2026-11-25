import { $, clamp } from "./display-math.js";
import { telemetry, send } from "./socket.js";
import { mapCanvas, mapStage, mapState, fitMap } from "./map-state.js";

function setFollow(on) {
  mapState.follow = !!on;
  $("btn-follow").classList.toggle("on", mapState.follow);
  $("btn-follow").setAttribute("aria-pressed", String(mapState.follow));
}

export function setPlacingMode(on) {
  mapState.placing = !!on;
  mapState.placeActive = false;
  mapState.placeWorld = null;
  mapStage.classList.toggle("placing", mapState.placing);
  $("btn-set-start").classList.toggle("active", mapState.placing);
  $("btn-set-start").classList.toggle("primary", !mapState.placing);
  $("btn-set-start-lbl").textContent = mapState.placing ? "CANCEL" : "SET START";
  $("map-mode-banner").hidden = !mapState.placing;
  updateMapPrompt();
}

export function updateMapPrompt() {
  const unplaced = telemetry && telemetry.pose && telemetry.pose.placed === false;
  $("map-prompt").hidden = !unplaced || mapState.placing;
}

function zoomAround(cx, cy, factor) {
  const oldZ = mapState.zoom;
  const newZ = clamp(oldZ * factor, 0.15, 8);
  const ratio = newZ / oldZ;
  mapState.panX = cx - (cx - mapState.panX) * ratio;
  mapState.panY = cy - (cy - mapState.panY) * ratio;
  mapState.zoom = newZ;
}
function zoomAroundCenter(factor) {
  const r = mapStage.getBoundingClientRect();
  zoomAround(r.width / 2, r.height / 2, factor);
}

function finishMapPointer(e) {
  mapState.pointers.delete(e.pointerId);
  if (mapState.placing && mapState.placeActive && mapState.placeWorld) {
    const { x, y } = mapState.placeWorld;
    const yaw = mapState.placeYaw;
    send({ t: "place", x: Number(x.toFixed(3)), y: Number(y.toFixed(3)), yaw: Number(yaw.toFixed(4)) });
    setPlacingMode(false);
    return;
  }
  if (mapState.pointers.size < 2) mapState.pinchDist = 0;
  mapState.draggingPan = false;
}

export function bindMapInput() {
  $("btn-set-start").addEventListener("click", () => setPlacingMode(!mapState.placing));
  $("map-prompt").addEventListener("click", () => setPlacingMode(true));
  $("btn-follow").addEventListener("click", () => setFollow(!mapState.follow));
  $("btn-fit").addEventListener("click", () => { setFollow(false); fitMap(); });
  $("btn-zoom-in").addEventListener("click", () => zoomAroundCenter(1.28));
  $("btn-zoom-out").addEventListener("click", () => zoomAroundCenter(1 / 1.28));

  mapCanvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const r = mapCanvas.getBoundingClientRect();
    const factor = e.deltaY < 0 ? 1.14 : 1 / 1.14;
    zoomAround(e.clientX - r.left, e.clientY - r.top, factor);
  }, { passive: false });

  mapCanvas.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    try { mapCanvas.setPointerCapture(e.pointerId); } catch {}
    mapState.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });

    if (mapState.placing) {
      const w = window.__remi.mapToWorld(e.clientX, e.clientY);
      mapState.placeActive = true;
      mapState.placeWorld = w;
      mapState.placeStartClient = { x: e.clientX, y: e.clientY };
      mapState.placeYaw = (telemetry && telemetry.pose && typeof telemetry.pose.yaw === "number") ? telemetry.pose.yaw : 0;
      return;
    }

    if (mapState.pointers.size === 2) {
      const pts = [...mapState.pointers.values()];
      mapState.pinchDist = Math.hypot(pts[1].x - pts[0].x, pts[1].y - pts[0].y);
      mapState.pinchZoom = mapState.zoom;
      mapState.draggingPan = false;
      return;
    }

    mapState.draggingPan = true;
    mapState.panStartX = e.clientX;
    mapState.panStartY = e.clientY;
    mapState.panOriginX = mapState.panX;
    mapState.panOriginY = mapState.panY;
  });

  mapCanvas.addEventListener("pointermove", (e) => {
    if (mapState.pointers.has(e.pointerId)) {
      mapState.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    }

    if (mapState.placing && mapState.placeActive && mapState.placeWorld) {
      const distPx = Math.hypot(e.clientX - mapState.placeStartClient.x, e.clientY - mapState.placeStartClient.y);
      if (distPx > 6) {
        const w2 = window.__remi.mapToWorld(e.clientX, e.clientY);
        mapState.placeYaw = Math.atan2(w2.y - mapState.placeWorld.y, w2.x - mapState.placeWorld.x);
      } else {
        mapState.placeYaw = (telemetry && telemetry.pose && typeof telemetry.pose.yaw === "number") ? telemetry.pose.yaw : 0;
      }
      return;
    }

    if (mapState.pointers.size === 2) {
      const pts = [...mapState.pointers.values()];
      const d = Math.hypot(pts[1].x - pts[0].x, pts[1].y - pts[0].y);
      if (mapState.pinchDist > 8) {
        const r = mapCanvas.getBoundingClientRect();
        const midX = (pts[0].x + pts[1].x) / 2 - r.left;
        const midY = (pts[0].y + pts[1].y) / 2 - r.top;
        const targetZ = clamp(mapState.pinchZoom * (d / mapState.pinchDist), 0.15, 8);
        zoomAround(midX, midY, targetZ / mapState.zoom);
      }
      return;
    }

    if (mapState.draggingPan) {
      const dx = e.clientX - mapState.panStartX;
      const dy = e.clientY - mapState.panStartY;
      if (Math.hypot(dx, dy) > 4) setFollow(false);
      mapState.panX = mapState.panOriginX + dx;
      mapState.panY = mapState.panOriginY + dy;
    }
  });
  mapCanvas.addEventListener("pointerup", finishMapPointer);
  mapCanvas.addEventListener("pointercancel", (e) => {
    mapState.pointers.delete(e.pointerId);
    mapState.placeActive = false;
    mapState.draggingPan = false;
  });
}
