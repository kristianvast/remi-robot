import { telemetry } from "./socket.js";
import { mapStage, mapCanvas, mapCtx, mapMetadata, mapState, fitMap, worldToMapPx, canvasToWorld, worldToCanvas } from "./map-state.js";
import { drawMapMarkers } from "./map-markers.js";

export function drawMapFrame(nowMs) {
  const rect = mapStage.getBoundingClientRect();
  if (rect.width < 4 || rect.height < 4) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const cw = Math.round(rect.width * dpr);
  const ch = Math.round(rect.height * dpr);
  if (mapCanvas.width !== cw || mapCanvas.height !== ch) {
    mapCanvas.width = cw;
    mapCanvas.height = ch;
    if (mapState.ready && !mapState.fitted) fitMap();
  }

  // Smoothly follow Remi if enabled & placed
  if (mapMetadata && mapState.follow && !mapState.placing && !mapState.draggingPan && telemetry && telemetry.pose && telemetry.pose.placed) {
    const { px, py } = worldToMapPx(telemetry.pose.x, telemetry.pose.y);
    const targetPanX = rect.width / 2 - px * mapState.zoom;
    const targetPanY = rect.height / 2 - py * mapState.zoom;
    mapState.panX += (targetPanX - mapState.panX) * 0.14;
    mapState.panY += (targetPanY - mapState.panY) * 0.14;
  }

  const ctx = mapCtx;
  ctx.save();
  ctx.scale(dpr, dpr);
  const W = rect.width, H = rect.height;
  ctx.clearRect(0, 0, W, H);

  // Deep tactical background + 1m/2m world grid
  ctx.fillStyle = "#02070e";
  ctx.fillRect(0, 0, W, H);

  if (mapMetadata) {
    const pxPerMeter = mapState.zoom / mapMetadata.resolution;
    const stepM = pxPerMeter < 22 ? 5 : pxPerMeter < 48 ? 2 : 1;
    const tl = canvasToWorld(0, 0);
    const br = canvasToWorld(W, H);
    const xStart = Math.floor(Math.min(tl.x, br.x) / stepM) * stepM;
    const xEnd = Math.ceil(Math.max(tl.x, br.x) / stepM) * stepM;
    const yStart = Math.floor(Math.min(tl.y, br.y) / stepM) * stepM;
    const yEnd = Math.ceil(Math.max(tl.y, br.y) / stepM) * stepM;

    ctx.strokeStyle = "rgba(38, 232, 255, 0.065)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let x = xStart; x <= xEnd; x += stepM) {
      const { cx } = worldToCanvas(x, 0);
      ctx.moveTo(cx, 0); ctx.lineTo(cx, H);
    }
    for (let y = yStart; y <= yEnd; y += stepM) {
      const { cy } = worldToCanvas(0, y);
      ctx.moveTo(0, cy); ctx.lineTo(W, cy);
    }
    ctx.stroke();

    // Recolored occupancy bitmap
    if (mapState.offscreen) {
      ctx.imageSmoothingEnabled = mapState.zoom < 1.5;
      ctx.drawImage(
        mapState.offscreen,
        mapState.panX,
        mapState.panY,
        mapMetadata.width_px * mapState.zoom,
        mapMetadata.height_px * mapState.zoom
      );
    }

    drawMapMarkers(ctx, pxPerMeter, nowMs);
  } else {
    ctx.fillStyle = "#446782";
    ctx.font = '500 11px "IBM Plex Mono", monospace';
    ctx.textAlign = "center";
    ctx.fillText("AWAITING FLOOR MAP TELEMETRY...", W / 2, H / 2);
  }

  ctx.restore();
}
