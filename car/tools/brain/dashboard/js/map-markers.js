import { compass, deg } from "./display-math.js";
import { telemetry } from "./socket.js";
import { mapMetadata, mapState, worldToCanvas } from "./map-state.js";

export function drawMapMarkers(ctx, pxPerMeter, nowMs) {
  // Room polygons & names
  const curRoom = telemetry && telemetry.pose ? telemetry.pose.room : null;
  for (const room of (mapMetadata.rooms || [])) {
    if (!room.polygon || room.polygon.length < 3) continue;
    const isCur = curRoom && room.name === curRoom;
    ctx.beginPath();
    room.polygon.forEach(([rx, ry], idx) => {
      const { cx, cy } = worldToCanvas(rx, ry);
      if (idx === 0) ctx.moveTo(cx, cy);
      else ctx.lineTo(cx, cy);
    });
    ctx.closePath();
    ctx.fillStyle = isCur ? "rgba(38, 232, 255, 0.12)" : "rgba(18, 92, 122, 0.05)";
    ctx.fill();
    ctx.strokeStyle = isCur ? "rgba(38, 232, 255, 0.72)" : "rgba(38, 232, 255, 0.28)";
    ctx.lineWidth = isCur ? 1.4 : 1;
    ctx.stroke();
  }

  // Place markers
  for (const pl of (mapMetadata.places || [])) {
    const { cx, cy } = worldToCanvas(pl.x, pl.y);
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(Math.PI / 4);
    ctx.strokeStyle = "rgba(38, 232, 255, 0.55)";
    ctx.fillStyle = "rgba(38, 232, 255, 0.16)";
    ctx.lineWidth = 1;
    ctx.fillRect(-3, -3, 6, 6);
    ctx.strokeRect(-3, -3, 6, 6);
    ctx.restore();
  }

  const pose = telemetry && telemetry.pose;
  const placedLabels = [];
  const overlapsAny = (x0, y0, x1, y1) => placedLabels.some((b) => !(x1 < b[0] || x0 > b[2] || y1 < b[1] || y0 > b[3]));

  // Home marker
  if (pose && pose.home) {
    const hc = worldToCanvas(pose.home.x, pose.home.y);
    if (pose.placed) {
      const rc = worldToCanvas(pose.x, pose.y);
      ctx.save();
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = "rgba(32, 232, 160, 0.42)";
      ctx.lineWidth = 1.1;
      ctx.beginPath();
      ctx.moveTo(rc.cx, rc.cy);
      ctx.lineTo(hc.cx, hc.cy);
      ctx.stroke();
      ctx.restore();
    }
    ctx.save();
    ctx.strokeStyle = "#20e8a0";
    ctx.fillStyle = "rgba(32, 232, 160, 0.22)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(hc.cx, hc.cy, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(hc.cx - 9, hc.cy); ctx.lineTo(hc.cx + 9, hc.cy);
    ctx.moveTo(hc.cx, hc.cy - 9); ctx.lineTo(hc.cx, hc.cy + 9);
    ctx.stroke();
    ctx.font = '600 9px "IBM Plex Mono", monospace';
    const hw = ctx.measureText("HOME").width + 6;
    const hx = hc.cx + 10;
    const hy = hc.cy + 10;
    ctx.fillStyle = "rgba(2, 10, 18, 0.82)";
    ctx.fillRect(hx - 3, hy - 6, hw, 12);
    ctx.fillStyle = "#20e8a0";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText("HOME", hx, hy);
    placedLabels.push([hx - 4, hy - 7, hx + hw, hy + 7]);
    ctx.restore();
  }

  // Room labels with collision avoidance
  ctx.font = '600 9.5px "Chakra Petch", sans-serif';
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  const sortedRooms = [...(mapMetadata.rooms || [])].sort((a, b) => {
    const aCur = curRoom && a.name === curRoom ? -1 : 0;
    const bCur = curRoom && b.name === curRoom ? -1 : 0;
    return aCur - bCur;
  });
  for (const room of sortedRooms) {
    if (!room.label) continue;
    const { cx, cy } = worldToCanvas(room.label[0], room.label[1]);
    const name = String(room.name || room.id).toUpperCase();
    const isCur = curRoom && room.name === curRoom;
    const tw = ctx.measureText(name).width + 8;
    const th = 12;
    let ly = cy - 11;
    for (const dy of [0, -13, 14, -26, 27]) {
      const yTest = cy - 11 + dy;
      if (!overlapsAny(cx - tw / 2 - 2, yTest - th / 2 - 1, cx + tw / 2 + 2, yTest + th / 2 + 1)) {
        ly = yTest;
        break;
      }
    }
    placedLabels.push([cx - tw / 2 - 2, ly - th / 2 - 1, cx + tw / 2 + 2, ly + th / 2 + 1]);
    ctx.fillStyle = "rgba(2, 8, 16, 0.8)";
    ctx.fillRect(cx - tw / 2, ly - th / 2, tw, th);
    ctx.fillStyle = isCur ? "#26e8ff" : "rgba(195, 236, 250, 0.88)";
    ctx.fillText(name, cx, ly + 0.5);
  }

  // Breadcrumb trail
  if (pose && pose.trail && pose.trail.length > 1) {
    ctx.save();
    ctx.strokeStyle = "rgba(38, 232, 255, 0.78)";
    ctx.lineWidth = 2;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    pose.trail.forEach(([tx, ty], i) => {
      const { cx, cy } = worldToCanvas(tx, ty);
      if (i === 0) ctx.moveTo(cx, cy);
      else ctx.lineTo(cx, cy);
    });
    ctx.stroke();
    ctx.restore();
  }

  // Car glowing arrow + uncertainty ring
  if (pose && pose.placed) {
    const { cx, cy } = worldToCanvas(pose.x, pose.y);
    const driftPx = Math.max(6, (pose.drift_m || 0) * pxPerMeter);
    const pulse = 0.72 + 0.28 * Math.sin(nowMs * 0.0045);

    ctx.save();
    // Uncertainty drift ring
    ctx.beginPath();
    ctx.arc(cx, cy, driftPx, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(255, 184, 56, 0.09)";
    ctx.fill();
    ctx.setLineDash([4, 3]);
    ctx.strokeStyle = `rgba(255, 184, 56, ${0.55 * pulse + 0.25})`;
    ctx.lineWidth = 1.2;
    ctx.stroke();
    ctx.setLineDash([]);

    // Heading ray & glowing arrow (world +y is up -> canvas -y)
    const canvasAng = -pose.yaw;
    ctx.translate(cx, cy);
    ctx.rotate(canvasAng);

    // Forward cone glow
    const grad = ctx.createRadialGradient(0, 0, 2, 0, 0, 34);
    grad.addColorStop(0, "rgba(38, 232, 255, 0.42)");
    grad.addColorStop(1, "rgba(38, 232, 255, 0)");
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.arc(0, 0, 34, -0.34, 0.34);
    ctx.closePath();
    ctx.fill();

    // Chevron arrow
    ctx.shadowColor = "#26e8ff";
    ctx.shadowBlur = 12;
    ctx.fillStyle = "#26e8ff";
    ctx.strokeStyle = "#e8fbff";
    ctx.lineWidth = 1.3;
    ctx.beginPath();
    ctx.moveTo(13, 0);
    ctx.lineTo(-8, -7.5);
    ctx.lineTo(-4, 0);
    ctx.lineTo(-8, 7.5);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  }

  // Set-start active drag preview arrow
  if (mapState.placing && mapState.placeActive && mapState.placeWorld) {
    const { cx, cy } = worldToCanvas(mapState.placeWorld.x, mapState.placeWorld.y);
    const ang = -mapState.placeYaw;
    ctx.save();
    ctx.translate(cx, cy);
    ctx.strokeStyle = "#ffb838";
    ctx.fillStyle = "rgba(255, 184, 56, 0.22)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(0, 0, 12, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    ctx.rotate(ang);
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(44, 0);
    ctx.stroke();
    ctx.fillStyle = "#ffb838";
    ctx.beginPath();
    ctx.moveTo(48, 0);
    ctx.lineTo(36, -6);
    ctx.lineTo(36, 6);
    ctx.closePath();
    ctx.fill();
    ctx.restore();

    const hdgDeg = ((Math.round(deg(mapState.placeYaw)) % 360) + 360) % 360;
    ctx.font = '600 11px "IBM Plex Mono", monospace';
    ctx.fillStyle = "#ffd98a";
    ctx.textAlign = "left";
    ctx.fillText(`${hdgDeg}° ${compass(mapState.placeYaw)}`, cx + 16, cy - 14);
  }
}
