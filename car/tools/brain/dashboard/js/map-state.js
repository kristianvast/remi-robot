import { $, clamp } from "./display-math.js";

export let mapMetadata = null;

export function setMapMetadata(map) {
  mapMetadata = map;
}

export const mapStage = $("map-stage");
export const mapCanvas = $("map-canvas");
export const mapCtx = mapCanvas.getContext("2d");

export const mapState = {
  offscreen: null,
  ready: false,
  fitted: false,
  zoom: 0.45,        // screen CSS px per map pixel
  panX: 0,           // screen CSS px offset of map pixel (0,0)
  panY: 0,
  follow: true,
  placing: false,
  draggingPan: false,
  panStartX: 0,
  panStartY: 0,
  panOriginX: 0,
  panOriginY: 0,
  // Set-start drag state
  placeActive: false,
  placeWorld: null,
  placeStartClient: null,
  placeYaw: 0,
  // Touch pinch
  pointers: new Map(),
  pinchDist: 0,
  pinchZoom: 1,
};

export function worldToMapPx(x, y) {
  if (!mapMetadata) return { px: 0, py: 0 };
  return {
    px: (x - mapMetadata.origin[0]) / mapMetadata.resolution,
    py: mapMetadata.height_px - (y - mapMetadata.origin[1]) / mapMetadata.resolution,
  };
}
function mapPxToWorld(px, py) {
  if (!mapMetadata) return { x: 0, y: 0 };
  return {
    x: mapMetadata.origin[0] + px * mapMetadata.resolution,
    y: mapMetadata.origin[1] + (mapMetadata.height_px - py) * mapMetadata.resolution,
  };
}
export function worldToCanvas(x, y) {
  const { px, py } = worldToMapPx(x, y);
  return {
    cx: mapState.panX + px * mapState.zoom,
    cy: mapState.panY + py * mapState.zoom,
  };
}
export function canvasToWorld(cx, cy) {
  const px = (cx - mapState.panX) / mapState.zoom;
  const py = (cy - mapState.panY) / mapState.zoom;
  return mapPxToWorld(px, py);
}

export function bindMapCoordinates() {
  window.__remi.mapToWorld = (clientX, clientY) => {
    const r = mapCanvas.getBoundingClientRect();
    return canvasToWorld(clientX - r.left, clientY - r.top);
  };
  window.__remi.worldToClient = (x, y) => {
    const r = mapCanvas.getBoundingClientRect();
    const { cx, cy } = worldToCanvas(x, y);
    return { x: r.left + cx, y: r.top + cy };
  };
}

export function fitMap() {
  if (!mapMetadata) return;
  const r = mapStage.getBoundingClientRect();
  if (r.width < 10 || r.height < 10) return;
  const b = mapState.bounds || { minX: 0, minY: 0, maxX: mapMetadata.width_px, maxY: mapMetadata.height_px };
  const bw = Math.max(40, b.maxX - b.minX);
  const bh = Math.max(40, b.maxY - b.minY);
  const pad = 16;
  const z = Math.min((r.width - pad * 2) / bw, (r.height - pad * 2) / bh);
  mapState.zoom = clamp(z, 0.12, 6);
  const cx = (b.minX + b.maxX) / 2;
  const cy = (b.minY + b.maxY) / 2;
  mapState.panX = r.width / 2 - cx * mapState.zoom;
  mapState.panY = r.height / 2 - cy * mapState.zoom;
  mapState.fitted = true;
}
