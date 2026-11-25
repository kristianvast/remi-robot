import { fitMap, mapState, mapMetadata, setMapMetadata } from "./map-state.js";

export function configureMap(map) {
  setMapMetadata(map);
  loadMapImage(map.image || "/map.png");
}

function loadMapImage(url) {
  const img = new Image();
  img.crossOrigin = "anonymous";
  img.onload = () => {
    const w = img.naturalWidth;
    const h = img.naturalHeight;
    if (mapMetadata && (!mapMetadata.width_px || !mapMetadata.height_px)) {
      mapMetadata.width_px = w;
      mapMetadata.height_px = h;
    }
    const raw = document.createElement("canvas");
    raw.width = w;
    raw.height = h;
    const rctx = raw.getContext("2d", { willReadFrequently: true });
    rctx.drawImage(img, 0, 0);
    const id = rctx.getImageData(0, 0, w, h);
    const d = id.data;

    // Separate wall layer for subtle glow + base recolored occupancy layer
    const wallCanvas = document.createElement("canvas");
    wallCanvas.width = w;
    wallCanvas.height = h;
    const wctx = wallCanvas.getContext("2d");
    const wid = wctx.createImageData(w, h);
    const wd = wid.data;

    let minX = w, minY = h, maxX = 0, maxY = 0;
    for (let i = 0, p = 0; i < d.length; i += 4, p++) {
      const v = d[i]; // grayscale occupancy: 205 unknown, 254 free, 0 wall
      const x = p % w;
      const y = (p - x) / w;
      if (v >= 230) {
        // Free space: dark translucent teal
        d[i] = 11;
        d[i + 1] = 56;
        d[i + 2] = 74;
        d[i + 3] = 178;
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      } else if (v <= 165) {
        // Occupied / wall: luminous cyan
        d[i] = 46;
        d[i + 1] = 236;
        d[i + 2] = 255;
        d[i + 3] = 248;
        wd[i] = 38;
        wd[i + 1] = 232;
        wd[i + 2] = 255;
        wd[i + 3] = 255;
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      } else {
        // Unknown (~205): transparent
        d[i + 3] = 0;
      }
    }
    rctx.putImageData(id, 0, 0);
    wctx.putImageData(wid, 0, 0);

    const composed = document.createElement("canvas");
    composed.width = w;
    composed.height = h;
    const cctx = composed.getContext("2d");
    cctx.drawImage(raw, 0, 0);
    cctx.save();
    cctx.filter = "blur(3px)";
    cctx.globalAlpha = 0.55;
    cctx.drawImage(wallCanvas, 0, 0);
    cctx.restore();
    cctx.drawImage(wallCanvas, 0, 0);

    mapState.offscreen = composed;
    mapState.bounds = maxX > minX ? { minX, minY, maxX, maxY } : { minX: 0, minY: 0, maxX: w, maxY: h };
    mapState.ready = true;
    fitMap();
  };
  img.src = url;
}
