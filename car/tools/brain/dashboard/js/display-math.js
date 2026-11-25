export const $ = (id) => document.getElementById(id);
const NS = "http://www.w3.org/2000/svg";
export const el = (tag, attrs = {}, parent) => {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (parent) parent.appendChild(n);
  return n;
};
export const rad = (d) => (d * Math.PI) / 180;
export const deg = (r) => (r * 180) / Math.PI;
export const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
export const compass = (yawRad) => {
  const dirs = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"];
  const idx = ((Math.round(deg(yawRad) / 45) % 8) + 8) % 8;
  return dirs[idx];
};
