import { $ } from "./display-math.js";
import { send } from "./socket.js";
import { mapState } from "./map-state.js";
import { setPlacingMode } from "./map-input.js";

const DRIVE = { w: "forward", s: "back", a: "left", d: "right" };
const held = [];
let pan = 0;
const pressed = new Set();
let hbTimer = null;

function keyEls(k) { return document.querySelectorAll(`.key[data-key="${k}"]`); }
function light(k, on) { keyEls(k).forEach((b) => b.classList.toggle("on", on)); }

function press(k) {
  if (pressed.has(k)) return;
  pressed.add(k);
  light(k, true);
  if (k in DRIVE) { held.push(k); sendDrive(); }
  else if (k === "ArrowLeft") { pan = 1; send({ t: "pan", dir: 1 }); }
  else if (k === "ArrowRight") { pan = -1; send({ t: "pan", dir: -1 }); }
  else if (k === "ArrowUp") send({ t: "center" });
  else if (k === "ArrowDown") send({ t: "scan" });
  else if (k === " ") estop();
  ensureHeartbeat();
}
function release(k) {
  if (!pressed.has(k)) return;
  pressed.delete(k);
  light(k, false);
  if (k in DRIVE) {
    const idx = held.indexOf(k);
    if (idx !== -1) held.splice(idx, 1);
    sendDrive();
  } else if ((k === "ArrowLeft" && pan === 1) || (k === "ArrowRight" && pan === -1)) {
    pan = 0;
    send({ t: "pan", dir: 0 });
  }
}
export function releaseAll(notify = true) {
  [...pressed].forEach((k) => { pressed.delete(k); light(k, false); });
  held.length = 0;
  pan = 0;
  if (notify) {
    send({ t: "drive", dir: null });
    send({ t: "pan", dir: 0 });
  }
}
function sendDrive() {
  const k = held[held.length - 1];
  send({ t: "drive", dir: k ? DRIVE[k] : null });
}
function ensureHeartbeat() {
  if (hbTimer) return;
  hbTimer = setInterval(() => {
    if (held.length || pan) send({ t: "hb" });
    else { clearInterval(hbTimer); hbTimer = null; }
  }, 120);
}
function estop() {
  releaseAll();
  send({ t: "estop" });
  const b = $("estop");
  b.classList.add("on");
  setTimeout(() => b.classList.remove("on"), 180);
}

const typing = () =>
  document.activeElement &&
  ["INPUT", "TEXTAREA"].includes(document.activeElement.tagName) &&
  document.activeElement.type !== "range";

function toggleMic() { send({ t: "mic", on: $("t-mic").getAttribute("aria-checked") !== "true" }); }

export function bindControls() {
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && mapState.placing) {
      setPlacingMode(false);
      return;
    }
    if (document.activeElement && document.activeElement.tagName === "SELECT" && e.key !== " ") return;
    if (typing() || e.metaKey || e.ctrlKey || e.altKey) return;
    const k = e.key.length === 1 ? e.key.toLowerCase() : e.key;
    if (k in DRIVE || k.startsWith("Arrow") || k === " ") {
      e.preventDefault();
      if (!e.repeat) press(k);
    } else if (k === "m" && !e.repeat) {
      toggleMic();
    }
  });
  document.addEventListener("keyup", (e) => {
    const k = e.key.length === 1 ? e.key.toLowerCase() : e.key;
    release(k);
  });
  window.addEventListener("blur", () => releaseAll());
  document.addEventListener("visibilitychange", () => { if (document.hidden) releaseAll(); });

  document.querySelectorAll(".key").forEach((b) => {
    const k = b.dataset.key;
    b.addEventListener("pointerdown", (e) => {
      e.preventDefault();
      try { b.setPointerCapture(e.pointerId); } catch {}
      press(k);
    });
    const up = () => release(k);
    b.addEventListener("pointerup", up);
    b.addEventListener("pointercancel", up);
    b.addEventListener("lostpointercapture", up);
    b.addEventListener("contextmenu", (e) => e.preventDefault());
  });

  $("estop").addEventListener("click", estop);
  $("t-auto").addEventListener("click", () => send({ t: "autonomy", on: $("t-auto").getAttribute("aria-checked") !== "true" }));
  $("t-initiative").addEventListener("click", () => send({ t: "initiative", on: $("t-initiative").getAttribute("aria-checked") !== "true" }));
  $("t-mic").addEventListener("click", toggleMic);
  $("voice").addEventListener("change", (e) => {
    send({ t: "voice", name: e.target.value });
    e.target.blur();  // hand the keys back to driving
  });
  $("speed").addEventListener("input", (e) => {
    $("speed-v").textContent = e.target.value;
    send({ t: "speed", value: +e.target.value });
  });
  $("say-in").addEventListener("input", () => { $("say-btn").disabled = !$("say-in").value.trim(); });
  $("say").addEventListener("submit", (e) => {
    e.preventDefault();
    const text = $("say-in").value.trim();
    if (!text) return;
    send({ t: "say", text });
    $("say-in").value = "";
    $("say-btn").disabled = true;
  });
}
