import { $ } from "./display-math.js";

export let geometry = null;
export let telemetry = null;
let ws = null;

export function connect({ onHello, render, releaseAll }) {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${proto}//${location.host}/ws`);
  ws.onopen = () => $("pane-off").classList.remove("on");
  ws.onclose = () => {
    $("pane-off").classList.add("on");
    releaseAll(false);
    setTimeout(() => connect({ onHello, render, releaseAll }), 1000);
  };
  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (msg.t === "hello") {
      geometry = msg.geometry;
      onHello(msg);
    } else if (msg.t === "state") {
      telemetry = msg;
      render(msg);
    }
  };
}
export const send = (obj) => { if (ws && ws.readyState === 1) ws.send(JSON.stringify(obj)); };
