import { $ } from "./display-math.js";

let convoKey = "";
export function renderConvo(items) {
  const key = items.map((i) => `${i.id}:${i.text.length}:${i.final}`).join("|");
  if (key === convoKey) return;
  convoKey = key;
  const box = $("convo");
  const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
  box.innerHTML = "";
  if (!items.length) {
    box.innerHTML = '<p class="empty-note">Awaiting voice or text channel.</p>';
    return;
  }
  for (const i of items) {
    const d = document.createElement("div");
    d.className = `msg ${i.role}${i.final ? "" : " interim"}`;
    const w = document.createElement("span");
    w.className = "who-l";
    w.textContent = i.role === "user" ? "You" : "Remi";
    d.append(w, document.createTextNode(i.text));
    box.appendChild(d);
  }
  if (atBottom) box.scrollTop = box.scrollHeight;
}

let toolsKey = "";
export function renderTools(items) {
  const key = items.map((i) => `${i.id}:${i.status}:${i.result ? i.result.length : 0}`).join("|");
  if (key === toolsKey) return;
  toolsKey = key;
  $("tools-count").textContent = `${items.length} OPS`;
  const ul = $("tools");
  ul.innerHTML = "";
  if (!items.length) {
    ul.innerHTML = '<li class="empty-note">Tool telemetry appears here.</li>';
    return;
  }
  for (const i of items.slice().reverse()) {
    const li = document.createElement("li");
    li.className = "tool";
    const icon = { running: "i-loader", done: "i-check", error: "i-alert" }[i.status] || "i-check";
    const dur = i.t1 ? `${(i.t1 - i.t0).toFixed(1)}s` : "RUN";
    li.innerHTML = `<svg class="ico ${i.status}" width="14" height="14"><use href="#${icon}"/></svg><span></span><span class="t num">${dur}</span>${i.result ? '<span class="res"></span>' : ""}`;
    li.children[1].textContent = i.label;
    if (i.result) li.querySelector(".res").textContent = i.result;
    ul.appendChild(li);
  }
}

export function toast(text, level = "info") {
  const t = document.createElement("div");
  t.className = `toast ${level}`;
  t.textContent = text;
  $("toasts").appendChild(t);
  setTimeout(() => { t.style.opacity = "0"; setTimeout(() => t.remove(), 400); }, 4000);
}
