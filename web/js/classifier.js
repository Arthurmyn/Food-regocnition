// Classifier page: upload / drop / camera -> prediction card, Grad-CAM, nearest training examples.
import { header, startCamera, stopCamera, snap, predict, explain, bars, neighbors, footer, esc, cap } from "./common.js";

header("classifier");
const $ = (id) => document.getElementById(id);
let stream = null, previewUrl = null, token = 0;

async function load(blob, name) {
  const mine = ++token;  // ignore stale responses if the user picks another image quickly
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(blob);
  $("preview").src = previewUrl;
  $("name").textContent = name;
  $("result").innerHTML = `<div class="pad"><p class="note">Analysing…</p></div>`;
  try {
    const r = await predict(blob);
    if (mine !== token) return;
    const top = r.top;
    $("result").innerHTML = `
      <div class="pad"><p class="eyebrow">Prediction</p>
        <div class="pred"><b>${esc(cap(r.classes[top]))}</b><span class="pct">${Math.round(r.probs[top] * 100)}%</span></div>
        ${bars(r.classes, r.probs, top)}</div>
      <div class="sep pad"><p class="eyebrow">Grad-CAM</p>
        <img id="cam2" class="img sq" alt="Grad-CAM heatmap" style="background:var(--track)">
        <p class="note">Warm regions pushed the score for <b>${esc(r.classes[top])}</b> up; cool regions barely mattered.</p></div>
      ${footer(r)}`;
    $("near").innerHTML = neighbors(r.neighbors);
    explain(blob, r.classes[top]).then((u) => { if (mine === token) $("cam2").src = u; }).catch(() => {});
  } catch (e) {
    if (mine === token) $("result").innerHTML = `<div class="pad"><p class="note">${esc(e.message)}</p></div>`;
  }
}

function stopLive() { stopCamera(stream); stream = null; $("live").hidden = true; $("camBtn").textContent = "Use camera"; }

$("file").addEventListener("change", (e) => { const f = e.target.files[0]; e.target.value = ""; if (f) { stopLive(); load(f, f.name); } });
const drop = $("drop");
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) { stopLive(); load(f, f.name); } });

$("camBtn").addEventListener("click", async () => {
  if (stream) return stopLive();
  $("camErr").hidden = true;
  try {
    $("live").hidden = false;
    stream = await startCamera($("cam"));
    $("camBtn").textContent = "Close camera";
  } catch (e) {
    stopLive();
    $("camErr").hidden = false;
    $("camErr").textContent = `Camera unavailable: ${e.message || "permission denied"}. Use "Browse files" instead.`;
  }
});
$("shutter").addEventListener("click", async () => { const b = await snap($("cam")); stopLive(); load(b, "camera.jpg"); });

fetch("/assets/example.jpg").then((r) => r.blob()).then((b) => load(b, "example.jpg"));
