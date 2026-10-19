// Classifier page: upload / drop / camera -> class, ranked probabilities, heatmap on the photo, nearest training photos.
import { header, startCamera, stopCamera, snap, predict, explain, neighbors, esc, cap } from "./common.js";

header("classifier");
const $ = (id) => document.getElementById(id);
let stream = null, previewUrl = null, token = 0;

const PIXEL = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7";
const skeleton = `<div class="skel"><i></i><i></i><i></i><i></i><i></i></div>`;

function readout(r) {
  const order = r.probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]);
  const [topP, top] = order[0];
  const pct = Math.round(topP * 100);
  const sure = topP >= 0.5 ? `<b class="num">${pct}%</b> sure.` : `Only <b class="num">${pct}%</b> sure, so treat this as a guess.`;
  return `
    <h2>${esc(cap(r.classes[top]))}</h2>
    <p class="sure">${sure}</p>
    <ul class="rank">${order.map(([p, i], k) => `
      <li class="${k === 0 ? "top" : ""}"><span>${esc(cap(r.classes[i]))}</span>
        <span class="bar"><i style="width:${(p * 100).toFixed(1)}%"></i></span>
        <span class="n">${Math.round(p * 100)}%</span></li>`).join("")}</ul>
    <div class="facts"><span>${Math.round(r.latency_ms)} ms</span><span>${esc(r.device)}</span><span>${esc(r.model)}</span></div>`;
}

async function load(blob, name) {
  const mine = ++token;  // ignore stale responses if the user picks another image quickly
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(blob);
  $("preview").src = previewUrl;
  $("heat").src = PIXEL;
  $("v-photo").checked = true;
  $("name").textContent = name;
  $("stageNote").textContent = "";
  $("result").innerHTML = skeleton;
  try {
    const r = await predict(blob);
    if (mine !== token) return;
    $("result").innerHTML = readout(r);
    $("stageNote").textContent = `Heatmap: warm areas pushed the score for ${r.classes[r.top]} up; cool areas barely mattered.`;
    $("near").innerHTML = neighbors(r.neighbors);
    explain(blob, r.classes[r.top]).then((u) => { if (mine === token) $("heat").src = u; }).catch(() => {});
  } catch (e) {
    if (mine === token) $("result").innerHTML = `<p class="note">${esc(e.message)}</p>`;
  }
}

function stopLive() { stopCamera(stream); stream = null; $("live").hidden = true; $("camBtn").textContent = "Use camera"; }

$("file").addEventListener("change", (e) => { const f = e.target.files[0]; e.target.value = ""; if (f) { stopLive(); load(f, f.name); } });
const drop = $("drop");
["dragenter", "dragover"].forEach((t) => document.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => document.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
document.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) { stopLive(); load(f, f.name); } });

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
    $("camErr").textContent = `Camera unavailable: ${e.message || "permission denied"}. Use "Choose photo" instead.`;
  }
});
$("shutter").addEventListener("click", async () => { const b = await snap($("cam")); stopLive(); load(b, "camera.jpg"); });

fetch("/assets/example.jpg").then((r) => r.blob()).then((b) => load(b, "example.jpg"));
