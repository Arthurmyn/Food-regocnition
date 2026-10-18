// Hot Dog or Not: opens straight on the camera. Photo -> "Evaluating..." -> green Hotdog / red Not hotdog.
import { header, startCamera, stopCamera, snap, predict, explain, bars, neighbors, footer, esc, cap } from "./common.js";

header("hotdog");
const THRESHOLD = 0.5;      // P(hotdog) needed for a green check; anything else is "Not hotdog"
const MIN_EVALUATING_MS = 1200;
const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

let stream = null, blob = null, result = null, camUrl = null, photoUrl = null;

const dots = Array.from({ length: 12 }, (_, k) => {
  const a = (k * Math.PI) / 6 - Math.PI / 2;
  return `<i style="left:${(60 * Math.cos(a)).toFixed(1)}px;top:${(60 * Math.sin(a)).toFixed(1)}px;animation-delay:${(-k / 12).toFixed(2)}s"></i>`;
}).join("");

function show(state) {
  const live = state === "camera";
  $("cam").hidden = !live;
  $("shutter").hidden = !live;
  $("photo").hidden = live;
  $("camControls").hidden = state === "result" || state === "evaluating";
  $("resControls").hidden = state !== "result";
  $("verdict").hidden = state !== "result";
  $("screen").classList.toggle("dim", state === "evaluating");
  if (state === "camera") { $("overlay").innerHTML = ""; $("info").hidden = true; $("infoBtn").textContent = "Additional information"; }
}
function message(title, text) {
  $("msg").hidden = false;
  $("msg").innerHTML = `${esc(title)}<small>${esc(text)}</small>`;
}

async function openCamera() {
  show("camera");
  $("msg").hidden = true;
  $("shutter").disabled = true;
  try {
    stream = await startCamera($("cam"));
    $("shutter").disabled = false;
  } catch (e) {
    message("Camera unavailable", `${e.message || "Permission denied."} You can upload a photo instead.`);
  }
}

async function evaluate(b) {
  blob = b; result = null; camUrl = null;
  if (photoUrl) URL.revokeObjectURL(photoUrl);
  photoUrl = URL.createObjectURL(b);
  $("photo").src = photoUrl;
  stopCamera(stream); stream = null;
  $("msg").hidden = true;
  show("evaluating");
  $("overlay").innerHTML = `<div class="spin">${dots}</div><div class="evaluating">Evaluating...</div>`;
  try {
    [result] = await Promise.all([predict(b), sleep(MIN_EVALUATING_MS)]);
  } catch (e) {
    $("overlay").innerHTML = "";
    message("Something went wrong", e.message);
    $("again").hidden = false; show("result"); $("verdict").hidden = true;
    return;
  }
  const hot = result.hotdog_prob >= THRESHOLD;
  $("overlay").innerHTML = hot
    ? `<div class="banner yes">Hotdog</div><div class="badge yes">✓</div>`
    : `<div class="banner no">Not hotdog</div><div class="badge no">✕</div>`;
  $("verdict").innerHTML = `Hot dog probability: <span class="mono">${Math.round(result.hotdog_prob * 100)}%</span>`;
  show("result");
}

async function toggleInfo() {
  const box = $("info");
  if (!box.hidden) { box.hidden = true; $("infoBtn").textContent = "Additional information"; return; }
  const r = result, top = r.top;
  box.hidden = false;
  $("infoBtn").textContent = "Hide information";
  box.innerHTML = `
    <div class="pad"><p class="eyebrow">Verdict</p>
      <p>P(hotdog) = <b class="mono">${Math.round(r.hotdog_prob * 100)}%</b>; it needs
      <span class="mono">${THRESHOLD * 100}%</span> for a green check. Best guess among the five classes:
      <b>${esc(cap(r.classes[top]))}</b> (<span class="mono">${Math.round(r.probs[top] * 100)}%</span>).</p>
      ${bars(r.classes, r.probs, top)}</div>
    <div class="sep pad two">
      <div><p class="eyebrow">Your photo</p><img class="img sq" src="${photoUrl}" alt="Your photo"></div>
      <div><p class="eyebrow">What made it decide? (Grad-CAM)</p><img id="cam2" class="img sq" alt="Grad-CAM heatmap"></div>
    </div>
    <div class="sep pad"><p class="eyebrow">Closest training photos</p><div class="grid5">${neighbors(r.neighbors)}</div></div>
    ${footer(r)}`;
  try {
    camUrl = camUrl || await explain(blob, "hotdog");
    $("cam2").src = camUrl;
  } catch (e) { $("cam2").alt = "Grad-CAM unavailable"; }
}

$("shutter").addEventListener("click", async () => { $("shutter").disabled = true; evaluate(await snap($("cam"))); });
$("file").addEventListener("change", (e) => { const f = e.target.files[0]; e.target.value = ""; if (f) evaluate(f); });
$("again").addEventListener("click", openCamera);
$("infoBtn").addEventListener("click", toggleInfo);
openCamera();
