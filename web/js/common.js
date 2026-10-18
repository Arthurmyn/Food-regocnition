// Shared helpers: header, API calls, camera, small HTML builders.

export const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
export const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);

export function header(active) {
  const links = [["/classifier", "Classifier", "classifier"], ["/", "Hot Dog or Not", "hotdog"], ["/synthetic", "Synthetic Data", "synthetic"]];
  document.getElementById("header").innerHTML = `
    <div class="head"><div class="head-row">
      <a class="brand" href="/"><i></i>Food Lens</a>
      <nav class="nav">${links.map(([h, t, k]) => `<a href="${h}" class="${k === active ? "on" : ""}">${t}</a>`).join("")}</nav>
      <span class="pill"><i></i>EfficientNet-B0 · Local model</span>
    </div></div>`;
}

async function post(url, blob, extra = {}) {
  const fd = new FormData();
  fd.append("file", blob, "photo.jpg");
  for (const [k, v] of Object.entries(extra)) fd.append(k, v);
  const r = await fetch(url, { method: "POST", body: fd });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `Request failed (${r.status})`);
  return r.json();
}
export const predict = (blob) => post("/api/predict", blob);
export const explain = (blob, cls) => post("/api/explain", blob, { cls }).then((j) => j.cam);

// Camera ---------------------------------------------------------------
export async function startCamera(video) {
  if (!navigator.mediaDevices?.getUserMedia) throw new Error("This browser has no camera access (HTTPS is required).");
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 } }, audio: false });
  video.srcObject = stream;
  await video.play();
  return stream;
}
export const stopCamera = (stream) => stream && stream.getTracks().forEach((t) => t.stop());

export function snap(video, maxSide = 1024) {
  const k = Math.min(1, maxSide / Math.max(video.videoWidth, video.videoHeight));
  const c = document.createElement("canvas");
  c.width = Math.round(video.videoWidth * k);
  c.height = Math.round(video.videoHeight * k);
  c.getContext("2d").drawImage(video, 0, 0, c.width, c.height);
  return new Promise((res) => c.toBlob(res, "image/jpeg", 0.9));
}

// HTML builders --------------------------------------------------------
export function bars(classes, probs, top) {
  return `<ul class="bars">${classes.map((c, i) => `
    <li class="${i === top ? "top" : ""}"><span>${esc(cap(c))}</span>
      <span class="track"><span class="fill" style="width:${(probs[i] * 100).toFixed(1)}%"></span></span>
      <span class="mono">${Math.round(probs[i] * 100)}%</span></li>`).join("")}</ul>`;
}
export function neighbors(list) {
  return list.map((n) => `<figure><img src="${esc(n.url)}" alt="${esc(n.label)}" loading="lazy">
    <figcaption><span>${esc(cap(n.label))}</span><span>${n.sim.toFixed(2)}</span></figcaption></figure>`).join("");
}
export const footer = (r) => `<div class="foot">Inference: ${Math.round(r.latency_ms)} ms · Device: ${esc(r.device)}</div>`;
