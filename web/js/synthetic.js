// Synthetic images page: real counts from the filter step, 6 labelled samples per class.
import { header, esc, cap } from "./common.js";

header("synthetic");
const d = await (await fetch("/api/synthetic")).json();
document.getElementById("stats").innerHTML = `
  <div><dt>Generated</dt><dd>${d.generated}</dd></div>
  <div><dt>Accepted after filtering</dt><dd style="color:var(--ok)">${d.accepted}</dd></div>
  <div><dt>Purpose</dt><dd class="text">Test whether synthetic images improve classification. In our runs they did not:
    the gain was within noise.</dd></div>`;
document.getElementById("classes").innerHTML = Object.entries(d.classes).map(([c, v], i) => `
  <div class="sec" ${i === 0 ? 'style="margin-top:0"' : ""}><h2>${esc(cap(c))}</h2><span>${v.accepted} / ${v.generated} accepted</span></div>
  <div class="syn-grid">${v.images.map((u) => `<div><img src="${esc(u)}" alt="Synthetic ${esc(c)}" loading="lazy"><span>Synthetic</span></div>`).join("")}</div>`).join("");
