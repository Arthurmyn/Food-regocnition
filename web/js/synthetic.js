// Synthetic images page: real counts from the filter step, six labelled samples per class.
import { header, esc, cap } from "./common.js";

header("synthetic");
const d = await (await fetch("/api/synthetic")).json();
const pct = Math.round((d.accepted / d.generated) * 100);
document.getElementById("lede").innerHTML =
  `SD-Turbo generated <b class="num">${d.generated}</b> images and a classifier trained on real photos kept ` +
  `<b class="num">${d.accepted}</b> (${pct}%). Adding them to training did not improve accuracy on real test photos: the change was within noise.`;
document.getElementById("classes").innerHTML = Object.entries(d.classes).map(([c, v]) => `
  <section class="syn-class">
    <div><h2>${esc(cap(c))}</h2>
      <div class="accept" role="img" aria-label="${v.accepted} of ${v.generated} accepted"><i style="width:${(v.accepted / v.generated) * 100}%"></i></div>
      <p class="of"><span class="num">${v.accepted}</span> of <span class="num">${v.generated}</span> accepted</p></div>
    <div class="tiles">${v.images.map((u) => `<div class="tile"><img src="${esc(u)}" alt="Synthetic ${esc(c)}" loading="lazy"><span>Synthetic</span></div>`).join("")}</div>
  </section>`).join("");
