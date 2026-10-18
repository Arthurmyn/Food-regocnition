# Look and feel shared by all pages: fonts, colours, header, cards, probability bars.
# Layout follows the supplied Lovable mock-up; fonts and colours are our own.
import html

import streamlit as st

import ui

FONTS = ("@import url('https://fonts.googleapis.com/css2?family=Schibsted+Grotesk:wght@400;500;600;700"
         "&family=JetBrains+Mono:wght@400;500&display=swap');")

LIGHT = """
:root{--bg:#EEF0EA;--card:#FBFBF8;--ink:#16201B;--muted:#5E6A63;--line:#D3D9CF;--accent:#C98F00;
  --ok:#1E6B47;--ok-soft:#D9EEE2;--syn:#B02672;--syn-soft:#F8E3EF;--track:#DFE4DA}
"""

DARK = """
:root{--bg:#0F1613;--card:#18221E;--ink:#EDF1EA;--muted:#8E9C94;--line:#2A3731;--accent:#E0A100;
  --ok:#5BC48B;--ok-soft:#183A2A;--syn:#E56BAA;--syn-soft:#3A1B2C;--track:#26322C}
.stApp{background:var(--bg)!important}
header[data-testid="stHeader"]{background:transparent!important}
[data-testid="stCameraInputWebcamStyledBox"],[data-testid="stCameraInput"] > div{background:var(--card)!important;
  border:1px solid var(--line);border-radius:6px;color:var(--ink)}
[data-testid="stCameraInput"] button{background:var(--accent)!important;color:#1A1300!important;border:0;
  font-weight:600}
[data-testid="stCameraInputWebcamComponent"] div{background:transparent!important}
[data-testid="stCameraInput"] a{color:var(--accent)!important}
[data-testid="stCameraInput"] svg{fill:var(--muted)!important}
[data-testid="stExpander"] details{border:1px solid var(--line);background:var(--card);border-radius:4px}
[data-testid="stFileUploaderDropzone"]{background:var(--card)!important}
"""

BASE = """
.stApp,.stApp p,.stApp h1,.stApp h2,.stApp h3,.stApp label,.stApp li,.stApp button,
.stApp div[data-testid="stMarkdownContainer"],.stApp [data-testid="stCaptionContainer"]{
  font-family:'Schibsted Grotesk',system-ui,sans-serif}
.stApp,.stApp p,.stApp h1,.stApp h2,.stApp h3,.stApp label,.stApp li,
.stApp [data-testid="stCaptionContainer"]{color:var(--ink)}
.stApp [data-testid="stCaptionContainer"]{color:var(--muted)}
.mono{font-family:'JetBrains Mono',ui-monospace,monospace}
.block-container{max-width:1100px;padding-top:2rem;padding-bottom:3rem}
h1,h2,h3{letter-spacing:-.015em;font-weight:600}
.fl-brand{display:flex;align-items:center;gap:8px;font-weight:700;font-size:17px;color:var(--ink)}
.fl-dot{width:13px;height:13px;border-radius:3px;background:var(--accent);display:inline-block}
.fl-pill{border:1px solid var(--line);border-radius:999px;padding:3px 10px;font-size:11px;
  color:var(--muted);white-space:nowrap;float:right}
.fl-pill i{display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--ok);margin-right:6px;
  vertical-align:middle}
.fl-rule{border:0;border-top:1px solid var(--line);margin:.6rem 0 1.4rem}
.fl-card{background:var(--card);border:1px solid var(--line);border-radius:4px;overflow:hidden}
.fl-pad{padding:18px 20px}
.fl-sep{border-top:1px solid var(--line)}
.fl-eyebrow{font-size:11.5px;text-transform:uppercase;letter-spacing:.09em;color:var(--muted);margin:0 0 8px}
.fl-pred{display:flex;justify-content:space-between;align-items:baseline}
.fl-pred b{font-size:34px;font-weight:600;letter-spacing:-.02em}
.fl-pct{font-family:'JetBrains Mono',monospace;font-size:14px;background:var(--ok-soft);color:var(--ok);
  padding:2px 8px;border-radius:3px}
.fl-bars{list-style:none;margin:18px 0 0;padding:0;display:grid;gap:10px}
.fl-bars li{display:grid;grid-template-columns:5.5rem 1fr 2.6rem;align-items:center;gap:12px;font-size:14px;
  color:var(--muted)}
.fl-bars li.top{color:var(--ink);font-weight:600}
.fl-track{height:6px;background:var(--track);border-radius:99px;overflow:hidden}
.fl-fill{display:block;height:100%;background:var(--muted);opacity:.45}
li.top .fl-fill{background:var(--ok);opacity:1}
.fl-bars .mono{text-align:right;font-size:12px;color:var(--muted);font-weight:400}
.fl-img{display:block;width:100%;object-fit:cover}
.fl-tag{display:inline-block;font-size:11px;background:var(--card);color:var(--muted);padding:2px 8px;
  border-radius:2px;border:1px solid var(--line)}
.fl-foot{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--muted);padding:12px 20px;
  border-top:1px solid var(--line)}
.fl-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:10px}
.fl-grid figure{margin:0;border:1px solid var(--line);background:var(--card)}
.fl-grid img{display:block;width:100%;aspect-ratio:1;object-fit:cover}
.fl-grid figcaption{display:flex;justify-content:space-between;padding:6px 8px;font-size:12.5px}
.fl-grid figcaption span:last-child{font-family:'JetBrains Mono',monospace;color:var(--muted)}
.fl-warn{border:1px solid var(--syn);background:var(--syn-soft);color:var(--syn);border-radius:3px;
  padding:10px 16px;font-size:14px}
.fl-warn b{letter-spacing:.06em}
.fl-syn-grid{display:grid;grid-template-columns:repeat(6,1fr);gap:8px}
.fl-syn-grid div{position:relative;aspect-ratio:1;overflow:hidden;background:var(--track)}
.fl-syn-grid img{width:100%;height:100%;object-fit:cover;display:block}
.fl-syn-grid span{position:absolute;left:4px;bottom:4px;background:var(--syn);color:#fff;font-size:10px;
  font-weight:600;padding:1px 6px;border-radius:2px}
.fl-sec{display:flex;justify-content:space-between;align-items:baseline;border-bottom:1px solid var(--line);
  padding-bottom:6px;margin:30px 0 12px}
.fl-sec h2{font-size:15px;margin:0}
.fl-sec span{font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--muted)}
.fl-stat dt{font-size:11.5px;text-transform:uppercase;letter-spacing:.09em;color:var(--muted)}
.fl-stat dd{margin:4px 0 0;font-family:'JetBrains Mono',monospace;font-size:26px}
.fl-stat div{padding:16px 18px;border-top:1px solid var(--line)}
.fl-stat div:first-child{border-top:0}
.stButton>button,.stApp [data-testid="stBaseButton-secondary"]{border-radius:4px;border:1px solid var(--line);
  background:var(--card);color:var(--ink)}
.stApp [data-testid="stPageLink"] a{border-radius:4px}
.stApp [data-testid="stPageLink"] a p{font-size:14px}
@media (max-width:760px){
  .fl-grid{grid-template-columns:repeat(2,1fr)}.fl-syn-grid{grid-template-columns:repeat(3,1fr)}
  .fl-pill{display:none}
}
"""


def apply(dark=False):
    st.html(f"<style>{FONTS}{LIGHT if not dark else DARK}{BASE}</style>")


def header(pages):
    """Brand on the left, page links in the middle, model pill on the right."""
    a, b, c = st.columns([2.2, 5, 3], vertical_alignment="center")
    a.html('<span class="fl-brand"><i class="fl-dot"></i>Food Lens</span>')
    with b:
        links = st.columns(len(pages))
        for col, page in zip(links, pages):
            col.page_link(page, label=page.title)
    c.html('<span class="fl-pill mono"><i></i>EfficientNet-B0 · Local model</span>')
    st.html('<hr class="fl-rule">')


def footer(device, ms=None):
    inf = f"Inference: {ms:.0f} ms · " if ms is not None else ""
    return f'<div class="fl-foot">{inf}Device: {html.escape(str(device))}</div>'


def prob_bars(classes, probs, top):
    rows = "".join(
        f'<li class="{"top" if i == top else ""}"><span>{html.escape(c.capitalize())}</span>'
        f'<span class="fl-track"><span class="fl-fill" style="width:{p * 100:.1f}%"></span></span>'
        f'<span class="mono">{p * 100:.0f}%</span></li>'
        for i, (c, p) in enumerate(zip(classes, probs)))
    return f'<ul class="fl-bars">{rows}</ul>'


def img_tag(img, cls="fl-img", style="", max_side=640):
    return f'<img class="{cls}" style="{style}" src="data:image/jpeg;base64,{ui.to_b64(img, max_side)}">'
