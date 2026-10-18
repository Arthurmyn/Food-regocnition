"""HTML/CSS for the joke "Hotdog or Not" phone screen (own artwork, no third-party assets)."""
import base64
import io
import math

from PIL import Image

CSS = """
<style>
.hd-phone{width:100%;max-width:380px;margin:0 auto;border:10px solid #111;border-radius:38px;
  background:#111;box-shadow:0 10px 30px rgba(0,0,0,.35);overflow:hidden}
.hd-screen{position:relative;height:540px;background:#222;overflow:hidden;border-radius:26px}
.hd-screen img{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;display:block}
.hd-dim img{filter:brightness(.45) saturate(.7)}
.hd-empty{display:flex;align-items:center;justify-content:center;flex-direction:column;
  height:100%;color:#fff;font:700 22px/1.3 system-ui,sans-serif;text-align:center;padding:0 24px}
.hd-empty span{font-size:64px;margin-bottom:12px}
.hd-spin{position:absolute;left:50%;top:38%;width:0;height:0}
.hd-spin i{position:absolute;width:20px;height:20px;margin:-10px;border-radius:50%;
  background:#e63950;border:3px solid #111;animation:hd-pulse 1s linear infinite}
@keyframes hd-pulse{0%{opacity:1;transform:scale(1.15)}100%{opacity:.15;transform:scale(.7)}}
.hd-text{position:absolute;left:0;right:0;top:58%;text-align:center;color:#fff;
  font:800 34px system-ui,sans-serif;-webkit-text-stroke:2px #111;paint-order:stroke fill;
  text-shadow:0 2px 0 #111}
.hd-banner{position:absolute;left:0;right:0;height:92px;display:flex;align-items:center;
  justify-content:center;color:#fff;font:800 40px system-ui,sans-serif;
  -webkit-text-stroke:2px #111;paint-order:stroke fill;text-shadow:0 2px 0 #111}
.hd-yes{top:0;background:#2db84b}
.hd-no{bottom:0;background:#b3101e}
.hd-badge{position:absolute;left:50%;transform:translateX(-50%);width:84px;height:84px;
  border-radius:50%;display:flex;align-items:center;justify-content:center;color:#fff;
  font:800 52px system-ui,sans-serif}
.hd-badge-yes{top:70px;background:#2db84b}
.hd-badge-no{bottom:70px;background:#b3101e}
</style>
"""

_DOTS = "".join(
    f'<i style="left:{60 * math.cos(k * math.pi / 6 - math.pi / 2):.1f}px;'
    f'top:{60 * math.sin(k * math.pi / 6 - math.pi / 2):.1f}px;animation-delay:{-k / 12:.2f}s"></i>'
    for k in range(12)
)


def to_b64(img, max_side=640):
    im = img.convert("RGB")
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode()


def phone(inner):
    return f'{CSS}<div class="hd-phone"><div class="hd-screen">{inner}</div></div>'


def screen_empty():
    return phone('<div class="hd-empty"><span>🌭</span>Point the camera at food<br>'
                 'or upload a photo</div>')


def screen_evaluating(b64):
    return phone(f'<div class="hd-dim"><img src="data:image/jpeg;base64,{b64}"></div>'
                 f'<div class="hd-spin">{_DOTS}</div><div class="hd-text">Evaluating...</div>')


def screen_result(b64, is_hotdog):
    img = f'<img src="data:image/jpeg;base64,{b64}">'
    if is_hotdog:
        return phone(f'{img}<div class="hd-banner hd-yes">Hotdog</div>'
                     '<div class="hd-badge hd-badge-yes">✓</div>')
    return phone(f'{img}<div class="hd-banner hd-no">Not hotdog</div>'
                 '<div class="hd-badge hd-badge-no">✕</div>')
