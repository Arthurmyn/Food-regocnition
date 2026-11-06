# Which browser and OS is the visitor using? Parsed from the User-Agent string (assignment: show the environment).
# The same rules are implemented in JS in web/js/common.js (browserLabel).
import re

_BROWSERS = (  # order matters: Edge and Chrome both say "Chrome", Chrome also says "Safari"
    ("Edge", r"Edg/(\d+)"),
    ("Firefox", r"(?:Firefox|FxiOS)/(\d+)"),
    ("Chrome", r"(?:Chrome|CriOS)/(\d+)"),
    ("Safari", r"Version/(\d+)[\d.]* (?:Mobile/\S+ )?Safari"),
)
_SYSTEMS = (  # iPhone UAs contain "Mac OS X", Android UAs contain "Linux": test the specific ones first
    ("iOS", r"iPhone|iPad|iPod"),
    ("Android", r"Android"),
    ("Windows", r"Windows"),
    ("macOS", r"Macintosh|Mac OS X"),
    ("Linux", r"Linux|X11"),
)


def browser_label(user_agent):
    ua = user_agent or ""
    browser = next((f"{name} {m.group(1)}" for name, pat in _BROWSERS if (m := re.search(pat, ua))), "unknown browser")
    system = next((name for name, pat in _SYSTEMS if re.search(pat, ua)), "unknown OS")
    return f"{browser} on {system}"
