"""Motion and accessibility rules for the shared stylesheet (web/css/lens.css), used by both frontends."""
import os
import re

from common import ROOT

CSS = open(os.path.join(ROOT, "web", "css", "lens.css")).read()


def test_no_transition_all_and_no_ease_in_on_ui():
    assert "transition:all" not in CSS.replace(" ", "")
    assert not re.search(r"ease-in(?!-out)", CSS)            # entrances and feedback use ease-out


def test_only_cheap_properties_are_animated():
    animated = re.findall(r"transition:([a-z-]+)", CSS)
    assert set(animated) <= {"transform", "opacity", "color", "border-color", "background-color", "box-shadow", "none"}


def _top_level_blocks(css):
    """Yield (prelude, body) for each top-level rule or at-rule, tracking brace depth."""
    depth, start, prelude = 0, 0, ""
    for i, ch in enumerate(css):
        if ch == "{":
            if depth == 0:
                prelude = css[start:i].strip()
                body_start = i + 1
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                yield prelude, css[body_start:i]
                start = i + 1


def test_hover_motion_is_limited_to_real_pointers():
    """A :hover rule that moves or scales something must sit inside @media (hover:hover) and (pointer:fine)."""
    css = re.sub(r"/\*.*?\*/", "", CSS, flags=re.S)
    bad = []
    for prelude, body in _top_level_blocks(css):
        if prelude.startswith("@media"):
            gated = "hover:hover" in prelude
            for inner_prelude, inner_body in _top_level_blocks(body):
                if ":hover" in inner_prelude and re.search(r"transform", inner_body) and not gated and "reduce" not in prelude:
                    bad.append(inner_prelude)
        elif ":hover" in prelude and "transform" in body:
            bad.append(prelude)
    assert bad == []


def test_user_preferences_are_respected():
    for query in ("prefers-reduced-motion:reduce", "prefers-reduced-transparency:reduce", "prefers-contrast:more"):
        assert query in CSS
    assert "animation-duration:.01ms" not in CSS             # reduced motion = gentler fades, not frozen UI
