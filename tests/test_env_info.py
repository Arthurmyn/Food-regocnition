import pytest

from env_info import browser_label

CASES = [
    ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36",
     "Chrome 141 on macOS"),
    ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36 Edg/141.0.0.0",
     "Edge 141 on Windows"),
    # iPhone UAs also say "Mac OS X"; Android UAs also say "Linux"; Chrome UAs also say "Safari"
    ("Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Mobile/15E148 Safari/604.1",
     "Safari 17 on iOS"),
    ("Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Mobile Safari/537.36",
     "Chrome 141 on Android"),
    ("Mozilla/5.0 (X11; Linux x86_64; rv:130.0) Gecko/20100101 Firefox/130.0", "Firefox 130 on Linux"),
    ("", "unknown browser on unknown OS"),
    (None, "unknown browser on unknown OS"),
]


@pytest.mark.parametrize("ua,expected", CASES)
def test_browser_label(ua, expected):
    assert browser_label(ua) == expected
