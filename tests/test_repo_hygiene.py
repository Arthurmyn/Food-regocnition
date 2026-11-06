"""Everything the deployed apps need must actually be committed (a .gitignore rule once hid the synthetic samples)."""
import shutil
import subprocess

import pytest

from common import ROOT


def _git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.split("\n")


pytestmark = pytest.mark.skipif(shutil.which("git") is None or not _git("rev-parse", "--git-dir")[0],
                                reason="not a git checkout")


def test_no_frontend_or_result_file_is_hidden_by_gitignore():
    hidden = [p for p in _git("ls-files", "--others", "--ignored", "--exclude-standard", "web", "app", "server", "src", "results")
              if p and "__pycache__" not in p]
    assert hidden == []


def test_synthetic_sample_images_are_tracked():
    assert len([p for p in _git("ls-files", "web/assets/synthetic") if p.endswith(".jpg")]) == 30   # 6 per class
    assert len([p for p in _git("ls-files", "web/assets/train") if p.endswith(".jpg")]) == 375
