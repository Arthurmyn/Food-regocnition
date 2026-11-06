"""The numbers quoted in the README/report/summary must come from the saved run files."""
import glob
import json
import os
import re

import numpy as np

from common import ROOT

A2 = os.path.join(ROOT, "results", "a2")


def _runs(name):
    return [json.load(open(p)) for p in sorted(glob.glob(os.path.join(A2, "runs", f"{name}_seed*.json")))]


def test_five_seeds_per_condition():
    for name in ("real_aug", "real_noaug", "real_plus_syn_all", "real_plus_syn_filtered"):
        assert len(_runs(name)) == 5


def test_summary_table_matches_run_files():
    summary = open(os.path.join(A2, "summary.md")).read()
    for name in ("real_aug", "real_noaug", "real_plus_syn_all", "real_plus_syn_filtered"):
        mean = np.mean([r["test"]["accuracy"] for r in _runs(name)])
        row = next(l for l in summary.splitlines() if l.startswith(f"| {name} "))
        shown = float(re.findall(r"\| (0\.\d{3}) ± ", row)[1])   # columns: val acc, test acc
        assert abs(shown - mean) < 0.0006


def test_synthetic_images_never_in_validation_or_test():
    for name in ("real_plus_syn_all", "real_plus_syn_filtered"):
        for r in _runs(name):
            assert r["n_train_real"] == 375 and r["n_train_synthetic"] in (200, 139)
            assert len(r["test"]["labels"]) == 79 and len(r["val"]["labels"]) == 81   # evaluation sets stay real and unchanged


def test_probe_selection_used_validation_only():
    j = json.load(open(os.path.join(A2, "linear_probe_metrics.json")))
    # C was chosen on validation accuracy: the first (smallest) C reaching the best validation score wins
    assert str(j["chosen_C"]) == next(c for c, v in j["selection"].items() if v == max(j["selection"].values()))
    best_val = max(j["selection"].values())
    assert j["val_acc_at_chosen_C"] == best_val
    assert j["export_check_max_prob_diff"] < 1e-6      # exported PyTorch head == scikit-learn head
