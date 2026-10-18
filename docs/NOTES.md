# A2 working notes (raw material for the report, not the report itself)

## 1. What was done

1. **Repo set up from A1.** The split regenerates exactly (375/81/79, seed 42). The saved A1 checkpoint
   `results/a1/efficientnet_best.pt` reproduces **63/79 = 79.7%** test accuracy.
2. **A1 inconsistency found.** `a1/train_efficientnet.py` in this repo (8 fine-tune epochs, LR 1e-5, no cosine,
   no early stopping, no BN freezing, no `augment` argument) does not match the A1 report/notebook
   (15 epochs, LR 1e-4, cosine, early stopping, BN eval, `main(augment=False, suffix="_noaug")`;
   78.5% / 89.9%). The `A1/` notebook is the newer one. The A1 numbers (78.5/89.9) cannot be
   reproduced from this repo's script. A2 numbers all come from `train.py`.
3. **New training script `train.py`**: same recipe as the repo's A1 script (head 10 ep LR 1e-3, then
   fine-tune 8 ep LR 1e-5, best epoch by val acc), plus `--seed`, `--no-augment`, `--synthetic <dir>`.
   Synthetic images are added to TRAIN only. Metrics: Top-1, Top-5, macro-F1, per-class recall.
4. **Baselines**: 5 seeds x {augmentation, no augmentation}, real data only.
5. **Synthetic data**: SD-Turbo (`stabilityai/sd-turbo`, 2 steps, 512px), 40 images per class = 200,
   prompts = subject x angle x setting + "amateur smartphone food photo". Prompts/seeds in
   `synthetic/metadata.csv`. Filtered with the A1 classifier (keep if predicted == intended class and
   confidence >= 0.5): 139 of 200 kept.
6. **Ablation**: real+aug vs real+all synthetic vs real+filtered synthetic, 5 seeds each.
7. **Retrieval** (`retrieval.py`): L2-normalised 1280-d features, index = real train, queries = real test,
   relevant = same class.
8. **Explainability**: Grad-CAM written in pure PyTorch (`explain.py`), last conv block.
9. **App** (`app.py`, Streamlit): upload/camera -> prediction -> Grad-CAM -> 5 nearest training images ->
   latency + device. Tabs: "Predict & explain", "Hotdog or Not" (joke mode, same model), "Synthetic
   training images" (labelled SYNTHETIC).

## 2. Results (all on the real 79-image test set; mean ± std over 5 seeds)

| Training set | train real/syn | Val acc | Test acc | Top-5 | Macro-F1 |
|---|---|---|---|---|---|
| Real + aug | 375/0 | 0.731 ± 0.016 | 0.800 ± 0.009 | 1.000 | 0.806 ± 0.008 |
| Real, no aug | 375/0 | 0.760 ± 0.020 | **0.838 ± 0.012** | 1.000 | 0.843 ± 0.012 |
| Real + all synthetic | 375/200 | 0.746 ± 0.015 | 0.808 ± 0.017 | 1.000 | 0.810 ± 0.016 |
| Real + filtered synthetic | 375/139 | 0.751 ± 0.009 | 0.785 ± 0.035 | 1.000 | 0.788 ± 0.035 |

Per-class recall (test): see `results/a2/summary.md`. Sandwich: 0.765 (aug), 0.871 (no aug),
0.788 (+all syn), 0.729 ± 0.142 (+filtered syn).
A1 baseline for reference: SVM+HOG 0.532 test acc, F1 0.502 (`results/a1/model_comparison.json`).

Paired per-seed differences in test accuracy:
- +all synthetic vs real+aug: +0.8 pts on average, per seed [+3.8, 0, 0, -1.3, +1.3] -> within noise.
- +filtered synthetic vs real+aug: -1.5 pts, per seed [-6.3, -2.5, +1.3, -3.8, +3.8] -> no gain.
- real+aug vs real no-aug: -3.8 pts, negative in all 5 seeds.

Retrieval P@k (train-set index, test queries):

| Features | P@1 | P@5 | P@10 |
|---|---|---|---|
| Frozen ImageNet EfficientNet-B0 | 0.873 | 0.805 | 0.762 |
| A1 fine-tuned model | 0.835 | 0.719 | 0.642 |

Latency (app, MPS, forward pass only, warm, mean of 10): about 6 ms. Measured while training was
running in the background, so re-measure on an idle machine for the report.

Synthetic filter: burger 37/40, hotdog 15/40, pizza 36/40, sandwich 15/40, sushi 36/40 accepted.
Hotdog was often rejected as "sushi" (14 of 40) and sandwich as "sushi" (10 of 40).

## 3. Honest conclusions / caveats

- Synthetic data **did not give a measurable improvement**. Do not claim otherwise.
- Std over seeds only measures training randomness. The test set is 79 images, so one image = 1.3 pts;
  differences below ~5 pts should not be called real (A1 bootstrap CI half-width was ~8-9 pts).
- Augmentation hurting accuracy is consistent across 5 seeds here (and in A1); val/test were not tuned for it.
- The filter uses a classifier trained on the same real train set, so it keeps "easy" images and
  rejects hard ones, which is the opposite of what hard-case synthesis wants. Possible explanation for
  the filtered set being worse. Not tested.
- Top-5 is always 1.0 with 5 classes, so it carries no information.
- Model choice for the app: the app uses the A1 fine-tuned checkpoint; its retrieval gallery is built from
  the fine-tuned features, which retrieve worse than the frozen ImageNet ones (table above).
- SD-Turbo is a **substitute** for the assigned TA-TiTok + MaskGen / ArtAug (no runnable public
  weights were used). Say so in the report and cite Neurohive for the assigned methods.
- Some real images come from the web (A1 report section 2.3 mentions a watermarked hotdog image).
- "Hotdog or Not" has only seen 4 other food classes; it will be confidently wrong on non-food.
- Grad-CAM example: the pizza test image is predicted as hotdog (33%), attention is on a bottle/label,
  not on the pizza. Good failure case to show.
- Not tested: browser upload/camera widgets in the app (only the loading, model, Grad-CAM, kNN and
  latency code paths were tested).

## 4. Figures available

- `results/a2/gradcam_examples.png`: one test image per class with Grad-CAM (includes a failure case).
- `results/a2/a2_training_curves.png`: mean train loss / val acc per condition (5 seeds).
- `results/a1/efficientnet_training_curves.png`, `results/a1/confusion_matrix_*_test.png`: A1.
- `results/a2/synthetic_contact_sheet.jpg`: 8 synthetic images per class.
- `results/a1/dataset_eda.png`: class balance / sizes.
- Still to make yourself: app screenshots, a nearest-neighbour gallery figure (take it from the app).

## 5. Files

`src/` (train, generate_synthetic, filter_synthetic, retrieval, explain, aggregate, make_figures),
`app/app.py`, `scripts/run_baselines.sh`, `scripts/run_ablation.sh`; results in `results/a2/`
(`runs/*.json` per-run metrics, `summary.md`, `summary.json`, `retrieval.json`, `synthetic_filter.json`).
A1 code in `a1/`, A1 results in `results/a1/`. How to run: `README.md`.

## 6. Web app: two services in one

- Page 1 (default): **Hotdog or Not**, joke service. Camera or upload -> "Evaluating..." spinner -> green
  "Hotdog" / red "Not hotdog" banner. "Additional information" button below: P(hotdog), class
  probabilities, Grad-CAM for hotdog, 5 nearest training photos, latency + device. Own artwork;
  inspired by the gag in the TV show Silicon Valley (credit it, do not use its logos).
- Page 2: **Food classifier**, the real service (Top-1 class, Grad-CAM, nearest neighbours, latency).
- Page 3: synthetic images, labelled SYNTHETIC.
- Joke rule: "Hotdog" iff P(hotdog) >= 0.5 (`HOTDOG_THRESHOLD` in `app/views/hotdog.py`). Anything else,
  including non-food, is "Not hotdog".
- Hotdog-vs-rest on the real test set (79 images, 14 hotdogs), `src/hotdog_eval.py`, thr 0.5:
  precision 1.00, recall 0.93 (13/14), accuracy 0.99. Thr 0.4: precision 0.88, recall 1.00.
  Tiny test set, so treat as indicative.
- Non-photo probes (noise, flat colours, gradient, text): P(hotdog) 0.11-0.21, all "Not hotdog".
  Real non-food objects (shoe, cup, person) were NOT tested; the model never saw them, so it can
  still be confidently wrong on them. Say so in the report.
