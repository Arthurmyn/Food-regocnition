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
- Model choice for the app: see section 7. The deployed model is the frozen-ImageNet linear probe, not the A1 fine-tuned checkpoint.
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

## 6. Web app ("Food Lens"): two services in one

There are two frontends on the same model. Public link: Streamlit (`app/`, runs on Streamlit Cloud). Alternative: FastAPI (`server/`) serves the model API and the static frontend in `web/` (plain HTML/CSS/JS,
no framework, no build step). The layout follows a Lovable mock-up supplied by the author (static React mock with
a fake classifier; only the layout was reused); fonts (Schibsted Grotesk, JetBrains Mono) and colours are our own.
API: `POST /api/predict` (class probabilities, mean latency of 10 warm runs, device, 5 nearest training photos),
`POST /api/explain` (Grad-CAM overlay for a class), `GET /api/synthetic` (counts + samples).

- **Hot Dog or Not** (joke service, home page, dark): the camera opens as soon as the page loads. Shutter ->
  "Evaluating..." spinner (at least 1.2 s) -> green "Hotdog" / red "Not hotdog" banner. Buttons: "Take another
  photo", "Additional information" (verdict, probabilities, Grad-CAM for hotdog, 5 nearest training photos,
  latency + device). Fallback: upload a photo if the camera is unavailable. Own artwork; inspired by the gag in
  the TV show Silicon Valley (credit it, no logos).
- **Classifier** (the real service): upload / drag-and-drop / camera -> Top-1 class + probability bars, Grad-CAM,
  nearest training examples, latency. The page opens with `example.jpg`, a real TEST photo (the most confident
  correct pizza), picked by hand, so say it is an illustrative example, not a typical one.
- **Synthetic Data**: SYNTHETIC MEDIA banner, real counts (200 generated, 139 accepted), 6 samples per class.
- Joke rule: "Hotdog" iff P(hotdog) >= 0.4 (`THRESHOLD` in `web/js/hotdog.js`, `HOTDOG_THRESHOLD` in the Streamlit view).
  Chosen on the validation set, where F1 is flat (0.87) for 0.35-0.5. Anything else, including non-food, is "Not hotdog".
- Tested end to end in headless Chrome with a fake camera feed (a real hotdog photo -> green, a real pizza
  photo -> red, "Take another photo" restarts the camera, classifier camera path, phone viewport).
  Not tested: a real phone camera, Docker build, a deployed instance.
- Hotdog-vs-rest of the DEPLOYED model on the real test set (79 images, 14 hotdogs), `src/hotdog_eval.py`:
  thr 0.5: precision 1.00, recall 0.93 (13/14); thr 0.4: precision 0.87, recall 0.93. Tiny test set, so indicative.
  On the 200 synthetic images (stress test, intended class as label): thr 0.5 P 0.97 / R 0.73, thr 0.4 P 0.89 /
  R 0.78, thr 0.3 P 0.85 / R 0.88. So 0.4 trades some precision for recall; for a joke app a missed hot dog is the
  worse error. The test-set numbers do not favour 0.4 over 0.5, say so if asked.
- Non-photo probes (noise, flat colours, gradient, text): P(hotdog) 0.01-0.06, all "Not hotdog".
  Real non-food objects (shoe, cup, person) were NOT tested; the model never saw them, so it can
  still be confidently wrong on them. Say so in the report.
- Privacy: photos are sent to the server for inference and not stored. The nearest-neighbour gallery shows
  training photos; at least one training photo shows an identifiable person. Review/remove such photos
  before sharing the repo or app publicly.

## 7. Model change after real-photo testing (read before writing the results section)

**Trigger.** The author tried real photos: a hot dog was called "Not hotdog" (P=29%) and the classifier answered
"sushi" for another photo. This was checked, not assumed.

**Findings (all reproducible with `src/train_linear_probe.py`, results in `results/a2/linear_probe_metrics.json`).**
- The A1 fine-tuned checkpoint (79.7% test) is fragile: on 200 SD-Turbo images it scores 0.78 overall, and
  hot dog recall is only 0.38 at P>=0.5 (hot dogs get called sushi/sandwich/burger).
- A logistic-regression head on FROZEN ImageNet EfficientNet-B0 features generalises better (C chosen on the
  validation set, test never used for any choice):

| Model | Val | Test (79 real) | Synthetic (200) | Test rotated 90° | Hot dog recall on synthetic (P>=0.5) |
|---|---|---|---|---|---|
| A1 fine-tuned | 0.753 | 0.797 | 0.780 | 0.709 | 0.38 |
| Linear probe, train only | 0.790 | 0.899 | 0.900 | | |
| Linear probe, train+val (DEPLOYED) | | 0.886 | 0.895 | 0.797 | 0.73 |

- Paired on the test set: probe right / A1 wrong on 9 images, A1 right / probe wrong on 2. Taken alone this
  is suggestive, not significant (exact sign test p ~ 0.07); the validation gap and the 200-image synthetic
  set point the same way. Synthetic images are NOT independent real photos, so they are a stress test only.
- Consistent with earlier results: frozen ImageNet features also retrieve better (P@5 0.81 vs 0.72), and in A1
  the best checkpoint was already "head only". Fine-tuning 4M parameters on 375 images hurt.
- Synthetic data as extra TRAINING data for the probe also gave no gain (test 0.886 with all, 0.899 with
  filtered, 0.899 real only): the earlier conclusion holds for this model too. The synthetic ablation in
  section 2 was run on the A1 fine-tuning recipe, not on the probe.
- **EXIF bug (fixed):** PIL ignores the orientation tag of phone photos, so uploads could reach the model
  rotated. Rotating the test set by 90° costs 8-9 points for both models. Test: the same hot dog scores
  P(hotdog)=0.99 upright / with a correct EXIF tag, and 0.46 ("sandwich") when the rotated pixels carry no tag.
  The camera path (canvas snapshot) has no EXIF, so it was never affected.

**What changed in the app.** Deployed model = frozen ImageNet EfficientNet-B0 + the logistic-regression head
(`results/a2/linear_probe.pt`, same architecture as A1, so Grad-CAM and retrieval code are unchanged); EXIF
orientation applied everywhere; hot dog threshold 0.4; nearest-neighbour index rebuilt from the same frozen features.

**Caveats to state.** (1) Fixing the model was not verified on the author's own photos. (2) A1 report numbers
(SVM 0.53, fine-tuned EfficientNet 78.5/89.9%) are unchanged and still describe the A1 work; the probe is a new,
better baseline found in A2. (3) 79 test images: differences under ~5 points are not reliable. (4) The remaining
weakness is the data: 535 images with ambiguous classes (burger / sandwich / hot dog) and a few mislabelled
or person-containing photos.
