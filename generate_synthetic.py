"""Generate synthetic food images (SD-Turbo, text-to-image) for training-set enrichment.

Prompts vary dish, plating, angle, lighting and background. `sandwich` and `burger` get extra
hard-case prompts (the main A1 confusion). Output: synthetic/raw/<class>/syn_<class>_NNN.png
plus synthetic/metadata.csv (prompt, seed). All images are synthetic and used for TRAIN only.

    python generate_synthetic.py --per-class 40
"""
import argparse
import csv
import os
import random

import torch
from diffusers import AutoPipelineForText2Image

from common import CLASSES

SUBJECTS = {
    "burger": ["a hamburger with a beef patty, cheese, lettuce and tomato in a sesame bun",
               "a double cheeseburger", "a homemade burger on a wooden board",
               "a chicken burger with sauce", "a burger with fries and a drink"],
    "hotdog": ["a hot dog with mustard and ketchup in a bun", "a grilled sausage hot dog with onions",
               "a chili cheese hot dog", "two hot dogs on a tray", "a street-vendor hot dog"],
    "pizza": ["a pepperoni pizza", "a margherita pizza with basil", "a slice of cheese pizza being lifted",
              "a whole pizza on a wooden peel", "a vegetable pizza"],
    "sandwich": ["a club sandwich cut in half with toothpicks", "a toasted ham and cheese sandwich",
                 "a sliced sandwich on sliced bread with lettuce and tomato",
                 "a deli sandwich on a baguette", "a grilled panini sandwich",
                 "a sandwich with fries on the side", "a submarine sandwich"],
    "sushi": ["a plate of assorted nigiri sushi", "salmon sushi rolls with chopsticks",
              "a maki sushi platter with wasabi and ginger", "tuna nigiri on a wooden board",
              "a sushi set in a restaurant"],
}
ANGLES = ["top-down view", "side view", "45 degree angle", "close-up", "low angle shot"]
SETTINGS = ["on a white plate", "on a wooden table", "in a restaurant", "on a kitchen counter",
            "on a paper wrapper", "outdoors at a picnic table", "in dim warm lighting",
            "under bright natural window light"]
STYLE = "amateur smartphone food photo, realistic"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=40)
    ap.add_argument("--out", default="synthetic")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sd-turbo", torch_dtype=torch.float16 if device != "cpu" else torch.float32).to(device)
    pipe.set_progress_bar_config(disable=True)

    rng = random.Random(args.seed)
    rows = []
    for cls in CLASSES:
        d = os.path.join(args.out, "raw", cls)
        os.makedirs(d, exist_ok=True)
        for i in range(args.per_class):
            prompt = f"{rng.choice(SUBJECTS[cls])}, {rng.choice(ANGLES)}, {rng.choice(SETTINGS)}, {STYLE}"
            seed = rng.randrange(10**9)
            g = torch.Generator("cpu").manual_seed(seed)
            img = pipe(prompt, num_inference_steps=args.steps, guidance_scale=0.0, generator=g,
                       height=512, width=512).images[0]
            fn = f"syn_{cls}_{i:03d}.png"
            img.save(os.path.join(d, fn))
            rows.append([cls, fn, prompt, seed])
            print(cls, i, prompt, flush=True)
    with open(os.path.join(args.out, "metadata.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "file", "prompt", "seed"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
