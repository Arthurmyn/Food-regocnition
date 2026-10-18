#!/bin/sh
# Builds ./space_build/ : exactly the files a Hugging Face Space (Docker SDK) needs.
# Run from the repo root, then upload the folder's contents to the Space (see docs/DEPLOY.md).
set -e
rm -rf space_build
mkdir -p space_build/results/a1 space_build/results/a2
cp -r Dockerfile server src web space_build/
cp results/a1/efficientnet_best.pt space_build/results/a1/
cp results/a2/index.npz results/a2/synthetic_filter.json space_build/results/a2/
find space_build -name __pycache__ -type d -prune -exec rm -rf {} +
# files over 10 MB must go through Git LFS on Hugging Face
printf '*.pt filter=lfs diff=lfs merge=lfs -text\n' > space_build/.gitattributes
cat > space_build/README.md <<'EOF'
---
title: Food Lens
emoji: 🌭
colorFrom: yellow
colorTo: green
sdk: docker
app_port: 7860
---

Food Lens: burger / hotdog / pizza / sandwich / sushi classifier with Grad-CAM, plus a joke
"Hot Dog or Not" camera page. Course project (Astana IT University, Computer Vision, Assignment 2).
EOF
du -sh space_build
