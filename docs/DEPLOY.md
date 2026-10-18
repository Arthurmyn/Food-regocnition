# Deploying Food Lens

The app is one FastAPI process: `uvicorn server.main:app` serves `/api/*` and the static pages in `web/`.
Browsers allow the camera only on **HTTPS or localhost**, so any of the hosts below (all HTTPS) works.
Streamlit Community Cloud cannot run it (it only runs Streamlit apps); delete the old Streamlit app there.

## Local

```bash
python3 -m pip install -r requirements.txt
uvicorn server.main:app --port 8000          # http://localhost:8000
```

## Hugging Face Spaces (free CPU, 16 GB RAM, Docker)

1. On huggingface.co: New Space -> SDK **Docker** (blank template) -> public or private.
2. From the repo root run `./scripts/package_space.sh`. It builds `space_build/` with exactly the needed files
   (Dockerfile, server, src, web, checkpoint, index) plus the Space `README.md` front matter and a Git LFS rule.
3. Upload the **contents** of `space_build/` to the Space:
   - easiest: Space page -> Files -> Add file -> Upload files, drag everything from `space_build/`
     (the website sends the 16 MB checkpoint through LFS automatically); or
   - with git: `git lfs install` (needs `brew install git-lfs`), then `cd space_build`, `git init`,
     `git remote add space https://huggingface.co/spaces/<user>/<space>`, `git add .`, `git commit -m deploy`,
     `git push space HEAD:main`. Files over 10 MB are rejected without LFS.
4. The Space builds the Dockerfile (installs CPU PyTorch, a few minutes), then the app is live at
   `https://<user>-<space>.hf.space`. After idle the free Space sleeps, so open it before a demo.

## Other hosts

Render / Railway / Fly.io can build the same `Dockerfile` (expose port 7860 or set `PORT`). Free tiers with
512 MB RAM may be too small for PyTorch.

## Verified / not verified

- Verified: `space_build/` runs on its own (`cd space_build && uvicorn server.main:app`): health, predict with
  neighbours, pages and static files.
- Not verified: the Docker build itself and a deployed instance (no Docker or Hugging Face account in the dev session).
