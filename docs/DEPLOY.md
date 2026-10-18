# Deploying Food Lens

The app is one FastAPI process: `uvicorn server.main:app` serves `/api/*` and the static pages in `web/`.
Browsers allow the camera only on **HTTPS or localhost**, so any of the hosts below (all HTTPS) works.

## Local

```bash
python3 -m pip install -r requirements.txt
uvicorn server.main:app --port 8000          # http://localhost:8000
```

## Hugging Face Spaces (free CPU, 16 GB RAM) with the Dockerfile

1. Create a Space, SDK **Docker**.
2. In the Space repo, the `README.md` must start with this front matter (add it only in the Space copy):
   ```
   ---
   title: Food Lens
   sdk: docker
   app_port: 7860
   ---
   ```
3. Push this repo's files to the Space (needs `results/a1/efficientnet_best.pt`, `results/a2/index.npz`,
   `results/a2/synthetic_filter.json`, `web/`, `server/`, `src/`, `Dockerfile`). The checkpoint is ~16 MB, so a
   normal git push works.
4. Wait for the build (installs CPU PyTorch, a few minutes). The first request after idle may be slow.

## Other hosts

Render / Railway / Fly.io can build the same `Dockerfile` (expose port 7860 or set `PORT`). Free tiers with
512 MB RAM may be too small for PyTorch.

## Notes

- Not verified here: the Docker build and a deployed instance (no Docker or account available in the dev session).
- The old Streamlit Cloud deployment points at `app/app.py`, which no longer exists. Delete that app.
