# Running and deploying Food Lens

The project has two frontends on the same model (`src/`, `results/`):

| Frontend | Stack | Run | Public hosting |
|---|---|---|---|
| `app/` | Streamlit | `streamlit run app/app.py` | **Streamlit Community Cloud (free)**, main file `app/app.py` |
| `web/` + `server/` | FastAPI + plain HTML/CSS/JS | `uvicorn server.main:app --port 8000` | needs a host that runs Python containers (see below) |

Browsers allow the camera only on **HTTPS or localhost**.

## Streamlit Cloud (free, current public link)

New app -> repository `Arthurmyn/Food-regocnition`, branch `main`, main file path **`app/app.py`**.
Dependencies come from `app/requirements.txt` (CPU PyTorch, unpinned because the platform uses Python 3.14).
Advanced settings: Python 3.12 is the safest choice. Push to `main` redeploys the app.

## FastAPI + web frontend

```bash
python3 -m pip install -r requirements.txt
uvicorn server.main:app --port 8000          # http://localhost:8000
```

Hosting options:
- **Hugging Face Spaces:** Docker Spaces now require a paid plan (only Static Spaces are free), so this is
  not a free route.
- Render / Railway / Fly.io can build the `Dockerfile` (port 7860 or `PORT`); free tiers have about 512 MB RAM,
  which may be too small for PyTorch. Not tried.
- Free alternative for a demo: run locally and expose it with a tunnel (Cloudflare Tunnel or ngrok); the
  tunnel gives HTTPS so the camera works.
- A fully static version (model in the browser via ONNX) could live on GitHub Pages or a free Static Space.
  Not built.

## Verified / not verified

- Verified locally: both frontends start and work (Streamlit app via its test harness and Chrome; FastAPI app
  end to end in headless Chrome with a fake camera feed).
- Not verified: the Docker build, any hosted instance of the FastAPI version, a real phone camera, and the
  current Streamlit Cloud build after this change (its Python 3.14 environment is not reproducible locally).
