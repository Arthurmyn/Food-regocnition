"""Food Lens API + static frontend.   Run from the repo root:  uvicorn server.main:app --port 8000"""
import io
import json
import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError

from server import inference

ROOT = inference.ROOT
WEB = os.path.join(ROOT, "web")
MAX_BYTES = 10 * 1024 * 1024

app = FastAPI(title="Food Lens", docs_url="/api/docs", openapi_url="/api/openapi.json")


async def _read_image(file: UploadFile):
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image is larger than 10 MB")
    try:
        return ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "Not a valid image")


@app.get("/api/health")
def health():
    return {"status": "ok", "device": str(inference.DEVICE), "model": inference.MODEL_NAME,
            "classes": inference.CLASSES}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    img = await _read_image(file)
    return await run_in_threadpool(inference.predict, img)


@app.post("/api/explain")
async def explain(file: UploadFile = File(...), cls: str = Form(...)):
    if cls not in inference.CLASSES:
        raise HTTPException(400, f"cls must be one of {inference.CLASSES}")
    img = await _read_image(file)
    cam = await run_in_threadpool(inference.explain, img, inference.CLASSES.index(cls))
    return {"cls": cls, "cam": cam}


@app.get("/api/synthetic")
def synthetic():
    stats = json.load(open(os.path.join(ROOT, "results", "a2", "synthetic_filter.json")))
    out = {}
    for cls in inference.CLASSES:
        d = os.path.join(WEB, "assets", "synthetic", cls)
        files = sorted(os.listdir(d)) if os.path.isdir(d) else []
        out[cls] = {"generated": stats[cls]["generated"], "accepted": stats[cls]["accepted"],
                    "images": [f"/assets/synthetic/{cls}/{f}" for f in files]}
    return {"generated": sum(v["generated"] for v in out.values()),
            "accepted": sum(v["accepted"] for v in out.values()), "classes": out}


@app.get("/classifier")
def classifier_page():
    return FileResponse(os.path.join(WEB, "classifier.html"))


@app.get("/synthetic")
def synthetic_page():
    return FileResponse(os.path.join(WEB, "synthetic.html"))


app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
