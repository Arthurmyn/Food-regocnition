# Food Lens: FastAPI API + static frontend. CPU-only PyTorch. Port 7860 (Hugging Face Spaces default).
FROM python:3.11-slim
WORKDIR /app
COPY server/requirements.txt server/requirements.txt
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r server/requirements.txt
COPY server server
COPY src src
COPY web web
COPY results/a2/linear_probe.pt results/a2/linear_probe.pt
COPY results/a2/index.npz results/a2/synthetic_filter.json results/a2/
RUN useradd -m app && chown -R app /app
USER app
EXPOSE 7860
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "7860"]
