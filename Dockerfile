# ABSA support-ticket router — FastAPI service.
# Build:  docker build -t absa-router .
# Run:    docker run -p 8000:8000 absa-router
# Note: the trained model (models/roberta-absa) must be present at build time
# (it is .gitignored; train it with `python scripts/train_absa.py` first, or
# mount it: docker run -v $(pwd)/models:/app/models ...).

FROM python:3.12-slim

WORKDIR /app

# CPU torch keeps the image small; swap for a CUDA base if GPU serving is needed.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY taxonomy.py .
COPY app/ ./app/
COPY models/ ./models/

EXPOSE 8000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
