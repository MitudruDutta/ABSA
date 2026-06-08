"""
FastAPI service for the ABSA support-ticket router.

POST /predict  {"body": "..."}  ->  per-aspect sentiment + route + priority
GET  /health

Run: uvicorn app.api:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI
from pydantic import BaseModel
from app.predict import predict, ASPECTS

app = FastAPI(title="ABSA Support-Ticket Router", version="1.0")


class Ticket(BaseModel):
    body: str


@app.get("/health")
def health():
    return {"status": "ok", "aspects": ASPECTS}


@app.post("/predict")
def route_ticket(t: Ticket):
    result = predict(t.body)
    return {"body": t.body, **result}
