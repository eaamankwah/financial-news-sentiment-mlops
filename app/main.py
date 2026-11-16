"""FastAPI sentiment service: inference, Prometheus metrics, JSON logs, A/B routing."""

import json
import logging
import os
import random
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel

from app.utils import load_classifier

load_dotenv()

# --------------------------------------------------------------------------- #
# Structured JSON logging: one JSON object per line on stdout
# --------------------------------------------------------------------------- #
_logger = logging.getLogger("finbert-api")
if not _logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(_handler)
    _logger.setLevel(logging.INFO)
    _logger.propagate = False


def log(level: str, message: str, **kwargs):
    """Emit a JSON log line with at least timestamp, level and message."""
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "message": message,
        **kwargs,
    }
    getattr(_logger, level.lower(), _logger.info)(json.dumps(payload, default=str))


# --------------------------------------------------------------------------- #
# Prometheus metrics (latency buckets are in milliseconds)
# --------------------------------------------------------------------------- #
LATENCY_BUCKETS_MS = (5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000)

PREDICTION_REQUESTS = Counter(
    "prediction_requests_total", "Total predictions by sentiment", ["sentiment"]
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_ms", "Prediction latency in milliseconds", buckets=LATENCY_BUCKETS_MS
)
PREDICTION_ERRORS = Counter("prediction_errors_total", "Total prediction errors")

# A/B testing: the same signals, split by model variant
AB_REQUESTS = Counter(
    "ab_predictions_total", "Predictions by model variant", ["variant", "sentiment"]
)
AB_LATENCY = Histogram(
    "ab_prediction_latency_ms",
    "Prediction latency in milliseconds by model variant",
    ["variant"],
    buckets=LATENCY_BUCKETS_MS,
)
AB_CONFIDENCE = Histogram(
    "ab_prediction_confidence",
    "Prediction confidence by model variant",
    ["variant"],
    buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
)

classifiers: dict = {}


class PredictRequest(BaseModel):
    text: str


class PredictBatchRequest(BaseModel):
    texts: list[str]


class PredictionResult(BaseModel):
    text: str
    sentiment: str
    confidence: float
    latency_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        classifiers["sentiment"] = load_classifier()
        log("INFO", "Model loaded")
    except Exception as exc:  # noqa: BLE001 - service must start and report 503
        log("ERROR", "Model load failed", error=str(exc))

    # Optional A/B candidate pulled from the MLflow registry
    if "sentiment" in classifiers and ab_percent() > 0:
        uri = os.getenv("AB_CANDIDATE_URI") or "models:/finbert@candidate"
        try:
            import mlflow
            import mlflow.transformers

            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI") or "http://localhost:5000")
            classifiers["candidate"] = mlflow.transformers.load_model(uri)
            log("INFO", "Candidate model loaded", uri=uri, traffic_percent=ab_percent())
        except Exception as exc:  # noqa: BLE001 - fall back to production only
            log("ERROR", "Candidate load failed; serving production only", uri=uri, error=str(exc))
    yield
    classifiers.clear()


app = FastAPI(title="Financial Sentiment API", lifespan=lifespan)


# --------------------------------------------------------------------------- #
# Inference helpers
# --------------------------------------------------------------------------- #
def run_prediction(text, variant: str = "production"):
    """Run the model on a text (or list of texts).

    Returns ``(predictions, latency_ms)``: the raw pipeline output (one
    ``{"label", "score"}`` dict per text) and the measured wall-clock latency.
    """
    texts = [text] if isinstance(text, str) else list(text)
    classifier = classifiers["candidate" if variant == "candidate" else "sentiment"]
    start = time.perf_counter()
    predictions = classifier(texts, truncation=True, max_length=512)
    latency_ms = (time.perf_counter() - start) * 1000
    return predictions, latency_ms


def run_predictions(texts: list[str], variant: str = "production") -> list[PredictionResult]:
    """Predict for many texts and record metrics and logs."""
    try:
        if any(len(t) > 2000 for t in texts):
            log("WARNING", "Headline exceeds 2000 characters; it will be truncated")
        raw, latency_ms = run_prediction(texts, variant)
        PREDICTION_LATENCY.observe(latency_ms)
        AB_LATENCY.labels(variant=variant).observe(latency_ms)
        results = []
        for text, item in zip(texts, raw):
            result = PredictionResult(
                text=text,
                sentiment=str(item["label"]).lower(),
                confidence=float(item["score"]),
                latency_ms=latency_ms,
            )
            PREDICTION_REQUESTS.labels(sentiment=result.sentiment).inc()
            AB_REQUESTS.labels(variant=variant, sentiment=result.sentiment).inc()
            AB_CONFIDENCE.labels(variant=variant).observe(result.confidence)
            log(
                "INFO",
                "Prediction",
                variant=variant,
                sentiment=result.sentiment,
                confidence=result.confidence,
                latency_ms=latency_ms,
            )
            results.append(result)
        return results
    except Exception as exc:
        PREDICTION_ERRORS.inc()
        log("ERROR", "Prediction failed", variant=variant, error=str(exc))
        raise


def ab_percent() -> float:
    """Share of traffic (0-100) routed to the candidate; re-read on every request."""
    try:
        return max(0.0, min(100.0, float(os.getenv("AB_CANDIDATE_PERCENT") or 0)))
    except ValueError:
        return 0.0


def selected_variant() -> str:
    if "candidate" in classifiers and random.random() * 100 < ab_percent():
        return "candidate"
    return "production"


def require_model():
    if "sentiment" not in classifiers:
        raise HTTPException(status_code=503, detail="Model unavailable")


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health():
    require_model()
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResult)
def predict(request: PredictRequest, response: Response):
    require_model()
    if not request.text.strip():
        raise HTTPException(status_code=422, detail="Text cannot be empty")
    variant = selected_variant()
    response.headers["X-Model-Variant"] = variant
    return run_predictions([request.text], variant)[0]


@app.post("/predict/batch", response_model=list[PredictionResult])
def predict_batch(request: PredictBatchRequest, response: Response):
    require_model()
    if not request.texts or any(not t.strip() for t in request.texts):
        raise HTTPException(status_code=422, detail="Texts must be a non-empty list of non-empty strings")
    variant = selected_variant()
    response.headers["X-Model-Variant"] = variant
    return run_predictions(request.texts, variant)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("API_HOST") or "0.0.0.0",
        port=int(os.getenv("API_PORT") or 8000),
    )
