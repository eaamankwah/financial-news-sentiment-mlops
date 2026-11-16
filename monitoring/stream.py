"""Simulate production traffic: stream headlines to the API and log windows to MLflow.

Run: python monitoring/stream.py
Env: API_URL, WINDOW_SIZE (default 50), SLEEP_MS (default 100),
     MLFLOW_TRACKING_URI, MLFLOW_MONITORING_EXPERIMENT
"""

import os
import time

import mlflow
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL") or "http://localhost:8000"
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE") or 50)
SLEEP_MS = int(os.getenv("SLEEP_MS") or 100)


def predict(text):
    """Call /predict; attach the A/B variant that served the request."""
    response = requests.post(f"{API_URL}/predict", json={"text": text}, timeout=30)
    response.raise_for_status()
    result = response.json()
    result["variant"] = response.headers.get("X-Model-Variant", "production")
    return result


def _mean(window, key):
    return sum(float(x[key]) for x in window) / len(window)


def log_window(window, window_idx):
    """Log aggregated metrics for one window; the window index is the MLflow step."""
    if not window:
        return
    n = len(window)
    metrics = {
        f"pct_{label}": 100 * sum(str(x["sentiment"]).lower() == label for x in window) / n
        for label in ("positive", "negative", "neutral")
    }
    metrics["avg_confidence"] = _mean(window, "confidence")
    metrics["avg_latency_ms"] = _mean(window, "latency_ms")

    # Per model version (A/B): lets production and candidate be compared in the UI
    for variant in {x.get("variant", "production") for x in window}:
        subset = [x for x in window if x.get("variant", "production") == variant]
        metrics[f"{variant}_requests"] = len(subset)
        metrics[f"{variant}_avg_confidence"] = _mean(subset, "confidence")
        metrics[f"{variant}_avg_latency_ms"] = _mean(subset, "latency_ms")

    mlflow.log_metrics(metrics, step=window_idx)


def main():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI") or "http://localhost:5000")
    mlflow.set_experiment(os.getenv("MLFLOW_MONITORING_EXPERIMENT") or "finbert-monitoring")
    texts = pd.read_csv("data/stream.csv")["text"].dropna().astype(str).tolist()

    with mlflow.start_run():
        mlflow.log_params({"window_size": WINDOW_SIZE, "api_url": API_URL, "n_headlines": len(texts)})
        window, idx, errors = [], 0, 0
        for text in texts:
            try:
                window.append(predict(text))
            except requests.RequestException as exc:
                errors += 1
                print(f"Request failed ({errors} so far): {exc}")
            if len(window) == WINDOW_SIZE:
                idx += 1
                log_window(window, idx)
                window = []
            time.sleep(SLEEP_MS / 1000)
        if window:
            log_window(window, idx + 1)
        mlflow.log_metric("failed_requests", errors)


if __name__ == "__main__":
    main()
