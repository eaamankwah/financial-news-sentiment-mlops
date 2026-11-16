"""API integration tests (run with MODEL_SOURCE=huggingface in CI)."""

import pytest

import app.main as main

RESULT_KEYS = {"text", "sentiment", "confidence", "latency_ms"}


@pytest.mark.parametrize(
    "text", ["Company profits increased", "Stock prices declined", "Markets traded sideways"]
)
def test_predict_returns_valid_response(client, text):
    response = client.post("/predict", json={"text": text})
    assert response.status_code == 200
    data = response.json()
    assert set(data) == RESULT_KEYS
    assert data["text"] == text
    assert data["sentiment"] in {"positive", "negative", "neutral"}
    assert 0 <= data["confidence"] <= 1
    assert data["latency_ms"] >= 0


def test_predict_batch_length_matches_input(client):
    texts = ["Profits rose", "Losses grew", "Shares were unchanged"]
    response = client.post("/predict/batch", json={"texts": texts})
    assert response.status_code == 200
    body = response.json()
    assert len(body) == len(texts)
    assert all(set(item) == RESULT_KEYS for item in body)


def test_predict_batch_empty_list_returns_422(client):
    assert client.post("/predict/batch", json={"texts": []}).status_code == 422


def test_predict_empty_text_returns_422(client):
    assert client.post("/predict", json={"text": ""}).status_code == 422
    assert client.post("/predict", json={"text": "   "}).status_code == 422


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_unloaded_returns_503(client, monkeypatch):
    monkeypatch.setattr(main, "classifiers", {})
    assert client.get("/health").status_code == 503
    assert client.post("/predict", json={"text": "Profits rise"}).status_code == 503
    assert client.post("/predict/batch", json={"texts": ["Profits rise"]}).status_code == 503


def test_metrics_exposes_prediction_series(client):
    client.post("/predict", json={"text": "Company profits increased"})
    body = client.get("/metrics").text
    assert "prediction_requests_total" in body
    assert "prediction_latency_ms_bucket" in body
    assert "prediction_errors_total" in body


def test_prediction_error_increments_error_counter(client, monkeypatch):
    def boom(texts, **kwargs):
        raise RuntimeError("model exploded")

    monkeypatch.setitem(main.classifiers, "sentiment", boom)
    with pytest.raises(RuntimeError):
        main.run_predictions(["anything"])
    assert main.PREDICTION_ERRORS._value.get() >= 1


def test_run_prediction_helper_returns_predictions_and_latency(client):
    predictions, latency_ms = main.run_prediction("Company profits increased")
    assert len(predictions) == 1 and {"label", "score"} <= set(predictions[0])
    assert latency_ms >= 0


# ----------------------------- A/B routing ---------------------------------- #
def _fake_candidate(texts, **kwargs):
    return [{"label": "NEGATIVE", "score": 0.5} for _ in texts]


def test_ab_routes_all_traffic_to_candidate_at_100_percent(client, monkeypatch):
    monkeypatch.setitem(main.classifiers, "candidate", _fake_candidate)
    monkeypatch.setenv("AB_CANDIDATE_PERCENT", "100")
    response = client.post("/predict", json={"text": "Company profits increased"})
    assert response.headers["X-Model-Variant"] == "candidate"
    assert response.json()["sentiment"] == "negative"
    assert 'variant="candidate"' in client.get("/metrics").text


def test_ab_routes_nothing_to_candidate_at_0_percent(client, monkeypatch):
    monkeypatch.setitem(main.classifiers, "candidate", _fake_candidate)
    monkeypatch.setenv("AB_CANDIDATE_PERCENT", "0")
    for _ in range(5):
        response = client.post("/predict", json={"text": "Company profits increased"})
        assert response.headers["X-Model-Variant"] == "production"


def test_ab_split_is_roughly_the_configured_percentage(client, monkeypatch):
    monkeypatch.setitem(main.classifiers, "candidate", _fake_candidate)
    monkeypatch.setenv("AB_CANDIDATE_PERCENT", "50")
    hits = sum(main.selected_variant() == "candidate" for _ in range(2000))
    assert 800 < hits < 1200
