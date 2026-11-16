"""
Model loading utilities.
"""

import json
import os

from transformers import pipeline

DEFAULT_HF_MODEL = "baptle/FinBERT_market_based"
FIXED_ID2LABEL = {"0": "negative", "1": "neutral", "2": "positive"}
FIXED_LABEL2ID = {"negative": "0", "neutral": "1", "positive": "2"}


def _patch_hf_label_config(model_id: str) -> None:
    """Fix the published config.json (id2label stored as floats) in the local HF cache.

    scripts/smoke_test.py does the same once on a developer machine. Fresh
    environments (CI runners, the Docker image, ECS) never run it, so the
    HuggingFace branch below repeats the fix. It is idempotent.
    """
    try:
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(repo_id=model_id, filename="config.json")
        with open(config_path) as f:
            config = json.load(f)
        if config.get("id2label") != FIXED_ID2LABEL:
            config["id2label"] = FIXED_ID2LABEL
            config["label2id"] = FIXED_LABEL2ID
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
    except Exception as exc:  # noqa: BLE001 - pipeline() will surface a real problem
        print(f"Could not patch label config for {model_id}: {exc}")


def load_classifier():
    """
    Load the sentiment classifier.

    Controlled by the MODEL_SOURCE environment variable:
      - "mlflow" (default) — loads from the MLflow Model Registry
      - "huggingface"       — loads directly from HuggingFace Hub (used in CI/prod)
    """
    model_source = os.getenv("MODEL_SOURCE") or "mlflow"

    if model_source == "huggingface":
        hf_model_id = os.getenv("HF_MODEL_ID") or DEFAULT_HF_MODEL
        print(f"Loading model from HuggingFace: {hf_model_id}")
        _patch_hf_label_config(hf_model_id)
        return pipeline(
            "text-classification",
            model=hf_model_id,
            tokenizer=hf_model_id,
            device="cpu",
        )

    import mlflow.transformers

    # `or` (not a getenv default) because .env.example ships these as empty strings
    model_name = os.getenv("MODEL_NAME") or "finbert"
    model_alias = os.getenv("MODEL_STAGE") or "production"
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or "http://localhost:5000"
    print(f"Loading model from MLflow: models:/{model_name}@{model_alias}")
    mlflow.set_tracking_uri(tracking_uri)
    return mlflow.transformers.load_model(f"models:/{model_name}@{model_alias}")
