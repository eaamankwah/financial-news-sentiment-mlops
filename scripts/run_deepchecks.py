"""Deepchecks quality gate: NLP property drift + prediction drift.

Compares the reference set (data/test.csv) with the production-like stream
(data/stream.csv):

* property drift   -> the NLP "Sentiment" property
* prediction drift -> classifier predictions on both datasets

Thresholds come from params.yaml (deepchecks.*).

Exit codes (used by CI and by the rollback workflow):
    0  both drift scores are within their thresholds
    1  drift gate FAILED (a score exceeded its threshold)
    2  the check itself could not run (infrastructure / data error)

A machine-readable summary is written to drift_result.json.

Run:  python scripts/run_deepchecks.py
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

EXIT_OK, EXIT_DRIFT, EXIT_ERROR = 0, 1, 2
REPORT_PATH = os.getenv("DRIFT_REPORT", "drift_result.json")
PROPERTY = "Sentiment"


def as_score(value):
    """Reduce a Deepchecks drift value (number or {class: number}) to one float."""
    if isinstance(value, dict):
        numbers = [
            float(v) for v in value.values() if isinstance(v, (int, float))
        ]
        if not numbers:
            raise RuntimeError(f"No numeric drift score in {value!r}")
        return max(numbers)
    if isinstance(value, (int, float)):
        return float(value)
    raise RuntimeError(f"Unexpected drift score type: {value!r}")


def read_texts(path: str) -> list[str]:
    import pandas as pd

    texts = pd.read_csv(path)["text"].dropna().astype(str).tolist()
    if not texts:
        raise RuntimeError(f"{path} contains no text rows")
    return texts


def run_gate() -> dict:
    import yaml
    from deepchecks.nlp import TextData
    from deepchecks.nlp.checks import PredictionDrift, PropertyDrift
    from dotenv import load_dotenv

    from app.utils import load_classifier

    load_dotenv()
    with open("params.yaml") as f:
        thresholds = yaml.safe_load(f)["deepchecks"]
    prop_limit = float(thresholds["property_drift_threshold"])
    pred_limit = float(thresholds["prediction_drift_threshold"])

    test_texts = read_texts("data/test.csv")
    stream_texts = read_texts("data/stream.csv")
    test_data = TextData(test_texts, task_type="text_classification")
    stream_data = TextData(stream_texts, task_type="text_classification")

    # Only the Sentiment property is needed (TextBlob based, no model download).
    # Deepchecks 0.19.1 always runs fasttext language detection as well, which is
    # why fasttext stays in requirements.txt; the English-only filter is switched
    # off so no sample is silently dropped from the comparison.
    for data in (test_data, stream_data):
        data.calculate_builtin_properties(
            include_properties=[PROPERTY],
            ignore_non_english_samples_for_english_properties=False,
        )

    prop_result = PropertyDrift(properties=[PROPERTY]).run(
        test_data, stream_data
    )
    property_score = as_score(prop_result.value[PROPERTY]["Drift score"])
    print(
        f"Property drift ({PROPERTY}): {property_score:.4f} (threshold {prop_limit})"
    )

    classifier = load_classifier()

    def predict(texts):
        out = classifier(texts, batch_size=32, truncation=True, max_length=512)
        return [str(item["label"]).lower() for item in out]

    test_pred, stream_pred = predict(test_texts), predict(stream_texts)
    classes = sorted(
        set(test_pred) | set(stream_pred)
    )  # Deepchecks needs sorted classes
    pred_result = PredictionDrift().run(
        test_data,
        stream_data,
        train_predictions=test_pred,
        test_predictions=stream_pred,
        model_classes=classes,
    )
    prediction_score = as_score(pred_result.value["Drift score"])
    print(f"Prediction drift: {prediction_score:.4f} (threshold {pred_limit})")

    return {
        "property": PROPERTY,
        "property_drift": property_score,
        "property_threshold": prop_limit,
        "prediction_drift": prediction_score,
        "prediction_threshold": pred_limit,
        "drift_failed": property_score > prop_limit
        or prediction_score > pred_limit,
    }


def main() -> int:
    try:
        report = run_gate()
    except (
        Exception
    ) as exc:  # noqa: BLE001 - any failure to run is an infrastructure error
        print(f"Deepchecks could not run: {exc!r}", file=sys.stderr)
        with open(REPORT_PATH, "w") as f:
            json.dump({"drift_failed": False, "error": str(exc)}, f, indent=2)
        return EXIT_ERROR

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    if report["drift_failed"]:
        print("Drift gate FAILED", file=sys.stderr)
        return EXIT_DRIFT
    print("Drift gates passed")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
