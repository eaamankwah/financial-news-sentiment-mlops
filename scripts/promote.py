"""Promote the latest evaluated model if its weighted F1 clears the gate in params.yaml."""

import os

import mlflow
import yaml
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

load_dotenv()


def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)["promote"]


def get_latest_f1(client, experiment_name):
    """Return (run_id, f1_weighted) of the most recent evaluation run."""
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"Evaluation experiment '{experiment_name}' not found; run evaluate.py first")
    runs = client.search_runs(
        [experiment.experiment_id], order_by=["attributes.start_time DESC"], max_results=100
    )
    for run in runs:
        if "f1_weighted" in run.data.metrics:
            return run.info.run_id, run.data.metrics["f1_weighted"]
    raise RuntimeError("No evaluation run with f1_weighted found")


def promote(client, model_name, run_id=None):
    """Point the `production` alias at the latest version registered by the evaluation run."""
    versions = list(client.search_model_versions(f"name='{model_name}'"))
    if run_id:
        versions = [v for v in versions if v.run_id == run_id]
    if not versions:
        raise RuntimeError("No registered version matches the evaluation run")
    latest = max(versions, key=lambda v: int(v.version))
    client.set_registered_model_alias(model_name, "production", latest.version)
    print(f"Production alias set to {model_name} version {latest.version}")


def main():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI") or "http://localhost:5000")
    client = MlflowClient()
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME") or "finbert-evaluation"
    model_name = os.getenv("MODEL_NAME") or "finbert"
    run_id, f1 = get_latest_f1(client, experiment_name)
    threshold = float(load_params()["f1_threshold"])
    if f1 >= threshold:
        promote(client, model_name, run_id)
    else:
        print(f"Model NOT promoted: f1_weighted {f1:.4f} is below the threshold {threshold:.4f}")


if __name__ == "__main__":
    main()
