"""Roll the `production` alias back to the previous registered model version.

Triggered automatically by .github/workflows/rollback.yml when the Deepchecks
drift gate fails, or by hand:

    python scripts/rollback.py                 # previous version
    python scripts/rollback.py --dry-run       # show what would happen
"""

import argparse
import os

import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient


def rollback(client=None, model_name=None, dry_run=False):
    """Move `production` to the greatest registered version below the current one.

    Returns the version it moved to. Raises if there is nothing to roll back to.
    """
    client = client or MlflowClient()
    model_name = model_name or os.getenv("MODEL_NAME") or "finbert"
    current = client.get_model_version_by_alias(model_name, "production")
    earlier = sorted(
        (
            int(v.version)
            for v in client.search_model_versions(f"name='{model_name}'")
            if int(v.version) < int(current.version)
        ),
        reverse=True,
    )
    if not earlier:
        raise RuntimeError("No earlier registered model version; refusing to roll back")
    target = earlier[0]
    if dry_run:
        print(f"[dry-run] would roll back {model_name} from {current.version} to {target}")
    else:
        client.set_registered_model_alias(model_name, "production", str(target))
        print(f"Rolled back {model_name} from {current.version} to {target}")
    return target


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI") or "http://localhost:5000")
    rollback(dry_run=args.dry_run)
