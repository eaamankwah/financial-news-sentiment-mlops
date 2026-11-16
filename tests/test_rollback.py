"""Unit tests for scripts/rollback.py using an in-memory fake MLflow client."""

from types import SimpleNamespace

import pytest

from scripts.rollback import rollback


class FakeClient:
    def __init__(self, versions, production):
        self.versions = versions
        self.alias = {"production": production}

    def get_model_version_by_alias(self, name, alias):
        return SimpleNamespace(version=str(self.alias[alias]))

    def search_model_versions(self, query):
        return [SimpleNamespace(version=str(v)) for v in self.versions]

    def set_registered_model_alias(self, name, alias, version):
        self.alias[alias] = version


def test_rolls_back_to_previous_version():
    client = FakeClient([1, 2, 3], production=3)
    assert rollback(client, "finbert") == 2
    assert client.alias["production"] == "2"


def test_skips_gaps_and_ignores_newer_versions():
    client = FakeClient([1, 4, 5, 7], production=5)
    assert rollback(client, "finbert") == 4


def test_refuses_when_no_earlier_version():
    client = FakeClient([1], production=1)
    with pytest.raises(RuntimeError):
        rollback(client, "finbert")
    assert client.alias["production"] == 1


def test_dry_run_changes_nothing():
    client = FakeClient([1, 2], production=2)
    assert rollback(client, "finbert", dry_run=True) == 1
    assert client.alias["production"] == 2
