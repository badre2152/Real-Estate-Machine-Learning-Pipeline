"""
test_mlflow_registry.py
-----------------------
Tests pour MLflowRegistry — mocking complet de MLflow.
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures & Mocks
# ─────────────────────────────────────────────────────────────────────────────

def _make_version(version="1", stage="None", run_id="abc123def456", description="test"):
    """Crée un mock de ModelVersion."""
    v = MagicMock()
    v.version = version
    v.current_stage = stage
    v.run_id = run_id
    v.description = description
    v.creation_timestamp = 1700000000000
    v.tags = {}
    return v


def _make_run(metrics: dict):
    """Crée un mock de Run avec des métriques."""
    run = MagicMock()
    run.data.metrics = metrics
    return run


@pytest.fixture
def mock_mlflow():
    """Patch mlflow et MlflowClient globalement."""
    with patch("mlflow_registry.MLFLOW_AVAILABLE", True), \
         patch("mlflow_registry.mlflow") as mock_mlf, \
         patch("mlflow_registry.MlflowClient") as mock_client_cls:

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_mlf.register_model.return_value = _make_version(version="1")

        yield mock_mlf, mock_client


# ─────────────────────────────────────────────────────────────────────────────
# Tests — register()
# ─────────────────────────────────────────────────────────────────────────────

class TestRegister:

    def test_register_returns_version(self, mock_mlflow):
        mock_mlf, mock_client = mock_mlflow
        mock_mlf.register_model.return_value = _make_version(version="2")

        from mlflow_registry import MLflowRegistry
        registry = MLflowRegistry("test-model")
        version = registry.register("run-abc", artifact_path="model")

        assert version == "2"
        mock_mlf.register_model.assert_called_once()

    def test_register_uses_correct_uri(self, mock_mlflow):
        mock_mlf, _ = mock_mlflow
        mock_mlf.register_model.return_value = _make_version(version="1")

        from mlflow_registry import MLflowRegistry
        registry = MLflowRegistry("test-model")
        registry.register("run-xyz", artifact_path="regression")

        call_args = mock_mlf.register_model.call_args
        assert "runs:/run-xyz/regression" in str(call_args)

    def test_register_returns_none_on_exception(self, mock_mlflow):
        mock_mlf, _ = mock_mlflow
        from mlflow.exceptions import MlflowException
        mock_mlf.register_model.side_effect = MlflowException("connection error")

        from mlflow_registry import MLflowRegistry
        registry = MLflowRegistry("test-model")
        version = registry.register("run-abc")

        assert version is None

    def test_register_sets_tag(self, mock_mlflow):
        mock_mlf, mock_client = mock_mlflow
        mock_mlf.register_model.return_value = _make_version(version="3")

        from mlflow_registry import MLflowRegistry
        registry = MLflowRegistry("test-model")
        registry.register("run-abc")

        mock_client.set_model_version_tag.assert_called()
        tag_call = mock_client.set_model_version_tag.call_args
        assert "registered_at" in str(tag_call)


# ─────────────────────────────────────────────────────────────────────────────
# Tests — promote_to_staging / promote_to_production
# ─────────────────────────────────────────────────────────────────────────────

class TestPromotions:

    def test_promote_to_staging(self, mock_mlflow):
        _, mock_client = mock_mlflow

        from mlflow_registry import MLflowRegistry, Stage
        registry = MLflowRegistry("test-model")
        ok = registry.promote_to_staging("1")

        assert ok is True
        mock_client.transition_model_version_stage.assert_called_with(
            name="test-model",
            version="1",
            stage=Stage.STAGING,
            archive_existing_versions=True,
        )

    def test_promote_to_production(self, mock_mlflow):
        _, mock_client = mock_mlflow

        from mlflow_registry import MLflowRegistry, Stage
        registry = MLflowRegistry("test-model")
        ok = registry.promote_to_production("2")

        assert ok is True
        mock_client.transition_model_version_stage.assert_called_with(
            name="test-model",
            version="2",
            stage=Stage.PRODUCTION,
            archive_existing_versions=True,
        )

    def test_promote_to_production_sets_tag(self, mock_mlflow):
        _, mock_client = mock_mlflow

        from mlflow_registry import MLflowRegistry
        registry = MLflowRegistry("test-model")
        registry.promote_to_production("1")

        tag_calls = [str(c) for c in mock_client.set_model_version_tag.call_args_list]
        assert any("promoted_to_production_at" in c for c in tag_calls)

    def test_archive(self, mock_mlflow):
        _, mock_client = mock_mlflow

        from mlflow_registry import MLflowRegistry, Stage
        registry = MLflowRegistry("test-model")
        ok = registry.archive("1")

        assert ok is True
        call_args = mock_client.transition_model_version_stage.call_args
        assert call_args.kwargs["stage"] == Stage.ARCHIVED


# ─────────────────────────────────────────────────────────────────────────────
# Tests — is_better_than_production()
# ─────────────────────────────────────────────────────────────────────────────

class TestComparison:

    def test_better_when_no_production(self, mock_mlflow):
        _, mock_client = mock_mlflow
        # Aucune version en Production
        mock_client.get_latest_versions.return_value = []

        from mlflow_registry import MLflowRegistry
        registry = MLflowRegistry("test-model")

        mock_client.get_run.return_value = _make_run({"reg/R2": 0.85})
        result = registry.is_better_than_production("run-abc", "reg/R2")

        assert result is True

    def test_better_higher_metric(self, mock_mlflow):
        mock_mlf, mock_client = mock_mlflow
        # Production a R2=0.75, nouveau run a R2=0.85
        prod_version = _make_version(version="1", stage="Production", run_id="prod-run")
        mock_client.get_latest_versions.return_value = [prod_version]
        mock_client.get_run.side_effect = lambda run_id: _make_run(
            {"reg/R2": 0.85} if run_id == "new-run" else {"reg/R2": 0.75}
        )

        from mlflow_registry import MLflowRegistry
        registry = MLflowRegistry("test-model")
        result = registry.is_better_than_production("new-run", "reg/R2", higher_is_better=True)

        assert result is True

    def test_not_better_lower_metric(self, mock_mlflow):
        _, mock_client = mock_mlflow
        # Production a R2=0.90, nouveau run a R2=0.80
        prod_version = _make_version(version="1", stage="Production", run_id="prod-run")
        mock_client.get_latest_versions.return_value = [prod_version]
        mock_client.get_run.side_effect = lambda run_id: _make_run(
            {"reg/R2": 0.80} if run_id == "new-run" else {"reg/R2": 0.90}
        )

        from mlflow_registry import MLflowRegistry
        registry = MLflowRegistry("test-model")
        result = registry.is_better_than_production("new-run", "reg/R2", higher_is_better=True)

        assert result is False

    def test_better_lower_is_better(self, mock_mlflow):
        _, mock_client = mock_mlflow
        # Pour MAE : Production=5000, nouveau=3000 (3000 < 5000 = meilleur)
        prod_version = _make_version(version="1", stage="Production", run_id="prod-run")
        mock_client.get_latest_versions.return_value = [prod_version]
        mock_client.get_run.side_effect = lambda run_id: _make_run(
            {"reg/MAE": 3000} if run_id == "new-run" else {"reg/MAE": 5000}
        )

        from mlflow_registry import MLflowRegistry
        registry = MLflowRegistry("test-model")
        result = registry.is_better_than_production("new-run", "reg/MAE", higher_is_better=False)

        assert result is True


# ─────────────────────────────────────────────────────────────────────────────
# Tests — auto_register_and_promote()
# ─────────────────────────────────────────────────────────────────────────────

class TestAutoRegisterAndPromote:

    def test_promotes_when_better(self, mock_mlflow):
        mock_mlf, mock_client = mock_mlflow
        mock_mlf.register_model.return_value = _make_version(version="1")
        # Pas de production existante
        mock_client.get_latest_versions.return_value = []
        mock_client.get_run.return_value = _make_run({"reg/R2": 0.85})

        from mlflow_registry import auto_register_and_promote
        result = auto_register_and_promote(
            run_id="run-abc",
            model_name="test-model",
            artifact_path="model",
            primary_metric="reg/R2",
        )

        assert result["promoted"] is True
        assert result["version"] == "1"

    def test_does_not_promote_when_worse(self, mock_mlflow):
        mock_mlf, mock_client = mock_mlflow
        mock_mlf.register_model.return_value = _make_version(version="2")
        # Production existante avec meilleure métrique
        prod_version = _make_version(version="1", stage="Production", run_id="prod-run")
        mock_client.get_latest_versions.return_value = [prod_version]
        mock_client.get_run.side_effect = lambda run_id: _make_run(
            {"reg/R2": 0.70} if run_id == "run-new" else {"reg/R2": 0.90}
        )

        from mlflow_registry import auto_register_and_promote
        result = auto_register_and_promote(
            run_id="run-new",
            model_name="test-model",
            artifact_path="model",
            primary_metric="reg/R2",
        )

        assert result["promoted"] is False
        assert result["version"] == "2"

    def test_returns_skipped_when_no_mlflow(self):
        with patch("mlflow_registry.MLFLOW_AVAILABLE", False):
            from mlflow_registry import auto_register_and_promote
            result = auto_register_and_promote(
                run_id="run-abc",
                model_name="test-model",
                artifact_path="model",
                primary_metric="reg/R2",
            )
            assert result["promoted"] is False
            assert "unavailable" in result["reason"]


# ─────────────────────────────────────────────────────────────────────────────
# Tests — get_latest_versions()
# ─────────────────────────────────────────────────────────────────────────────

class TestGetVersions:

    def test_get_latest_versions_returns_list(self, mock_mlflow):
        _, mock_client = mock_mlflow
        mock_client.get_latest_versions.return_value = [
            _make_version("1", "Production"),
            _make_version("2", "Staging"),
        ]

        from mlflow_registry import MLflowRegistry
        registry = MLflowRegistry("test-model")
        versions = registry.get_latest_versions()

        assert len(versions) == 2
        assert versions[0]["stage"] == "Production"
        assert versions[1]["stage"] == "Staging"

    def test_get_production_info_returns_first(self, mock_mlflow):
        _, mock_client = mock_mlflow
        mock_client.get_latest_versions.return_value = [
            _make_version("3", "Production"),
        ]

        from mlflow_registry import MLflowRegistry
        registry = MLflowRegistry("test-model")
        info = registry.get_production_info()

        assert info is not None
        assert info["version"] == "3"
        assert info["stage"] == "Production"

    def test_get_production_info_none_when_empty(self, mock_mlflow):
        _, mock_client = mock_mlflow
        mock_client.get_latest_versions.return_value = []

        from mlflow_registry import MLflowRegistry
        registry = MLflowRegistry("test-model")
        info = registry.get_production_info()

        assert info is None

    def test_no_mlflow_returns_empty(self):
        with patch("mlflow_registry.MLFLOW_AVAILABLE", False):
            from mlflow_registry import MLflowRegistry
            registry = MLflowRegistry("test-model")
            assert registry.get_latest_versions() == []
            assert registry.get_production_info() is None
