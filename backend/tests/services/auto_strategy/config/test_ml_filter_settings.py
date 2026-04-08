from app.services.auto_strategy.config.ga import GAConfig
from app.services.auto_strategy.config.ml_filter_settings import (
    normalize_ml_gate_fields,
    resolve_ml_gate_settings,
)


class TestMLFilterSettings:
    def test_resolve_ml_gate_settings_supports_legacy_aliases(self):
        settings = resolve_ml_gate_settings(
            {
                "ml_filter_enabled": True,
                "ml_model_path": "/tmp/legacy-model.pkl",
            }
        )

        assert settings.enabled is True
        assert settings.model_path == "/tmp/legacy-model.pkl"

    def test_resolve_ml_gate_settings_prefers_canonical_model_path(self):
        settings = resolve_ml_gate_settings(
            {
                "volatility_gate_enabled": True,
                "volatility_model_path": "/tmp/canonical.pkl",
                "ml_model_path": "/tmp/legacy.pkl",
            }
        )

        assert settings.enabled is True
        assert settings.model_path == "/tmp/canonical.pkl"
