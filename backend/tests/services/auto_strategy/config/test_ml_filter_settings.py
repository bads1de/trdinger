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

    def test_normalize_ml_gate_fields_returns_synced_aliases(self):
        normalized = normalize_ml_gate_fields(
            {
                "ml_filter_enabled": True,
                "ml_model_path": "/tmp/legacy-model.pkl",
            }
        )

        assert normalized == {
            "volatility_gate_enabled": True,
            "ml_filter_enabled": True,
            "volatility_model_path": "/tmp/legacy-model.pkl",
            "ml_model_path": "/tmp/legacy-model.pkl",
        }

    def test_ga_config_post_init_normalizes_legacy_ml_fields(self):
        config = GAConfig(
            ml_filter_enabled=True,
            ml_model_path="/tmp/legacy-model.pkl",
        )

        assert config.volatility_gate_enabled is True
        assert config.ml_filter_enabled is True
        assert config.volatility_model_path == "/tmp/legacy-model.pkl"
        assert config.ml_model_path == "/tmp/legacy-model.pkl"
