from app.services.auto_strategy.config.helpers import (
    normalize_ml_gate_fields,
    resolve_ml_gate_settings,
)


class TestMLGateSettings:
    def test_resolve_ml_gate_settings_from_dict(self):
        settings = resolve_ml_gate_settings(
            {
                "volatility_gate_enabled": True,
                "volatility_model_path": "/tmp/model.pkl",
            }
        )

        assert settings.enabled is True
        assert settings.model_path == "/tmp/model.pkl"

    def test_resolve_ml_gate_settings_disabled(self):
        settings = resolve_ml_gate_settings(
            {
                "volatility_gate_enabled": False,
            }
        )

        assert settings.enabled is False
        assert settings.model_path is None

    def test_resolve_ml_gate_settings_from_object(self):
        class Config:
            volatility_gate_enabled = True
            volatility_model_path = "/tmp/obj-model.pkl"

        settings = resolve_ml_gate_settings(Config())
        assert settings.enabled is True
        assert settings.model_path == "/tmp/obj-model.pkl"

    def test_resolve_ml_gate_settings_with_hybrid_config(self):
        settings = resolve_ml_gate_settings(
            {
                "hybrid_config": {
                    "volatility_gate_enabled": True,
                    "volatility_model_path": "/tmp/hybrid.pkl",
                }
            }
        )

        assert settings.enabled is True
        assert settings.model_path == "/tmp/hybrid.pkl"

    def test_normalize_ml_gate_fields(self):
        result = normalize_ml_gate_fields(
            {
                "volatility_gate_enabled": True,
                "volatility_model_path": "/tmp/model.pkl",
            }
        )

        assert result["volatility_gate_enabled"] is True
        assert result["volatility_model_path"] == "/tmp/model.pkl"
