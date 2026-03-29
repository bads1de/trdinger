from app.services.auto_strategy.config.ga import GAConfig
from app.services.auto_strategy.config.validators import ConfigValidator


def test_from_dict_expands_robustness_regime_windows():
    config = GAConfig.from_dict(
        {
            "robustness_config": {
                "regime_windows": [
                    {
                        "name": "bull",
                        "start_date": "2024-01-01 00:00:00",
                        "end_date": "2024-03-01 00:00:00",
                    }
                ]
            }
        }
    )

    assert config.robustness_regime_windows == [
        {
            "name": "bull",
            "start_date": "2024-01-01 00:00:00",
            "end_date": "2024-03-01 00:00:00",
        }
    ]


def test_validator_rejects_invalid_regime_window_order():
    config = GAConfig(
        robustness_regime_windows=[
            {
                "name": "invalid",
                "start_date": "2024-04-01 00:00:00",
                "end_date": "2024-03-01 00:00:00",
            }
        ]
    )

    is_valid, errors = ConfigValidator.validate(config)

    assert is_valid is False
    assert any("regime window" in error for error in errors)
