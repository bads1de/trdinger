from app.services.auto_strategy.config.ga import GAConfig
from app.services.auto_strategy.config.robustness_windows import (
    normalize_robustness_regime_windows,
)
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

    assert config.robustness_config.regime_windows == [
        {
            "name": "bull",
            "start_date": "2024-01-01 00:00:00",
            "end_date": "2024-03-01 00:00:00",
        }
    ]


def test_validator_rejects_invalid_regime_window_order():
    from app.services.auto_strategy.config.sub_configs import RobustnessConfig

    config = GAConfig(
        robustness_config=RobustnessConfig(
            regime_windows=[
                {
                    "name": "invalid",
                    "start_date": "2024-04-01 00:00:00",
                    "end_date": "2024-03-01 00:00:00",
                }
            ]
        )
    )

    is_valid, errors = ConfigValidator.validate(config)

    assert is_valid is False
    assert any("regime window" in error for error in errors)


def test_normalize_robustness_regime_windows_strips_and_skips_invalid_entries():
    normalized = normalize_robustness_regime_windows(
        [
            {
                "name": " bear ",
                "start_date": " 2024-07-01 00:00:00 ",
                "end_date": " 2024-08-01 00:00:00 ",
            },
            "ignored",
            {
                "name": "",
                "start_date": "2024-09-01 00:00:00",
                "end_date": "2024-10-01 00:00:00",
            },
        ]
    )

    assert [window.signature for window in normalized] == [
        ("bear", "2024-07-01 00:00:00", "2024-08-01 00:00:00")
    ]
