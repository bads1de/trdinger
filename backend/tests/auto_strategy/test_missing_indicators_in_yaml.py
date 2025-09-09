import yaml
from pathlib import Path

from app.services.auto_strategy.constants import CURATED_TECHNICAL_INDICATORS


def test_all_curated_indicators_present_in_yaml():
    """technical_indicators_config.yaml に Curated 指標が全て定義されていることを検証する"""
    repo_root = Path(__file__).resolve().parents[2]
    yaml_path = (
        repo_root
        / "app"
        / "services"
        / "auto_strategy"
        / "config"
        / "technical_indicators_config.yaml"
    )
    assert yaml_path.exists(), f"YAML config not found at {yaml_path}"

    data = yaml.safe_load(yaml_path.read_text())
    defined = set((data or {}).get("indicators", {}).keys())

    missing = set(CURATED_TECHNICAL_INDICATORS) - defined
    assert (
        not missing
    ), f"Missing indicators in technical_indicators_config.yaml: {sorted(missing)}"
