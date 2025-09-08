"""
yaml_utilsの統合テスト
"""

import importlib.util
import os
import tempfile
import yaml
from pathlib import Path

# モジュール直接インポート
spec = importlib.util.spec_from_file_location(
    "yaml_utils",
    os.path.join(os.path.dirname(__file__), "../../../app/services/auto_strategy/utils/yaml_utils.py")
)
yaml_utils_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(yaml_utils_module)

YamlIndicatorUtils = yaml_utils_module.YamlIndicatorUtils
YamlLoadUtils = yaml_utils_module.YamlLoadUtils
YamlTestUtils = yaml_utils_module.YamlTestUtils
MockIndicatorGene = yaml_utils_module.MockIndicatorGene


def test_integration_yaml_loading_and_generation():
    """YAML読み込みから特性生成までの統合テスト"""
    # サンプルYAML作成
    sample_config = {
        "indicators": {
            "sma": {
                "type": "moving_average",
                "scale_type": "price_absolute",
                "thresholds": {
                    "normal": {
                        "long_gt": 0.01,
                        "short_lt": -0.01
                    }
                },
                "conditions": {
                    "long": "value > 70",
                    "short": "value < 30"
                }
            },
            "rsi": {
                "type": "oscillator",
                "scale_type": "oscillator_0_100",
                "thresholds": {
                    "normal": {
                        "overbought": 70,
                        "oversold": 30
                    }
                },
                "conditions": {
                    "long": "value < 30",
                    "short": "value > 70"
                }
            }
        }
    }

    # 一時ファイルにYAML書き込み
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        yaml.dump(sample_config, f)
        temp_file = f.name

    try:
        print("=== YAML読み込みテスト ===")
        # YAML読み込みテスト
        loaded_config = YamlLoadUtils.load_yaml_config(temp_file)
        print(f"読み込んだ設定: {loaded_config}")

        # 特性生成テスト
        characteristics = YamlIndicatorUtils.generate_characteristics_from_yaml(temp_file)
        print(f"生成された特性: {characteristics}")

        # 検証テスト
        is_valid, errors = YamlLoadUtils.validate_yaml_config(loaded_config)
        print(f"設定妥当性: {is_valid}, エラー: {errors}")

        # 指標取得テスト
        indicator_names = YamlLoadUtils.get_all_indicator_names(loaded_config)
        print(f"指標名リスト: {indicator_names}")

        for name in indicator_names:
            config = YamlLoadUtils.get_indicator_config(loaded_config, name)
            print(f"{name}の設定: {config}")

        # MockIndicatorGeneテスト
        print("\n=== MockIndicatorGeneテスト ===")
        mock_gene = MockIndicatorGene("sma", enabled=True, parameters={"period": 20})
        print(f"MockIndicatorGene: type={mock_gene.type}, enabled={mock_gene.enabled}, parameters={mock_gene.parameters}")

        # YAML設定取得テスト
        yaml_config_from_file = YamlIndicatorUtils.get_indicator_config_from_yaml(loaded_config, "rsi")
        print(f"YAMLから上昇した設定: {yaml_config_from_file}")

        print("\n=== 統合テスト完了 ===")
        return True

    except Exception as e:
        print(f"統合テスト失敗: {e}")
        return False
    finally:
        os.unlink(temp_file)


if __name__ == "__main__":
    success = test_integration_yaml_loading_and_generation()
    print(f"統合テスト結果: {'成功' if success else '失敗'}")