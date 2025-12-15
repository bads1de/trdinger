"""
Auto Strategy テストユーティリティ

テストで使用されるヘルパークラスやモックオブジェクト定義。
"""

import logging
from typing import Any, Dict, List, Optional

from app.services.auto_strategy.utils.yaml_utils import YamlLoadUtils

logger = logging.getLogger(__name__)


class MockIndicatorGene:
    """テスト用のモックIndicatorGene"""

    def __init__(
        self,
        type: str,
        enabled: bool = True,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        self.type = type
        self.enabled = enabled
        self.parameters = parameters or {}


class YamlTestUtils:
    """YAML条件生成テストユーティリティ"""

    @staticmethod
    def test_yaml_based_conditions(
        yaml_config: Dict[str, Any],
        condition_generator_class,
        test_indicators: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """YAMLベースの条件生成をテスト

        Args:
            yaml_config: YAML設定データ
            condition_generator_class: テスト対象の条件generatorクラス
            test_indicators: テスト対象のindicators（指定なしの場合は全て）

        Returns:
            テスト結果
        """
        result = {
            "success": False,
            "tested_indicators": [],
            "errors": [],
            "summary": {},
        }

        try:
            # ConditionGeneratorの作成
            generator = condition_generator_class()
            generator.yaml_config = yaml_config  # YAML設定をセット

            # テスト対象のindicators決定
            if test_indicators:
                indicators_to_test = test_indicators
            else:
                indicators_to_test = YamlLoadUtils.get_all_indicator_names(yaml_config)

            # 各indicatorでテスト実行
            successful_tests = 0
            total_tests = 0

            for indicator_name in indicators_to_test:
                try:
                    # Mock IndicatorGene作成
                    mock_indicator = MockIndicatorGene(indicator_name, enabled=True)

                    # YAML設定取得
                    config = YamlLoadUtils.get_indicator_config(
                        yaml_config, indicator_name
                    )
                    if not config:
                        result["errors"].append(f"YAML設定なし: {indicator_name}")
                        continue

                    # 条件生成テスト
                    long_conditions = generator._generate_yaml_based_conditions(
                        mock_indicator, "long"
                    )
                    short_conditions = generator._generate_yaml_based_conditions(
                        mock_indicator, "short"
                    )

                    result["tested_indicators"].append(
                        {
                            "name": indicator_name,
                            "long_conditions_count": len(long_conditions),
                            "short_conditions_count": len(short_conditions),
                            "type": config.get("type", "unknown"),
                        }
                    )

                    successful_tests += 1
                    total_tests += 1

                except Exception as e:
                    result["errors"].append(f"{indicator_name} テスト失敗: {e}")
                    total_tests += 1

            result["success"] = successful_tests == total_tests and total_tests > 0
            result["summary"] = {
                "total_tested": total_tests,
                "successful": successful_tests,
                "success_rate": (
                    successful_tests / total_tests if total_tests > 0 else 0.0
                ),
            }

        except Exception as e:
            result["errors"].append(f"全体テスト失敗: {e}")

        return result




