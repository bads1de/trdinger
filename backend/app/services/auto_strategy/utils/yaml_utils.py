"""
YAML関連ユーティリティ関数

yamlファイルの読み込みと処理を専門に行うユーティリティを提供します。
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, List, Union
import logging


class YamlIndicatorUtils:
    """YAMLベースの指標特性ユーティリティ"""

    @classmethod
    def generate_characteristics_from_yaml(cls, yaml_file_path: str) -> Dict[str, Dict]:
        """
        YAMLファイルから指標特性を動的に生成

        Args:
            yaml_file_path: YAML設定ファイルのパス

        Returns:
            各指標の特性定義を含む辞書
        """
        characteristics = {}

        try:
            with open(yaml_file_path, "r", encoding="utf-8") as file:
                config = yaml.safe_load(file)

            if "indicators" not in config:
                print(f"警告: {yaml_file_path}に'indicators'セクションが見つかりません")
                return characteristics

            config["indicators"]

            for indicator_name, indicator_config in config["indicators"].items():
                if not isinstance(indicator_config, dict):
                    continue

                # 基本構造
                char = {
                    "type": indicator_config.get("type", "unknown"),
                    "scale_type": indicator_config.get("scale_type", "price_absolute"),
                }

                # thresholdsの処理
                thresholds = indicator_config.get("thresholds", {})
                if thresholds:
                    if isinstance(thresholds, dict):
                        # riskレベルの処理
                        for risk_level, risk_config in thresholds.items():
                            if risk_level in ["aggressive", "normal", "conservative"]:
                                char[f"{risk_level}_config"] = risk_config
                            elif risk_level == "all":
                                char.update(cls._process_thresholds(risk_config))
                            else:
                                # その他の特殊なしきい値設定
                                char.update(
                                    cls._process_thresholds({risk_level: risk_config})
                                )

                    # 既存の互換性維持
                    char.update(
                        cls._extract_oscillator_settings(
                            char, indicator_config, thresholds
                        )
                    )

                # 特性辞書に追加
                characteristics[indicator_name] = char
        except FileNotFoundError:
            print(f"エラー: YAMLファイルが見つかりません: {yaml_file_path}")
        except yaml.YAMLError as e:
            print(f"エラー: YAMLファイルの解析に失敗しました: {e}")
        except Exception as e:
            print(f"エラー: 特性生成中に予期しないエラーが発生しました: {e}")

        return characteristics

    @classmethod
    def _process_thresholds(cls, thresholds: Dict[str, Any]) -> Dict[str, Any]:
        """しきい値の設定を処理"""
        processed = {}

        for key, value in thresholds.items():
            if key.endswith("_lt"):
                processed[f'{key.rstrip("_lt")}_oversold'] = value
            elif key.endswith("_gt"):
                processed[f'{key.rstrip("_gt")}_overbought'] = value
            elif key in ["long_gt", "short_lt"]:
                processed[key.replace("_", "_signal_")] = value
            else:
                processed[key] = value

        return processed

    @classmethod
    def _extract_oscillator_settings(
        cls, char: Dict, indicator_config: Dict, thresholds: Dict
    ) -> Dict[str, Any]:
        """オシレーター設定を抽出して互換性のための形式に変換"""
        settings = {}

        # スケールタイプに基づいてデフォルトの設定を適用
        scale_type = indicator_config.get("scale_type", "price_absolute")

        if scale_type == "oscillator_0_100":
            settings.update(
                {
                    "range": (0, 100),
                    "oversold_threshold": 30,
                    "overbought_threshold": 70,
                    "neutral_zone": (40, 60),
                }
            )
        elif scale_type == "oscillator_plus_minus_100":
            settings.update({"range": (-100, 100), "neutral_zone": (-20, 20)})
        elif scale_type == "momentum_zero_centered":
            settings.update({"range": None, "zero_cross": True})

        # 条件に基づいてゾーン設定を上書き
        conditions = indicator_config.get("conditions", {})
        if conditions:
            cls._apply_condition_based_settings(settings, conditions, thresholds)

        return settings

    @classmethod
    def _apply_condition_based_settings(
        cls, settings: Dict, conditions: Dict, thresholds: Dict
    ) -> Dict[str, Any]:
        """条件ベースの設定を適用"""
        if "long_lt" in str(conditions.get("long", "")) and "short_gt" in str(
            conditions.get("short", "")
        ):
            # オーバーソールド/オーバーバウト設定
            settings["oversold_based"] = True
            settings["overbought_based"] = True
        elif "long_gt" in str(conditions.get("long", "")) and "short_lt" in str(
            conditions.get("short", "")
        ):
            # ゼロクロス設定
            settings["zero_cross"] = True

        return settings

    @classmethod
    def _merge_characteristics(cls, existing: Dict, yaml_based: Dict) -> Dict:
        """既存の特性とYAMLベースの特性をマージ"""
        merged = existing.copy()

        for indicator_name, yaml_config in yaml_based.items():
            if indicator_name in merged:
                # 既存のエントリをYAMLの設定で更新
                merged[indicator_name].update(yaml_config)
            else:
                # 新しいエントリを追加
                merged[indicator_name] = yaml_config

        return merged

    @classmethod
    def initialize_yaml_based_characteristics(
        cls, existing_characteristics: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        YAML設定に基づいて特性を生成してマージ

        Args:
            existing_characteristics: 既存のINDICATOR_CHARACTERISTICS

        Returns:
            マージされた特性データ
        """
        CONFIG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        YAML_CONFIG_PATH = os.path.join(
            CONFIG_DIR, "config", "technical_indicators_config.yaml"
        )

        if os.path.exists(YAML_CONFIG_PATH):
            YAML_BASED_CHARACTERISTICS = cls.generate_characteristics_from_yaml(
                YAML_CONFIG_PATH
            )
            # 既存の特性とマージして動的に更新
            return cls._merge_characteristics(
                existing_characteristics, YAML_BASED_CHARACTERISTICS
            )
        else:
            print(f"警告: YAML設定ファイルが見つかりません: {YAML_CONFIG_PATH}")
            return existing_characteristics

    @classmethod
    def load_yaml_config_for_indicators(cls) -> Dict[str, Any]:
        """技術指標のYAML設定を読み込み（ConditionGenerator用）"""
        config_path = (
            Path(__file__).parent.parent / "config" / "technical_indicators_config.yaml"
        )
        # YamlLoadUtilsを使用して読み込み
        config = YamlLoadUtils.load_yaml_config(config_path)
        return config

    @classmethod
    def get_indicator_config_from_yaml(
        cls, yaml_config: Dict[str, Any], indicator_name: str
    ) -> Optional[Dict[str, Any]]:
        """YAMLから指標設定を取得"""
        logger = logging.getLogger(__name__)
        logger.debug(f"Looking for indicator config: {indicator_name}")
        indicators_config = yaml_config.get("indicators", {})
        logger.debug(f"Indicators config keys: {list(indicators_config.keys())}")
        config = indicators_config.get(indicator_name)
        if config is None:
            logger.warning(f"No config found for indicator: {indicator_name}")
        else:
            logger.debug(f"Found config for {indicator_name}: {config}")
        return config

    @classmethod
    def get_threshold_from_yaml(
        cls,
        yaml_config: Dict[str, Any],
        config: Dict[str, Any],
        side: str,
        context: Dict[str, Any],
    ) -> Any:
        """YAMLから閾値取得"""
        logger = logging.getLogger(__name__)

        if not config:
            logger.debug("YAML config is None")
            return None

        thresholds = config.get("thresholds", {})
        if not thresholds:
            logger.debug("thresholds not found in config")
            return None

        profile = context.get("threshold_profile", "normal")
        logger.debug(f"Using profile: {profile}, side: {side}")

        if profile in thresholds and thresholds[profile]:
            profile_config = thresholds[profile]
            logger.debug(f"Found profile_config: {profile_config}")
            if side == "long" and (
                "long_gt" in profile_config or "long_lt" in profile_config
            ):
                threshold = profile_config.get("long_gt", profile_config.get("long_lt"))
                logger.debug(f"Long threshold: {threshold}")
                return threshold
            elif side == "short" and (
                "short_lt" in profile_config or "short_gt" in profile_config
            ):
                threshold = profile_config.get(
                    "short_lt", profile_config.get("short_gt")
                )
                logger.debug(f"Short threshold: {threshold}")
                return threshold

        if "all" in thresholds and thresholds["all"]:
            all_config = thresholds["all"]
            logger.debug(f"Using 'all' config: {all_config}")
            if side == "long":
                threshold = all_config.get("pos_threshold")
                logger.debug(f"Long threshold from all: {threshold}")
                return threshold
            else:
                threshold = all_config.get("neg_threshold")
                logger.debug(f"Short threshold from all: {threshold}")
                return threshold

        logger.debug("No threshold found")
        return None

    @classmethod
    def test_yaml_conditions_with_generator(
        cls, yaml_config: Dict[str, Any], test_indicators: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """YAMLベースの条件生成テスト（ConditionGenerator用）"""
        try:
            return {
                "error": "Generator class must be passed externally to avoid circular import"
            }
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"YAMLテストエラー: {e}")
            return {"error": str(e)}


class YamlLoadUtils:
    """YAMLローディングユーティリティ"""

    @staticmethod
    def load_yaml_config(
        config_path: Union[str, Path], fallback: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """YAML設定ファイルを安全に読み込み

        Args:
            config_path: YAMLファイルのパス
            fallback: 読み込み失敗時のフォールバックデータ

        Returns:
            読み込んだ設定データ
        """
        if fallback is None:
            fallback = {"indicators": {}}

        logger = logging.getLogger(__name__)

        try:
            path = Path(config_path) if isinstance(config_path, str) else config_path

            if not path.exists():
                logger.warning(f"YAML設定ファイルが見つかりません: {path}")
                return fallback

            with open(path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            if config is None:
                logger.warning(f"YAML設定ファイルが空です: {path}")
                return fallback

            # 基本構造の検証
            if not isinstance(config, dict):
                logger.error(f"無効なYAML構造: {path}")
                return fallback

            return config

        except yaml.YAMLError as e:
            logger.error(f"YAML構文エラー: {config_path}, {e}")
            return fallback
        except Exception as e:
            logger.error(f"YAML読み込みエラー: {config_path}, {e}")
            return fallback

    @staticmethod
    def validate_yaml_config(config: Dict[str, Any]) -> tuple[bool, List[str]]:
        """YAML設定データの妥当性を検証

        Args:
            config: 検証対象の設定データ

        Returns:
            (妥当性, エラーメッセージリスト) のタプル
        """
        errors = []

        try:
            # indicators セクションの存在確認
            if "indicators" not in config:
                errors.append("indicatorsセクションが必須です")

            indicators = config.get("indicators", {})

            # 各indicatorの構造検証
            for indicator_name, indicator_config in indicators.items():
                if not isinstance(indicator_config, dict):
                    errors.append(
                        f"indicator {indicator_name}: 辞書形式である必要があります"
                    )
                    continue

                # 必須フィールドチェック
                required_fields = ["type", "scale_type", "thresholds", "conditions"]
                for field in required_fields:
                    if field not in indicator_config:
                        errors.append(
                            f"indicator {indicator_name}: {field}フィールドが必須です"
                        )

                # conditionsの検証
                conditions = indicator_config.get("conditions", {})
                if conditions is None:
                    errors.append(
                        f"indicator {indicator_name}: conditionsフィールドはNoneにできません"
                    )
                elif isinstance(conditions, dict):
                    for side in ["long", "short"]:
                        if side not in conditions:
                            continue  # オプション
                        condition_template = conditions[side]
                        if not isinstance(condition_template, str):
                            errors.append(
                                f"indicator {indicator_name}: {side}条件は文字列テンプレートである必要があります"
                            )

        except Exception as e:
            errors.append(f"設定検証エラー: {e}")

        return len(errors) == 0, errors

    @staticmethod
    def get_indicator_config(
        yaml_config: Dict[str, Any], indicator_name: str
    ) -> Optional[Dict[str, Any]]:
        """YAML設定から特定のindicator設定を取得

        Args:
            yaml_config: YAML設定データ
            indicator_name: 取得対象のindicator名

        Returns:
            indicator設定
        """
        indicators = yaml_config.get("indicators", {})
        return indicators.get(indicator_name)

    @staticmethod
    def get_all_indicator_names(yaml_config: Dict[str, Any]) -> List[str]:
        """YAML設定に含まれる全indicator名を取得

        Args:
            yaml_config: YAML設定データ

        Returns:
            indicator名リスト
        """
        indicators = yaml_config.get("indicators", {})
        return list(indicators.keys())


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
