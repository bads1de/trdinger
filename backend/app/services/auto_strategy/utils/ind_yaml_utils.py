"""
YAMLベースの指標特性ユーティリティ
"""
import os
import yaml
from typing import Dict, Any


def generate_characteristics_from_yaml(yaml_file_path: str) -> Dict[str, Dict]:
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

        indicators_config = config["indicators"]

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
                            char.update(_process_thresholds(risk_config))
                        else:
                            # その他の特殊なしきい値設定
                            char.update(_process_thresholds({risk_level: risk_config}))

                # 既存の互換性維持
                char.update(
                    _extract_oscillator_settings(char, indicator_config, thresholds)
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


def _process_thresholds(thresholds: Dict[str, Any]) -> Dict[str, Any]:
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


def _extract_oscillator_settings(
    char: Dict, indicator_config: Dict, thresholds: Dict
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
        _apply_condition_based_settings(settings, conditions, thresholds)

    return settings


def _apply_condition_based_settings(
    settings: Dict, conditions: Dict, thresholds: Dict
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


def _merge_characteristics(existing: Dict, yaml_based: Dict) -> Dict:
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


def initialize_yaml_based_characteristics(existing_characteristics: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    YAML設定に基づいて特性を生成してマージ

    Args:
        existing_characteristics: 既存のINDICATOR_CHARACTERISTICS

    Returns:
        マージされた特性データ
    """
    CONFIG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    YAML_CONFIG_PATH = os.path.join(CONFIG_DIR, "config", "technical_indicators_config.yaml")

    if os.path.exists(YAML_CONFIG_PATH):
        YAML_BASED_CHARACTERISTICS = generate_characteristics_from_yaml(YAML_CONFIG_PATH)
        # 既存の特性とマージして動的に更新
        return _merge_characteristics(
            existing_characteristics, YAML_BASED_CHARACTERISTICS
        )
    else:
        print(f"警告: YAML設定ファイルが見つかりません: {YAML_CONFIG_PATH}")
        return existing_characteristics
from typing import Dict, List, Any, Optional
from pathlib import Path
from .common_utils import YamlLoadUtils, YamlTestUtils



def load_yaml_config_for_indicators() -> Dict[str, Any]:
    """技術指標のYAML設定を読み込み（ConditionGenerator用）"""
    config_path = (
        Path(__file__).parent.parent / "config" / "technical_indicators_config.yaml"
    )
    # YamlLoadUtilsを使用して読み込み
    config = YamlLoadUtils.load_yaml_config(config_path)
    return config


def get_indicator_config_from_yaml(
    yaml_config: Dict[str, Any], indicator_name: str
) -> Optional[Dict[str, Any]]:
    """YAMLから指標設定を取得"""
    import logging
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


def get_threshold_from_yaml(
    yaml_config: Dict[str, Any],
    config: Dict[str, Any],
    side: str,
    context: Dict[str, Any]
) -> Any:
    """YAMLから閾値取得"""
    import logging
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


def test_yaml_conditions_with_generator(
    yaml_config: Dict[str, Any],
    test_indicators: Optional[List[str]] = None
) -> Dict[str, Any]:
    """YAMLベースの条件生成テスト（ConditionGenerator用）"""
    try:
        # 遅延 import to avoid circular import
        from ..generators.condition_generator import ConditionGenerator

        return YamlTestUtils.test_yaml_based_conditions(
            yaml_config=yaml_config,
            condition_generator_class=ConditionGenerator,
            test_indicators=test_indicators,
        )
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"YAMLテストエラー: {e}")
        return {"error": str(e)}