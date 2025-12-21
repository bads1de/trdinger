"""
指標関連ユーティリティ関数

指標のリスト取得、カテゴリ分類、設定の読み込みなどの機能を提供します。
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from app.services.indicators import TechnicalIndicatorService
from app.services.indicators.config.indicator_config import indicator_registry


# =============================================================================
# 指標リスト取得関連
# =============================================================================


def _load_indicator_registry():
    """indicator_registry を取得"""
    from app.services.indicators.config.indicator_config import (
        indicator_registry,
    )

    return indicator_registry


def indicators_by_category(category: str) -> List[str]:
    """レジストリに登録済みのインジケーターからカテゴリ別に主名称とエイリアスを抽出"""
    registry = _load_indicator_registry()
    seen = set()
    results: List[str] = []
    for name, cfg in registry._configs.items():  # type: ignore[attr-defined]
        try:
            if cfg and getattr(cfg, "category", None) == category:
                # 主名称を追加
                primary = getattr(cfg, "indicator_name", name)
                if primary not in seen:
                    seen.add(primary)
                    results.append(primary)
                # エイリアスも追加（MDなど）
                if cfg.aliases:
                    for alias in cfg.aliases:
                        if alias not in seen:
                            seen.add(alias)
                            results.append(alias)
        except Exception:
            continue
    results.sort()
    return results


def get_all_indicators(include_composite: bool = True) -> List[str]:
    """
    全指標タイプを取得

    Args:
        include_composite: 複合指標（COMPOSITE_INDICATORS）を含めるか
    """
    registry = _load_indicator_registry()
    all_types = registry.list_indicators()

    if include_composite:
        from ..config.constants import COMPOSITE_INDICATORS

        all_types.extend(COMPOSITE_INDICATORS)

    # 重複除去して順序維持
    seen = set()
    return [x for x in all_types if not (x in seen or seen.add(x))]


def get_volume_indicators() -> List[str]:
    return indicators_by_category("volume")


def get_momentum_indicators() -> List[str]:
    return indicators_by_category("momentum")


def get_trend_indicators() -> List[str]:
    return indicators_by_category("trend")


def get_volatility_indicators() -> List[str]:
    return indicators_by_category("volatility")


def get_original_indicators() -> List[str]:
    return indicators_by_category("original")


def get_all_indicator_ids() -> Dict[str, int]:
    """
    全指標のIDマッピングを取得

    テクニカル指標のIDマッピングを提供します。
    """
    try:
        indicator_service = TechnicalIndicatorService()
        technical_indicators = list(indicator_service.get_supported_indicators().keys())

        # IDマッピングを作成（空文字列は0、その他は1から開始）
        return {"": 0, **{ind: i + 1 for i, ind in enumerate(technical_indicators)}}
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"指標ID取得エラー: {e}")
        return {"": 0}


def get_valid_indicator_types() -> List[str]:
    """有効な指標タイプを取得"""
    return get_all_indicators(include_composite=True)


# =============================================================================
# 設定・特性関連ユーティリティ
# =============================================================================


def registry_to_config(registry=None) -> Dict[str, Any]:
    """
    レジストリ設定を設定辞書形式に変換
    """
    return IndicatorCharacteristics._generate_config_from_registry()


class IndicatorCharacteristics:
    """指標特性および設定管理ユーティリティ"""

    _cached_characteristics: Optional[Dict[str, Dict]] = None

    @classmethod
    def get_characteristics(cls) -> Dict[str, Dict]:
        """キャッシュされた指標特性を取得"""
        if cls._cached_characteristics is None:
            cls._cached_characteristics = cls.initialize_config_based_characteristics(
                {}
            )
        return cls._cached_characteristics

    @classmethod
    def generate_characteristics_from_file(cls, file_path: str) -> Dict[str, Dict]:
        """
        設定ファイルから指標特性を動的に生成
        """
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                config = yaml.safe_load(file)
            return cls._parse_indicator_config(config)
        except Exception as e:
            print(f"エラー: 設定ファイル読み込み失敗 ({file_path}): {e}")
            return {}

    @classmethod
    def _parse_indicator_config(cls, config: Dict[str, Any]) -> Dict[str, Dict]:
        """設定データから指標特性を解析"""
        characteristics = {}
        indicators = config.get("indicators", {})

        for name, cfg in indicators.items():
            if not isinstance(cfg, dict):
                continue

            char = {
                "type": cfg.get("type", "unknown"),
                "scale_type": cfg.get("scale_type", "price_absolute"),
            }

            thresholds = cfg.get("thresholds", {})
            if thresholds and isinstance(thresholds, dict):
                for level, risk_cfg in thresholds.items():
                    if level in ["aggressive", "normal", "conservative"]:
                        char[f"{level}_config"] = risk_cfg
                    elif level == "all":
                        char.update(cls._process_thresholds(risk_cfg))
                    else:
                        char.update(cls._process_thresholds({level: risk_cfg}))
                char.update(cls._extract_oscillator_settings(char, cfg, thresholds))

            characteristics[name] = char
        return characteristics

    @classmethod
    def _process_thresholds(cls, thresholds: Dict[str, Any]) -> Dict[str, Any]:
        """しきい値設定の構造化処理"""
        processed = {}
        for key, value in thresholds.items():
            if key in ["long_gt", "short_lt"]:
                processed[key.replace("_", "_signal_")] = value
            elif key.endswith("_lt"):
                processed[f"{key.removesuffix('_lt')}_oversold"] = value
            elif key.endswith("_gt"):
                processed[f"{key.removesuffix('_gt')}_overbought"] = value
            else:
                processed[key] = value
        return processed

    @classmethod
    def _extract_oscillator_settings(
        cls, char: Dict, indicator_config: Dict, thresholds: Dict
    ) -> Dict[str, Any]:
        """オシレーター設定を抽出"""
        settings = {}
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
            settings["oversold_based"] = True
            settings["overbought_based"] = True
        elif "long_gt" in str(conditions.get("long", "")) and "short_lt" in str(
            conditions.get("short", "")
        ):
            settings["zero_cross"] = True
        return settings

    @classmethod
    def _merge_characteristics(cls, existing: Dict, config_based: Dict) -> Dict:
        """既存の特性と設定ベースの特性をマージ"""
        merged = existing.copy()
        for name, cfg in config_based.items():
            if name in merged:
                merged[name].update(cfg)
            else:
                merged[name] = cfg
        return merged

    @classmethod
    def _generate_config_from_registry(cls) -> Dict[str, Any]:
        """レジストリから設定辞書を生成"""
        indicators = {}

        for name, config in indicator_registry.get_all_indicators().items():
            scale_type = config.scale_type.value

            # スケールタイプに基づくデフォルト設定
            thresholds = {}
            conditions = {}

            if scale_type == "oscillator_0_100":
                thresholds = {
                    "aggressive": {"long_lt": 35, "short_gt": 65},
                    "normal": {"long_lt": 30, "short_gt": 70},
                    "conservative": {"long_lt": 25, "short_gt": 75},
                }
                conditions = {
                    "long": "{left_operand} < {threshold}",
                    "short": "{left_operand} > {threshold}",
                }
            elif scale_type == "oscillator_plus_minus_100":
                thresholds = {"normal": {"long_lt": -80, "short_gt": 80}}
                conditions = {
                    "long": "{left_operand} < {threshold}",
                    "short": "{left_operand} > {threshold}",
                }
            elif scale_type == "momentum_zero_centered":
                thresholds = {"normal": {"long_gt": 0, "short_lt": 0}}
                conditions = {
                    "long": "{left_operand} > 0",
                    "short": "{left_operand} < 0",
                }
            elif scale_type == "price_absolute":
                # 価格との比較など
                conditions = {
                    "long": "close > {left_operand}",
                    "short": "close < {left_operand}",
                }

            indicators[name] = {
                "type": config.category or "technical",
                "scale_type": scale_type,
                "thresholds": thresholds,
                "conditions": conditions,
            }

        return {"indicators": indicators}

    @classmethod
    def initialize_config_based_characteristics(
        cls, existing_characteristics: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """設定に基づいて特性を生成してマージ"""
        config = registry_to_config()
        config_based = cls._parse_indicator_config(config)
        return cls._merge_characteristics(existing_characteristics, config_based)

    @classmethod
    def load_indicator_config(cls) -> Dict[str, Any]:
        """メタデータから技術指標設定を提供"""
        return registry_to_config()

    @classmethod
    def get_indicator_config(
        cls, config: Dict[str, Any], indicator_name: str
    ) -> Optional[Dict[str, Any]]:
        """指標設定を取得"""
        indicators_config = config.get("indicators", {})
        return indicators_config.get(indicator_name)

    @classmethod
    def get_threshold_from_config(
        cls,
        config_dict: Dict[str, Any],  # unused but kept for compatibility
        indicator_config: Dict[str, Any],
        side: str,
        context: Dict[str, Any],
    ) -> Any:
        """設定から閾値取得"""
        if not indicator_config:
            return None

        thresholds = indicator_config.get("thresholds", {})
        if not thresholds:
            return None

        if isinstance(thresholds, str):
            return thresholds

        profile = context.get("threshold_profile", "normal")

        if profile in thresholds and thresholds[profile]:
            profile_config = thresholds[profile]
            if side == "long" and (
                "long_gt" in profile_config or "long_lt" in profile_config
            ):
                return profile_config.get("long_gt", profile_config.get("long_lt"))
            elif side == "short" and (
                "short_lt" in profile_config or "short_gt" in profile_config
            ):
                return profile_config.get("short_lt", profile_config.get("short_gt"))

        if "all" in thresholds and thresholds["all"]:
            all_config = thresholds["all"]
            if side == "long":
                return all_config.get("pos_threshold")
            else:
                return all_config.get("neg_threshold")

        return None


class ConfigFileUtils:
    """設定ファイルローディングユーティリティ"""

    @staticmethod
    def load_config(
        config_path: Union[str, Path], fallback: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """設定ファイルを安全に読み込み"""
        if fallback is None:
            fallback = {"indicators": {}}

        logger = logging.getLogger(__name__)

        try:
            path = Path(config_path) if isinstance(config_path, str) else config_path

            if not path.exists():
                logger.warning(f"設定ファイルが見つかりません: {path}")
                return fallback

            with open(path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            if config is None:
                logger.warning(f"設定ファイルが空です: {path}")
                return fallback

            if not isinstance(config, dict):
                logger.error(f"無効な設定構造: {path}")
                return fallback

            return config

        except yaml.YAMLError as e:
            logger.error(f"YAML構文エラー: {config_path}, {e}")
            return fallback
        except Exception as e:
            logger.error(f"設定読み込みエラー: {config_path}, {e}")
            return fallback

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> tuple[bool, List[str]]:
        """設定データの妥当性を検証"""
        errors = []

        try:
            if "indicators" not in config:
                errors.append("indicatorsセクションが必須です")

            indicators = config.get("indicators", {})

            for indicator_name, indicator_config in indicators.items():
                if not isinstance(indicator_config, dict):
                    errors.append(
                        f"indicator {indicator_name}: 辞書形式である必要があります"
                    )
                    continue

                required_fields = ["type", "scale_type", "thresholds", "conditions"]
                for field in required_fields:
                    if field not in indicator_config:
                        errors.append(
                            f"indicator {indicator_name}: {field}フィールドが必須です"
                        )

                conditions = indicator_config.get("conditions", {})
                if conditions is None:
                    errors.append(
                        f"indicator {indicator_name}: conditionsフィールドはNoneにできません"
                    )
                elif isinstance(conditions, dict):
                    for side in ["long", "short"]:
                        if side not in conditions:
                            continue
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
        config: Dict[str, Any], indicator_name: str
    ) -> Optional[Dict[str, Any]]:
        """設定から特定のindicator設定を取得"""
        indicators = config.get("indicators", {})
        return indicators.get(indicator_name)

    @staticmethod
    def get_all_indicator_names(config: Dict[str, Any]) -> List[str]:
        """設定に含まれる全indicator名を取得"""
        indicators = config.get("indicators", {})
        return list(indicators.keys())