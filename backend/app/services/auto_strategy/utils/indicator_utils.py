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


def indicators_by_category(category: str) -> List[str]:
    """レジストリに登録済みのインジケーターからカテゴリ別に主名称とエイリアスを抽出"""
    indicator_registry.ensure_initialized()
    seen = set()
    results: List[str] = []
    
    for name, cfg in indicator_registry.get_all_indicators().items():
        try:
            if cfg and getattr(cfg, "category", None) == category:
                # 主名称を追加
                primary = cfg.indicator_name
                if primary not in seen:
                    seen.add(primary)
                    results.append(primary)
                # エイリアスも追加
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
    indicator_registry.ensure_initialized()
    all_types = indicator_registry.list_indicators()

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


def get_valid_indicator_types() -> List[str]:
    """有効な指標タイプを取得"""
    return get_all_indicators(include_composite=True)


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


# =============================================================================
# 設定・特性関連ユーティリティ
# =============================================================================


class IndicatorCharacteristics:
    """指標特性および設定管理ユーティリティ (indicator_registryへの委譲版)"""

    @classmethod
    def load_indicator_config(cls) -> Dict[str, Any]:
        """メタデータから全技術指標設定を提供（辞書形式）"""
        indicator_registry.ensure_initialized()
        indicators = {}
        
        for name, config in indicator_registry.get_all_indicators().items():
            indicators[name] = {
                "type": config.category or "technical",
                "scale_type": config.scale_type.value,
                "thresholds": config.thresholds,
            }
            
        return {"indicators": indicators}

    @classmethod
    def get_indicator_config(
        cls, config_dict: Dict[str, Any], indicator_name: str
    ) -> Optional[Dict[str, Any]]:
        """指標設定を取得"""
        # config_dictが渡されている場合はそこから（後方互換性）
        if config_dict and "indicators" in config_dict:
            return config_dict["indicators"].get(indicator_name)
            
        # そうでなければレジストリから直接取得
        indicator_registry.ensure_initialized()
        config = indicator_registry.get_indicator_config(indicator_name)
        if config:
            return {
                "type": config.category or "technical",
                "scale_type": config.scale_type.value,
                "thresholds": config.thresholds,
            }
        return None

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

        profile = context.get("threshold_profile", "normal")

        if profile in thresholds and thresholds[profile]:
            profile_config = thresholds[profile]
            if side == "long":
                return profile_config.get("long_gt") or profile_config.get("long_lt")
            elif side == "short":
                return profile_config.get("short_lt") or profile_config.get("short_gt")

        return None

    @classmethod
    def get_characteristics(cls) -> Dict[str, Dict]:
        """指標特性を取得（後方互換用）"""
        config_data = cls.load_indicator_config()
        characteristics = {}
        for name, cfg in config_data.get("indicators", {}).items():
            characteristics[name] = {
                "type": cfg["type"],
                "scale_type": cfg["scale_type"]
            }
        return characteristics


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
