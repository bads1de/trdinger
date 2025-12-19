"""
YAML関連ユーティリティ関数

yamlファイルの読み込みと処理を専門に行うユーティリティを提供します。
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from app.services.indicators.config.indicator_config import indicator_registry


class YamlIndicatorUtils:
    """YAMLベースの指標特性ユーティリティ"""

    _cached_characteristics: Optional[Dict[str, Dict]] = None

    @classmethod
    def get_characteristics(cls) -> Dict[str, Dict]:
        """キャッシュされた指標特性を取得"""
        if cls._cached_characteristics is None:
            cls._cached_characteristics = cls.initialize_yaml_based_characteristics({})
        return cls._cached_characteristics

    @classmethod
    def generate_characteristics_from_yaml(cls, yaml_file_path: str) -> Dict[str, Dict]:
        """
        YAMLファイルから指標特性を動的に生成
        """
        try:
            with open(yaml_file_path, "r", encoding="utf-8") as file:
                config = yaml.safe_load(file)
            return cls._parse_indicator_config(config)
        except Exception as e:
            print(f"エラー: YAML読み込み失敗 ({yaml_file_path}): {e}")
            return {}

    @classmethod
    def _parse_indicator_config(cls, config: Dict[str, Any]) -> Dict[str, Dict]:
        """YAML設定データから指標特性を解析"""
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
    def _merge_characteristics(cls, existing: Dict, yaml_based: Dict) -> Dict:
        """既存の特性とYAMLベースの特性をマージ"""
        merged = existing.copy()
        for name, cfg in yaml_based.items():
            if name in merged:
                merged[name].update(cfg)
            else:
                merged[name] = cfg
        return merged

    @classmethod
    def _generate_yaml_from_registry(cls) -> Dict[str, Any]:
        """レジストリからYAML互換の辞書を生成"""
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
                    "conservative": {"long_lt": 25, "short_gt": 75}
                }
                conditions = {
                    "long": "{left_operand} < {threshold}",
                    "short": "{left_operand} > {threshold}"
                }
            elif scale_type == "oscillator_plus_minus_100":
                thresholds = {
                    "normal": {"long_lt": -80, "short_gt": 80}
                }
                conditions = {
                    "long": "{left_operand} < {threshold}",
                    "short": "{left_operand} > {threshold}"
                }
            elif scale_type == "momentum_zero_centered":
                thresholds = {"normal": {"long_gt": 0, "short_lt": 0}}
                conditions = {
                    "long": "{left_operand} > 0",
                    "short": "{left_operand} < 0"
                }
            elif scale_type == "price_absolute":
                # 価格との比較など
                conditions = {
                    "long": "close > {left_operand}",
                    "short": "close < {left_operand}"
                }
            
            indicators[name] = {
                "type": config.category or "technical",
                "scale_type": scale_type,
                "thresholds": thresholds,
                "conditions": conditions
            }
            
        return {"indicators": indicators}

    @classmethod
    def initialize_yaml_based_characteristics(
        cls, existing_characteristics: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """YAML設定に基づいて特性を生成してマージ"""
        yaml_config = manifest_to_yaml_dict()
        yaml_based = cls._parse_indicator_config(yaml_config)
        return cls._merge_characteristics(existing_characteristics, yaml_based)

    @classmethod
    def load_yaml_config_for_indicators(cls) -> Dict[str, Any]:
        """メタデータから技術指標設定を提供"""
        return manifest_to_yaml_dict()

    @classmethod
    def get_indicator_config_from_yaml(
        cls, yaml_config: Dict[str, Any], indicator_name: str
    ) -> Optional[Dict[str, Any]]:
        """YAMLから指標設定を取得"""
        indicators_config = yaml_config.get("indicators", {})
        return indicators_config.get(indicator_name)

    @classmethod
    def get_threshold_from_yaml(
        cls,
        yaml_config: Dict[str, Any],
        config: Dict[str, Any],
        side: str,
        context: Dict[str, Any],
    ) -> Any:
        """YAMLから閾値取得"""
        if not config:
            return None

        thresholds = config.get("thresholds", {})
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
                return profile_config.get(
                    "short_lt", profile_config.get("short_gt")
                )

        if "all" in thresholds and thresholds["all"]:
            all_config = thresholds["all"]
            if side == "long":
                return all_config.get("pos_threshold")
            else:
                return all_config.get("neg_threshold")

        return None


class IndicatorConfigProvider:
    """指標設定プロバイダー（インスタンスベース）"""

    def __init__(self, config_path: Optional[str] = None):
        """初期化"""
        self._logger = logging.getLogger(__name__)
        self.config_path = Path(config_path) if config_path else None
        self.config = self._load_yaml_config()

    def _load_yaml_config(self) -> Dict[str, Any]:
        """YAML設定を読み込み"""
        if self.config_path and self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as file:
                    return yaml.safe_load(file) or {"indicators": {}}
            except (yaml.YAMLError, OSError) as exc:
                self._logger.warning("YAMLロード失敗: %s", exc)
        return manifest_to_yaml_dict()

    def get_available_indicators(self) -> List[str]:
        """利用可能な指標名のリストを取得"""
        return list(self.config.get("indicators", {}).keys())

    def get_indicator_info(self, indicator_name: str) -> Dict[str, Any]:
        """指標の設定情報を取得"""
        indicators = self.config.get("indicators", {})
        if indicator_name not in indicators:
            raise ValueError(f"Unknown indicator: {indicator_name}")
        return indicators[indicator_name].copy()


class YamlLoadUtils:
    """YAMLローディングユーティリティ"""

    @staticmethod
    def load_yaml_config(
        config_path: Union[str, Path], fallback: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """YAML設定ファイルを安全に読み込み"""
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
        """YAML設定データの妥当性を検証"""
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
        yaml_config: Dict[str, Any], indicator_name: str
    ) -> Optional[Dict[str, Any]]:
        """YAML設定から特定のindicator設定を取得"""
        indicators = yaml_config.get("indicators", {})
        return indicators.get(indicator_name)

    @staticmethod
    def get_all_indicator_names(yaml_config: Dict[str, Any]) -> List[str]:
        """YAML設定に含まれる全indicator名を取得"""
        indicators = yaml_config.get("indicators", {})
        return list(indicators.keys())


def manifest_to_yaml_dict(registry=None) -> Dict[str, Any]:
    """
    レジストリ設定をYAML互換の辞書形式に変換（後方互換性のため）
    """
    return YamlIndicatorUtils._generate_yaml_from_registry()
