"""
AutoStrategyConfigクラス

オートストラテジー統合設定クラスを提供します。
"""

import json
import logging
from typing import Any, Dict, List, Tuple

from dataclasses import dataclass, field


from .trading import TradingSettings
from .indicators import IndicatorSettings
from .ga import GASettings
from .tpsl import TPSLSettings
from .position_sizing import PositionSizingSettings
from ..constants import (
    ERROR_CODES,
    THRESHOLD_RANGES,
)

logger = logging.getLogger(__name__)


@dataclass
class AutoStrategyConfig:
    """オートストラテジー統合設定

    このクラスはオートストラテジーの全ての設定を一元管理します。
    """

    # 設定グループ
    trading: TradingSettings = field(default_factory=TradingSettings)
    indicators: IndicatorSettings = field(default_factory=IndicatorSettings)
    ga: GASettings = field(default_factory=GASettings)
    tpsl: TPSLSettings = field(default_factory=TPSLSettings)
    position_sizing: PositionSizingSettings = field(
        default_factory=PositionSizingSettings
    )

    # 共通設定
    enable_caching: bool = True
    cache_ttl_hours: int = 24
    enable_async_processing: bool = False
    log_level: str = "WARNING"

    # エラーハンドリング設定
    error_codes: Dict[str, str] = field(default_factory=lambda: ERROR_CODES.copy())
    threshold_ranges: Dict[str, List] = field(
        default_factory=lambda: THRESHOLD_RANGES.copy()
    )

    # 設定検証ルール
    validation_rules: Dict[str, Any] = field(
        default_factory=lambda: {
            "required_fields": [],
            "ranges": {
                "cache_ttl_hours": [1, 168],  # 1時間から1週間
            },
            "types": {
                "enable_caching": bool,
                "enable_async_processing": bool,
                "log_level": str,
            },
        }
    )

    def get_default_values(self) -> Dict[str, Any]:
        """デフォルト値を取得（サブコンポーネント付き）"""
        # 自動生成したデフォルト値にサブコンポーネントを統合
        return {
            "enable_caching": True,
            "cache_ttl_hours": 24,
            "enable_async_processing": False,
            "log_level": "WARNING",
            "error_codes": ERROR_CODES.copy(),
            "threshold_ranges": THRESHOLD_RANGES.copy(),
            "trading": TradingSettings().get_default_values(),
            "indicators": IndicatorSettings().get_default_values(),
            "ga": GASettings().get_default_values(),
            "tpsl": TPSLSettings().get_default_values(),
            "position_sizing": PositionSizingSettings().get_default_values(),
        }

    def validate(self) -> Tuple[bool, List[str]]:
        """設定の妥当性を検証"""
        errors = []

        try:
            # 必須フィールドチェック
            required_fields = self.validation_rules.get("required_fields", [])
            for field_name in required_fields:
                if not hasattr(self, field_name) or getattr(self, field_name) is None:
                    errors.append(f"必須フィールド '{field_name}' が設定されていません")

            # 範囲チェック
            range_rules = self.validation_rules.get("ranges", {})
            for field_name, (min_val, max_val) in range_rules.items():
                if hasattr(self, field_name):
                    value = getattr(self, field_name)
                    if isinstance(value, (int, float)) and not (
                        min_val <= value <= max_val
                    ):
                        errors.append(
                            f"'{field_name}' は {min_val} から {max_val} の範囲で設定してください"
                        )

            # 型チェック
            type_rules = self.validation_rules.get("types", {})
            for field_name, expected_type in type_rules.items():
                if hasattr(self, field_name):
                    value = getattr(self, field_name)
                    if value is not None and not isinstance(value, expected_type):
                        errors.append(
                            f"'{field_name}' は {expected_type.__name__} 型である必要があります"
                        )

            # カスタム検証
            custom_errors = self._custom_validation()
            errors.extend(custom_errors)

        except Exception as e:
            logger.error(f"AutoStrategyConfig検証中にエラーが発生: {e}", exc_info=True)
            errors.append(f"検証処理エラー: {e}")

        return len(errors) == 0, errors

    def _custom_validation(self) -> List[str]:
        """カスタム検証（サブクラスでオーバーライド可能）"""
        errors = []

        # cache_ttl_hoursの検証
        if isinstance(self.cache_ttl_hours, (int, float)) and self.cache_ttl_hours < 0:
            errors.append("キャッシュTTLは正の数である必要があります")

        # log_levelの検証
        valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level not in valid_log_levels:
            errors.append(f"無効なログレベル: {self.log_level}")

        return errors

    def validate_all(self) -> Tuple[bool, Dict[str, List[str]]]:
        """全ての設定グループを検証"""
        all_errors = {}
        is_valid = True

        # 各設定グループの検証
        settings_groups = {
            "trading": self.trading,
            "indicators": self.indicators,
            "ga": self.ga,
            "tpsl": self.tpsl,
            "position_sizing": self.position_sizing,
        }

        for group_name, group_config in settings_groups.items():
            valid, errors = group_config.validate()
            if not valid:
                all_errors[group_name] = errors
                is_valid = False

        # メイン設定の検証
        main_valid, main_errors = self.validate()
        if not main_valid:
            all_errors["main"] = main_errors
            is_valid = False

        return is_valid, all_errors

    def to_nested_dict(self) -> Dict[str, Any]:
        """ネストされた辞書形式に変換"""
        try:
            result = {}

            # 各設定グループを辞書化
            result["trading"] = self.trading.to_dict()
            result["indicators"] = self.indicators.to_dict()
            result["ga"] = self.ga.to_dict()
            result["tpsl"] = self.tpsl.to_dict()
            result["position_sizing"] = self.position_sizing.to_dict()

            # 共通設定
            result["enable_caching"] = self.enable_caching
            result["cache_ttl_hours"] = self.cache_ttl_hours
            result["enable_async_processing"] = self.enable_async_processing
            result["log_level"] = self.log_level
            result["error_codes"] = self.error_codes
            result["threshold_ranges"] = self.threshold_ranges

            return result
        except Exception as e:
            logger.error(f"設定の辞書変換エラー: {e}", exc_info=True)
            return {}

    @classmethod
    def from_nested_dict(cls, data: Dict[str, Any]) -> "AutoStrategyConfig":
        """ネストされた辞書から設定オブジェクトを作成"""
        try:
            # 設定グループの作成
            trading_data = data.get("trading", {})
            indicators_data = data.get("indicators", {})
            ga_data = data.get("ga", {})
            tpsl_data = data.get("tpsl", {})
            position_sizing_data = data.get("position_sizing", {})

            # 各設定グループのインスタンス化（適切な型にキャスト）
            trading = (
                TradingSettings.from_dict(trading_data)
                if trading_data
                else TradingSettings()
            )
            indicators = (
                IndicatorSettings.from_dict(indicators_data)
                if indicators_data
                else IndicatorSettings()
            )
            ga = GASettings.from_dict(ga_data) if ga_data else GASettings()
            tpsl = TPSLSettings.from_dict(tpsl_data) if tpsl_data else TPSLSettings()
            position_sizing = (
                PositionSizingSettings.from_dict(position_sizing_data)
                if position_sizing_data
                else PositionSizingSettings()
            )

            # メイン設定の作成
            instance = cls(
                trading=trading,
                indicators=indicators,
                ga=ga,
                tpsl=tpsl,
                position_sizing=position_sizing,
                enable_caching=data.get("enable_caching", True),
                cache_ttl_hours=data.get("cache_ttl_hours", 24),
                enable_async_processing=data.get("enable_async_processing", False),
                log_level=data.get("log_level", "WARNING"),
                error_codes=data.get("error_codes", ERROR_CODES.copy()),
                threshold_ranges=data.get("threshold_ranges", THRESHOLD_RANGES.copy()),
            )

            return instance
        except Exception as e:
            logger.error(f"設定オブジェクト作成エラー: {e}", exc_info=True)
            raise ValueError(f"設定の作成に失敗しました: {e}")

    def save_to_json(self, filepath: str) -> bool:
        """設定をJSONファイルに保存"""
        try:
            data = self.to_nested_dict()
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"設定のJSON保存エラー: {e}", exc_info=True)
            return False

    @classmethod
    def load_from_json(cls, filepath: str) -> "AutoStrategyConfig":
        """JSONファイルから設定を読み込み"""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls.from_nested_dict(data)
        except Exception as e:
            logger.error(f"設定のJSON読み込みエラー: {e}", exc_info=True)
            raise ValueError(f"設定ファイルの読み込みに失敗しました: {e}")


# デフォルト設定インスタンス
DEFAULT_AUTO_STRATEGY_CONFIG = AutoStrategyConfig()


def get_default_config() -> AutoStrategyConfig:
    """デフォルト設定を取得"""
    return DEFAULT_AUTO_STRATEGY_CONFIG


def create_config_from_file(filepath: str) -> AutoStrategyConfig:
    """設定ファイルを読み込んでAutoStrategyConfigを作成"""
    return AutoStrategyConfig.load_from_json(filepath)


def validate_config_file(filepath: str) -> Tuple[bool, Dict[str, List[str]]]:
    """設定ファイルの妥当性を検証"""
    try:
        config = AutoStrategyConfig.load_from_json(filepath)
        return config.validate_all()
    except Exception as e:
        return False, {"file_error": [f"設定ファイル読み込みエラー: {e}"]}
