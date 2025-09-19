"""
AutoStrategyConfigクラス

オートストラテジー統合設定クラスを提供します。
"""

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
