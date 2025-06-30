"""
指標初期化器

指標の初期化とアダプター統合を担当するモジュール。
TALibAdapterシステムとの統合を重視した実装です。
"""

import logging
import pandas as pd
from typing import Dict, Any, Optional

from ..models.strategy_gene import IndicatorGene
from .indicator_calculator import IndicatorCalculator
from app.core.services.indicators.config import indicator_registry
from app.core.utils.data_utils import convert_to_series

logger = logging.getLogger(__name__)


class IndicatorInitializer:
    """
    指標初期化器

    指標の初期化と戦略への統合を担当します。
    計算ロジックはIndicatorCalculatorに委譲します。
    """

    def __init__(self):
        """初期化"""
        self.indicator_calculator = IndicatorCalculator()

    def calculate_indicator_only(
        self, indicator_type: str, parameters: Dict[str, Any], data: pd.DataFrame
    ) -> tuple:
        """
        指標計算のみを行う（戦略インスタンスへの追加は行わない）
        """
        try:
            resolved_indicator_type = indicator_registry.resolve_indicator_type(
                indicator_type
            )
            if not resolved_indicator_type:
                logger.warning(f"未対応の指標タイプ（代替なし）: {indicator_type}")
                return None, None
            indicator_type = resolved_indicator_type

            close_data = pd.Series(data["close"].values, index=data.index)
            high_data = pd.Series(data["high"].values, index=data.index)
            low_data = pd.Series(data["low"].values, index=data.index)
            volume_data = pd.Series(data["volume"].values, index=data.index)
            open_data = pd.Series(data["open"].values, index=data.index)

            return self.indicator_calculator.calculate_indicator(
                indicator_type,
                parameters,
                close_data,
                high_data,
                low_data,
                volume_data,
                open_data,
            )

        except Exception as e:
            logger.error(f"指標計算エラー ({indicator_type}): {e}")
            return None, None

    def initialize_indicator(
        self, indicator_gene: IndicatorGene, data, strategy_instance
    ) -> Optional[str]:
        """
        単一指標の初期化
        """
        try:
            indicator_type = indicator_gene.type
            parameters = indicator_gene.parameters
            original_type = indicator_type

            indicator_type = indicator_registry.resolve_indicator_type(indicator_type)
            if not indicator_type:
                return None

            close_data = convert_to_series(data.Close)
            high_data = convert_to_series(data.High)
            low_data = convert_to_series(data.Low)
            volume_data = convert_to_series(data.Volume)
            open_data = convert_to_series(data.Open) if hasattr(data, "Open") else None

            result, indicator_name = self.indicator_calculator.calculate_indicator(
                indicator_type,
                parameters,
                close_data,
                high_data,
                low_data,
                volume_data,
                open_data,
            )

            if result is not None and indicator_name is not None:
                json_indicator_name = indicator_registry.generate_json_name(
                    original_type
                )

                if isinstance(result, dict):
                    indicator_config = indicator_registry.get_indicator_config(
                        original_type
                    )
                    if indicator_config and indicator_config.result_handler:
                        handler_key = indicator_config.result_handler
                        indicator_values = result.get(
                            handler_key, list(result.values())[0]
                        )
                    else:
                        indicator_values = list(result.values())[0]
                else:
                    indicator_values = (
                        result.values if hasattr(result, "values") else result
                    )

                strategy_instance.indicators[json_indicator_name] = strategy_instance.I(
                    lambda vals=indicator_values: vals, name=json_indicator_name
                )

                legacy_indicator_name = self._get_legacy_indicator_name(
                    original_type, parameters
                )
                if legacy_indicator_name != json_indicator_name:
                    strategy_instance.indicators[legacy_indicator_name] = (
                        strategy_instance.indicators[json_indicator_name]
                    )

                logger.debug(
                    f"指標初期化完了: {json_indicator_name} (実装: {indicator_type})"
                )
                return json_indicator_name

            return None

        except Exception as e:
            logger.error(f"指標初期化エラー ({indicator_gene.type}): {e}")
            return None

    def _get_legacy_indicator_name(self, indicator_type: str, parameters: dict) -> str:
        """レガシー形式の指標名を生成（後方互換性用）"""
        try:
            return indicator_registry.generate_legacy_name(indicator_type, parameters)
        except Exception as e:
            logger.warning(f"レガシー指標名生成エラー ({indicator_type}): {e}")
            return indicator_type

    def get_supported_indicators(self) -> list:
        """サポートされている指標のリストを取得"""
        return list(indicator_registry.get_supported_indicator_names())

    def is_supported_indicator(self, indicator_type: str) -> bool:
        """指標がサポートされているかチェック（代替指標も含む）"""
        return indicator_registry.is_indicator_supported(indicator_type)
