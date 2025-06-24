"""
指標初期化器

指標の初期化とアダプター統合を担当するモジュール。
TALibAdapterシステムとの統合を重視した実装です。
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from ..models.strategy_gene import IndicatorGene
from .indicator_calculator import IndicatorCalculator

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
        self.fallback_indicators = self._setup_fallback_indicators()

    def _setup_fallback_indicators(self) -> Dict[str, str]:
        """未対応指標の代替指標マッピングを設定"""
        return {
            "STOCHF": "STOCH",
            "ROCP": "ROC",
            "ROCR": "ROC",
            "AROONOSC": "AROON",
            "PLUS_DI": "ADX",
            "MINUS_DI": "ADX",
            "MOMENTUM": "MOM",
        }

    def calculate_indicator_only(
        self, indicator_type: str, parameters: Dict[str, Any], data: pd.DataFrame
    ) -> tuple:
        """
        指標計算のみを行う（戦略インスタンスへの追加は行わない）
        """
        try:
            fallback_indicator_type = self._get_fallback_indicator(indicator_type)
            if not fallback_indicator_type:
                return None, None
            indicator_type = fallback_indicator_type

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

            indicator_type = self._get_fallback_indicator(indicator_type)
            if not indicator_type:
                return None

            close_data = self._convert_to_series(data.Close)
            high_data = self._convert_to_series(data.High)
            low_data = self._convert_to_series(data.Low)
            volume_data = self._convert_to_series(data.Volume)
            # open_dataは任意
            open_data = (
                self._convert_to_series(data.Open) if hasattr(data, "Open") else None
            )

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
                final_indicator_name = self._get_final_indicator_name(
                    original_type, indicator_type, indicator_name, parameters
                )

                indicator_values = (
                    result.values if hasattr(result, "values") else result
                )

                strategy_instance.indicators[final_indicator_name] = (
                    strategy_instance.I(
                        lambda: indicator_values, name=final_indicator_name
                    )
                )

                logger.debug(
                    f"指標初期化完了: {final_indicator_name} (実装: {indicator_type})"
                )
                return final_indicator_name

            return None

        except Exception as e:
            logger.error(f"指標初期化エラー ({indicator_gene.type}): {e}")
            return None

    def _get_fallback_indicator(self, indicator_type: str) -> Optional[str]:
        """代替指標を取得。なければNone"""
        supported_indicators = self.indicator_calculator.indicator_adapters.keys()
        if indicator_type not in supported_indicators:
            if indicator_type in self.fallback_indicators:
                fallback_type = self.fallback_indicators[indicator_type]
                logger.info(f"未対応指標 {indicator_type} を {fallback_type} で代替")
                return fallback_type
            else:
                logger.warning(f"未対応の指標タイプ（代替なし）: {indicator_type}")
                return None
        return indicator_type

    def _get_final_indicator_name(
        self, original_type, indicator_type, indicator_name, parameters
    ):
        """最終的な指標名を取得"""
        if original_type != indicator_type:
            if original_type == "MAMA":
                return "MAMA"
            else:
                period = parameters.get("period", 14)
                return f"{original_type}_{period}"
        return indicator_name

    def _convert_to_series(self, data) -> pd.Series:
        """backtesting.pyの_ArrayをPandas Seriesに変換"""
        try:
            if hasattr(data, "_data"):
                return pd.Series(data._data)
            elif hasattr(data, "values"):
                return pd.Series(data.values)
            elif isinstance(data, (list, np.ndarray)):
                return pd.Series(data)
            else:
                return pd.Series(data)
        except Exception as e:
            logger.error(f"データ変換エラー: {e}")
            return pd.Series([])

    def get_supported_indicators(self) -> list:
        """サポートされている指標のリストを取得"""
        return list(self.indicator_calculator.indicator_adapters.keys())

    def is_supported_indicator(self, indicator_type: str) -> bool:
        """指標がサポートされているかチェック（代替指標も含む）"""
        return (
            indicator_type in self.indicator_calculator.indicator_adapters
            or indicator_type in self.fallback_indicators
        )
