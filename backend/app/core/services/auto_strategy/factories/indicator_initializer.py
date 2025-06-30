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
        self.fallback_indicators = self._setup_fallback_indicators()

    def _setup_fallback_indicators(self) -> Dict[str, str]:
        """未対応指標の代替指標マッピングを設定（オートストラテジー用）"""
        return {
            # 削除された指標を利用可能な指標で代替
            "WMA": "SMA",
            "HMA": "EMA",
            "KAMA": "EMA",
            "TEMA": "EMA",
            "DEMA": "EMA",
            "T3": "EMA",
            "MAMA": "EMA",
            "ZLEMA": "EMA",
            "MIDPOINT": "SMA",
            "MIDPRICE": "SMA",
            "TRIMA": "SMA",
            "VWMA": "SMA",
            "STOCHRSI": "RSI",
            "STOCHF": "STOCH",
            "WILLR": "CCI",
            "MOMENTUM": "RSI",
            "MOM": "RSI",
            "ROC": "RSI",
            "ROCP": "RSI",
            "ROCR": "RSI",
            "AROON": "ADX",
            "AROONOSC": "ADX",
            "MFI": "RSI",
            "CMO": "RSI",
            "TRIX": "RSI",
            "ULTOSC": "RSI",
            "BOP": "RSI",
            "APO": "MACD",
            "PPO": "MACD",
            "DX": "ADX",
            "ADXR": "ADX",
            "PLUS_DI": "ADX",
            "MINUS_DI": "ADX",
            "NATR": "ATR",
            "TRANGE": "ATR",
            "KELTNER": "BB",
            "STDDEV": "ATR",
            "DONCHIAN": "BB",
            "AD": "OBV",
            "ADOSC": "OBV",
            "VWAP": "OBV",
            "PVT": "OBV",
            "EMV": "OBV",
            "AVGPRICE": "SMA",
            "MEDPRICE": "SMA",
            "TYPPRICE": "SMA",
            "WCLPRICE": "SMA",
            "PSAR": "SMA",
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

            close_data = convert_to_series(data.Close)
            high_data = convert_to_series(data.High)
            low_data = convert_to_series(data.Low)
            volume_data = convert_to_series(data.Volume)
            # open_dataは任意
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
                # JSON形式の指標名を使用（パラメータなし）
                json_indicator_name = original_type

                # 指標値の処理（辞書形式の指標に対応）
                if isinstance(result, dict):
                    # 辞書形式の場合（STOCH、MACD等）
                    # 最初の値を使用するか、適切なキーを選択
                    if original_type == "STOCH":
                        # STOCHの場合は%Kを使用
                        indicator_values = result.get(
                            "k_percent", list(result.values())[0]
                        )
                    elif original_type == "MACD":
                        # MACDの場合はMACDラインを使用
                        indicator_values = result.get("macd", list(result.values())[0])
                    else:
                        # その他の辞書形式は最初の値を使用
                        indicator_values = list(result.values())[0]
                else:
                    # Series形式の場合
                    indicator_values = (
                        result.values if hasattr(result, "values") else result
                    )

                # JSON形式で指標を登録
                strategy_instance.indicators[json_indicator_name] = strategy_instance.I(
                    lambda vals=indicator_values: vals, name=json_indicator_name
                )

                # 後方互換性のためレガシー形式でも登録
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

    def _get_legacy_indicator_name(self, indicator_type: str, parameters: dict) -> str:
        """レガシー形式の指標名を生成（後方互換性用）"""
        try:
            # indicator_registryから一元的に名前を生成
            return indicator_registry.generate_legacy_name(indicator_type, parameters)
        except Exception as e:
            logger.warning(f"レガシー指標名生成エラー ({indicator_type}): {e}")
            # フォールバック
            return indicator_type

    def get_supported_indicators(self) -> list:
        """サポートされている指標のリストを取得"""
        return list(self.indicator_calculator.indicator_adapters.keys())

    def is_supported_indicator(self, indicator_type: str) -> bool:
        """指標がサポートされているかチェック（代替指標も含む）"""
        return (
            indicator_type in self.indicator_calculator.indicator_adapters
            or indicator_type in self.fallback_indicators
        )
