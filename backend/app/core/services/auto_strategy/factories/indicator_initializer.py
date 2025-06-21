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

try:
    from app.core.services.indicators.adapters.trend_adapter import TrendAdapter
    from app.core.services.indicators.adapters.momentum_adapter import MomentumAdapter
    from app.core.services.indicators.adapters.volatility_adapter import (
        VolatilityAdapter,
    )
    from app.core.services.indicators.adapters.volume_adapter import VolumeAdapter
    from app.core.services.indicators.adapters.price_transform_adapter import (
        PriceTransformAdapter,
    )
except ImportError:
    # テスト環境での代替インポート
    TrendAdapter = None
    MomentumAdapter = None
    VolatilityAdapter = None
    VolumeAdapter = None
    PriceTransformAdapter = None

logger = logging.getLogger(__name__)


class IndicatorInitializer:
    """
    指標初期化器

    指標の初期化とアダプター統合を担当します。
    """

    def __init__(self):
        """初期化"""
        self.indicator_cache = {}

        # 指標タイプとアダプターのマッピング
        self.indicator_adapters = self._setup_indicator_adapters()

    def _setup_indicator_adapters(self) -> Dict[str, Any]:
        """指標アダプターのマッピングを設定"""
        adapters = {}

        if TrendAdapter:
            adapters.update(
                {
                    "SMA": TrendAdapter.sma,
                    "EMA": TrendAdapter.ema,
                    "TEMA": TrendAdapter.tema,
                    "DEMA": TrendAdapter.dema,
                    "T3": TrendAdapter.t3,
                    "WMA": TrendAdapter.wma,
                    "HMA": TrendAdapter.hma,
                    "KAMA": TrendAdapter.kama,
                    "ZLEMA": TrendAdapter.zlema,
                    "VWMA": TrendAdapter.vwma,
                    "MIDPOINT": TrendAdapter.midpoint,
                    "MIDPRICE": TrendAdapter.midprice,
                    "TRIMA": TrendAdapter.trima,
                }
            )

        if MomentumAdapter:
            adapters.update(
                {
                    "RSI": MomentumAdapter.rsi,
                    "STOCH": MomentumAdapter.stochastic,
                    "STOCHRSI": MomentumAdapter.stochastic_rsi,
                    "CCI": MomentumAdapter.cci,
                    "WILLR": MomentumAdapter.williams_r,
                    "WILLIAMS": MomentumAdapter.williams_r,
                    "ADX": MomentumAdapter.adx,
                    "AROON": MomentumAdapter.aroon,
                    "MFI": MomentumAdapter.mfi,
                    "MOM": MomentumAdapter.momentum,
                    "ROC": MomentumAdapter.roc,
                    "ULTOSC": MomentumAdapter.ultimate_oscillator,
                    "CMO": MomentumAdapter.cmo,
                    "TRIX": MomentumAdapter.trix,
                    "BOP": MomentumAdapter.bop,
                    "APO": MomentumAdapter.apo,
                    "PPO": MomentumAdapter.ppo,
                    "DX": MomentumAdapter.dx,
                    "ADXR": MomentumAdapter.adxr,
                    "MACD": MomentumAdapter.macd,
                }
            )

        if VolatilityAdapter:
            adapters.update(
                {
                    "ATR": VolatilityAdapter.atr,
                    "NATR": VolatilityAdapter.natr,
                    "TRANGE": VolatilityAdapter.trange,
                    "STDDEV": VolatilityAdapter.stddev,
                    "BB": VolatilityAdapter.bollinger_bands,
                    "KELTNER": VolatilityAdapter.keltner_channels,
                    "DONCHIAN": VolatilityAdapter.donchian_channels,
                }
            )

        if VolumeAdapter:
            adapters.update(
                {
                    "OBV": VolumeAdapter.obv,
                    "AD": VolumeAdapter.ad,
                    "ADOSC": VolumeAdapter.adosc,
                    "VWAP": VolumeAdapter.vwap,
                    "PVT": VolumeAdapter.pvt,
                    "EMV": VolumeAdapter.emv,
                }
            )

        if PriceTransformAdapter:
            adapters.update(
                {
                    "AVGPRICE": PriceTransformAdapter.avgprice,
                    "MEDPRICE": PriceTransformAdapter.medprice,
                    "TYPPRICE": PriceTransformAdapter.typprice,
                    "WCLPRICE": PriceTransformAdapter.wclprice,
                }
            )

        return adapters

    def initialize_indicator(
        self, indicator_gene: IndicatorGene, data, strategy_instance
    ) -> Optional[str]:
        """
        単一指標の初期化

        Args:
            indicator_gene: 指標遺伝子
            data: 価格データ
            strategy_instance: 戦略インスタンス

        Returns:
            初期化された指標名（失敗時はNone）
        """
        try:
            indicator_type = indicator_gene.type
            parameters = indicator_gene.parameters

            if indicator_type not in self.indicator_adapters:
                logger.warning(f"未対応の指標タイプ: {indicator_type}")
                return None

            # backtesting.pyの_ArrayをPandas Seriesに変換
            close_data = self._convert_to_series(data.Close)
            high_data = self._convert_to_series(data.High)
            low_data = self._convert_to_series(data.Low)
            volume_data = self._convert_to_series(data.Volume)

            # 指標を計算（指標タイプに応じて引数を調整）
            result, indicator_name = self._calculate_indicator(
                indicator_type, parameters, close_data, high_data, low_data, volume_data
            )

            if result is not None and indicator_name is not None:
                # resultがSeriesの場合は値のみを取得
                if hasattr(result, "values"):
                    indicator_values = result.values
                else:
                    indicator_values = result

                strategy_instance.indicators[indicator_name] = strategy_instance.I(
                    lambda: indicator_values, name=indicator_name
                )

                logger.debug(f"指標初期化完了: {indicator_name}")
                return indicator_name

            return None

        except Exception as e:
            logger.error(f"指標初期化エラー ({indicator_gene.type}): {e}")
            return None

    def _calculate_indicator(
        self,
        indicator_type: str,
        parameters: Dict[str, Any],
        close_data: pd.Series,
        high_data: pd.Series,
        low_data: pd.Series,
        volume_data: pd.Series,
    ) -> tuple:
        """
        指標を計算

        Args:
            indicator_type: 指標タイプ
            parameters: パラメータ
            close_data: 終値データ
            high_data: 高値データ
            low_data: 安値データ
            volume_data: 出来高データ

        Returns:
            (計算結果, 指標名) のタプル
        """
        try:
            # 複合指標の処理
            if indicator_type in ["MACD", "BB"]:
                result = self.indicator_adapters[indicator_type](
                    close_data, **parameters
                )
                return self._handle_complex_indicator(
                    result, indicator_type, parameters
                )

            elif indicator_type == "STOCHRSI":
                # Stochastic RSI（DataFrameを返す）
                period = int(parameters.get("period", 14))
                fastk_period = int(parameters.get("fastk_period", 3))
                fastd_period = int(parameters.get("fastd_period", 3))
                result = self.indicator_adapters[indicator_type](
                    close_data, period, fastk_period, fastd_period
                )
                return self._handle_complex_indicator(
                    result, indicator_type, parameters
                )

            elif indicator_type in ["KELTNER", "DONCHIAN"]:
                # High, Low, Closeが必要な複合指標
                period = int(parameters.get("period", 20))
                result = self.indicator_adapters[indicator_type](
                    high_data, low_data, close_data, period
                )
                return self._handle_complex_indicator(
                    result, indicator_type, parameters
                )

            # elif indicator_type == "PSAR":
            #     # PSARはHigh, Lowが必要
            #     acceleration = float(parameters.get("acceleration", 0.02))
            #     maximum = float(parameters.get("maximum", 0.2))
            #     result = self.indicator_adapters[indicator_type](
            #         high_data, low_data, acceleration, maximum
            #     )
            #     indicator_name = f"{indicator_type}_{acceleration}_{maximum}"
            #     return result, indicator_name

            elif indicator_type in ["ATR", "NATR", "TRANGE"]:
                # High, Low, Closeが必要な指標
                if indicator_type == "TRANGE":
                    result = self.indicator_adapters[indicator_type](
                        high_data, low_data, close_data
                    )
                    indicator_name = indicator_type
                else:
                    period = int(parameters.get("period", 14))
                    result = self.indicator_adapters[indicator_type](
                        high_data, low_data, close_data, period
                    )
                    indicator_name = f"{indicator_type}_{period}"
                return result, indicator_name

            elif indicator_type in ["OBV", "AD", "PVT", "EMV"]:
                # ボリューム系指標
                if indicator_type in ["OBV", "AD"]:
                    result = self.indicator_adapters[indicator_type](
                        close_data, volume_data
                    )
                    indicator_name = indicator_type
                elif indicator_type == "PVT":
                    result = self.indicator_adapters[indicator_type](
                        close_data, volume_data
                    )
                    indicator_name = indicator_type
                elif indicator_type == "EMV":
                    period = int(parameters.get("period", 14))
                    result = self.indicator_adapters[indicator_type](
                        high_data, low_data, volume_data, period
                    )
                    indicator_name = f"{indicator_type}_{period}"
                return result, indicator_name

            elif indicator_type in ["VWAP", "ADOSC"]:
                # 特殊なボリューム系指標
                if indicator_type == "VWAP":
                    result = self.indicator_adapters[indicator_type](
                        high_data, low_data, close_data, volume_data
                    )
                    indicator_name = indicator_type
                elif indicator_type == "ADOSC":
                    fast_period = int(parameters.get("fast_period", 3))
                    slow_period = int(parameters.get("slow_period", 10))
                    result = self.indicator_adapters[indicator_type](
                        high_data,
                        low_data,
                        close_data,
                        volume_data,
                        fast_period,
                        slow_period,
                    )
                    indicator_name = f"{indicator_type}_{fast_period}_{slow_period}"
                return result, indicator_name

            elif indicator_type in ["AVGPRICE", "MEDPRICE", "TYPPRICE", "WCLPRICE"]:
                # 価格変換系指標
                result = self.indicator_adapters[indicator_type](
                    high_data, low_data, close_data
                )
                indicator_name = indicator_type
                return result, indicator_name

            else:
                # 単一パラメータ指標（期間のみ）
                period = int(parameters.get("period", 14))
                result = self.indicator_adapters[indicator_type](close_data, period)
                indicator_name = f"{indicator_type}_{period}"
                return result, indicator_name

        except Exception as e:
            logger.error(f"指標計算エラー ({indicator_type}): {e}")
            return None, None

    def _handle_complex_indicator(
        self, result: Any, indicator_type: str, parameters: Dict[str, Any]
    ) -> tuple:
        """複合指標の結果を処理"""
        try:
            if indicator_type == "MACD":
                if isinstance(result, tuple) and len(result) >= 3:
                    macd_line, signal_line, histogram = result[:3]
                    period = int(parameters.get("fast_period", 12))
                    return macd_line, f"MACD_{period}"
                else:
                    period = int(parameters.get("fast_period", 12))
                    return result, f"MACD_{period}"

            elif indicator_type == "BB":
                if isinstance(result, tuple) and len(result) >= 3:
                    upper, middle, lower = result[:3]
                    period = int(parameters.get("period", 20))
                    return middle, f"BB_MIDDLE_{period}"
                else:
                    period = int(parameters.get("period", 20))
                    return result, f"BB_{period}"

            elif indicator_type in ["KELTNER", "DONCHIAN"]:
                if isinstance(result, tuple) and len(result) >= 3:
                    upper, middle, lower = result[:3]
                    period = int(parameters.get("period", 20))
                    return middle, f"{indicator_type}_MIDDLE_{period}"
                else:
                    period = int(parameters.get("period", 20))
                    return result, f"{indicator_type}_{period}"

            elif indicator_type == "STOCHRSI":
                # Stochastic RSI（DataFrameを返す）
                if isinstance(result, pd.DataFrame) and "fastk" in result.columns:
                    period = int(parameters.get("period", 14))
                    return result["fastk"], f"STOCHRSI_FASTK_{period}"
                else:
                    period = int(parameters.get("period", 14))
                    return result, f"STOCHRSI_{period}"

            return result, indicator_type

        except Exception as e:
            logger.error(f"複合指標処理エラー ({indicator_type}): {e}")
            return None, None

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
        return list(self.indicator_adapters.keys())

    def is_supported_indicator(self, indicator_type: str) -> bool:
        """指標がサポートされているかチェック"""
        return indicator_type in self.indicator_adapters
