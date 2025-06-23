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


from app.core.services.indicators.adapters.trend_adapter import TrendAdapter
from app.core.services.indicators.adapters.momentum_adapter import MomentumAdapter
from app.core.services.indicators.adapters.volatility_adapter import (
    VolatilityAdapter,
)
from app.core.services.indicators.adapters.volume_adapter import VolumeAdapter
from app.core.services.indicators.adapters.price_transform_adapter import (
    PriceTransformAdapter,
)


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

        # 未対応指標の代替マッピング
        self.fallback_indicators = self._setup_fallback_indicators()

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
                    "MAMA": TrendAdapter.mama,  # MAMA指標を追加
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
                    "PSAR": VolatilityAdapter.psar,
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

    def _setup_fallback_indicators(self) -> Dict[str, str]:
        """未対応指標の代替指標マッピングを設定"""
        return {
            # 未実装のモメンタム系指標の代替
            "STOCHF": "STOCH",  # Stochastic Fast → Stochastic
            "ROCP": "ROC",  # Rate of Change Percentage → Rate of Change
            "ROCR": "ROC",  # Rate of Change Ratio → Rate of Change
            "AROONOSC": "AROON",  # Aroon Oscillator → Aroon
            "PLUS_DI": "ADX",  # Plus DI → ADX
            "MINUS_DI": "ADX",  # Minus DI → ADX
            "MOMENTUM": "MOM",  # MOMENTUM → MOM
            # 未実装のトレンド系指標の代替（将来の拡張用）
            # "FUTURE_INDICATOR": "EMA",
        }

    def calculate_indicator_only(
        self, indicator_type: str, parameters: Dict[str, Any], data
    ) -> tuple:
        """
        指標計算のみを行う（戦略インスタンスへの追加は行わない）

        Args:
            indicator_type: 指標タイプ
            parameters: パラメータ辞書
            data: 価格データ（DataFrame）

        Returns:
            (計算結果, 指標名) のタプル
        """
        try:
            original_type = indicator_type

            # データを一時的に保存（BOP、AVGPRICEで使用）
            self._current_data = data

            # 未対応指標の代替処理
            if indicator_type not in self.indicator_adapters:
                if indicator_type in self.fallback_indicators:
                    fallback_type = self.fallback_indicators[indicator_type]
                    logger.info(
                        f"未対応指標 {indicator_type} を {fallback_type} で代替"
                    )
                    indicator_type = fallback_type
                else:
                    logger.warning(f"未対応の指標タイプ（代替なし）: {indicator_type}")
                    return None, None

            # DataFrameからSeriesを取得
            close_data = pd.Series(data["close"].values, index=data.index)
            high_data = pd.Series(data["high"].values, index=data.index)
            low_data = pd.Series(data["low"].values, index=data.index)
            volume_data = pd.Series(data["volume"].values, index=data.index)

            # 指標を計算
            result, indicator_name = self._calculate_indicator(
                indicator_type, parameters, close_data, high_data, low_data, volume_data
            )

            return result, indicator_name

        except Exception as e:
            logger.error(f"指標計算エラー ({indicator_type}): {e}")
            return None, None

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
            original_type = indicator_type

            # 未対応指標の代替処理
            if indicator_type not in self.indicator_adapters:
                if indicator_type in self.fallback_indicators:
                    fallback_type = self.fallback_indicators[indicator_type]
                    logger.info(
                        f"未対応指標 {indicator_type} を {fallback_type} で代替"
                    )
                    indicator_type = fallback_type
                else:
                    logger.warning(f"未対応の指標タイプ（代替なし）: {indicator_type}")
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
                # 代替指標の場合は元の指標名を使用
                if original_type != indicator_type:
                    # 代替指標の場合、元の指標名を保持
                    if original_type == "MAMA":
                        final_indicator_name = "MAMA"
                    else:
                        # パラメータがある場合は元の指標名_パラメータ形式
                        period = parameters.get("period", 14)
                        final_indicator_name = f"{original_type}_{period}"
                else:
                    final_indicator_name = indicator_name

                # resultがSeriesの場合は値のみを取得
                if hasattr(result, "values"):
                    indicator_values = result.values
                else:
                    indicator_values = result

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
            if indicator_type == "MACD":
                fast_period = int(parameters.get("fast_period", 12))
                slow_period = int(parameters.get("slow_period", 26))
                signal_period = int(parameters.get("signal_period", 9))
                result = self.indicator_adapters[indicator_type](
                    close_data, fast_period, slow_period, signal_period
                )
                return self._handle_complex_indicator(
                    result, indicator_type, parameters
                )
            elif indicator_type == "BB":
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

            elif indicator_type == "KELTNER":
                # Keltner Channels: High, Low, Close, period
                period = int(parameters.get("period", 20))
                result = self.indicator_adapters[indicator_type](
                    high_data, low_data, close_data, period
                )
                return self._handle_complex_indicator(
                    result, indicator_type, parameters
                )
            elif indicator_type == "DONCHIAN":
                # Donchian Channels: High, Low, period (Closeは不要)
                period = int(parameters.get("period", 20))
                result = self.indicator_adapters[indicator_type](
                    high_data, low_data, period
                )
                return self._handle_complex_indicator(
                    result, indicator_type, parameters
                )

            elif indicator_type == "PSAR":
                # PSARはHigh, Lowが必要
                # periodパラメータからaccelerationとmaximumを生成
                period = parameters.get("period", 12)
                acceleration = 0.02 + (period - 1) * 0.001  # 0.02-0.03の範囲
                maximum = 0.15 + (period - 1) * 0.005  # 0.15-0.25の範囲
                result = self.indicator_adapters[indicator_type](
                    high_data, low_data, acceleration, maximum
                )
                indicator_name = f"PSAR_{period}"
                return result, indicator_name

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
                if indicator_type == "OBV":
                    result = self.indicator_adapters[indicator_type](
                        close_data, volume_data
                    )
                    indicator_name = indicator_type
                elif indicator_type == "AD":
                    # AD: High, Low, Close, Volume
                    result = self.indicator_adapters[indicator_type](
                        high_data, low_data, close_data, volume_data
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
                    # VWAP: High, Low, Close, Volume, period
                    period = int(parameters.get("period", 20))
                    result = self.indicator_adapters[indicator_type](
                        high_data, low_data, close_data, volume_data, period
                    )
                    indicator_name = f"{indicator_type}_{period}"
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

            elif indicator_type == "MAMA":
                # MAMA指標（特殊処理）
                fast_limit = parameters.get("fast_limit", 0.5)
                slow_limit = parameters.get("slow_limit", 0.05)
                result = self.indicator_adapters[indicator_type](
                    close_data, fast_limit, slow_limit
                )
                # MAMAは辞書を返すので、mama値を取得
                if isinstance(result, dict) and "mama" in result:
                    indicator_name = "MAMA"
                    return result["mama"], indicator_name
                else:
                    indicator_name = "MAMA"
                    return result, indicator_name

            elif indicator_type in ["AVGPRICE", "MEDPRICE", "TYPPRICE", "WCLPRICE"]:
                # 価格変換系指標
                if indicator_type == "AVGPRICE":
                    # AVGPRICE: Open, High, Low, Close
                    # calculate_indicator_onlyメソッドではdataパラメータを使用
                    if (
                        hasattr(self, "_current_data")
                        and "open" in self._current_data.columns
                    ):
                        open_data = pd.Series(
                            self._current_data["open"].values,
                            index=self._current_data.index,
                        )
                    else:
                        open_data = close_data  # フォールバック
                    result = self.indicator_adapters[indicator_type](
                        open_data, high_data, low_data, close_data
                    )
                elif indicator_type == "MEDPRICE":
                    # MEDPRICE: High, Low
                    result = self.indicator_adapters[indicator_type](
                        high_data, low_data
                    )
                else:
                    # TYPPRICE, WCLPRICE: High, Low, Close
                    result = self.indicator_adapters[indicator_type](
                        high_data, low_data, close_data
                    )
                indicator_name = indicator_type
                return result, indicator_name

            # 特殊な引数が必要な指標
            elif indicator_type == "STOCH":
                # Stochastic: High, Low, Close, k_period, d_period
                k_period = int(parameters.get("k_period", 14))
                d_period = int(parameters.get("d_period", 3))
                result = self.indicator_adapters[indicator_type](
                    high_data, low_data, close_data, k_period, d_period
                )
                indicator_name = f"STOCH_{k_period}_{d_period}"
                return result, indicator_name

            elif indicator_type in ["CCI", "WILLR"]:
                # CCI, WILLR: High, Low, Close, period
                period = int(parameters.get("period", 14))
                result = self.indicator_adapters[indicator_type](
                    high_data, low_data, close_data, period
                )
                indicator_name = f"{indicator_type}_{period}"
                return result, indicator_name

            elif indicator_type in ["ADX", "DX", "ADXR"]:
                # ADX系: High, Low, Close, period
                period = int(parameters.get("period", 14))
                result = self.indicator_adapters[indicator_type](
                    high_data, low_data, close_data, period
                )
                indicator_name = f"{indicator_type}_{period}"
                return result, indicator_name

            elif indicator_type == "MFI":
                # MFI: High, Low, Close, Volume, period
                period = int(parameters.get("period", 14))
                result = self.indicator_adapters[indicator_type](
                    high_data, low_data, close_data, volume_data, period
                )
                indicator_name = f"MFI_{period}"
                return result, indicator_name

            elif indicator_type == "ULTOSC":
                # Ultimate Oscillator: High, Low, Close, period1, period2, period3
                period1 = int(parameters.get("period1", 7))
                period2 = int(parameters.get("period2", 14))
                period3 = int(parameters.get("period3", 28))
                result = self.indicator_adapters[indicator_type](
                    high_data, low_data, close_data, period1, period2, period3
                )
                indicator_name = f"ULTOSC_{period1}_{period2}_{period3}"
                return result, indicator_name

            elif indicator_type == "BOP":
                # BOP: Open, High, Low, Close
                # calculate_indicator_onlyメソッドではdataパラメータを使用
                if hasattr(self, "_current_data") and hasattr(
                    self._current_data, "open"
                ):
                    open_data = pd.Series(
                        self._current_data["open"].values,
                        index=self._current_data.index,
                    )
                else:
                    open_data = close_data  # フォールバック
                result = self.indicator_adapters[indicator_type](
                    open_data, high_data, low_data, close_data
                )
                indicator_name = "BOP"
                return result, indicator_name

            elif indicator_type in ["APO", "PPO"]:
                # APO, PPO: Close, fast_period, slow_period
                fast_period = int(parameters.get("fast_period", 12))
                slow_period = int(parameters.get("slow_period", 26))
                result = self.indicator_adapters[indicator_type](
                    close_data, fast_period, slow_period
                )
                indicator_name = f"{indicator_type}_{fast_period}_{slow_period}"
                return result, indicator_name

            elif indicator_type == "AROON":
                # AROON: High, Low, period
                period = int(parameters.get("period", 14))
                result = self.indicator_adapters[indicator_type](
                    high_data, low_data, period
                )
                # ARROONは辞書を返すので、適切に処理
                if isinstance(result, dict):
                    # aroon_upを使用
                    return result.get("aroon_up", result), f"AROON_UP_{period}"
                else:
                    return result, f"AROON_{period}"

            elif indicator_type == "VWMA":
                # VWMA: Price, Volume, period
                period = int(parameters.get("period", 20))
                result = self.indicator_adapters[indicator_type](
                    close_data, volume_data, period
                )
                indicator_name = f"VWMA_{period}"
                return result, indicator_name

            elif indicator_type == "MIDPRICE":
                # MIDPRICE: High, Low, period
                period = int(parameters.get("period", 14))
                result = self.indicator_adapters[indicator_type](
                    high_data, low_data, period
                )
                indicator_name = f"MIDPRICE_{period}"
                return result, indicator_name

            elif indicator_type == "T3":
                # T3: Close, period, vfactor
                period = int(parameters.get("period", 20))
                vfactor = float(parameters.get("vfactor", 0.7))
                result = self.indicator_adapters[indicator_type](
                    close_data, period, vfactor
                )
                indicator_name = f"T3_{period}"
                return result, indicator_name

            else:
                # 特別な引数が必要な指標を除外
                special_indicators = {
                    "BOP",
                    "AVGPRICE",
                    "MEDPRICE",
                    "TYPPRICE",
                    "WCLPRICE",
                    "STOCH",
                    "CCI",
                    "WILLR",
                    "ADX",
                    "DX",
                    "ADXR",
                    "MFI",
                    "ULTOSC",
                    "APO",
                    "PPO",
                    "AROON",
                    "VWMA",
                    "MIDPRICE",
                    "T3",
                    "ATR",
                    "NATR",
                    "TRANGE",
                    "OBV",
                    "AD",
                    "PVT",
                    "EMV",
                    "VWAP",
                    "ADOSC",
                    "MAMA",
                    "PSAR",
                    "KELTNER",
                    "DONCHIAN",
                }

                if indicator_type in special_indicators:
                    logger.error(
                        f"指標 {indicator_type} は特別処理が必要ですが、条件分岐に含まれていません"
                    )
                    return None, None

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
        """指標がサポートされているかチェック（代替指標も含む）"""
        return (
            indicator_type in self.indicator_adapters
            or indicator_type in self.fallback_indicators
        )
