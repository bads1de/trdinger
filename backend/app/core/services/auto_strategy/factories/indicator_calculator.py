"""
指標計算機（オートストラテジー最適化版）

numpy配列ベースの新しいTa-lib指標クラスを使用し、
backtesting.pyとの完全な互換性を提供します。
pandas Seriesの変換は一切行いません。
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, Union

# 新しいnumpy配列ベース指標クラス
from app.core.services.indicators.trend import TrendIndicators
from app.core.services.indicators.momentum import MomentumIndicators
from app.core.services.indicators.volatility import VolatilityIndicators
from app.core.services.indicators.utils import TALibError, ensure_numpy_array

logger = logging.getLogger(__name__)


class IndicatorCalculator:
    """
    指標計算を担当するクラス（オートストラテジー最適化版）

    責務:
    - numpy配列ベースの高速指標計算
    - backtesting.pyとの完全な互換性
    - pandas Seriesの変換を一切行わない最適化

    特徴:
    - Ta-lib直接呼び出しによる最大パフォーマンス
    - numpy配列ネイティブ処理
    - メモリ効率の最大化
    """

    def __init__(self):
        """初期化"""
        self.indicator_map = {
            # トレンド系指標
            "SMA": TrendIndicators.sma,
            "EMA": TrendIndicators.ema,
            "TEMA": TrendIndicators.tema,
            "DEMA": TrendIndicators.dema,
            "WMA": TrendIndicators.wma,
            "TRIMA": TrendIndicators.trima,
            "KAMA": TrendIndicators.kama,
            "T3": TrendIndicators.t3,
            "MIDPOINT": TrendIndicators.midpoint,
            "MIDPRICE": TrendIndicators.midprice,
            "SAR": TrendIndicators.sar,
            "SAREXT": TrendIndicators.sarext,
            # モメンタム系指標
            "RSI": MomentumIndicators.rsi,
            "MACD": MomentumIndicators.macd,
            "MACDEXT": MomentumIndicators.macdext,
            "MACDFIX": MomentumIndicators.macdfix,
            "STOCH": MomentumIndicators.stoch,
            "STOCHF": MomentumIndicators.stochf,
            "STOCHRSI": MomentumIndicators.stochrsi,
            "WILLR": MomentumIndicators.williams_r,
            "CCI": MomentumIndicators.cci,
            "CMO": MomentumIndicators.cmo,
            "ROC": MomentumIndicators.roc,
            "ROCP": MomentumIndicators.rocp,
            "ROCR": MomentumIndicators.rocr,
            "ROCR100": MomentumIndicators.rocr100,
            "MOM": MomentumIndicators.mom,
            # ボラティリティ系指標
            "ATR": VolatilityIndicators.atr,
            "NATR": VolatilityIndicators.natr,
            "TRANGE": VolatilityIndicators.trange,
            "BBANDS": VolatilityIndicators.bollinger_bands,
            "STDDEV": VolatilityIndicators.stddev,
            "VAR": VolatilityIndicators.var,
            "ADX": VolatilityIndicators.adx,
            "ADXR": VolatilityIndicators.adxr,
            "DX": VolatilityIndicators.dx,
            "MINUS_DI": VolatilityIndicators.minus_di,
            "PLUS_DI": VolatilityIndicators.plus_di,
            "MINUS_DM": VolatilityIndicators.minus_dm,
            "PLUS_DM": VolatilityIndicators.plus_dm,
            "AROON": VolatilityIndicators.aroon,
            "AROONOSC": VolatilityIndicators.aroonosc,
        }

    def calculate_indicator(
        self,
        indicator_type: str,
        parameters: Dict[str, Any],
        close_data: Union[pd.Series, np.ndarray],
        high_data: Optional[Union[pd.Series, np.ndarray]] = None,
        low_data: Optional[Union[pd.Series, np.ndarray]] = None,
        volume_data: Optional[Union[pd.Series, np.ndarray]] = None,
        open_data: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> Tuple[Optional[Union[np.ndarray, tuple]], Optional[str]]:
        """
        指標を計算（numpy配列最適化版）

        Args:
            indicator_type: 指標タイプ
            parameters: パラメータ辞書
            close_data: 終値データ（numpy配列またはpandas Series）
            high_data: 高値データ（オプション）
            low_data: 安値データ（オプション）
            volume_data: 出来高データ（オプション）
            open_data: 始値データ（オプション）

        Returns:
            tuple: (計算結果, 指標名) または (None, None)
        """
        try:
            # データをnumpy配列に変換（最適化）
            close_array = ensure_numpy_array(close_data)
            high_array = (
                ensure_numpy_array(high_data) if high_data is not None else None
            )
            low_array = ensure_numpy_array(low_data) if low_data is not None else None
            volume_array = (
                ensure_numpy_array(volume_data) if volume_data is not None else None
            )

            # 指標タイプに応じて適切な関数を呼び出し
            if indicator_type not in self.indicator_map:
                logger.warning(f"未対応の指標タイプ: {indicator_type}")
                return None, None

            func = self.indicator_map[indicator_type]

            # 指標タイプに応じて適切な引数で呼び出し（numpy配列直接渡し）
            result = self._call_indicator_function(
                func,
                indicator_type,
                parameters,
                close_array,
                high_array,
                low_array,
                volume_array,
            )

            return result, indicator_type

        except TALibError as e:
            logger.error(f"Ta-lib計算エラー ({indicator_type}): {e}")
            return None, None
        except Exception as e:
            logger.error(f"指標計算エラー ({indicator_type}): {e}")
            return None, None

    def _call_indicator_function(
        self,
        func,
        indicator_type: str,
        parameters: Dict[str, Any],
        close_array: np.ndarray,
        high_array: Optional[np.ndarray] = None,
        low_array: Optional[np.ndarray] = None,
        volume_array: Optional[np.ndarray] = None,
    ) -> Union[np.ndarray, tuple]:
        """
        指標関数を適切な引数で呼び出し

        Args:
            func: 指標関数
            indicator_type: 指標タイプ
            parameters: パラメータ辞書
            close_array: 終値データ
            high_array: 高値データ（オプション）
            low_array: 安値データ（オプション）
            volume_array: 出来高データ（オプション）

        Returns:
            指標計算結果
        """
        # 単一価格データを使用する指標
        single_price_indicators = {
            "SMA",
            "EMA",
            "TEMA",
            "DEMA",
            "WMA",
            "TRIMA",
            "KAMA",
            "T3",
            "MIDPOINT",
            "RSI",
            "CMO",
            "ROC",
            "ROCP",
            "ROCR",
            "ROCR100",
            "MOM",
            "STDDEV",
            "VAR",
        }

        # OHLC価格データを使用する指標
        ohlc_indicators = {
            "ATR",
            "NATR",
            "TRANGE",
            "ADX",
            "ADXR",
            "DX",
            "MINUS_DI",
            "PLUS_DI",
            "CCI",
            "WILLR",
            "STOCH",
            "STOCHF",
        }

        # HL価格データを使用する指標
        hl_indicators = {
            "MIDPRICE",
            "SAR",
            "SAREXT",
            "MINUS_DM",
            "PLUS_DM",
            "AROON",
            "AROONOSC",
        }

        # 複合指標（特別な処理が必要）
        complex_indicators = {"MACD", "MACDEXT", "MACDFIX", "STOCHRSI", "BBANDS"}

        try:
            if indicator_type in single_price_indicators:
                # 単一価格データ（通常はclose）
                return func(close_array, **parameters)

            elif indicator_type in ohlc_indicators:
                # OHLC価格データが必要
                if high_array is None or low_array is None:
                    raise TALibError(f"{indicator_type}には高値・安値データが必要です")
                return func(high_array, low_array, close_array, **parameters)

            elif indicator_type in hl_indicators:
                # HL価格データが必要
                if high_array is None or low_array is None:
                    raise TALibError(f"{indicator_type}には高値・安値データが必要です")
                return func(high_array, low_array, **parameters)

            elif indicator_type in complex_indicators:
                # 複合指標の特別処理
                if indicator_type in ["MACD", "MACDEXT", "MACDFIX"]:
                    return func(close_array, **parameters)
                elif indicator_type == "STOCHRSI":
                    return func(close_array, **parameters)
                elif indicator_type == "BBANDS":
                    return func(close_array, **parameters)

            else:
                # デフォルト処理（close価格のみ）
                logger.warning(
                    f"未知の指標タイプ {indicator_type}、デフォルト処理を使用"
                )
                return func(close_array, **parameters)

        except Exception as e:
            raise TALibError(f"{indicator_type}計算エラー: {e}")
