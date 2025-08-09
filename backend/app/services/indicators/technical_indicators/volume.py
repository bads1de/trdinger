"""
出来高系テクニカル指標（pandas-ta移行版）

このモジュールはpandas-taライブラリを使用し、
backtesting.pyとの完全な互換性を提供します。
numpy配列ベースのインターフェースを維持しています。
"""

from typing import cast

import numpy as np

from ..utils import (
    PandasTAError,
    ensure_numpy_array,
    format_indicator_result,
    handle_pandas_ta_errors,
    validate_input,
    validate_multi_input,
)
from ..pandas_ta_utils import (
    ad as pandas_ta_ad,
    adosc as pandas_ta_adosc,
    obv as pandas_ta_obv,
)


class VolumeIndicators:
    """
    出来高系指標クラス（オートストラテジー最適化）

    全ての指標はnumpy配列を直接処理し、Ta-libの性能を最大限活用します。
    backtesting.pyでの使用に最適化されています。
    """

    @staticmethod
    def ad(
        high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray
    ) -> np.ndarray:
        """
        Chaikin A/D Line (チャイキンA/Dライン) - pandas-ta版

        Args:
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            close: 終値データ（numpy配列）
            volume: 出来高データ（numpy配列）

        Returns:
            AD値のnumpy配列
        """
        return pandas_ta_ad(high, low, close, volume)

    @staticmethod
    def adosc(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        fastperiod: int = 3,
        slowperiod: int = 10,
    ) -> np.ndarray:
        """
        Chaikin A/D Oscillator (チャイキンA/Dオシレーター) - pandas-ta版

        Args:
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            close: 終値データ（numpy配列）
            volume: 出来高データ（numpy配列）
            fastperiod: 高速期間（デフォルト: 3）
            slowperiod: 低速期間（デフォルト: 10）

        Returns:
            ADOSC値のnumpy配列
        """
        return pandas_ta_adosc(high, low, close, volume, fastperiod, slowperiod)

    @staticmethod
    def obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """
        On Balance Volume (オンバランスボリューム) - pandas-ta版

        Args:
            close: 終値データ（numpy配列）
            volume: 出来高データ（numpy配列）

        Returns:
            OBV値のnumpy配列
        """
        return pandas_ta_obv(close, volume)
