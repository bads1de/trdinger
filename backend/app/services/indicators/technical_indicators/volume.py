"""
出来高系テクニカル指標

このモジュールはnumpy配列ベースでTa-libを直接使用し、
backtesting.pyとの完全な互換性を提供します。
pandas Seriesの変換は一切行いません。
"""

from typing import cast

import numpy as np

from ..utils import (
    TALibError,
    ensure_numpy_array,
    format_indicator_result,
    handle_talib_errors,
    validate_input,
    validate_multi_input,
)


class VolumeIndicators:
    """
    出来高系指標クラス（オートストラテジー最適化）

    全ての指標はnumpy配列を直接処理し、Ta-libの性能を最大限活用します。
    backtesting.pyでの使用に最適化されています。
    """

    @staticmethod
    @handle_talib_errors
    def ad(
        high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray
    ) -> np.ndarray:
        """
        Chaikin A/D Line (チャイキンA/Dライン)

        Args:
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            close: 終値データ（numpy配列）
            volume: 出来高データ（numpy配列）

        Returns:
            AD値のnumpy配列
        """
        high = ensure_numpy_array(high)
        low = ensure_numpy_array(low)
        close = ensure_numpy_array(close)
        volume = ensure_numpy_array(volume)

        validate_multi_input(high, low, close, 1)
        if len(volume) != len(close):
            raise TALibError(
                f"出来高データの長さが一致しません。Volume: {len(volume)}, Close: {len(close)}"
            )

        result = talib.AD(high, low, close, volume)
        return cast(np.ndarray, format_indicator_result(result, "AD"))

    @staticmethod
    @handle_talib_errors
    def adosc(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        fastperiod: int = 3,
        slowperiod: int = 10,
    ) -> np.ndarray:
        """
        Chaikin A/D Oscillator (チャイキンA/Dオシレーター)

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
        high = ensure_numpy_array(high)
        low = ensure_numpy_array(low)
        close = ensure_numpy_array(close)
        volume = ensure_numpy_array(volume)

        validate_multi_input(high, low, close, max(fastperiod, slowperiod))
        if len(volume) != len(close):
            raise TALibError(
                f"出来高データの長さが一致しません。Volume: {len(volume)}, Close: {len(close)}"
            )

        result = talib.ADOSC(
            high, low, close, volume, fastperiod=fastperiod, slowperiod=slowperiod
        )
        return cast(np.ndarray, format_indicator_result(result, "ADOSC"))

    @staticmethod
    @handle_talib_errors
    def obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """
        On Balance Volume (オンバランスボリューム)

        Args:
            close: 終値データ（numpy配列）
            volume: 出来高データ（numpy配列）

        Returns:
            OBV値のnumpy配列
        """
        close = ensure_numpy_array(close)
        volume = ensure_numpy_array(volume)

        validate_input(close, 1)
        if len(volume) != len(close):
            raise TALibError(
                f"出来高データの長さが一致しません。Volume: {len(volume)}, Close: {len(close)}"
            )

        result = talib.OBV(close, volume)
        return cast(np.ndarray, format_indicator_result(result, "OBV"))
