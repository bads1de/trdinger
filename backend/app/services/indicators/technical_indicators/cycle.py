"""
サイクル系テクニカル指標

このモジュールはnumpy配列ベースでTa-libを直接使用し、
backtesting.pyとの完全な互換性を提供します。
pandas Seriesの変換は一切行いません。
"""

from typing import Tuple, cast

import numpy as np
import talib

from ..utils import (
    ensure_numpy_array,
    format_indicator_result,
    handle_talib_errors,
    validate_input,
)


class CycleIndicators:
    """
    サイクル系指標クラス（オートストラテジー最適化）

    全ての指標はnumpy配列を直接処理し、Ta-libの性能を最大限活用します。
    backtesting.pyでの使用に最適化されています。
    """

    @staticmethod
    @handle_talib_errors
    def ht_dcperiod(data: np.ndarray) -> np.ndarray:
        """
        Hilbert Transform - Dominant Cycle Period (ヒルベルト変換支配的サイクル期間)

        Args:
            data: 価格データ（numpy配列）

        Returns:
            HT_DCPERIOD値のnumpy配列
        """
        data = ensure_numpy_array(data)
        validate_input(data, 2)
        result = talib.HT_DCPERIOD(data)
        return cast(np.ndarray, format_indicator_result(result, "HT_DCPERIOD"))

    @staticmethod
    @handle_talib_errors
    def ht_dcphase(data: np.ndarray) -> np.ndarray:
        """
        Hilbert Transform - Dominant Cycle Phase (ヒルベルト変換支配的サイクル位相)

        Args:
            data: 価格データ（numpy配列）

        Returns:
            HT_DCPHASE値のnumpy配列
        """
        data = ensure_numpy_array(data)
        validate_input(data, 2)
        result = talib.HT_DCPHASE(data)
        return cast(np.ndarray, format_indicator_result(result, "HT_DCPHASE"))

    @staticmethod
    @handle_talib_errors
    def ht_phasor(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Hilbert Transform - Phasor Components (ヒルベルト変換フェーザー成分)

        Args:
            data: 価格データ（numpy配列）

        Returns:
            (InPhase, Quadrature)のtuple
        """
        data = ensure_numpy_array(data)
        validate_input(data, 2)
        inphase, quadrature = talib.HT_PHASOR(data)
        return cast(
            Tuple[np.ndarray, np.ndarray],
            format_indicator_result((inphase, quadrature), "HT_PHASOR"),
        )

    @staticmethod
    @handle_talib_errors
    def ht_sine(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Hilbert Transform - SineWave (ヒルベルト変換サイン波)

        Args:
            data: 価格データ（numpy配列）

        Returns:
            (Sine, LeadSine)のtuple
        """
        data = ensure_numpy_array(data)
        validate_input(data, 2)
        sine, leadsine = talib.HT_SINE(data)
        return cast(
            Tuple[np.ndarray, np.ndarray],
            format_indicator_result((sine, leadsine), "HT_SINE"),
        )

    @staticmethod
    @handle_talib_errors
    def ht_trendmode(data: np.ndarray) -> np.ndarray:
        """
        Hilbert Transform - Trend vs Cycle Mode (ヒルベルト変換トレンド対サイクルモード)

        Args:
            data: 価格データ（numpy配列）

        Returns:
            HT_TRENDMODE値のnumpy配列（0=サイクル、1=トレンド）
        """
        data = ensure_numpy_array(data)
        validate_input(data, 2)
        result = talib.HT_TRENDMODE(data)
        return cast(np.ndarray, format_indicator_result(result, "HT_TRENDMODE"))
