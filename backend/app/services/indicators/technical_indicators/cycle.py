"""
サイクル系テクニカル指標（scipy実装版）

このモジュールはnumpy配列ベースでscipyを使用し、
backtesting.pyとの完全な互換性を提供します。
pandas Seriesの変換は一切行いません。
"""

from typing import Tuple, cast

import numpy as np
from scipy import signal

from ..utils import (
    PandasTAError,
    ensure_numpy_array,
    format_indicator_result,
    handle_pandas_ta_errors,
    validate_input,
)


class CycleIndicators:
    """
    サイクル系指標クラス（scipy実装版）

    全ての指標はnumpy配列を直接処理し、scipyを使用した実装を提供します。
    backtesting.pyでの使用に最適化されています。
    """

    @staticmethod
    @handle_pandas_ta_errors
    def ht_dcperiod(data: np.ndarray) -> np.ndarray:
        """
        Hilbert Transform - Dominant Cycle Period (ヒルベルト変換支配的サイクル期間)
        scipy実装版

        Args:
            data: 価格データ（numpy配列）

        Returns:
            HT_DCPERIOD値のnumpy配列
        """
        data = ensure_numpy_array(data)
        validate_input(data, 2)

        # scipyのヒルベルト変換を使用した簡易実装
        analytic_signal = signal.hilbert(data)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi)

        # 周期は周波数の逆数
        result = np.full_like(data, np.nan)
        result[1:] = np.where(
            instantaneous_frequency > 0, 1.0 / instantaneous_frequency, np.nan
        )

        # 異常値をクリップ
        result = np.clip(result, 6, 50)

        return cast(np.ndarray, format_indicator_result(result, "HT_DCPERIOD"))

    @staticmethod
    @handle_pandas_ta_errors
    def ht_dcphase(data: np.ndarray) -> np.ndarray:
        """
        Hilbert Transform - Dominant Cycle Phase (ヒルベルト変換支配的サイクル位相)
        scipy実装版

        Args:
            data: 価格データ（numpy配列）

        Returns:
            HT_DCPHASE値のnumpy配列
        """
        data = ensure_numpy_array(data)
        validate_input(data, 2)

        # scipyのヒルベルト変換を使用
        analytic_signal = signal.hilbert(data)
        instantaneous_phase = np.angle(analytic_signal)

        # 位相を度数に変換
        result = np.degrees(instantaneous_phase)

        return cast(np.ndarray, format_indicator_result(result, "HT_DCPHASE"))

    @staticmethod
    @handle_pandas_ta_errors
    def ht_phasor(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Hilbert Transform - Phasor Components (ヒルベルト変換フェーザー成分)
        scipy実装版

        Args:
            data: 価格データ（numpy配列）

        Returns:
            (InPhase, Quadrature)のtuple
        """
        data = ensure_numpy_array(data)
        validate_input(data, 2)

        # scipyのヒルベルト変換を使用
        analytic_signal = signal.hilbert(data)
        inphase = np.real(analytic_signal)  # 実部
        quadrature = np.imag(analytic_signal)  # 虚部

        return cast(
            Tuple[np.ndarray, np.ndarray],
            format_indicator_result((inphase, quadrature), "HT_PHASOR"),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def ht_sine(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Hilbert Transform - SineWave (ヒルベルト変換サイン波)
        scipy実装版

        Args:
            data: 価格データ（numpy配列）

        Returns:
            (Sine, LeadSine)のtuple
        """
        data = ensure_numpy_array(data)
        validate_input(data, 2)

        # scipyのヒルベルト変換を使用
        analytic_signal = signal.hilbert(data)
        instantaneous_phase = np.angle(analytic_signal)

        # サイン波とリードサイン波を計算
        sine = np.sin(instantaneous_phase)
        leadsine = np.sin(instantaneous_phase + np.pi / 4)  # 45度位相を進める

        return cast(
            Tuple[np.ndarray, np.ndarray],
            format_indicator_result((sine, leadsine), "HT_SINE"),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def ht_trendmode(data: np.ndarray) -> np.ndarray:
        """
        Hilbert Transform - Trend vs Cycle Mode (ヒルベルト変換トレンド対サイクルモード)
        scipy実装版

        Args:
            data: 価格データ（numpy配列）

        Returns:
            HT_TRENDMODE値のnumpy配列（0=サイクル、1=トレンド）
        """
        data = ensure_numpy_array(data)
        validate_input(data, 2)

        # scipyのヒルベルト変換を使用
        analytic_signal = signal.hilbert(data)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi)

        # トレンドモードの判定（簡易実装）
        # 周波数の変動が小さい場合はトレンド、大きい場合はサイクル
        result = np.full_like(data, 0)
        if len(instantaneous_frequency) > 0:
            freq_std = np.std(instantaneous_frequency)
            threshold = 0.01  # 閾値は調整可能
            result[1:] = (freq_std < threshold).astype(int)

        return cast(np.ndarray, format_indicator_result(result, "HT_TRENDMODE"))
