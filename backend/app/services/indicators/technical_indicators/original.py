"""
独自テクニカル指標モジュール

現在の実装:
- FRAMA (Fractal Adaptive Moving Average)
- SUPER_SMOOTHER (Ehlers 2-Pole Super Smoother Filter)
- ELDER_RAY (Elder Ray Index)
- PRIME_OSC (Prime Number Oscillator)
- FIBO_CYCLE (Fibonacci Cycle Indicator)
- ADAPTIVE_ENTROPY (Adaptive Entropy Oscillator)
- QUANTUM_FLOW (Quantum-inspired Flow Analysis)
- HARMONIC_RESONANCE (Harmonic Resonance Indicator)
- CHAOS_FRACTAL_DIM (Chaos Theory Fractal Dimension)
- MCGINLEY_DYNAMIC (McGinley Dynamic)
- KAUFMAN_EFFICIENCY_RATIO (Kaufman Efficiency Ratio)
- CHANDE_KROLL_STOP (Chande Kroll Stop)
- TREND_INTENSITY_INDEX (Trend Intensity Index)
- CONNORS_RSI (Connors RSI)
- GRI (Gopalakrishnan Range Index)
"""

from __future__ import annotations

import logging
from typing import Final, Tuple

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy.fft import fft

from ..data_validation import (
    handle_pandas_ta_errors,
    validate_multi_series_params,
    validate_series_params,
)

logger = logging.getLogger(__name__)


class OriginalIndicators:
    """新規の独自指標を提供するクラス"""

    _ALPHA_MIN: Final[float] = 0.01
    _ALPHA_MAX: Final[float] = 1.0

    @staticmethod
    @handle_pandas_ta_errors
    def frama(close: pd.Series, length: int = 16, slow: int = 200) -> pd.Series:
        """Fractal Adaptive Moving Average (FRAMA)"""
        if length < 4:
            length = 4
        if length % 2 != 0:
            # 奇数の場合は偶数に調整（バックテスト中断を防ぐため）
            length += 1
            
        if slow < 1:
            slow = 1

        validation = validate_series_params(close, length, min_data_length=length)
        if validation is not None:
            return pd.Series(
                np.full(len(close), np.nan), index=close.index, name="FRAMA"
            )

        prices = close.astype(float).to_numpy(copy=True)
        result = np.empty_like(prices)
        result[:] = np.nan

        half = length // 2
        log2 = np.log(2.0)
        slow_float = float(slow)
        w = 2.303 * np.log(2.0 / (slow_float + 1.0))

        # ウォームアップ期間は元の価格をそのまま返す
        warmup_end = length - 1
        result[:warmup_end] = prices[:warmup_end]

        for idx in range(warmup_end, len(prices)):
            window = prices[idx - length + 1 : idx + 1]
            first_half = window[:half]
            second_half = window[half:]

            n1 = (np.max(first_half) - np.min(first_half)) / half
            n2 = (np.max(second_half) - np.min(second_half)) / half
            n3 = (np.max(window) - np.min(window)) / length

            if n1 > 1e-9 and n2 > 1e-9 and n3 > 1e-9:
                dimen = (np.log(n1 + n2) - np.log(n3)) / log2
            else:
                dimen = 1.0

            alpha = float(np.exp(w * (dimen - 1.0)))
            alpha = float(
                np.clip(
                    alpha, OriginalIndicators._ALPHA_MIN, OriginalIndicators._ALPHA_MAX
                )
            )

            prev_value = result[idx - 1] if np.isfinite(result[idx - 1]) else window[-1]
            current_price = window[-1]
            result[idx] = alpha * current_price + (1.0 - alpha) * prev_value

        return pd.Series(result, index=close.index, name="FRAMA")

    @staticmethod
    @handle_pandas_ta_errors
    def super_smoother(close: pd.Series, length: int = 10) -> pd.Series:
        """Ehlers 2-Pole Super Smoother Filter"""
        if length < 2:
            raise ValueError("length must be >= 2")

        validation = validate_series_params(close, length, min_data_length=length)
        if validation is not None:
            return pd.Series(
                np.full(len(close), np.nan), index=close.index, name="SUPER_SMOOTHER"
            )

        prices = close.astype(float).to_numpy(copy=True)
        result = np.empty_like(prices)
        result[:] = np.nan

        warmup = min(len(prices), 2)
        result[:warmup] = prices[:warmup]

        sqrt_two = np.sqrt(2.0)
        f = (sqrt_two * np.pi) / float(length)
        a = float(np.exp(-f))
        c2 = 2.0 * a * float(np.cos(f))
        c3 = -(a**2)
        c1 = 1.0 - c2 - c3

        for idx in range(2, len(prices)):
            current = prices[idx]
            previous = prices[idx - 1]
            val = (
                0.5 * c1 * (current + previous)
                + c2 * result[idx - 1]
                + c3 * result[idx - 2]
            )
            if np.isfinite(val):
                result[idx] = val
            else:
                result[idx] = result[idx - 1]  # Fallback to previous value

        return pd.Series(result, index=close.index, name="SUPER_SMOOTHER")

    @staticmethod
    @handle_pandas_ta_errors
    def elder_ray(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 13,
        ema_length: int = 16,
    ) -> Tuple[pd.Series, pd.Series]:
        """Elder Ray Index"""
        if length <= 0:
            raise ValueError("length must be positive")
        if ema_length <= 0:
            raise ValueError("ema_length must be positive")

        validation = validate_multi_series_params(
            {"high": high, "low": low, "close": close}, ema_length
        )
        if validation is not None:
            nan_series = pd.Series(np.full(len(close), np.nan), index=close.index)
            return nan_series, nan_series

        # EMAを計算
        ema = close.ewm(span=ema_length, adjust=False).mean()

        # ブルパワー: 高値 - EMA
        bull_power = high - ema

        # ベアパワー: 安値 - EMA
        bear_power = low - ema

        return bull_power, bear_power

    @staticmethod
    def calculate_elder_ray(data, length=13, ema_length=16):
        """Elder Ray Index計算のラッパーメソッド"""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be pandas DataFrame")

        required_columns = ["high", "low", "close"]
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        high = data["high"]
        low = data["low"]
        close = data["close"]

        bull_power, bear_power = OriginalIndicators.elder_ray(
            high, low, close, length, ema_length
        )

        result = pd.DataFrame(
            {
                f"Elder_Ray_Bull_{length}_{ema_length}": bull_power,
                f"Elder_Ray_Bear_{length}_{ema_length}": bear_power,
            },
            index=data.index,
        )

        return result

    @staticmethod
    def _is_prime(n: int) -> bool:
        """素数判定ヘルパー関数"""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True

    @staticmethod
    def _get_prime_sequence(length: int) -> list[int]:
        """指定された長さの素数列を生成"""
        primes = []
        num = 2
        while len(primes) < length:
            if OriginalIndicators._is_prime(num):
                primes.append(num)
            num += 1
        return primes

    @staticmethod
    def _entropy(data: np.ndarray, window: int) -> np.ndarray:
        """エントロピー計算のヘルパー関数"""
        if len(data) < window:
            return np.full_like(data, np.nan)

        result = np.empty_like(data)
        result[: window - 1] = np.nan

        for i in range(window - 1, len(data)):
            window_data = data[i - window + 1 : i + 1]
            # ヒストグラムを作成
            hist, _ = np.histogram(
                window_data, bins=min(10, len(window_data)), density=True
            )
            # ゼロを避ける
            hist = hist[hist > 0]
            if len(hist) > 0:
                # Shannonエントロピーを計算
                entropy = -np.sum(hist * np.log(hist))
                result[i] = entropy
            else:
                result[i] = np.nan

        return result

    @staticmethod
    def _simple_wavelet_transform(data: np.ndarray, scale: int) -> np.ndarray:
        """簡単なウェーブレット変換の近似"""
        if len(data) < scale:
            return np.full_like(data, np.nan)

        result = np.empty_like(data)
        result[: scale - 1] = np.nan

        # Haarウェーブレットの近似
        for i in range(scale - 1, len(data)):
            window_start = max(0, i - scale + 1)
            window_data = data[window_start : i + 1]

            # 簡単なウェーブレット計算
            half = len(window_data) // 2
            if half > 0:
                first_half = window_data[:half]
                second_half = window_data[-half:]

                # 差分と平均を組み合わせた特徴量
                diff = np.mean(second_half) - np.mean(first_half)

                result[i] = diff * np.sqrt(len(window_data))
            else:
                result[i] = 0.0

        return result

    @staticmethod
    def _find_dominant_frequencies(
        prices: np.ndarray, max_freq: int = 50
    ) -> np.ndarray:
        """主要周波数を検出するヘルパー関数"""
        # FFTを計算
        n = len(prices)
        if n < 4:
            return np.array([])

        # ハミングウィンドウを適用
        window = scipy_signal.windows.hamming(n)
        windowed_prices = prices * window

        # FFTと周波数軸の計算
        fft_result = fft(windowed_prices)
        frequencies = np.fft.fftfreq(n)[: n // 2]
        magnitude = np.abs(fft_result[: n // 2])

        # 主要ピークを検出
        peaks, _ = scipy_signal.find_peaks(
            magnitude[:max_freq], height=np.mean(magnitude[:max_freq])
        )

        if len(peaks) == 0:
            return np.array([0.1, 0.2, 0.3])  # デフォルト周波数

        # 上位3つの周波数を選択
        peak_magnitudes = magnitude[peaks]
        sorted_indices = np.argsort(peak_magnitudes)[-3:][::-1]

        return frequencies[peaks[sorted_indices]]

    @staticmethod
    def _calculate_correlation_dimension(
        prices: np.ndarray, embedding_dim: int = 3, time_delay: int = 1
    ) -> float:
        """相関次元の近似計算"""
        if len(prices) < embedding_dim * 2:
            return 1.0

        # タイムディレイ埋め込み
        try:
            n_points = len(prices) - (embedding_dim - 1) * time_delay
            if n_points <= 0:
                return 1.0

            # 埋め込みベクトルの作成
            embedded = np.zeros((n_points, embedding_dim))
            for i in range(embedding_dim):
                start_idx = i * time_delay
                embedded[:, i] = prices[start_idx : start_idx + n_points]

            # 相関積分の計算
            distances = []
            for i in range(n_points):
                for j in range(i + 1, n_points):
                    dist = np.linalg.norm(embedded[i] - embedded[j])
                    distances.append(dist)

            if not distances:
                return 1.0

            # 距離の分布を分析
            distances = np.array(distances)
            sorted_distances = np.sort(distances)

            # 相関次元の推定 (簡易版)
            # log(C(r)) ≈ D * log(r) の関係を利用
            if len(sorted_distances) > 10:
                # 上位50%と下位50%の距離で回帰
                mid_point = len(sorted_distances) // 2
                if mid_point > 1:
                    low_distances = sorted_distances[1:mid_point]
                    high_distances = sorted_distances[mid_point:]

                    if len(low_distances) > 1 and len(high_distances) > 1:
                        low_mean = np.mean(np.log(low_distances))
                        high_mean = np.mean(np.log(high_distances))

                        low_log_c = np.log(mid_point / len(sorted_distances))
                        high_log_c = np.log(0.5)

                        if high_mean != low_mean:
                            dimension = (high_log_c - low_log_c) / (
                                high_mean - low_mean
                            )
                            return max(1.0, min(5.0, dimension))  # 制限範囲

            return 1.0

        except Exception:
            return 1.0

    @staticmethod
    @handle_pandas_ta_errors
    def prime_oscillator(
        close: pd.Series, length: int = 14, signal_length: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """Prime Number Oscillator (素数オシレーター)"""
        if length < 2:
            raise ValueError("length must be >= 2")

        validation = validate_series_params(close, length, min_data_length=length)
        if validation is not None:
            nan_osc = pd.Series(
                np.full(len(close), np.nan),
                index=close.index,
                name=f"PRIME_OSC_{length}",
            )
            nan_sig = pd.Series(
                np.full(len(close), np.nan),
                index=close.index,
                name=f"PRIME_SIGNAL_{length}_{signal_length}",
            )
            return nan_osc, nan_sig

        if signal_length < 2:
            raise ValueError("signal_length must be >= 2")

        prices = close.astype(float).to_numpy()
        result = np.empty_like(prices)
        result[:] = np.nan

        # 指定期間内の素数列を生成
        primes = OriginalIndicators._get_prime_sequence(length)
        if not primes:
            nan_series = pd.Series(np.full(len(close), np.nan), index=close.index)
            return nan_series, nan_series

        max_prime = max(primes)

        if len(prices) < max_prime:
            nan_series = pd.Series(result, index=close.index)
            return nan_series, nan_series

        # 素数位置の重みを計算（素数の逆数）
        weights = [1.0 / p for p in primes]
        weight_sum = sum(weights)

        # Prime Oscillatorの計算
        for i in range(max_prime, len(prices)):
            # 各素数位置の価格変化率を計算
            price_changes = []
            for prime in primes:
                if i >= prime:
                    prev_price = prices[i - prime]
                    current_price = prices[i]
                    if prev_price != 0:
                        change_rate = (current_price - prev_price) / prev_price
                        price_changes.append(change_rate)
                    else:
                        price_changes.append(0.0)
                else:
                    price_changes.append(0.0)

            # 加重平均を計算
            weighted_sum = sum(w * change for w, change in zip(weights, price_changes))
            oscillator_value = weighted_sum / weight_sum

            # 正規化（過去200期間の標準偏差でスケーリング）
            lookback = min(200, i)
            if lookback >= max_prime:
                recent_changes = []
                for j in range(i - lookback + 1, i + 1):
                    for prime in primes:
                        if j >= prime:
                            prev_price = prices[j - prime]
                            current_price = prices[j]
                            if prev_price != 0:
                                change_rate = (current_price - prev_price) / prev_price
                                recent_changes.append(change_rate)

                if recent_changes:
                    std_dev = np.std(recent_changes)
                    if std_dev > 0:
                        oscillator_value = oscillator_value / std_dev * 100

            result[i] = oscillator_value

        oscillator = pd.Series(result, index=close.index, name=f"PRIME_OSC_{length}")

        # Signal Lineの計算（SMA）
        signal = oscillator.rolling(window=signal_length).mean()
        signal.name = f"PRIME_SIGNAL_{length}_{signal_length}"

        return oscillator, signal

    @staticmethod
    @handle_pandas_ta_errors
    def calculate_prime_oscillator(data, length=14, signal_length=3):
        """Prime Number Oscillator計算のラッパーメソッド"""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be pandas DataFrame")

        required_columns = ["close"]
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        close = data["close"]
        oscillator, signal = OriginalIndicators.prime_oscillator(
            close, length, signal_length
        )

        result = pd.DataFrame(
            {
                oscillator.name: oscillator,
                signal.name: signal,
            },
            index=data.index,
        )

        return result

    @staticmethod
    def _generate_fibonacci_sequence(count: int) -> list[int]:
        """フィボナッチ数列を生成"""
        if count <= 0:
            return []
        elif count == 1:
            return [1]
        elif count == 2:
            return [1, 1]

        sequence = [1, 1]
        for i in range(2, count):
            sequence.append(sequence[i - 1] + sequence[i - 2])
        return sequence

    @staticmethod
    @handle_pandas_ta_errors
    def fibonacci_cycle(
        close: pd.Series,
        cycle_periods: list[int] = None,
        fib_ratios: list[float] = None,
    ) -> Tuple[pd.Series, pd.Series]:
        """Fibonacci Cycle Indicator (フィボナッチサイクルインジケーター)"""
        if cycle_periods is None:
            cycle_periods = [8, 13, 21, 34, 55]
        if fib_ratios is None:
            fib_ratios = [0.618, 1.0, 1.618, 2.618]

        if not cycle_periods:
            raise ValueError("cycle_periods must not be empty")
        if not fib_ratios:
            raise ValueError("fib_ratios must not be empty")

        validation = validate_series_params(
            close, max(cycle_periods), min_data_length=max(cycle_periods)
        )
        if validation is not None:
            nan_cycle = pd.Series(
                np.full(len(close), np.nan),
                index=close.index,
                name=f"FIBO_CYCLE_{len(cycle_periods)}",
            )
            nan_sig = pd.Series(
                np.full(len(close), np.nan),
                index=close.index,
                name=f"FIBO_SIGNAL_{len(cycle_periods)}",
            )
            return nan_cycle, nan_sig

        prices = close.astype(float).to_numpy()
        result = np.empty_like(prices)
        result[:] = np.nan

        # 最大期間まで計算不能
        max_period = max(cycle_periods)

        for i in range(max_period, len(prices)):
            cycle_values = []

            # 各フィボナッチ期間で計算
            for period in cycle_periods:
                if i >= period:
                    # 期間内の価格変化率を計算
                    period_prices = prices[i - period + 1 : i + 1]
                    if period_prices[0] != 0:
                        period_return = (
                            period_prices[-1] - period_prices[0]
                        ) / period_prices[0]

                        # フィボナッチ比率を基準に正規化
                        normalized_values = []
                        for ratio in fib_ratios:
                            normalized = period_return / ratio
                            normalized_values.append(normalized)

                        # 各比率の平均を取る
                        if normalized_values:
                            avg_normalized = np.mean(normalized_values)
                            cycle_values.append(avg_normalized)

            # 調和平均を計算（負の値を除外）
            positive_values = [abs(v) for v in cycle_values if v != 0]
            if positive_values:
                # 調和平均: n / Σ(1/xi)
                harmonic_mean = len(positive_values) / sum(
                    1.0 / v for v in positive_values
                )

                # 符号を保持
                sign_sum = sum(1 for v in cycle_values if v > 0) - sum(
                    1 for v in cycle_values if v < 0
                )
                final_value = harmonic_mean * (1 if sign_sum >= 0 else -1)

                # 時間的フィルタリング（過去の値との平滑化）
                if i > max_period:
                    prev_value = result[i - 1]
                    if not np.isnan(prev_value):
                        # 指数平滑化を適用
                        alpha = 0.3
                        result[i] = alpha * final_value + (1 - alpha) * prev_value
                    else:
                        result[i] = final_value
                else:
                    result[i] = final_value

        fibonacci_cycle = pd.Series(
            result, index=close.index, name=f"FIBO_CYCLE_{len(cycle_periods)}"
        )

        # Signal Lineの計算（SMA）
        signal = fibonacci_cycle.rolling(window=3).mean()
        signal.name = f"FIBO_SIGNAL_{len(cycle_periods)}"

        return fibonacci_cycle, signal

    @staticmethod
    @handle_pandas_ta_errors
    def calculate_fibonacci_cycle(
        data, cycle_periods: list[int] = None, fib_ratios: list[float] = None
    ):
        """Fibonacci Cycle Indicator計算のラッパーメソッド"""
        # SeriesとDataFrameの両方に対応
        if isinstance(data, pd.Series):
            close = data
        elif isinstance(data, pd.DataFrame):
            required_columns = ["close"]
            for col in required_columns:
                if col not in data.columns:
                    # カラム名が大文字の場合も考慮
                    if col.capitalize() in data.columns:
                        col = col.capitalize()
                    else:
                        raise ValueError(f"Missing required column: {col}")
            close = data[col]
        else:
            raise TypeError("data must be pandas Series or DataFrame")

        cycle, signal = OriginalIndicators.fibonacci_cycle(
            close, cycle_periods, fib_ratios
        )

        result = pd.DataFrame(
            {
                cycle.name: cycle,
                signal.name: signal,
            },
            index=close.index,
        )

        return result

    @staticmethod
    @handle_pandas_ta_errors
    def adaptive_entropy(
        close: pd.Series,
        short_length: int = 14,
        long_length: int = 28,
        signal_length: int = 5,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Adaptive Entropy Oscillator (適応的エントロピーオシレーター)"""
        if short_length < 5:
            raise ValueError("short_length must be >= 5")
        if long_length < 10:
            raise ValueError("long_length must be >= 10")
        if signal_length < 2:
            raise ValueError("signal_length must be >= 2")
        if short_length >= long_length:
            raise ValueError("short_length must be < long_length")

        validation = validate_series_params(
            close, long_length, min_data_length=long_length
        )
        if validation is not None:
            nan_osc = pd.Series(
                np.full(len(close), np.nan),
                index=close.index,
                name=f"ADAPTIVE_ENTROPY_OSC_{short_length}_{long_length}",
            )
            nan_sig = pd.Series(
                np.full(len(close), np.nan),
                index=close.index,
                name=f"ADAPTIVE_ENTROPY_SIGNAL_{short_length}_{long_length}_{signal_length}",
            )
            nan_ratio = pd.Series(
                np.full(len(close), np.nan),
                index=close.index,
                name=f"ADAPTIVE_ENTROPY_RATIO_{short_length}_{long_length}",
            )
            return nan_osc, nan_sig, nan_ratio

        prices = close.astype(float).to_numpy()

        # 短期と長期のエントロピーを計算
        short_entropy = OriginalIndicators._entropy(prices, short_length)
        long_entropy = OriginalIndicators._entropy(prices, long_length)

        # エントロピー比を計算 (短期/長期)
        with np.errstate(divide="ignore", invalid="ignore"):
            entropy_ratio = short_entropy / long_entropy

        # 結果を正規化 (逆相関スケール: 高い値 = より混雑)
        normalized_osc = (entropy_ratio - 0.5) * 2.0

        # Signal Lineの計算（SMA）
        signal = (
            pd.Series(normalized_osc, index=close.index)
            .rolling(window=signal_length)
            .mean()
        )

        # 結果をPandas Seriesに変換
        oscillator = pd.Series(
            normalized_osc,
            index=close.index,
            name=f"ADAPTIVE_ENTROPY_OSC_{short_length}_{long_length}",
        )
        signal.name = (
            f"ADAPTIVE_ENTROPY_SIGNAL_{short_length}_{long_length}_{signal_length}"
        )
        ratio = pd.Series(
            entropy_ratio,
            index=close.index,
            name=f"ADAPTIVE_ENTROPY_RATIO_{short_length}_{long_length}",
        )

        return oscillator, signal, ratio

    @staticmethod
    @handle_pandas_ta_errors
    def calculate_adaptive_entropy(
        data, short_length=14, long_length=28, signal_length=5
    ):
        """Adaptive Entropy Oscillator計算のラッパーメソッド"""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be pandas DataFrame")

        required_columns = ["close"]
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        close = data["close"]
        oscillator, signal, ratio = OriginalIndicators.adaptive_entropy(
            close, short_length, long_length, signal_length
        )

        result = pd.DataFrame(
            {
                oscillator.name: oscillator,
                signal.name: signal,
                ratio.name: ratio,
            },
            index=data.index,
        )

        return result

    @staticmethod
    @handle_pandas_ta_errors
    def quantum_flow(
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        volume: pd.Series,
        length: int = 14,
        flow_length: int = 9,
    ) -> Tuple[pd.Series, pd.Series]:
        """Quantum Flow Analysis (量子インスパイアード・フローアナリシス)"""
        if length < 5:
            raise ValueError("length must be >= 5")
        if flow_length < 3:
            raise ValueError("flow_length must be >= 3")

        validation = validate_multi_series_params(
            {"close": close, "high": high, "low": low, "volume": volume},
            length,
            min_data_length=length,
        )
        if validation is not None:
            nan_flow = pd.Series(
                np.full(len(close), np.nan), index=close.index, name="QUANTUM_FLOW"
            )
            nan_sig = pd.Series(
                np.full(len(close), np.nan),
                index=close.index,
                name="QUANTUM_FLOW_SIGNAL",
            )
            return nan_flow, nan_sig

        # 価格データの前処理
        prices = close.astype(float).to_numpy()
        highs = high.astype(float).to_numpy()
        lows = low.astype(float).to_numpy()
        volumes = volume.astype(float).to_numpy()

        # ウェーブレット変換で多スケール特性を抽出
        wavelet_result = OriginalIndicators._simple_wavelet_transform(prices, length)

        # 価格変動とVolumeの相関を計算
        price_change = np.diff(prices, prepend=prices[0])
        volume_change = np.diff(volumes, prepend=volumes[0])

        # 相関スコアの計算 (簡易版)
        correlation_score = np.zeros_like(prices)
        for i in range(length, len(prices)):
            price_window = price_change[i - length + 1 : i + 1]
            volume_window = volume_change[i - length + 1 : i + 1]

            if np.std(price_window) > 0 and np.std(volume_window) > 0:
                corr = np.corrcoef(price_window, volume_window)[0, 1]
                correlation_score[i] = corr
            else:
                correlation_score[i] = 0.0

        # ボラティリティスコアの計算
        volatility = (highs - lows) / prices

        # Quantum Flowの統合計算
        quantum_flow = np.zeros_like(prices)
        for i in range(length, len(prices)):
            wavelet_component = (
                wavelet_result[i] if np.isfinite(wavelet_result[i]) else 0
            )
            corr_component = correlation_score[i]
            vol_component = volatility[i]

            integrated = (
                wavelet_component * 0.4 + corr_component * 0.3 + vol_component * 0.3
            )

            lookback = min(200, i)
            if lookback >= length:
                recent_values = quantum_flow[i - lookback + 1 : i + 1]
                if len(recent_values) > 0 and np.std(recent_values) > 0:
                    integrated = integrated / np.std(recent_values) * 0.5

            quantum_flow[i] = integrated

        # Signal Line (SMA)
        signal = (
            pd.Series(quantum_flow, index=close.index)
            .rolling(window=flow_length)
            .mean()
        )

        # 結果をPandas Seriesに変換
        flow_series = pd.Series(quantum_flow, index=close.index, name="QUANTUM_FLOW")
        signal.name = "QUANTUM_FLOW_SIGNAL"

        return flow_series, signal

    @staticmethod
    @handle_pandas_ta_errors
    def calculate_quantum_flow(data, length=14, flow_length=9):
        """Quantum Flow Analysis計算のラッパーメソッド"""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be pandas DataFrame")

        required_columns = ["close", "high", "low", "volume"]
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        close = data["close"]
        high = data["high"]
        low = data["low"]
        volume = data["volume"]

        flow, signal = OriginalIndicators.quantum_flow(
            close, high, low, volume, length, flow_length
        )

        result = pd.DataFrame(
            {
                flow.name: flow,
                signal.name: signal,
            },
            index=data.index,
        )

        return result

    @staticmethod
    def _find_dominant_frequencies(
        prices: np.ndarray, max_freq: int = 50
    ) -> np.ndarray:
        """主要周波数を検出するヘルパー関数"""
        n = len(prices)
        if n < 4:
            return np.array([])

        window = scipy_signal.windows.hamming(n)
        windowed_prices = prices * window

        fft_result = fft(windowed_prices)
        frequencies = np.fft.fftfreq(n)[: n // 2]
        magnitude = np.abs(fft_result[: n // 2])

        peaks, _ = scipy_signal.find_peaks(
            magnitude[:max_freq], height=np.mean(magnitude[:max_freq])
        )

        if len(peaks) == 0:
            return np.array([0.1, 0.2, 0.3])

        peak_magnitudes = magnitude[peaks]
        sorted_indices = np.argsort(peak_magnitudes)[-3:][::-1]

        return frequencies[peaks[sorted_indices]]

    @staticmethod
    @handle_pandas_ta_errors
    def harmonic_resonance(
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        length: int = 20,
        resonance_bands: int = 5,
        signal_length: int = 3,
    ) -> Tuple[pd.Series, pd.Series]:
        """Harmonic Resonance Indicator (HRI)"""
        if length < 10:
            raise ValueError("length must be >= 10")
        if resonance_bands < 3 or resonance_bands > 10:
            raise ValueError("resonance_bands must be between 3 and 10")
        if signal_length < 2:
            raise ValueError("signal_length must be >= 2")

        validation = validate_multi_series_params(
            {"close": close, "high": high, "low": low}, length, min_data_length=length
        )
        if validation is not None:
            nan_hri = pd.Series(
                np.full(len(close), np.nan),
                index=close.index,
                name="HARMONIC_RESONANCE",
            )
            nan_sig = pd.Series(
                np.full(len(close), np.nan), index=close.index, name="HRI_SIGNAL"
            )
            return nan_hri, nan_sig

        prices = close.astype(float).to_numpy()
        result = np.empty_like(prices)
        result[:] = np.nan

        min_period = max(length, 30)

        for i in range(min_period, len(prices)):
            window_start = i - length + 1
            price_window = prices[window_start : i + 1]

            dominant_freqs = OriginalIndicators._find_dominant_frequencies(price_window)

            if len(dominant_freqs) == 0:
                result[i] = 0.0
                continue

            resonance_score = 0.0
            for freq in dominant_freqs[: min(resonance_bands, len(dominant_freqs))]:
                if freq <= 0:
                    continue

                try:
                    nyquist = 0.5
                    lowcut = max(freq * 0.8, 0.01)
                    highcut = min(freq * 1.2, nyquist * 0.9)

                    if lowcut < highcut:
                        b, a = scipy_signal.butter(
                            3, [lowcut, highcut], btype="band", fs=1.0
                        )
                        filtered = scipy_signal.filtfilt(b, a, price_window)

                        correlation = (
                            np.corrcoef(filtered[:-1], filtered[1:], rowvar=False)[0, 1]
                            if len(filtered) > 1
                            else 0
                        )

                        amplitude = np.std(filtered)
                        freq_weight = 1.0 / (1.0 + freq * 10)

                        resonance_score += abs(correlation) * amplitude * freq_weight

                except Exception:
                    resonance_score += 0.1

            lookback = min(200, i)
            if lookback >= min_period:
                recent_scores = [
                    result[j]
                    for j in range(i - lookback + 1, i)
                    if not np.isnan(result[j])
                ]
                if recent_scores:
                    mean_score = np.mean(recent_scores)
                    std_score = np.std(recent_scores)
                    if std_score > 0:
                        result[i] = (resonance_score - mean_score) / std_score
                    else:
                        result[i] = resonance_score
                else:
                    result[i] = resonance_score
            else:
                result[i] = resonance_score

        hri_series = pd.Series(result, index=close.index, name="HARMONIC_RESONANCE")
        signal = hri_series.rolling(window=signal_length, min_periods=1).mean()
        signal.name = "HRI_SIGNAL"

        return hri_series, signal

    @staticmethod
    @handle_pandas_ta_errors
    def calculate_harmonic_resonance(
        data, length=20, resonance_bands=5, signal_length=3
    ):
        """Harmonic Resonance Indicator計算のラッパーメソッド"""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be pandas DataFrame")

        required_columns = ["close", "high", "low"]
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        close = data["close"]
        high = data["high"]
        low = data["low"]

        hri, signal = OriginalIndicators.harmonic_resonance(
            close, high, low, length, resonance_bands, signal_length
        )

        result = pd.DataFrame(
            {
                hri.name: hri,
                signal.name: signal,
            },
            index=data.index,
        )

        return result

    @staticmethod
    def _calculate_correlation_dimension(
        prices: np.ndarray, embedding_dim: int = 3, time_delay: int = 1
    ) -> float:
        """相関次元の近似計算"""
        if len(prices) < embedding_dim * 2:
            return 1.0

        # タイムディレイ埋め込み
        try:
            n_points = len(prices) - (embedding_dim - 1) * time_delay
            if n_points <= 0:
                return 1.0

            # 埋め込みベクトルの作成
            embedded = np.zeros((n_points, embedding_dim))
            for i in range(embedding_dim):
                start_idx = i * time_delay
                embedded[:, i] = prices[start_idx : start_idx + n_points]

            # 相関積分の計算
            distances = []
            for i in range(n_points):
                for j in range(i + 1, n_points):
                    dist = np.linalg.norm(embedded[i] - embedded[j])
                    distances.append(dist)

            if not distances:
                return 1.0

            # 距離の分布を分析
            distances = np.array(distances)
            sorted_distances = np.sort(distances)

            # 相関次元の推定 (簡易版)
            # log(C(r)) ≈ D * log(r) の関係を利用
            if len(sorted_distances) > 10:
                # 上位50%と下位50%の距離で回帰
                mid_point = len(sorted_distances) // 2
                if mid_point > 1:
                    low_distances = sorted_distances[1:mid_point]
                    high_distances = sorted_distances[mid_point:]

                    if len(low_distances) > 1 and len(high_distances) > 1:
                        low_mean = np.mean(np.log(low_distances))
                        high_mean = np.mean(np.log(high_distances))

                        low_log_c = np.log(mid_point / len(sorted_distances))
                        high_log_c = np.log(0.5)

                        if high_mean != low_mean:
                            dimension = (high_log_c - low_log_c) / (
                                high_mean - low_mean
                            )
                            return max(1.0, min(5.0, dimension))  # 制限範囲

            return 1.0

        except Exception:
            return 1.0

    @staticmethod
    @handle_pandas_ta_errors
    def chaos_fractal_dimension(
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        volume: pd.Series,
        length: int = 25,
        embedding_dim: int = 3,
        signal_length: int = 4,
    ) -> Tuple[pd.Series, pd.Series]:
        """Chaos Theory Fractal Dimension (CTFD)"""
        if length < 15:
            raise ValueError("length must be >= 15")
        if embedding_dim < 2 or embedding_dim > 5:
            raise ValueError("embedding_dim must be between 2 and 5")
        if signal_length < 2:
            raise ValueError("signal_length must be >= 2")

        validation = validate_multi_series_params(
            {"close": close, "high": high, "low": low, "volume": volume},
            length,
            min_data_length=length,
        )
        if validation is not None:
            nan_ctf = pd.Series(
                np.full(len(close), np.nan), index=close.index, name="CHAOS_FRACTAL_DIM"
            )
            nan_sig = pd.Series(
                np.full(len(close), np.nan), index=close.index, name="CTFD_SIGNAL"
            )
            return nan_ctf, nan_sig

        prices = close.astype(float).to_numpy()
        volumes = volume.astype(float).to_numpy()

        result = np.empty_like(prices)
        result[:] = np.nan

        min_period = max(length, 30)

        for i in range(min_period, len(prices)):
            window_start = i - length + 1
            price_window = prices[window_start : i + 1]
            volume_window = volumes[window_start : i + 1]

            price_change = np.diff(price_window, prepend=price_window[0])
            volume_change = np.diff(volume_window, prepend=volume_window[0])

            correlation_dim = OriginalIndicators._calculate_correlation_dimension(
                price_window, embedding_dim
            )

            if len(price_change) > 5 and len(volume_change) > 5:
                try:
                    poly_fit = np.polyfit(price_change, volume_change, 2)
                    poly_predict = np.polyval(poly_fit, price_change)
                    nonlinear_residual = np.std(volume_change - poly_predict)

                    price_squared = price_change**2
                    linear_corr = (
                        np.corrcoef(price_change, price_squared)[0, 1]
                        if len(price_change) > 1
                        else 0
                    )

                    chaos_score = (
                        correlation_dim * 0.4
                        + abs(linear_corr) * 0.3
                        + (nonlinear_residual / (np.std(volume_change) + 1e-6)) * 0.3
                    )

                except Exception:
                    chaos_score = correlation_dim * 0.7 + 0.3
            else:
                chaos_score = correlation_dim

            predictability = 1.0 / (1.0 + chaos_score)

            lookback = min(200, i)
            if lookback >= min_period:
                recent_scores = []
                for j in range(i - lookback + 1, i):
                    if not np.isnan(result[j]):
                        recent_scores.append(result[j])

                if len(recent_scores) > 10:
                    mean_score = np.mean(recent_scores)
                    std_score = np.std(recent_scores)
                    if std_score > 0:
                        normalized_score = (predictability - mean_score) / std_score
                        result[i] = np.clip(normalized_score, -1.0, 1.0)
                    else:
                        result[i] = predictability
                else:
                    result[i] = predictability
            else:
                result[i] = predictability

        ctf_series = pd.Series(result, index=close.index, name="CHAOS_FRACTAL_DIM")
        signal = ctf_series.rolling(window=signal_length, min_periods=1).mean()
        signal.name = "CTFD_SIGNAL"

        return ctf_series, signal

    @staticmethod
    @handle_pandas_ta_errors
    def calculate_chaos_fractal_dimension(
        data, length=25, embedding_dim=3, signal_length=4
    ):
        """Chaos Theory Fractal Dimension計算のラッパーメソッド"""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be pandas DataFrame")

        required_columns = ["close", "high", "low", "volume"]
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        close = data["close"]
        high = data["high"]
        low = data["low"]
        volume = data["volume"]

        ctf, signal = OriginalIndicators.chaos_fractal_dimension(
            close, high, low, volume, length, embedding_dim, signal_length
        )

        result = pd.DataFrame(
            {
                ctf.name: ctf,
                signal.name: signal,
            },
            index=data.index,
        )

        return result

    @staticmethod
    @handle_pandas_ta_errors
    def mcginley_dynamic(
        close: pd.Series, length: int = 10, k: float = 0.6
    ) -> pd.Series:
        """McGinley Dynamic (MD)"""
        if length < 1:
            raise ValueError("length must be >= 1")

        validation = validate_series_params(close, length, min_data_length=length)
        if validation is not None:
            return pd.Series(
                np.full(len(close), np.nan),
                index=close.index,
                name=f"MCGINLEY_{length}",
            )

        if k <= 0:
            raise ValueError("k must be > 0")

        prices = close.astype(float).to_numpy(copy=True)
        result = np.empty_like(prices)
        result[:] = np.nan

        # 初期値は最初の価格
        result[0] = prices[0]

        # McGinley Dynamicの計算
        for i in range(1, len(prices)):
            price = prices[i]
            prev_md = result[i - 1]

            if np.isnan(prev_md) or prev_md == 0:
                result[i] = price
                continue

            ratio = price / prev_md
            ratio = np.clip(ratio, 0.1, 10.0)
            denominator = k * length * (ratio**4)

            if denominator < 1e-10:
                denominator = 1e-10

            md_change = (price - prev_md) / denominator
            result[i] = prev_md + md_change

        return pd.Series(result, index=close.index, name=f"MCGINLEY_{length}")

    @staticmethod
    @handle_pandas_ta_errors
    def calculate_mcginley_dynamic(data, length=10, k=0.6):
        """McGinley Dynamic計算のラッパーメソッド"""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be pandas DataFrame")

        required_columns = ["close"]
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        close = data["close"]
        md = OriginalIndicators.mcginley_dynamic(close, length, k)

        result = pd.DataFrame(
            {
                md.name: md,
            },
            index=data.index,
        )

        return result

    @staticmethod
    @handle_pandas_ta_errors
    def chande_kroll_stop(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        p: int = 10,
        x: float = 1.0,
        q: int = 9,
    ) -> Tuple[pd.Series, pd.Series]:
        """Chande Kroll Stop"""
        if p < 1:
            raise ValueError("p must be >= 1")
        if q < 1:
            raise ValueError("q must be >= 1")

        validation = validate_multi_series_params(
            {"high": high, "low": low, "close": close},
            max(p, q),
            min_data_length=max(p, q),
        )
        if validation is not None:
            nan_long = pd.Series(
                np.full(len(close), np.nan), index=close.index, name=f"CKS_LONG_{p}"
            )
            nan_short = pd.Series(
                np.full(len(close), np.nan), index=close.index, name=f"CKS_SHORT_{p}"
            )
            return nan_long, nan_short

        if x <= 0:
            raise ValueError("x must be > 0")

        # ATRの計算
        tr = pd.DataFrame(
            {
                "hl": high - low,
                "hc": abs(high - close.shift(1)),
                "lc": abs(low - close.shift(1)),
            }
        ).max(axis=1)
        atr = tr.rolling(window=p).mean()

        highest_high = high.rolling(window=p).max()
        lowest_low = low.rolling(window=p).min()

        long_stop_initial = highest_high - x * atr
        short_stop_initial = lowest_low + x * atr

        long_stop = long_stop_initial.rolling(window=q).mean()
        short_stop = short_stop_initial.rolling(window=q).mean()

        long_stop.name = f"CKS_LONG_{p}"
        short_stop.name = f"CKS_SHORT_{p}"

        return long_stop, short_stop

    @staticmethod
    @handle_pandas_ta_errors
    def calculate_chande_kroll_stop(data, p=10, x=1.0, q=9):
        """Chande Kroll Stop計算のラッパーメソッド"""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be pandas DataFrame")

        required_columns = ["high", "low", "close"]
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        high = data["high"]
        low = data["low"]
        close = data["close"]

        long_stop, short_stop = OriginalIndicators.chande_kroll_stop(
            high, low, close, p, x, q
        )

        result = pd.DataFrame(
            {
                long_stop.name: long_stop,
                short_stop.name: short_stop,
            },
            index=data.index,
        )

        return result

    @staticmethod
    @handle_pandas_ta_errors
    def trend_intensity_index(
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        length: int = 14,
        sma_length: int = 30,
    ) -> pd.Series:
        """Trend Intensity Index (TII)"""
        if length < 1:
            raise ValueError("length must be >= 1")
        if sma_length < 1:
            raise ValueError("sma_length must be >= 1")

        validation = validate_multi_series_params(
            {"close": close, "high": high, "low": low},
            max(length, sma_length),
            min_data_length=max(length, sma_length),
        )
        if validation is not None:
            return pd.Series(
                np.full(len(close), np.nan),
                index=close.index,
                name=f"TII_{length}_{sma_length}",
            )

        # SMAの計算
        sma = close.rolling(window=sma_length).mean()

        # 終値がSMAより上かどうか
        above_sma = (close > sma).astype(int)

        # length期間内での上の日数をカウント
        count_above = above_sma.rolling(window=length).sum()

        # TIIの計算（パーセンテージ）
        tii = (count_above / length) * 100

        return pd.Series(tii, index=close.index, name=f"TII_{length}_{sma_length}")

    @staticmethod
    @handle_pandas_ta_errors
    def calculate_trend_intensity_index(data, length=14, sma_length=30):
        """Trend Intensity Index計算のラッパーメソッド"""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be pandas DataFrame")

        required_columns = ["close", "high", "low"]
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        close = data["close"]
        high = data["high"]
        low = data["low"]

        tii = OriginalIndicators.trend_intensity_index(
            close, high, low, length, sma_length
        )

        result = pd.DataFrame(
            {
                tii.name: tii,
            },
            index=data.index,
        )

        return result

    @staticmethod
    @handle_pandas_ta_errors
    def connors_rsi(
        close: pd.Series,
        rsi_periods: int = 3,
        streak_periods: int = 2,
        rank_periods: int = 100,
    ) -> pd.Series:
        """Connors RSI (ローレンス・コナーズ RSI)"""
        if rsi_periods < 2:
            raise ValueError("rsi_periods must be >= 2")
        if streak_periods < 1:
            raise ValueError("streak_periods must be >= 1")
        if rank_periods < 2:
            raise ValueError("rank_periods must be >= 2")

        max_period = max(rsi_periods, streak_periods, rank_periods)
        validation = validate_series_params(
            close, max_period, min_data_length=max_period
        )
        if validation is not None:
            return pd.Series(
                np.full(len(close), np.nan),
                index=close.index,
                name=f"CONNORS_RSI_{rsi_periods}_{streak_periods}_{rank_periods}",
            )

        prices = close.astype(float).to_numpy()
        result = np.empty_like(prices)
        result[:] = np.nan

        close_rsi = np.empty_like(prices)
        streak_rsi = np.empty_like(prices)
        rank_values = np.empty_like(prices)

        # 1. Close RSIの計算
        close_rsi[:] = np.nan
        if len(prices) >= rsi_periods + 1:
            for i in range(rsi_periods, len(prices)):
                window = prices[i - rsi_periods + 1 : i + 1]
                gains = []
                losses = []

                for j in range(1, len(window)):
                    change = window[j] - window[j - 1]
                    if change > 0:
                        gains.append(change)
                    elif change < 0:
                        losses.append(abs(change))

                if len(gains) > 0 and len(losses) > 0:
                    avg_gain = np.mean(gains)
                    avg_loss = np.mean(losses)
                    rs = avg_gain / avg_loss
                    rsi_value = 100 - (100 / (1 + rs))
                    close_rsi[i] = rsi_value
                elif len(gains) > 0:
                    close_rsi[i] = 100
                else:
                    close_rsi[i] = 0

        # 2. Streak RSIの計算
        streak_rsi[:] = np.nan
        if len(prices) >= streak_periods + 1:
            streaks = np.zeros_like(prices)
            for i in range(1, len(prices)):
                if prices[i] > prices[i - 1]:
                    streaks[i] = max(streaks[i - 1] + 1, 1)
                elif prices[i] < prices[i - 1]:
                    streaks[i] = min(streaks[i - 1] - 1, -1)
                else:
                    streaks[i] = 0

            for i in range(streak_periods, len(prices)):
                window = streaks[i - streak_periods + 1 : i + 1]
                gains = []
                losses = []

                for j in range(1, len(window)):
                    change = window[j] - window[j - 1]
                    if change > 0:
                        gains.append(change)
                    elif change < 0:
                        losses.append(abs(change))

                if len(gains) > 0 and len(losses) > 0:
                    avg_gain = np.mean(gains)
                    avg_loss = np.mean(losses)
                    rs = avg_gain / avg_loss
                    streak_rsi[i] = 100 - (100 / (1 + rs))
                elif len(gains) > 0:
                    streak_rsi[i] = 100
                else:
                    streak_rsi[i] = 0

        # 3. Rankの計算
        rank_values[:] = np.nan
        if len(prices) >= rank_periods:
            for i in range(rank_periods, len(prices)):
                window = prices[i - rank_periods + 1 : i + 1]
                current_price = prices[i]
                count_lower = np.sum(window <= current_price)
                total_count = len(window)
                if total_count > 0:
                    percentile = (count_lower / total_count) * 100
                    rank_values[i] = percentile

        # Connors RSIの統合計算
        for i in range(max(rsi_periods, streak_periods, rank_periods), len(prices)):
            valid_components = []
            if not np.isnan(close_rsi[i]):
                valid_components.append(close_rsi[i])
            if not np.isnan(streak_rsi[i]):
                valid_components.append(streak_rsi[i])
            if not np.isnan(rank_values[i]):
                valid_components.append(rank_values[i])

            if valid_components:
                if len(valid_components) == 3:
                    connors_value = np.mean(valid_components)
                elif len(valid_components) == 2:
                    connors_value = np.mean(valid_components) * (
                        3.0 / len(valid_components)
                    )
                else:
                    connors_value = valid_components[0]
                result[i] = max(0, min(100, connors_value))

        return pd.Series(
            result,
            index=close.index,
            name=f"CONNORS_RSI_{rsi_periods}_{streak_periods}_{rank_periods}",
        )

    @staticmethod
    @handle_pandas_ta_errors
    def calculate_connors_rsi(data, rsi_periods=3, streak_periods=2, rank_periods=100):
        """Connors RSI計算のラッパーメソッド"""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be pandas DataFrame")

        required_columns = ["close"]
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        close = data["close"]
        connors_rsi = OriginalIndicators.connors_rsi(
            close, rsi_periods, streak_periods, rank_periods
        )

        result = pd.DataFrame(
            {
                connors_rsi.name: connors_rsi,
            },
            index=data.index,
        )

        return result

    @staticmethod
    @handle_pandas_ta_errors
    def gri(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 14,
        offset: int = 0,
    ) -> pd.Series:
        """Gopalakrishnan Range Index (GRI)

        期間内の最高値と最安値のレンジを分析し、市場のボラティリティ/フラクタル特性を測定する。
        GRI = log(max(High, n) - min(Low, n)) / log(n)

        Args:
            high: 高値
            low: 安値
            close: 終値
            length: 期間（デフォルト: 14）
            offset: シフト量

        Returns:
            GRI シリーズ
        """
        validation = validate_multi_series_params(
            {"high": high, "low": low, "close": close}, length
        )
        if validation is not None:
            return validation

        # GRI の計算: log(MaxHigh_n - MinLow_n) / log(n)
        hh = high.rolling(window=length).max()
        ll = low.rolling(window=length).min()

        # ゼロ以下にならないよう微小値を加算
        tr = (hh - ll).replace(0, 1e-9)

        result = np.log(tr) / np.log(float(length))

        if offset != 0:
            result = result.shift(offset)

        return result
