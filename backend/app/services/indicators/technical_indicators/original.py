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
"""

from __future__ import annotations

import logging
from typing import Final, Tuple

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy.fft import fft

logger = logging.getLogger(__name__)


class OriginalIndicators:
    """新規の独自指標を提供するクラス"""

    _ALPHA_MIN: Final[float] = 0.01
    _ALPHA_MAX: Final[float] = 1.0

    @staticmethod
    def frama(close: pd.Series, length: int = 16, slow: int = 200) -> pd.Series:
        """Fractal Adaptive Moving Average (FRAMA)

        John Ehlers が提案した適応型移動平均で、ウィンドウのフラクタル次元に応じてスムージング係数を調整する。
        ETFHQ による改良案に従い、slow パラメータで最大スムージング長を調整できるようにする。

        参考文献:
            - "Ehler's Fractal Adaptive Moving Average (FRAMA)", ProRealCode, 2016-11-24
              https://www.prorealcode.com/prorealtime-indicators/ehlers-fractal-adaptive-moving-average/

        Args:
            close: クローズ価格の系列
            length: フラクタル次元を評価するローリングウィンドウ長（偶数、>=4）
            slow: スムージング係数の下限を決める最大期間（>=1）

        Returns:
            FRAMA 値を表す Pandas Series
        """

        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")
        if length < 4:
            raise ValueError("length must be >= 4")
        if length % 2 != 0:
            raise ValueError("length must be an even number")
        if slow < 1:
            raise ValueError("slow must be >= 1")

        if close.empty:
            return pd.Series(np.full(0, np.nan), index=close.index, name="FRAMA")
        if len(close) < length:
            logger.warning(
                "FRAMA: insufficient data length (%s) for window size %s",
                len(close),
                length,
            )
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

            if n1 > 0 and n2 > 0 and n3 > 0:
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
    def super_smoother(close: pd.Series, length: int = 10) -> pd.Series:
        """Ehlers 2-Pole Super Smoother Filter

        John Ehlers が提案したバターワース型の2極フィルターで、0.5*(x[n] + x[n-1])を入力としたIIR構造を用いて
        高周波ノイズを抑制しつつ遅延を最小化する。

        参考実装:
            - "Ehler´s Super Smoothers.", ProRealCode Forum, 2018-04-06
              https://prorealcode.com/topic/ehlers-super-smoothers/

        Args:
            close: クローズ価格の系列
            length: フィルター期間（>=2）

        Returns:
            Super Smoother による平滑化結果
        """

        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")
        if length < 2:
            raise ValueError("length must be >= 2")

        if close.empty:
            return pd.Series(
                np.full(0, np.nan), index=close.index, name="SUPER_SMOOTHER"
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
            result[idx] = (
                0.5 * c1 * (current + previous)
                + c2 * result[idx - 1]
                + c3 * result[idx - 2]
            )

        return pd.Series(result, index=close.index, name="SUPER_SMOOTHER")

    @staticmethod
    def elder_ray(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 13,
        ema_length: int = 16,
    ) -> Tuple[pd.Series, pd.Series]:
        """Elder Ray Index

        Dr. Alexander Elderが開発したモメンタムインジケーター。
        ブルパワー（高値 - EMA）とベアパワー（安値 - EMA）を計算し、
        市場の買いと売りの勢いを測定する。

        計算式:
        - Bull Power = High - EMA(close, ema_length)
        - Bear Power = Low - EMA(close, ema_length)

        Args:
            high: 高値の系列
            low: 安値の系列
            close: 終値の系列
            length: 計算期間（未使用、将来の拡張用）
            ema_length: EMA計算期間 (default: 16)

        Returns:
            Tuple[pd.Series, pd.Series]: (Bull Power, Bear Power)

        References:
            - Elder, Alexander. Trading for a Living (1993)
        """
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        # データ長の検証
        series_lengths = [len(high), len(low), len(close)]
        if not all(length == series_lengths[0] for length in series_lengths):
            raise ValueError("Elder Ray requires all series to have the same length")

        if length <= 0:
            raise ValueError(f"length must be positive: {length}")
        if ema_length <= 0:
            raise ValueError(f"ema_length must be positive: {ema_length}")

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
    def prime_oscillator(
        close: pd.Series, length: int = 14, signal_length: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """Prime Number Oscillator (素数オシレーター)

        素数の特性を利用して価格の周期性を検出する独自のオシレーター。
        素数間隔で価格変化をサンプリングし、それらの加重平均を計算することで、
        通常の移動平均では捉えきれない市場の周期的なパターンを捉える。

        計算方法:
        1. 指定期間内の素数列を生成 (例: 14期間なら [2,3,5,7,11,13])
        2. 各素数位置の価格変化率を計算
        3. 素数の逆数を重みとして加重平均を計算
        4. 結果を正規化してオシレーターとして表示

        Args:
            close: クローズ価格の系列
            length: 計算期間（素数列の最大値、>=2）
            signal_length: 信号線の平滑化期間（>=2）

        Returns:
            Tuple[pd.Series, pd.Series]: (Prime Oscillator, Signal Line)

        特徴:
        - 素数間隔による非周期的サンプリング
        - 市場の隠れた周期性の検出
        - 過去の価格パターンとの相関分析に有用
        """
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")
        if length < 2:
            raise ValueError("length must be >= 2")
        if signal_length < 2:
            raise ValueError("signal_length must be >= 2")

        if close.empty or len(close) < length:
            empty_osc = pd.Series(
                np.full(len(close), np.nan),
                index=close.index,
                name=f"PRIME_OSC_{length}",
            )
            empty_signal = pd.Series(
                np.full(len(close), np.nan),
                index=close.index,
                name=f"PRIME_SIGNAL_{length}_{signal_length}",
            )
            return empty_osc, empty_signal

        prices = close.astype(float).to_numpy()
        result = np.empty_like(prices)
        result[:] = np.nan

        # 指定期間内の素数列を生成
        primes = OriginalIndicators._get_prime_sequence(length)
        max_prime = max(primes)

        if len(prices) < max_prime:
            logger.warning(
                "Prime Oscillator: insufficient data length (%s) for max prime %s",
                len(prices),
                max_prime,
            )
            return (
                pd.Series(result, index=close.index, name=f"PRIME_OSC_{length}"),
                pd.Series(
                    result,
                    index=close.index,
                    name=f"PRIME_SIGNAL_{length}_{signal_length}",
                ),
            )

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
    def fibonacci_cycle(
        close: pd.Series,
        cycle_periods: list[int] = None,
        fib_ratios: list[float] = None,
    ) -> Tuple[pd.Series, pd.Series]:
        """Fibonacci Cycle Indicator (フィボナッチサイクルインジケーター)

        フィボナッチ数列と黄金比を利用して市場のサイクルを検出する独自のオシレーター。
        複数のフィボナッチ期間で価格変動を分析し、調和平均で統合することで、
        市場の潜在的な時間的パターンとサイクルを特定する。

        計算方法:
        1. 指定されたフィボナッチ期間で価格変化率を計算
        2. フィボナッチ比率 (0.618, 1.0, 1.618, 2.618) を基準に正規化
        3. 各期間の結果を調和平均で統合
        4. 時間的フィルタリングを適用してノイズを低減

        Args:
            close: クローズ価格の系列
            cycle_periods: 使用するフィボナッチ期間のリスト（例: [8, 13, 21, 34, 55]）
            fib_ratios: フィボナッチ比率のリスト（例: [0.618, 1.0, 1.618, 2.618]）

        Returns:
            Tuple[pd.Series, pd.Series]: (Fibonacci Cycle, Signal Line)

        特徴:
        - フィボナッチ数列に基づく時間的分析
        - 市場サイクルの検出
        - 黄金比を利用した正規化
        - 複数時間枠の統合分析
        """
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        if cycle_periods is None:
            cycle_periods = [8, 13, 21, 34, 55]
        if fib_ratios is None:
            fib_ratios = [0.618, 1.0, 1.618, 2.618]

        if not cycle_periods or not fib_ratios:
            raise ValueError("cycle_periods and fib_ratios must not be empty")

        if close.empty or len(close) < max(cycle_periods):
            empty_cycle = pd.Series(
                np.full(len(close), np.nan),
                index=close.index,
                name=f"FIBO_CYCLE_{len(cycle_periods)}",
            )
            empty_signal = pd.Series(
                np.full(len(close), np.nan),
                index=close.index,
                name=f"FIBO_SIGNAL_{len(cycle_periods)}",
            )
            return empty_cycle, empty_signal

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
    def calculate_fibonacci_cycle(
        data, cycle_periods: list[int] = None, fib_ratios: list[float] = None
    ):
        """Fibonacci Cycle Indicator計算のラッパーメソッド"""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be pandas DataFrame")

        required_columns = ["close"]
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        close = data["close"]
        cycle, signal = OriginalIndicators.fibonacci_cycle(
            close, cycle_periods, fib_ratios
        )

        result = pd.DataFrame(
            {
                cycle.name: cycle,
                signal.name: signal,
            },
            index=data.index,
        )

        return result

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
    def adaptive_entropy(
        close: pd.Series,
        short_length: int = 14,
        long_length: int = 28,
        signal_length: int = 5,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Adaptive Entropy Oscillator (適応的エントロピーオシレーター)

        時系列データの複雑さと予測可能性を測定する独自のオシレーター。
        シャノンエントロピーを利用して市場の混雑度と予測可能性を定量化し、
        短期と長期のエントロピー比から市場の状態を判断する。

        計算方法:
        1. 短期ウィンドウと長期ウィンドウでエントロピーを計算
        2. エントロピー比を計算 (短期/長期)
        3. 結果を正規化して逆相関スケールに変換
        4. 信号線で平滑化

        Args:
            close: クローズ価格の系列
            short_length: 短期エントロピー計算期間（>=5）
            long_length: 長期エントロピー計算期間（>=10）
            signal_length: 信号線平滑化期間（>=2）

        Returns:
            Tuple[pd.Series, pd.Series, pd.Series]: (Entropy Oscillator, Signal Line, Raw Entropy Ratio)

        特徴:
        - 市場の混乱度と予測可能性の定量化
        - 異常値に対してロバスト
        - トレンド変化の早期検出
        - ボラティリティと相関の組み合わせ分析
        """
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")
        if short_length < 5:
            raise ValueError("short_length must be >= 5")
        if long_length < 10:
            raise ValueError("long_length must be >= 10")
        if signal_length < 2:
            raise ValueError("signal_length must be >= 2")
        if short_length >= long_length:
            raise ValueError("short_length must be < long_length")

        if close.empty or len(close) < long_length:
            empty_osc = pd.Series(
                np.full(len(close), np.nan),
                index=close.index,
                name=f"ADAPTIVE_ENTROPY_OSC_{short_length}_{long_length}",
            )
            empty_signal = pd.Series(
                np.full(len(close), np.nan),
                index=close.index,
                name=f"ADAPTIVE_ENTROPY_SIGNAL_{short_length}_{long_length}_{signal_length}",
            )
            empty_ratio = pd.Series(
                np.full(len(close), np.nan),
                index=close.index,
                name=f"ADAPTIVE_ENTROPY_RATIO_{short_length}_{long_length}",
            )
            return empty_osc, empty_signal, empty_ratio

        prices = close.astype(float).to_numpy()

        # 短期と長期のエントロピーを計算
        short_entropy = OriginalIndicators._entropy(prices, short_length)
        long_entropy = OriginalIndicators._entropy(prices, long_length)

        # エントロピー比を計算 (短期/長期)
        with np.errstate(divide="ignore", invalid="ignore"):
            entropy_ratio = short_entropy / long_entropy

        # 結果を正規化 (逆相関スケール: 高い値 = より混雑)
        # 0.5を基準として、1.0以上で混雑、0.5以下で秩序と解釈
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
    def quantum_flow(
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        volume: pd.Series,
        length: int = 14,
        flow_length: int = 9,
    ) -> Tuple[pd.Series, pd.Series]:
        """Quantum Flow Analysis (量子インスパイアード・フローアナリシス)

        量子力学の概念から着想を得た独自のフローアナリシス指標。
        価格、ボラティリティ、Volumeの相互作用をウェーブレット変換で解析し、
        市場のエネルギー流動を測定する。波動関数の確率解釈を応用した、
        独特な市場状態評価手法。

        計算方法:
        1. 価格のウェーブレット変換で多スケール特性を抽出
        2. 高低価格差とVolumeの相関を計算
        3. 結果を統合してフロースコアを生成
        4. 時間的平滑化でノイズを低減

        Args:
            close: クローズ価格の系列
            high: 高値の系列
            low: 安値の系列
            volume: 出来高の系列
            length: ウェーブレット計算期間（>=5）
            flow_length: フロースコア計算期間（>=3）

        Returns:
            Tuple[pd.Series, pd.Series]: (Quantum Flow, Signal Line)

        特徴:
        - 量子力学の概念を応用した独自アルゴリズム
        - 多スケール市場分析
        - ボリュームと価格の相関分析
        - 波動的市場状態の可視化
        """
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(volume, pd.Series):
            raise TypeError("volume must be pandas Series")

        # データ長の検証
        series_lengths = [len(close), len(high), len(low), len(volume)]
        if not all(length == series_lengths[0] for length in series_lengths):
            raise ValueError("All series must have the same length")

        if length < 5:
            raise ValueError("length must be >= 5")
        if flow_length < 3:
            raise ValueError("flow_length must be >= 3")

        if close.empty or len(close) < length:
            empty_flow = pd.Series(
                np.full(len(close), np.nan), index=close.index, name="QUANTUM_FLOW"
            )
            empty_signal = pd.Series(
                np.full(len(close), np.nan),
                index=close.index,
                name="QUANTUM_FLOW_SIGNAL",
            )
            return empty_flow, empty_signal

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
                # 簡単な相関計算
                corr = np.corrcoef(price_window, volume_window)[0, 1]
                correlation_score[i] = corr
            else:
                correlation_score[i] = 0.0

        # ボラティリティスコアの計算
        volatility = (highs - lows) / prices

        # Quantum Flowの統合計算
        quantum_flow = np.zeros_like(prices)
        for i in range(length, len(prices)):
            # ウェーブレット結果、相関、ボラティリティを統合
            wavelet_component = (
                wavelet_result[i] if np.isfinite(wavelet_result[i]) else 0
            )
            corr_component = correlation_score[i]
            vol_component = volatility[i]

            # 統合スコア (正規化して[-1, 1]スケールに)
            integrated = (
                wavelet_component * 0.4 + corr_component * 0.3 + vol_component * 0.3
            )

            # 正規化 (過去200期間の標準偏差でスケーリング)
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
    def harmonic_resonance(
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        length: int = 20,
        resonance_bands: int = 5,
        signal_length: int = 3,
    ) -> Tuple[pd.Series, pd.Series]:
        """Harmonic Resonance Indicator (HRI)

        多重時間軸の共振パターンを検出する独自指標。
        フーリエ変換ベースの周波数分析で市場の潜在的周期性と調和を測定し、
        複数の時間スケールでの共振度合いを定量化する。

        計算方法:
        1. 価格データに対してFFTを実行し主要周波数を特定
        2. 複数のバンドパスフィルターで共振度を計算
        3. 異なる時間スケールでの調和度を統合
        4. 過去の統計に基づいて正規化

        Args:
            close: クローズ価格の系列
            high: 高値の系列
            low: 安値の系列
            length: 主計算期間（>=10）
            resonance_bands: 共振バンド数（>=3, <=10）
            signal_length: 信号線平滑化期間（>=2）

        Returns:
            Tuple[pd.Series, pd.Series]: (Harmonic Resonance, Signal Line)

        特徴:
        - フーリエ変換ベースの周波数分析
        - 多重時間軸の共振パターン検出
        - 市場の潜在的周期性の可視化
        - 調和度と不調和度の定量化
        """
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")

        # データ長の検証
        series_lengths = [len(close), len(high), len(low)]
        if not all(length == series_lengths[0] for length in series_lengths):
            raise ValueError("All series must have the same length")

        if length < 10:
            raise ValueError("length must be >= 10")
        if resonance_bands < 3 or resonance_bands > 10:
            raise ValueError("resonance_bands must be between 3 and 10")
        if signal_length < 2:
            raise ValueError("signal_length must be >= 2")

        if close.empty or len(close) < length:
            empty_hri = pd.Series(
                np.full(len(close), np.nan),
                index=close.index,
                name="HARMONIC_RESONANCE",
            )
            empty_signal = pd.Series(
                np.full(len(close), np.nan), index=close.index, name="HRI_SIGNAL"
            )
            return empty_hri, empty_signal

        # 価格データの前処理
        prices = close.astype(float).to_numpy()

        result = np.empty_like(prices)
        result[:] = np.nan

        # 最小要件を満たす期間から計算開始
        min_period = max(length, 30)

        for i in range(min_period, len(prices)):
            # ローリングウインドウで分析
            window_start = i - length + 1
            price_window = prices[window_start : i + 1]

            # 主要周波数を検出
            dominant_freqs = OriginalIndicators._find_dominant_frequencies(price_window)

            if len(dominant_freqs) == 0:
                result[i] = 0.0
                continue

            # 共振度の計算
            resonance_score = 0.0
            for freq in dominant_freqs[: min(resonance_bands, len(dominant_freqs))]:
                if freq <= 0:
                    continue

                # バンドパスフィルターを適用
                try:
                    # カーソルフィルター設計
                    nyquist = 0.5
                    lowcut = max(freq * 0.8, 0.01)
                    highcut = min(freq * 1.2, nyquist * 0.9)

                    if lowcut < highcut:
                        b, a = scipy_signal.butter(
                            3, [lowcut, highcut], btype="band", fs=1.0
                        )
                        filtered = scipy_signal.filtfilt(b, a, price_window)

                        # 共振強度を計算
                        correlation = (
                            np.corrcoef(filtered[:-1], filtered[1:], rowvar=False)[0, 1]
                            if len(filtered) > 1
                            else 0
                        )

                        # 振幅と周波数の重み付け
                        amplitude = np.std(filtered)
                        freq_weight = 1.0 / (1.0 + freq * 10)  # 高周波数にペナルティ

                        resonance_score += abs(correlation) * amplitude * freq_weight

                except Exception:
                    # フィルター設計失敗時のフォールバック
                    resonance_score += 0.1

            # 正規化 (過去期間の統計を使用)
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

        # Signal Lineの計算
        signal = (
            pd.Series(result, index=close.index)
            .rolling(window=signal_length, min_periods=1)
            .mean()
        )

        # 結果をPandas Seriesに変換
        hri_series = pd.Series(result, index=close.index, name="HARMONIC_RESONANCE")
        signal.name = "HRI_SIGNAL"

        return hri_series, signal

    @staticmethod
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
    def chaos_fractal_dimension(
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        volume: pd.Series,
        length: int = 25,
        embedding_dim: int = 3,
        signal_length: int = 4,
    ) -> Tuple[pd.Series, pd.Series]:
        """Chaos Theory Fractal Dimension (CTFD)

        カオス理論に基づく独自のフラクタル次元指標。
        リアプノフ指数の近似計算と相関次元分析で市場のカオス性を測定し、
        予測可能性と市場の複雑性を定量化する。

        計算方法:
        1. タイムディレイ埋め込みで位相空間を構築
        2. 相関次元の近似計算
        3. 価格変動の非線形性とカオス性を評価
        4. 予測可能性の逆スケールで正規化

        Args:
            close: クローズ価格の系列
            high: 高値の系列
            low: 安値の系列
            volume: 出来高の系列
            length: 主計算期間（>=15）
            embedding_dim: 埋め込み次元数（>=2, <=5）
            signal_length: 信号線平滑化期間（>=2）

        Returns:
            Tuple[pd.Series, pd.Series]: (Chaos Fractal Dimension, Signal Line)

        特徴:
        - カオス理論と非線形動力学の応用
        - 市場の予測可能性の定量化
        - 複雑性とカオス性の測定
        - 非線形相関の分析
        """
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(volume, pd.Series):
            raise TypeError("volume must be pandas Series")

        # データ長の検証
        series_lengths = [len(close), len(high), len(low), len(volume)]
        if not all(length == series_lengths[0] for length in series_lengths):
            raise ValueError("All series must have the same length")

        if length < 15:
            raise ValueError("length must be >= 15")
        if embedding_dim < 2 or embedding_dim > 5:
            raise ValueError("embedding_dim must be between 2 and 5")
        if signal_length < 2:
            raise ValueError("signal_length must be >= 2")

        if close.empty or len(close) < length:
            empty_ctfd = pd.Series(
                np.full(len(close), np.nan), index=close.index, name="CHAOS_FRACTAL_DIM"
            )
            empty_signal = pd.Series(
                np.full(len(close), np.nan), index=close.index, name="CTFD_SIGNAL"
            )
            return empty_ctfd, empty_signal

        # 価格データの前処理
        prices = close.astype(float).to_numpy()
        volumes = volume.astype(float).to_numpy()

        result = np.empty_like(prices)
        result[:] = np.nan

        # 最小要件を満たす期間から計算開始
        min_period = max(length, 30)

        for i in range(min_period, len(prices)):
            # ローリングウインドウで分析
            window_start = i - length + 1
            price_window = prices[window_start : i + 1]
            volume_window = volumes[window_start : i + 1]

            # 基本統計量の計算
            price_change = np.diff(price_window, prepend=price_window[0])
            volume_change = np.diff(volume_window, prepend=volume_window[0])

            # 相関次元の計算
            correlation_dim = OriginalIndicators._calculate_correlation_dimension(
                price_window, embedding_dim
            )

            # 非線形性の測定
            # 価格変動と出来高変動の非線形相関を計算
            if len(price_change) > 5 and len(volume_change) > 5:
                # ローカル回帰の残差を計算
                try:
                    # 簡単な多項式フィッティング
                    poly_fit = np.polyfit(price_change, volume_change, 2)
                    poly_predict = np.polyval(poly_fit, price_change)
                    nonlinear_residual = np.std(volume_change - poly_predict)

                    # 価格の自己相関の非線形性
                    price_squared = price_change**2
                    linear_corr = (
                        np.corrcoef(price_change, price_squared)[0, 1]
                        if len(price_change) > 1
                        else 0
                    )

                    # カオス性スコアの計算
                    chaos_score = (
                        correlation_dim * 0.4
                        + abs(linear_corr) * 0.3
                        + (nonlinear_residual / (np.std(volume_change) + 1e-6)) * 0.3
                    )

                except Exception:
                    # フィッティング失敗時のフォールバック
                    chaos_score = correlation_dim * 0.7 + 0.3
            else:
                chaos_score = correlation_dim

            # 予測可能性への変換 (逆スケール)
            # 高い値 = より予測可能、低い値 = よりカオス的
            predictability = 1.0 / (1.0 + chaos_score)

            # 正規化 (過去期間の統計を使用)
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
                        # z-score正規化
                        normalized_score = (predictability - mean_score) / std_score
                        # [-1, 1]スケールに制限
                        result[i] = np.clip(normalized_score, -1.0, 1.0)
                    else:
                        result[i] = predictability
                else:
                    result[i] = predictability
            else:
                result[i] = predictability

        # Signal Lineの計算
        signal = (
            pd.Series(result, index=close.index)
            .rolling(window=signal_length, min_periods=1)
            .mean()
        )

        # 結果をPandas Seriesに変換
        ctf_series = pd.Series(result, index=close.index, name="CHAOS_FRACTAL_DIM")
        signal.name = "CTFD_SIGNAL"

        return ctf_series, signal

    @staticmethod
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
    def mcginley_dynamic(
        close: pd.Series, length: int = 10, k: float = 0.6
    ) -> pd.Series:
        """McGinley Dynamic (MD)

        John R. McGinley Jr.が1990年に開発した適応型移動平均線。
        従来の移動平均線の遅延を最小化し、価格の変動に自動的に追従する。
        市場の速度に応じて平滑化係数を動的に調整することで、
        トレンド転換を早期に検出しながらノイズを効果的に除去する。

        計算式:
        MD[i] = MD[i-1] + ((Price - MD[i-1]) / (k * N * (Price/MD[i-1])^4))

        ここで:
        - N: 期間長（length）
        - k: 調整係数（デフォルト0.6、範囲0.5-0.7）
        - Price: 現在の価格
        - MD[i-1]: 前のMcGinley Dynamic値

        Args:
            close: クローズ価格の系列
            length: 計算期間（>=1、通常10-20）
            k: 適応係数（>0、デフォルト0.6）
                - 小さい値(0.5): より反応が速い
                - 大きい値(0.7): よりスムーズ

        Returns:
            McGinley Dynamic値を表す Pandas Series

        特徴:
        - 価格変動に自動追従
        - 従来のMAより遅延が少ない
        - トレンド方向の判定に有効
        - ストップロスレベルの設定に有用

        References:
            - McGinley, John R. (1990) "Market Timing with McGinley Dynamic"
              Technical Analysis of Stocks & Commodities Magazine
        """
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")
        if length < 1:
            raise ValueError("length must be >= 1")
        if k <= 0:
            raise ValueError("k must be > 0")

        if close.empty:
            return pd.Series(
                np.full(0, np.nan), index=close.index, name=f"MCGINLEY_{length}"
            )

        prices = close.astype(float).to_numpy(copy=True)
        result = np.empty_like(prices)
        result[:] = np.nan

        # 初期値は最初の価格
        result[0] = prices[0]

        # McGinley Dynamicの計算
        for i in range(1, len(prices)):
            price = prices[i]
            prev_md = result[i - 1]

            if np.isnan(prev_md):
                result[i] = price
                continue

            # ゼロ除算を防止
            if prev_md == 0:
                result[i] = price
                continue

            # McGinley Dynamic式
            # MD = MD[i-1] + ((Price - MD[i-1]) / (k * N * (Price/MD[i-1])^4))
            ratio = price / prev_md

            # オーバーフローを防ぐためにratioを制限
            ratio = np.clip(ratio, 0.1, 10.0)

            denominator = k * length * (ratio**4)

            # 非常に小さい値での除算を防止
            if denominator < 1e-10:
                denominator = 1e-10

            md_change = (price - prev_md) / denominator
            result[i] = prev_md + md_change

        return pd.Series(result, index=close.index, name=f"MCGINLEY_{length}")

    @staticmethod
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
    def kaufman_efficiency_ratio(close: pd.Series, length: int = 10) -> pd.Series:
        """Kaufman Efficiency Ratio (KER)

        Perry Kaufmanが開発したトレンド効率性を測定する指標。
        価格変動の方向性とノイズの比率を計算することで、
        市場がトレンド状態かレンジ状態かを判定する。

        計算式:
        ER = (Net Change) / (Total Movement)
        - Net Change = |Close[n] - Close[0]|
        - Total Movement = Σ|Close[i] - Close[i-1]|

        Args:
            close: クローズ価格の系列
            length: 計算期間（>=2）

        Returns:
            Efficiency Ratio（0-1の範囲）を表す Pandas Series

        特徴:
        - 0に近い: レンジ相場（ノイズが多い）
        - 1に近い: 強いトレンド（効率的な動き）
        - Kaufman's Adaptive Moving Average (KAMA)の基礎

        References:
            - Kaufman, Perry J. (1995) "Smarter Trading"
        """
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")
        if length < 2:
            raise ValueError("length must be >= 2")

        if close.empty or len(close) < length:
            return pd.Series(
                np.full(len(close), np.nan), index=close.index, name=f"KER_{length}"
            )

        prices = close.astype(float).to_numpy()
        result = np.empty_like(prices)
        result[:] = np.nan

        for i in range(length - 1, len(prices)):
            window_start = i - length + 1
            window = prices[window_start : i + 1]

            # Net Change（方向性）
            net_change = abs(window[-1] - window[0])

            # Total Movement（総変動）
            total_movement = np.sum(np.abs(np.diff(window)))

            # Efficiency Ratio計算
            if total_movement > 0:
                er = net_change / total_movement
                result[i] = np.clip(er, 0.0, 1.0)
            else:
                result[i] = 0.0

        return pd.Series(result, index=close.index, name=f"KER_{length}")

    @staticmethod
    def calculate_kaufman_efficiency_ratio(data, length=10):
        """Kaufman Efficiency Ratio計算のラッパーメソッド"""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be pandas DataFrame")

        required_columns = ["close"]
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        close = data["close"]
        ker = OriginalIndicators.kaufman_efficiency_ratio(close, length)

        result = pd.DataFrame(
            {
                ker.name: ker,
            },
            index=data.index,
        )

        return result

    @staticmethod
    def chande_kroll_stop(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        p: int = 10,
        x: float = 1.0,
        q: int = 9,
    ) -> Tuple[pd.Series, pd.Series]:
        """Chande Kroll Stop

        Tushar ChandeとStanley Krollが開発した動的ストップロス指標。
        ATRベースでボラティリティに応じたストップレベルを自動計算し、
        トレンドフォロー戦略のリスク管理に使用される。

        計算式:
        Long Stop = Highest(p) - x * ATR(p)
        Short Stop = Lowest(p) + x * ATR(p)
        最終値 = それぞれのq期間SMA

        Args:
            high: 高値の系列
            low: 安値の系列
            close: 終値の系列
            p: 初期ストップ計算期間（>=1）
            x: ATR乗数（>0、通常1-3）
            q: 平滑化期間（>=1）

        Returns:
            Tuple[pd.Series, pd.Series]: (Long Stop, Short Stop)

        特徴:
        - ボラティリティ適応型ストップロス
        - トレンド継続判定
        - ポジション保護に有効

        References:
            - Chande, Tushar & Kroll, Stanley (1994)
              "The New Technical Trader"
        """
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        # データ長の検証
        series_lengths = [len(high), len(low), len(close)]
        if not all(length == series_lengths[0] for length in series_lengths):
            raise ValueError("All series must have the same length")

        if p < 1:
            raise ValueError("p must be >= 1")
        if x <= 0:
            raise ValueError("x must be > 0")
        if q < 1:
            raise ValueError("q must be >= 1")

        if close.empty or len(close) < max(p, q):
            empty_long = pd.Series(
                np.full(len(close), np.nan), index=close.index, name=f"CKS_LONG_{p}"
            )
            empty_short = pd.Series(
                np.full(len(close), np.nan), index=close.index, name=f"CKS_SHORT_{p}"
            )
            return empty_long, empty_short

        # ATRの計算
        tr = pd.DataFrame(
            {
                "hl": high - low,
                "hc": abs(high - close.shift(1)),
                "lc": abs(low - close.shift(1)),
            }
        ).max(axis=1)
        atr = tr.rolling(window=p).mean()

        # Highest High と Lowest Low
        highest_high = high.rolling(window=p).max()
        lowest_low = low.rolling(window=p).min()

        # ストップレベルの計算
        long_stop_initial = highest_high - x * atr
        short_stop_initial = lowest_low + x * atr

        # q期間で平滑化
        long_stop = long_stop_initial.rolling(window=q).mean()
        short_stop = short_stop_initial.rolling(window=q).mean()

        long_stop.name = f"CKS_LONG_{p}"
        short_stop.name = f"CKS_SHORT_{p}"

        return long_stop, short_stop

    @staticmethod
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
    def trend_intensity_index(
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        length: int = 14,
        sma_length: int = 30,
    ) -> pd.Series:
        """Trend Intensity Index (TII)

        M.H. Peeが開発したトレンドの強さを測定する指標。
        価格が移動平均線より上にある期間の割合を計算し、
        トレンドの強さと方向性を0-100のスケールで表示する。

        計算式:
        1. SMA(close, sma_length)を計算
        2. length期間内で終値がSMAより上の日数をカウント
        3. TII = (上の日数 / length) * 100

        Args:
            close: クローズ価格の系列
            high: 高値の系列
            low: 安値の系列
            length: カウント期間（>=1）
            sma_length: SMA計算期間（>=1）

        Returns:
            Trend Intensity Index（0-100）を表す Pandas Series

        特徴:
        - 80以上: 強い上昇トレンド
        - 20以下: 強い下降トレンド
        - 40-60: レンジ相場
        - トレンド転換の早期検出

        References:
            - Pee, M.H. "Trend Intensity Index"
              Technical Analysis of Stocks & Commodities (2002)
        """
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")

        # データ長の検証
        series_lengths = [len(close), len(high), len(low)]
        if not all(length == series_lengths[0] for length in series_lengths):
            raise ValueError("All series must have the same length")

        if length < 1:
            raise ValueError("length must be >= 1")
        if sma_length < 1:
            raise ValueError("sma_length must be >= 1")

        if close.empty or len(close) < max(length, sma_length):
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
    def connors_rsi(
        close: pd.Series,
        rsi_periods: int = 3,
        streak_periods: int = 2,
        rank_periods: int = 100,
    ) -> pd.Series:
        """Connors RSI (ローレンス・コナーズ RSI)

        Laurence Connorsによって開発された複合モメンタム指標。
        3つの異なる時間スケールの相対強度を組み合わせて市場の過熱・過冷状態を測定する。

        計算式:
        1. Close RSI: 通常の終値RSI (短期)
        2. Streak RSI: 連勝/連敗ストリークのRSI
        3. Rank: 価格変動のパーセンタイルランク
        4. Connors RSI = (Close RSI + Streak RSI + Rank) / 3

        Args:
            close: クローズ価格の系列
            rsi_periods: Close RSIの計算期間（通常3）
            streak_periods: Streak RSIの計算期間（通常2）
            rank_periods: ランク計算の期間（通常100）

        Returns:
            pd.Series: Connors RSI値（0-100スケール）

        特徴:
        - 70以上: 過熱（売りシグナル）
        - 30以下: 過冷（買いシグナル）
        - 複数時間スケールの統合分析
        - スウィングトレードに特に有効

        References:
            - Connors, Laurence (2013) "Connors RSI" Technical Analysis
        """
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")
        if rsi_periods < 2:
            raise ValueError("rsi_periods must be >= 2")
        if streak_periods < 1:
            raise ValueError("streak_periods must be >= 1")
        if rank_periods < 2:
            raise ValueError("rank_periods must be >= 2")

        if close.empty or len(close) < max(rsi_periods, streak_periods, rank_periods):
            return pd.Series(
                np.full(len(close), np.nan),
                index=close.index,
                name=f"CONNORS_RSI_{rsi_periods}_{streak_periods}_{rank_periods}",
            )

        prices = close.astype(float).to_numpy()
        result = np.empty_like(prices)
        result[:] = np.nan

        # Connors RSI = (Close RSI + Streak RSI + Rank) / 3
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

        # 2. Streak RSIの計算（連勝/連敗ストリーク）
        streak_rsi[:] = np.nan
        if len(prices) >= streak_periods + 1:
            # ストリークの計算
            streaks = np.zeros_like(prices)
            for i in range(1, len(prices)):
                if prices[i] > prices[i - 1]:
                    streaks[i] = max(streaks[i - 1] + 1, 1)
                elif prices[i] < prices[i - 1]:
                    streaks[i] = min(streaks[i - 1] - 1, -1)
                else:
                    streaks[i] = 0

            # ストリークのRSIを計算
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

        # 3. Rankの計算（価格変動のパーセンタイル）
        rank_values[:] = np.nan
        if len(prices) >= rank_periods:
            for i in range(rank_periods, len(prices)):
                window = prices[i - rank_periods + 1 : i + 1]
                current_price = prices[i]

                # 現在価格が窓内の何％の価格以下かを計算
                count_lower = np.sum(window <= current_price)
                total_count = len(window)

                if total_count > 0:
                    percentile = (count_lower / total_count) * 100
                    rank_values[i] = percentile

        # Connors RSIの統合計算
        for i in range(max(rsi_periods, streak_periods, rank_periods), len(prices)):
            # 有効な値のみを使用
            valid_components = []
            if not np.isnan(close_rsi[i]):
                valid_components.append(close_rsi[i])
            if not np.isnan(streak_rsi[i]):
                valid_components.append(streak_rsi[i])
            if not np.isnan(rank_values[i]):
                valid_components.append(rank_values[i])

            if valid_components:
                # 3つの成分の平均を取る
                if len(valid_components) == 3:
                    connors_value = np.mean(valid_components)
                elif len(valid_components) == 2:
                    # 2つのみ有効な場合はフォールバック（平均に大きな重みを付ける）
                    connors_value = np.mean(valid_components) * (
                        3.0 / len(valid_components)
                    )
                else:
                    connors_value = valid_components[0]

                # 0-100の範囲にクランプ
                result[i] = max(0, min(100, connors_value))

        return pd.Series(
            result,
            index=close.index,
            name=f"CONNORS_RSI_{rsi_periods}_{streak_periods}_{rank_periods}",
        )

    @staticmethod
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
