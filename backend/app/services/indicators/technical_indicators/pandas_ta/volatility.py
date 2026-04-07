"""
ボラティリティ系テクニカル指標 (Volatility Indicators)

pandas-ta の volatility カテゴリに対応。
市場の変動性とリスク評価に使用する指標群。

登録してあるテクニカルの一覧:
- ATR (Average True Range)
- NATR (Normalized ATR)
- Bollinger Bands
- Keltner Channels
- Donchian Channels
- Acceleration Bands
- Ulcer Index
- RVI (Relative Volatility Index)
- True Range
- Mass Index
- Aberration
- HWC (Holt-Winter Channel)
- PDIST (Price Distance)
- Thermo
"""

import logging
from typing import Tuple, cast

import pandas as pd
import pandas_ta_classic as ta

from ...data_validation import (
    create_nan_series_bundle,
    handle_pandas_ta_errors,
    run_multi_series_indicator,
    run_series_indicator,
)

logger = logging.getLogger(__name__)


class VolatilityIndicators:
    """
    ボラティリティ系指標クラス

    ATR, Bollinger Bandsなどのボラティリティ系テクニカル指標を提供。
    市場の変動性とリスク評価に使用します。
    """

    @staticmethod
    @handle_pandas_ta_errors
    def atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 14,
    ) -> pd.Series:
        """平均真の値幅"""

        def compute():
            """ATR を計算するヘルパー関数"""
            result = ta.atr(high=high, low=low, close=close, length=length)
            if result is None:
                logger.error("ATR: Calculation returned None - returning NaN series")
            return result

        return run_multi_series_indicator(
            {"high": high, "low": low, "close": close},
            length,
            compute,
            min_data_length=length,
        )

    @staticmethod
    @handle_pandas_ta_errors
    def natr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 14,
    ) -> pd.Series:
        """Normalized Average True Range"""
        return run_multi_series_indicator(
            {"high": high, "low": low, "close": close},
            length,
            lambda: ta.natr(high=high, low=low, close=close, length=length),
            min_data_length=length,
        )

    @staticmethod
    @handle_pandas_ta_errors
    def bbands(
        data: pd.Series, length: int = 20, std: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """ボリンジャーバンド"""
        result = run_series_indicator(
            data,
            length,
            lambda: ta.bbands(data, length=length, std=std),
            fallback_factory=lambda: cast(
                Tuple[pd.Series, pd.Series, pd.Series],
                create_nan_series_bundle(data, 3),
            ),
        )

        if isinstance(result, tuple):
            return cast(Tuple[pd.Series, pd.Series, pd.Series], result)

        if result is None:
            logger.error("BBands: Calculation returned None - returning NaN series")
            return cast(
                Tuple[pd.Series, pd.Series, pd.Series],
                create_nan_series_bundle(data, 3),
            )

        # 列名を動的に取得（pandas-taのバージョンによって異なる可能性がある）
        columns = result.columns.tolist()

        # 上位、中位、下位バンドを特定
        upper_col = [col for col in columns if "BBU" in col][0]
        middle_col = [col for col in columns if "BBM" in col][0]
        lower_col = [col for col in columns if "BBL" in col][0]

        return (
            result[upper_col],
            result[middle_col],
            result[lower_col],
        )

    @staticmethod
    @handle_pandas_ta_errors
    def keltner(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 20,
        scalar: float = 2.0,
        mamode: str = "sma",
        std_dev: bool = False,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Keltner Channels: returns (upper, middle, lower)"""

        def nan_result() -> Tuple[pd.Series, pd.Series, pd.Series]:
            """計算失敗時に NaN の Series を返すヘルパー関数"""
            return cast(
                Tuple[pd.Series, pd.Series, pd.Series],
                create_nan_series_bundle(close, 3),
            )

        df = run_multi_series_indicator(
            {"high": high, "low": low, "close": close},
            period,
            lambda: ta.kc(
                high=high,
                low=low,
                close=close,
                length=period,
                scalar=scalar,
                mamode=mamode,
            ),
            fallback_factory=nan_result,
        )

        if isinstance(df, tuple):
            return cast(Tuple[pd.Series, pd.Series, pd.Series], df)

        if df.empty:
            return nan_result()

        # カラム名: KC{mamode[0]}_{length}_{scalar}
        m = mamode[0].lower()
        # pandas-ta は整数として埋め込む場合と浮動小数点として埋め込む場合があるため、両方試行
        try:
            # 浮動小数点形式 (例: 2.0)
            return (
                df[f"KCU{m}_{period}_{float(scalar)}"],
                df[f"KCB{m}_{period}_{float(scalar)}"],
                df[f"KCL{m}_{period}_{float(scalar)}"],
            )
        except KeyError:
            try:
                # 整数形式 (例: 2)
                return (
                    df[f"KCU{m}_{period}_{int(scalar)}"],
                    df[f"KCB{m}_{period}_{int(scalar)}"],
                    df[f"KCL{m}_{period}_{int(scalar)}"],
                )
            except (KeyError, Exception):
                return nan_result()

    @staticmethod
    @handle_pandas_ta_errors
    def donchian(
        high: pd.Series,
        low: pd.Series,
        length: int = 20,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Donchian Channels: returns (upper, middle, lower)"""

        def nan_result() -> Tuple[pd.Series, pd.Series, pd.Series]:
            return cast(
                Tuple[pd.Series, pd.Series, pd.Series],
                create_nan_series_bundle(high, 3),
            )

        df = run_multi_series_indicator(
            {"high": high, "low": low},
            length,
            lambda: ta.donchian(high=high, low=low, length=length),
            fallback_factory=nan_result,
        )

        if isinstance(df, tuple):
            return cast(Tuple[pd.Series, pd.Series, pd.Series], df)

        if df.empty:
            return nan_result()

        # カラム名: DCU_{length}_{length}, DCM_{length}_{length}, DCL_{length}_{length}
        try:
            return (
                df[f"DCU_{length}_{length}"],
                df[f"DCM_{length}_{length}"],
                df[f"DCL_{length}_{length}"],
            )
        except (KeyError, Exception):
            return nan_result()

    @staticmethod
    @handle_pandas_ta_errors
    def accbands(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 20,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Acceleration Bands: returns (upper, middle, lower)"""

        def nan_result() -> Tuple[pd.Series, pd.Series, pd.Series]:
            return cast(
                Tuple[pd.Series, pd.Series, pd.Series],
                create_nan_series_bundle(close, 3),
            )

        result = run_multi_series_indicator(
            {"high": high, "low": low, "close": close},
            period,
            lambda: ta.accbands(high=high, low=low, close=close, length=period),
            fallback_factory=nan_result,
        )

        if isinstance(result, tuple):
            return cast(Tuple[pd.Series, pd.Series, pd.Series], result)

        if result.empty:
            return nan_result()

        # カラム名: ACCBU_{length}, ACCBM_{length}, ACCBL_{length}
        try:
            return (
                result[f"ACCBU_{period}"],
                result[f"ACCBM_{period}"],
                result[f"ACCBL_{period}"],
            )
        except (KeyError, Exception):
            return nan_result()

    @staticmethod
    @handle_pandas_ta_errors
    def ui(data: pd.Series, period: int = 14) -> pd.Series:
        """Ulcer Index"""
        return run_series_indicator(data, None, lambda: ta.ui(data, window=period))

    @staticmethod
    @handle_pandas_ta_errors
    def rvi(
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        length: int = 14,
        scalar: float = 100.0,
        refined: bool = False,
        thirds: bool = False,
        mamode: str | None = None,
        drift: int | None = None,
        offset: int | None = None,
    ) -> pd.Series:
        """Relative Volatility Index"""
        return run_multi_series_indicator(
            {"close": close, "high": high, "low": low},
            length,
            lambda: ta.rvi(
                close=close,
                high=high,
                low=low,
                length=length,
                scalar=scalar,
                refined=refined,
                thirds=thirds,
                mamode=mamode,
                drift=drift,
                offset=offset,
            ),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def true_range(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        drift: int = 1,
    ) -> pd.Series:
        """True Range"""
        return run_multi_series_indicator(
            {"high": high, "low": low, "close": close},
            None,
            lambda: ta.true_range(high=high, low=low, close=close, drift=drift),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def massi(
        high: pd.Series,
        low: pd.Series,
        fast: int = 9,
        slow: int = 25,
    ) -> pd.Series:
        """Mass Index

        トレンドの反転を予測するためのボラティリティ指標。
        高値と安値のレンジ拡大パターンを検出。

        Args:
            high: 高値
            low: 安値
            fast: 高速 EMA 期間（デフォルト: 9）
            slow: 低速 EMA 期間（デフォルト: 25）

        Returns:
            Mass Index
        """
        if fast <= 0:
            raise ValueError("fast must be positive")

        return run_multi_series_indicator(
            {"high": high, "low": low},
            slow,
            lambda: ta.massi(high=high, low=low, fast=fast, slow=slow),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def aberration(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 5,
        atr_length: int = 15,
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """Aberration"""
        result = run_multi_series_indicator(
            {"high": high, "low": low, "close": close},
            max(length, atr_length),
            lambda: ta.aberration(
                high=high, low=low, close=close, length=length, atr_length=atr_length
            ),
            fallback_factory=lambda: cast(
                Tuple[pd.Series, pd.Series, pd.Series, pd.Series],
                create_nan_series_bundle(close, 4),
            ),
        )

        if isinstance(result, tuple):
            return cast(Tuple[pd.Series, pd.Series, pd.Series, pd.Series], result)

        if result.empty:
            return cast(
                Tuple[pd.Series, pd.Series, pd.Series, pd.Series],
                create_nan_series_bundle(close, 4),
            )

        # Returns multiple columns. Usually ZG, SG, XG, ATR
        return (
            result.iloc[:, 0],
            result.iloc[:, 1],
            result.iloc[:, 2],
            result.iloc[:, 3],
        )

    @staticmethod
    @handle_pandas_ta_errors
    def hwc(
        close: pd.Series,
        na: int = 2,
        nb: int = 3,
        nc: int = 4,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Holt-Winter Channel"""
        result = run_series_indicator(
            close,
            max(na, nb, nc),
            lambda: ta.hwc(close=close, na=na, nb=nb, nc=nc),
            fallback_factory=lambda: cast(
                Tuple[pd.Series, pd.Series, pd.Series],
                create_nan_series_bundle(close, 3),
            ),
        )

        if isinstance(result, tuple):
            return cast(Tuple[pd.Series, pd.Series, pd.Series], result)

        if result.empty:
            return cast(
                Tuple[pd.Series, pd.Series, pd.Series],
                create_nan_series_bundle(close, 3),
            )

        return result.iloc[:, 0], result.iloc[:, 1], result.iloc[:, 2]

    @staticmethod
    @handle_pandas_ta_errors
    def pdist(
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> pd.Series:
        """Price Distance"""
        return run_multi_series_indicator(
            {"open_": open_, "high": high, "low": low, "close": close},
            None,
            lambda: ta.pdist(open_=open_, high=high, low=low, close=close),
        )

    @staticmethod
    @handle_pandas_ta_errors
    def thermo(
        high: pd.Series,
        low: pd.Series,
        length: int = 20,
        long_: int = 2,
        short: int = 2,
        mamode: str = "ema",
        drift: int = 1,
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """Thermo"""
        result = run_multi_series_indicator(
            {"high": high, "low": low},
            length,
            lambda: ta.thermo(
                high=high,
                low=low,
                length=length,
                long=long_,
                short=short,
                mamode=mamode,
                drift=drift,
            ),
            fallback_factory=lambda: cast(
                Tuple[pd.Series, pd.Series, pd.Series, pd.Series],
                create_nan_series_bundle(high, 4),
            ),
        )

        if isinstance(result, tuple):
            return cast(Tuple[pd.Series, pd.Series, pd.Series, pd.Series], result)

        if result.empty:
            return cast(
                Tuple[pd.Series, pd.Series, pd.Series, pd.Series],
                create_nan_series_bundle(high, 4),
            )

        # Returns Thermo, ThermoMa, ThermoLa, ThermoSa
        return (
            result.iloc[:, 0],
            result.iloc[:, 1],
            result.iloc[:, 2],
            result.iloc[:, 3],
        )
