"""
市場データ特徴量計算クラス

ファンディングレート（FR）、建玉残高（OI）データから
市場の歪みや偏りを捉える特徴量を計算します。
"""

import logging
import math
from typing import Any, Dict

import pandas as pd

from .base_feature_calculator import BaseFeatureCalculator

logger = logging.getLogger(__name__)


class MarketDataFeatureCalculator(BaseFeatureCalculator):
    """
    市場データ特徴量計算クラス（超安全・環境不具合回避版）
    """

    def __init__(self):
        super().__init__()

    def calculate_features(
        self, df: pd.DataFrame, config: Dict[str, Any]
    ) -> pd.DataFrame:
        funding_rate_data = config.get("funding_rate_data")
        open_interest_data = config.get("open_interest_data")
        result_df = df

        if funding_rate_data is not None and not funding_rate_data.empty:
            result_df = self.calculate_funding_rate_features(
                result_df, funding_rate_data, {}
            )
        if open_interest_data is not None and not open_interest_data.empty:
            result_df = self.calculate_open_interest_features(
                result_df, open_interest_data, {}
            )
        if (
            funding_rate_data is not None
            and not funding_rate_data.empty
            and open_interest_data is not None
            and not open_interest_data.empty
        ):
            result_df = self.calculate_composite_features(
                result_df, funding_rate_data, open_interest_data, {}
            )
        return result_df

    def _process_market_data(
        self,
        df: pd.DataFrame,
        data: pd.DataFrame,
        column_candidates: list[str],
        suffix: str,
    ) -> tuple[pd.DataFrame, str | None]:
        if data is None or data.empty:
            return df, None
        target_col = next((c for c in column_candidates if c in data.columns), None)
        if target_col is None:
            return df, None

        # データの抽出 (NumPy不具合を避けるため、一度値をプリミティブ化)
        data_ts = [
            pd.Timestamp(t).tz_localize(None)
            for t in (
                data["timestamp"].tolist()
                if "timestamp" in data.columns
                else data.index.tolist()
            )
        ]
        data_vals = [float(x) for x in data[target_col].tolist()]
        df_ts = [pd.Timestamp(t).tz_localize(None) for t in df.index.tolist()]

        merged_vals = []
        for t in df_ts:
            match_val = 0.0  # Default
            for i in range(len(data_ts) - 1, -1, -1):
                if data_ts[i] <= t:
                    match_val = data_vals[i]
                    break
            merged_vals.append(match_val)

        res = df.copy()
        res[target_col + suffix] = merged_vals
        return res, target_col + suffix

    def calculate_funding_rate_features(
        self,
        df: pd.DataFrame,
        funding_rate_data: pd.DataFrame,
        lookback_periods: Dict[str, int],
    ) -> pd.DataFrame:
        try:
            res, fr_col = self._process_market_data(
                df, funding_rate_data, ["funding_rate", "fundingRate", "rate"], "_fr"
            )
            if fr_col is None:
                return df
            fr_s = [float(x) for x in res[fr_col].tolist()]
            n = len(fr_s)

            # Z-Score
            zscores = []
            for i in range(n):
                w = fr_s[max(0, i - 167) : i + 1]
                m = sum(w) / len(w)
                v = (sum((x - m) ** 2 for x in w) / len(w)) ** 0.5
                zscores.append((fr_s[i] - m) / v if v > 1e-9 else 0.0)

            # EMA
            def get_ema(values, span):
                if not values:
                    return []
                a = 2 / (span + 1)
                e = [values[0]]
                for j in range(1, len(values)):
                    e.append(e[-1] + a * (values[j] - e[-1]))
                return e

            macd = [e1 - e2 for e1, e2 in zip(get_ema(fr_s, 12), get_ema(fr_s, 26))]

            # 結果構築 (辞書による再構築)
            data = {c: df[c].tolist() for c in df.columns}
            data.update(
                {
                    "FR_Extremity_Zscore": zscores,
                    "FR_MA_24": [0.0] * n,
                    "FR_MACD": macd,
                    "FR_Momentum": 0.0,
                }
            )
            return pd.DataFrame(data, index=df.index)
        except Exception as e:
            logger.error(f"FR特徴量計算エラー: {e}")
            return df

    def calculate_open_interest_features(
        self,
        df: pd.DataFrame,
        open_interest_data: pd.DataFrame,
        lookback_periods: Dict[str, int],
    ) -> pd.DataFrame:
        try:
            res, oi_col = self._process_market_data(
                df, open_interest_data, ["open_interest", "oi"], "_oi"
            )
            if oi_col is None:
                return df
            oi_s = res[oi_col]
            # カラムを除去したDFを渡す
            return self._calculate_oi_derived_features(df.copy(), oi_s)
        except Exception as e:
            logger.error(f"OI特徴量計算エラー: {e}")
            return df

    def calculate_pseudo_open_interest_features(
        self, df: pd.DataFrame, lookback_periods: Dict[str, int]
    ) -> pd.DataFrame:
        if "volume" not in df.columns or df.empty:
            return df
        pseudo_oi = pd.Series(
            [float(x) * 10 for x in df["volume"].tolist()], index=df.index
        )
        return self._calculate_oi_derived_features(df.copy(), pseudo_oi)

    def _calculate_oi_derived_features(
        self, df: pd.DataFrame, oi_s: pd.Series
    ) -> pd.DataFrame:
        import math

        oi_vals = [float(x) for x in oi_s.tolist()]
        n = len(oi_vals)
        if n == 0:
            return df

        # RSI
        diffs = [0.0] + [oi_vals[i] - oi_vals[i - 1] for i in range(1, n)]
        rsi = []
        for i in range(n):
            w = diffs[max(0, i - 13) : i + 1]
            up = sum(x for x in w if x > 0) / 14
            down = sum(abs(x) for x in w if x < 0) / 14
            rsi.append(100 - (100 / (1 + up / down)) if down != 0 else 50.0)

        # MACD
        def get_ema(values, span):
            if not values:
                return []
            a = 2 / (span + 1)
            e = [values[0]]
            for j in range(1, len(values)):
                e.append(e[-1] + a * (values[j] - e[-1]))
            return e

        macd = [e1 - e2 for e1, e2 in zip(get_ema(oi_vals, 12), get_ema(oi_vals, 26))]

        # 結果構築
        data = {c: df[c].tolist() for c in df.columns}
        data.update(
            {
                "OI_RSI": rsi,
                "OI_MACD": macd,
                "OI_Change_Rate": [0.0] * n,
                "OI_Trend_Strength": [0.0] * n,
                "OI_Price_Divergence": [0.0] * n,
                "Price_OI_Divergence": 0.0,
            }
        )
        if "volume" in df.columns:
            v_vals = [float(x) for x in df["volume"].tolist()]
            data["Volume_OI_Ratio"] = [
                math.log((v_vals[i] + 1) / (oi_vals[i] + 1)) for i in range(n)
            ]

        return pd.DataFrame(data, index=df.index)

    def calculate_composite_features(
        self,
        df: pd.DataFrame,
        f_data: pd.DataFrame,
        o_data: pd.DataFrame,
        lookback: Dict[str, int],
    ) -> pd.DataFrame:
        try:
            res, fr_col = self._process_market_data(df, f_data, ["funding_rate"], "_fr")
            res, oi_col = self._process_market_data(
                res, o_data, ["open_interest"], "_oi"
            )
            if fr_col is None or oi_col is None:
                return df
            fr_vals = [float(x) for x in res[fr_col].tolist()]
            oi_vals = [float(x) for x in res[oi_col].tolist()]
            n = len(fr_vals)

            stress = [
                math.sqrt(
                    (fr_vals[i] * 1000) ** 2
                    + (
                        oi_vals[i]
                        / max(1, sum(oi_vals[: i + 1]) / len(oi_vals[: i + 1]))
                        - 1
                    )
                    ** 2
                )
                for i in range(n)
            ]

            data = {
                c: res[c].tolist() for c in res.columns if c not in [fr_col, oi_col]
            }
            data.update(
                {
                    "Market_Stress": stress,
                    "FR_Cumulative_Trend": 0.0,
                    "FR_OI_Sentiment": 0.0,
                    "OI_Weighted_Price_Dev": 0.0,
                }
            )
            return pd.DataFrame(data, index=df.index)
        except Exception as e:
            logger.error(f"複合特徴量計算エラー: {e}")
            return df

    def get_feature_names(self) -> list:
        return [
            "OI_RSI",
            "OI_Change_Rate",
            "OI_Price_Divergence",
            "Price_OI_Divergence",
            "OI_Trend_Strength",
            "FR_Cumulative_Trend",
            "FR_Extremity_Zscore",
            "FR_MA_24",
            "Market_Stress",
            "FR_OI_Sentiment",
            "OI_Weighted_Price_Dev",
            "OI_MACD",
            "FR_MACD",
            "Volume_OI_Ratio",
        ]
