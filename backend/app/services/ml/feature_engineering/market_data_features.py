"""
市場データ特徴量計算クラス

ファンディングレート（FR）、建玉残高（OI）データから
市場の歪みや偏りを捉える特徴量を計算します。
"""

import logging
import math
from typing import Any, Dict

import numpy as np
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

        # データの準備 (タイムゾーンを考慮しつつベクトル化)
        # 元のDFのインデックスを保持
        res = df.copy()

        # 結合用の一時的なDFを作成
        df_temp = pd.DataFrame(index=df.index)
        df_temp["__orig_index__"] = df.index
        # タイムゾーンを排除して正規化（結合のため）
        df_temp.index = df_temp.index.tz_localize(None)

        data_temp = data.copy()
        if "timestamp" in data_temp.columns:
            data_temp["timestamp"] = pd.to_datetime(
                data_temp["timestamp"]
            ).dt.tz_localize(None)
            data_temp = data_temp.sort_values("timestamp")
            on_col = "timestamp"
        else:
            data_temp.index = data_temp.index.tz_localize(None)
            data_temp = data_temp.sort_index()
            on_col = None

        # pd.merge_asof を使用して高速に結合
        merged = pd.merge_asof(
            df_temp,
            data_temp[[target_col, on_col] if on_col else [target_col]],
            left_index=True,
            right_on=on_col if on_col else None,
            right_index=True if on_col is None else False,
            direction="backward",
        )

        # 元のインデックスに戻す
        merged.index = merged["__orig_index__"]
        res[target_col + suffix] = merged[target_col].fillna(0.0)

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

            # Seriesとして取得
            fr_series = res[fr_col]

            # Z-Score (ベクトル化)
            roll = fr_series.rolling(window=168, min_periods=1)
            m = roll.mean()
            s = roll.std(ddof=0)  # 元の実装に合わせる
            zscores = ((fr_series - m) / s).fillna(0.0).replace([np.inf, -np.inf], 0.0)

            # EMA
            ema12 = fr_series.ewm(span=12, adjust=False).mean()
            ema26 = fr_series.ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26

            # 結果構築
            res["FR_Extremity_Zscore"] = zscores
            res["FR_MA_24"] = fr_series.rolling(window=24, min_periods=1).mean()
            res["FR_MACD"] = macd
            res["FR_Momentum"] = fr_series.diff().fillna(0.0)

            # 不要な中間カラムを削除
            if fr_col in res.columns:
                del res[fr_col]

            return res
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
        n = len(oi_s)
        if n == 0:
            return df

        # RSI (ベクトル化)
        delta = oi_s.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(50.0)

        # MACD (ベクトル化)
        ema12 = oi_s.ewm(span=12, adjust=False).mean()
        ema26 = oi_s.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26

        # 結果構築
        res = df.copy()
        res["OI_RSI"] = rsi
        res["OI_MACD"] = macd
        res["OI_Change_Rate"] = oi_s.pct_change().fillna(0.0)
        res["OI_Trend_Strength"] = (
            oi_s.rolling(window=20)
            .corr(pd.Series(range(n), index=oi_s.index))
            .fillna(0.0)
        )
        res["OI_Price_Divergence"] = (
            df["close"].pct_change() - res["OI_Change_Rate"]
        ).fillna(0.0)
        res["Price_OI_Divergence"] = (
            df["close"].rolling(window=20).corr(oi_s).fillna(0.0)
        )

        if "volume" in df.columns:
            res["Volume_OI_Ratio"] = np.log((df["volume"] + 1) / (oi_s + 1))

        return res

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

            fr_vals = res[fr_col]
            oi_vals = res[oi_col]

            # Stress (ベクトル化)
            # OIの移動平均を計算
            oi_mean = oi_vals.expanding(min_periods=1).mean()
            stress = np.sqrt(
                (fr_vals * 1000) ** 2 + (oi_vals / oi_mean.replace(0, 1) - 1) ** 2
            )

            res["Market_Stress"] = stress
            res["FR_Cumulative_Trend"] = fr_vals.cumsum()
            res["FR_OI_Sentiment"] = fr_vals * oi_vals.pct_change().fillna(0.0)
            res["OI_Weighted_Price_Dev"] = (
                oi_vals * (df["close"] - df["close"].rolling(window=24).mean())
            ).fillna(0.0)

            # 不要な中間カラムを削除
            if fr_col in res.columns:
                del res[fr_col]
            if oi_col in res.columns:
                del res[oi_col]

            return res
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
