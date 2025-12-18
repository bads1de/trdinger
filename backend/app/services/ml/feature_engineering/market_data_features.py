"""
市場データ特徴量計算クラス

ファンディングレート（FR）、建玉残高（OI）データから
市場の歪みや偏りを捉える特徴量を計算します。
"""

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

from .base_feature_calculator import BaseFeatureCalculator

logger = logging.getLogger(__name__)


class MarketDataFeatureCalculator(BaseFeatureCalculator):
    """
    市場データ特徴量計算クラス

    ファンディングレート、建玉残高データから特徴量を計算します。
    """

    def __init__(self):
        """初期化"""
        super().__init__()

    def calculate_features(
        self, df: pd.DataFrame, config: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        市場データ特徴量を計算（BaseFeatureCalculatorの抽象メソッド実装）

        Args:
            df: OHLCV価格データ
            config: 計算設定（lookback_periods、funding_rate_data、open_interest_dataを含む）

        Returns:
            市場データ特徴量が追加されたDataFrame
        """
        lookback_periods = config.get("lookback_periods", {})
        funding_rate_data = config.get("funding_rate_data")
        open_interest_data = config.get("open_interest_data")

        result_df = df

        if funding_rate_data is not None:
            result_df = self.calculate_funding_rate_features(
                result_df, funding_rate_data, lookback_periods
            )

        if open_interest_data is not None:
            result_df = self.calculate_open_interest_features(
                result_df, open_interest_data, lookback_periods
            )

        if funding_rate_data is not None and open_interest_data is not None:
            result_df = self.calculate_composite_features(
                result_df, funding_rate_data, open_interest_data, lookback_periods
            )
            # 市場ダイナミクス特徴量は削除

        return result_df

    # calculate_market_dynamics_features は削除されました

    def _process_market_data(
        self,
        df: pd.DataFrame,
        data: pd.DataFrame,
        column_candidates: list[str],
        suffix: str,
    ) -> tuple[pd.DataFrame, str | None]:
        """市場データをマージし、カラムを特定して前処理を行う"""
        if "timestamp" in data.columns:
            data = data.set_index("timestamp")

        # タイムゾーン調整
        if df.index.tz is not None and data.index.tz is None:
            data.index = data.index.tz_localize("UTC").tz_convert(df.index.tz)
        elif df.index.tz is None and data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        elif df.index.tz is not None and data.index.tz is not None:
            data.index = data.index.tz_convert(df.index.tz)

        merged = df.join(data, how="left", rsuffix=suffix)
        target_column = next((c for c in column_candidates if c in merged.columns), None)
        if target_column:
            merged[target_column] = merged[target_column].ffill()

        return merged, target_column

    def calculate_funding_rate_features(
        self,
        df: pd.DataFrame,
        funding_rate_data: pd.DataFrame,
        lookback_periods: Dict[str, int],
    ) -> pd.DataFrame:
        """ファンディングレート特徴量を計算"""
        try:
            res, fr_col = self._process_market_data(
                df, funding_rate_data, ["funding_rate", "fundingRate", "rate"], "_fr"
            )
            if fr_col is None:
                return df

            # FR Extremity (Z-Score)
            fr_s = res[fr_col]
            fr_m = fr_s.rolling(168, min_periods=1).mean()
            fr_v = fr_s.rolling(168, min_periods=1).std().replace(0, np.nan)
            res["FR_Extremity_Zscore"] = ((fr_s - fr_m) / fr_v).fillna(0.0)

            # FR Momentum & MA & MACD
            fr_cr = fr_s.pct_change(8)
            res["FR_Momentum"] = (fr_cr - fr_cr.shift(8)).fillna(0.0)
            res["FR_MA_24"] = fr_s.rolling(24, min_periods=1).mean().fillna(0.0)
            res["FR_MACD"] = fr_s.ewm(span=12).mean() - fr_s.ewm(span=26).mean()

            return res.drop(columns=[fr_col])
        except Exception as e:
            logger.error(f"FR特徴量計算エラー: {e}")
            return df

    def calculate_open_interest_features(
        self,
        df: pd.DataFrame,
        open_interest_data: pd.DataFrame,
        lookback_periods: Dict[str, int],
    ) -> pd.DataFrame:
        """建玉残高特徴量を計算"""
        try:
            res, oi_col = self._process_market_data(
                df, open_interest_data, ["open_interest", "openInterest", "oi", "open_interest_value"], "_oi"
            )
            if oi_col is None:
                return df

            oi_s = res[oi_col]
            res = self._calculate_oi_derived_features(res.drop(columns=[oi_col]), oi_s)
            return res
        except Exception as e:
            logger.error(f"OI特徴量計算エラー: {e}")
            return df

    def calculate_pseudo_open_interest_features(
        self, df: pd.DataFrame, lookback_periods: Dict[str, int]
    ) -> pd.DataFrame:
        """建玉残高疑似特徴量を生成"""
        if "volume" not in df.columns:
            return df
        try:
            pseudo_oi = df["volume"].rolling(24).mean() * 10
            return self._calculate_oi_derived_features(df.copy(), pseudo_oi)
        except Exception as e:
            logger.error(f"疑似OI生成エラー: {e}")
            return df

    def _calculate_oi_derived_features(
        self, df: pd.DataFrame, oi_s: pd.Series
    ) -> pd.DataFrame:
        """建玉残高から派生特徴量を計算"""
        res = df
        # OI RSI
        d = oi_s.diff()
        gain, loss = d.clip(lower=0).rolling(14).mean(), d.clip(upper=0).abs().rolling(14).mean()
        res["OI_RSI"] = (100 - (100 / (1 + gain / loss.replace(0, np.nan)))).fillna(50.0)

        # Ratio & Change
        res["Volume_OI_Ratio"] = np.log((df["volume"] + 1) / (oi_s + 1)).fillna(0) if "volume" in df else 0.0
        res["OI_Change_Rate"] = oi_s.pct_change().fillna(0)
        res["OI_Price_Divergence"] = df["close"].rolling(14).corr(oi_s).fillna(0)
        res["Price_OI_Divergence"] = res["OI_Price_Divergence"]

        # Trend & Correlation
        oi_m24 = oi_s.rolling(24).mean()
        res["OI_Trend_Strength"] = ((oi_s - oi_m24) / oi_m24).fillna(0)
        res["OI_Volume_Correlation"] = df["volume"].rolling(24).corr(oi_s).fillna(0) if "volume" in df else 0.0

        # Momentum & Risk
        p_vol = df["close"].pct_change().abs().rolling(14).mean()
        res["OI_Momentum_Ratio"] = (oi_s.pct_change().abs().rolling(14).mean() / p_vol).fillna(0)
        
        oi_m168 = oi_s.rolling(168, min_periods=1).mean()
        oi_s168 = oi_s.rolling(168, min_periods=1).std().replace(0, np.nan)
        res["OI_Liquidation_Risk"] = (((oi_s - oi_m168) / oi_s168).fillna(0) * p_vol).fillna(0)
        res["Liquidation_Risk"] = res["OI_Liquidation_Risk"]

        # MACD & BB
        ema12, ema26 = oi_s.ewm(span=12).mean(), oi_s.ewm(span=26).mean()
        res["OI_MACD"] = ema12 - ema26
        res["OI_MACD_Hist"] = res["OI_MACD"] - res["OI_MACD"].ewm(span=9).mean()
        
        oi_m20, oi_s20 = oi_s.rolling(20).mean(), oi_s.rolling(20).std()
        res["OI_BB_Position"] = ((oi_s - (oi_m20 - 2*oi_s20)) / (4*oi_s20)).fillna(0.5)
        res["OI_BB_Width"] = (4*oi_s20 / oi_m20).fillna(0)

        return res

    def calculate_composite_features(
        self,
        df: pd.DataFrame,
        funding_rate_data: pd.DataFrame,
        open_interest_data: pd.DataFrame,
        lookback_periods: Dict[str, int],
    ) -> pd.DataFrame:
        """複合特徴量を計算（FR + OI）"""
        try:
            # データの準備
            res, fr_col = self._process_market_data(df, funding_rate_data, ["funding_rate", "fundingRate", "rate"], "_fr")
            res, oi_col = self._process_market_data(res, open_interest_data, ["open_interest", "openInterest", "oi"], "_oi")
            if fr_col is None or oi_col is None:
                return df

            fr_s, oi_s = res[fr_col], res[oi_col]
            
            # FR Cumulative Trend
            fr_c24 = fr_s.rolling(24).sum()
            fr_cm = fr_c24.rolling(168, min_periods=1).mean()
            res["FR_Cumulative_Trend"] = ((fr_c24 - fr_cm) / fr_cm).fillna(0) # 簡略化

            # OI Normalized
            oi_m, oi_v = oi_s.rolling(168, min_periods=1).mean(), oi_s.rolling(168, min_periods=1).std().replace(0, np.nan)
            oi_norm = ((oi_s - oi_m) / oi_v).fillna(0)

            # Market Stress
            fr_norm = res["FR_Extremity_Zscore"] if "FR_Extremity_Zscore" in res else 0.0
            res["Market_Stress"] = np.sqrt(fr_norm**2 + oi_norm**2)

            # Sentiment & Risk
            res["FR_OI_Sentiment"] = (np.sign(fr_s).replace(0, 1) * oi_s.pct_change().fillna(0)).rolling(8).mean().fillna(0)
            res["Liquidation_Risk"] = (df["close"].pct_change().abs().rolling(4).mean() * oi_norm).fillna(0)

            # OI Weighted Price & Efficiency
            oi_w24 = oi_s.rolling(24).sum()
            oi_wp = (df["close"] * oi_s).rolling(24).sum() / oi_w24
            res["OI_Weighted_Price_Dev"] = (df["close"] - oi_wp) / oi_wp
            res["FR_Volatility"] = fr_s.rolling(24).std().fillna(0)
            res["OI_Trend_Efficiency"] = (df["close"].pct_change().abs() / oi_s.pct_change().abs().replace(0, np.nan)).fillna(0)
            
            if "volume" in res:
                res["Volume_OI_Ratio"] = (res["volume"] / oi_s).fillna(0)

            return res.drop(columns=[fr_col, oi_col], errors="ignore")
        except Exception as e:
            logger.error(f"複合特徴量計算エラー: {e}")
            return df

    def get_feature_names(self) -> list:
        """
        生成される市場データ特徴量名のリストを取得

        Returns:
            特徴量名のリスト
        """
        return [
            # 建玉残高特徴量
            "OI_RSI",
            "OI_Change_Rate",
            "OI_Price_Divergence",  # Group A: 1
            "Price_OI_Divergence",  # Alias
            "OI_Trend_Strength",  # Group A: 2
            "OI_Volume_Correlation",  # Group A: 3
            "OI_Momentum_Ratio",  # Group A: 4
            "OI_Liquidation_Risk",  # Group A: 5
            "Liquidation_Risk",  # Alias
            # 複合特徴量
            "FR_Cumulative_Trend",
            "FR_Extremity_Zscore",
            "FR_Momentum",
            "FR_MA_24",
            "Market_Stress",
            "FR_OI_Sentiment",
            "OI_Weighted_Price_Dev",
            "FR_Volatility",
            "OI_Trend_Efficiency",
            "Volume_OI_Ratio",
            # Group B
            "OI_MACD",
            "OI_MACD_Hist",
            "OI_BB_Position",
            "OI_BB_Width",
            "FR_MACD",
        ]
