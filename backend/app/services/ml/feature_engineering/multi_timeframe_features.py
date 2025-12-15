"""
Multi-Timeframe Features

複数時間足の情報を統合し、より広いコンテキストを提供する特徴量。
上位時間足のトレンドとの一致が強力な予測因子となることが実証されています。
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd


class MultiTimeframeFeatureCalculator:
    """
    複数時間足の特徴量を計算するクラス

    1時間足データから4時間足・1日足の情報を合成し、
    時間足間のアライメント（方向一致）を検出します。
    """

    def calculate_features(
        self, df: pd.DataFrame, config: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Multi-Timeframe特徴量を計算

        Args:
            df: 1時間足のOHLCV DataFrame
            config: 設定（オプション）

        Returns:
            Multi-Timeframe特徴量を含むDataFrame
        """
        result = pd.DataFrame(index=df.index)

        if len(df) < 200:  # 最低限のデータ量が必要
            return result

        # === 4時間足の情報を合成 ===
        df_4h = self._resample_to_timeframe(df, "4h")

        # 【重要】データリーク防止のためのシフト処理
        # resampleしたデータ（index=00:00）は00:00-04:00のデータを含むため、
        # 決定するのは04:00時点。したがって1期間（4時間）シフトして
        # データが利用可能になる時刻に合わせる必要がある。
        df_4h = df_4h.shift(1)

        # 4時間足のトレンド指標
        df_4h["SMA_50"] = df_4h["close"].rolling(50).mean()
        df_4h["SMA_200"] = df_4h["close"].rolling(200).mean()
        df_4h["EMA_20"] = df_4h["close"].ewm(span=20).mean()

        # 4時間足トレンド方向
        df_4h["trend_direction"] = np.where(
            df_4h["SMA_50"] > df_4h["SMA_200"],
            1,
            np.where(df_4h["SMA_50"] < df_4h["SMA_200"], -1, 0),
        )

        # 4時間足トレンド強度（SMA間の距離）
        df_4h["trend_strength"] = (df_4h["SMA_50"] - df_4h["SMA_200"]) / df_4h[
            "SMA_200"
        ]

        # 4時間足のRSI
        df_4h["RSI"] = self._calculate_rsi(df_4h["close"], period=14)

        # === 1日足の情報を合成 ===
        df_1d = self._resample_to_timeframe(df, "1D")

        # 【重要】データリーク防止のためのシフト処理
        df_1d = df_1d.shift(1)

        # 1日足のトレンド指標
        df_1d["SMA_50"] = df_1d["close"].rolling(50).mean()
        df_1d["SMA_200"] = df_1d["close"].rolling(200).mean()

        # 1日足トレンド方向
        df_1d["trend_direction"] = np.where(
            df_1d["SMA_50"] > df_1d["SMA_200"],
            1,
            np.where(df_1d["SMA_50"] < df_1d["SMA_200"], -1, 0),
        )

        # 1日足トレンド強度
        df_1d["trend_strength"] = (df_1d["SMA_50"] - df_1d["SMA_200"]) / df_1d[
            "SMA_200"
        ]

        # === 1時間足のトレンド方向 ===
        df["SMA_50_1h"] = df["close"].rolling(50).mean()
        df["SMA_200_1h"] = df["close"].rolling(200).mean()
        df["trend_direction_1h"] = np.where(
            df["SMA_50_1h"] > df["SMA_200_1h"],
            1,
            np.where(df["SMA_50_1h"] < df["SMA_200_1h"], -1, 0),
        )

        # === 1時間足インデックスに統合 ===
        # 4時間足データを1時間足に展開
        df_4h_expanded = df_4h.reindex(df.index, method="ffill")
        df_1d_expanded = df_1d.reindex(df.index, method="ffill")

        # === Multi-Timeframe特徴量を生成 ===

        # 1. 4時間足トレンド方向
        result["HTF_4h_Trend_Direction"] = df_4h_expanded["trend_direction"]

        # 2. 4時間足トレンド強度
        result["HTF_4h_Trend_Strength"] = df_4h_expanded["trend_strength"]

        # 3. 4時間足RSI
        result["HTF_4h_RSI"] = df_4h_expanded["RSI"]

        # 4. 1日足トレンド方向
        result["HTF_1d_Trend_Direction"] = df_1d_expanded["trend_direction"]

        # 5. 1日足トレンド強度
        result["HTF_1d_Trend_Strength"] = df_1d_expanded["trend_strength"]

        # 6. Timeframe Alignment Score（時間足アライメント）
        # 全時間足が同じ方向を向いているか
        alignment_score = (
            (df["trend_direction_1h"] == df_4h_expanded["trend_direction"]).astype(
                float
            )
            + (df["trend_direction_1h"] == df_1d_expanded["trend_direction"]).astype(
                float
            )
            + (
                df_4h_expanded["trend_direction"] == df_1d_expanded["trend_direction"]
            ).astype(float)
        ) / 3.0  # 0.0～1.0の範囲に正規化

        result["Timeframe_Alignment_Score"] = alignment_score

        # 7. Timeframe Alignment Direction（アライメントの方向）
        # 全時間足が一致している場合、その方向を返す
        all_bullish = (
            (df["trend_direction_1h"] == 1)
            & (df_4h_expanded["trend_direction"] == 1)
            & (df_1d_expanded["trend_direction"] == 1)
        )

        all_bearish = (
            (df["trend_direction_1h"] == -1)
            & (df_4h_expanded["trend_direction"] == -1)
            & (df_1d_expanded["trend_direction"] == -1)
        )

        result["Timeframe_Alignment_Direction"] = np.where(
            all_bullish, 1, np.where(all_bearish, -1, 0)
        )

        # 8. Higher Timeframe Divergence（上位足との乖離）
        # 1時間足と4時間足の方向が逆 = 逆張りシグナル
        result["HTF_4h_Divergence"] = (
            df["trend_direction_1h"] != df_4h_expanded["trend_direction"]
        ).astype(float)

        result["HTF_1d_Divergence"] = (
            df["trend_direction_1h"] != df_1d_expanded["trend_direction"]
        ).astype(float)

        # 9. 価格の4時間足SMAからの距離
        result["Price_Distance_From_4h_SMA50"] = (
            df["close"] - df_4h_expanded["SMA_50"]
        ) / df_4h_expanded["SMA_50"]

        # 10. 価格の1日足SMAからの距離
        result["Price_Distance_From_1d_SMA50"] = (
            df["close"] - df_1d_expanded["SMA_50"]
        ) / df_1d_expanded["SMA_50"]

        # 欠損値・inf値の処理
        result = result.replace([np.inf, -np.inf], np.nan)
        result = result.fillna(0)

        return result

    def _resample_to_timeframe(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        1時間足データを指定時間足にリサンプル

        Args:
            df: 1時間足OHLCV DataFrame
            timeframe: 目標時間足（'4h', '1D' など）

        Returns:
            リサンプルされたDataFrame
        """
        resampled = pd.DataFrame()

        resampled["open"] = df["open"].resample(timeframe).first()
        resampled["high"] = df["high"].resample(timeframe).max()
        resampled["low"] = df["low"].resample(timeframe).min()
        resampled["close"] = df["close"].resample(timeframe).last()
        resampled["volume"] = df["volume"].resample(timeframe).sum()

        # 欠損値を前方埋め
        resampled = resampled.fillna(method="ffill")

        return resampled

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """RSIを計算"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        return rsi



