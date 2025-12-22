"""
時間的アノマリー特徴量計算モジュール

市場参加者の交代（セッション）、カレンダー効果、周期性など、
時間に起因する市場のアノマリーを特徴量化します。
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .base_feature_calculator import BaseFeatureCalculator

logger = logging.getLogger(__name__)


class TimeAnomalyFeatures(BaseFeatureCalculator):
    """
    時間的アノマリー特徴量計算クラス
    """

    def calculate_features(
        self, df: pd.DataFrame, config: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        時間関連の特徴量を計算

        Args:
            df: 入力データ（DatetimeIndexを持つことを想定）
            config: 設定辞書

        Returns:
            特徴量が追加されたDataFrame
        """
        if not self.validate_input_data(df):
            return df

        # DatetimeIndexの確認
        if not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" in df.columns:
                temp_df = df.set_index("timestamp")
            else:
                logger.warning("DatetimeIndexが見つかりません。時間特徴量を計算できません。")
                return df
        else:
            temp_df = df

        index = temp_df.index
        new_features = {}

        # 1. 周期性特徴量 (Cyclical Encoding)
        # 23時から0時への連続性をモデルに理解させる
        new_features["time_hour_sin"] = np.sin(2 * np.pi * index.hour / 24)
        new_features["time_hour_cos"] = np.cos(2 * np.pi * index.hour / 24)
        new_features["time_day_sin"] = np.sin(2 * np.pi * index.dayofweek / 7)
        new_features["time_day_cos"] = np.cos(2 * np.pi * index.dayofweek / 7)

        # 2. 市場セッション (UTC基準)
        # 仮想通貨は24時間動くが、法定通貨ペアや機関投資家の動きは主要市場に依存する
        hours = index.hour
        # アジア (東京/香港): 00:00 - 09:00 UTC
        new_features["time_session_asia"] = ((hours >= 0) & (hours < 9)).astype(int)
        # 欧州 (ロンドン): 08:00 - 16:00 UTC
        new_features["time_session_europe"] = ((hours >= 8) & (hours < 16)).astype(int)
        # 米国 (ニューヨーク): 13:00 - 21:00 UTC
        new_features["time_session_us"] = ((hours >= 13) & (hours < 21)).astype(int)
        # セッションの重なり (最も流動性が高まりやすい)
        new_features["time_session_overlap"] = (
            ((hours >= 8) & (hours < 9)) | ((hours >= 13) & (hours < 16))
        ).astype(int)

        # 3. カレンダーアノマリー
        # 週末フラグ (土日)
        new_features["time_is_weekend"] = (index.dayofweek >= 5).astype(int)
        
        # 月末フラグ (最後の3日間)
        # 明日の月が変わる、あるいは明後日の月が変わる...
        next_day = index + pd.Timedelta(days=1)
        new_features["time_is_month_end"] = (index.month != next_day.month).astype(int)
        
        # 週初め・週末の特定の動き (月曜の窓埋め、金曜の手仕舞い)
        new_features["time_is_monday"] = (index.dayofweek == 0).astype(int)
        new_features["time_is_friday"] = (index.dayofweek == 4).astype(int)

        # 4. 特定の時間帯のボラティリティ傾向 (ダミーフラグ)
        # 00:00 UTC (日足確定前後)
        new_features["time_is_daily_close"] = (index.hour == 0).astype(int)

        # 5. 相互作用特徴量 (Interaction Features)
        # 時間帯ごとの出来高やボラティリティの強さを捉える
        if "volume" in df.columns:
            # 正規化された出来高（対数変換など）を使用するのが理想だが、ここでは簡易的に比率で扱う
            # ただし、生の出来高はスケールが大きすぎるため、移動平均との比率などを使うのがベター
            # ここでは単純な積算を行い、後段のスケーリングに任せる
            vol = df["volume"]
            new_features["time_interaction_vol_asia"] = vol * new_features["time_session_asia"]
            new_features["time_interaction_vol_europe"] = vol * new_features["time_session_europe"]
            new_features["time_interaction_vol_us"] = vol * new_features["time_session_us"]

        # ボラティリティとの相互作用 (US時間のボラティリティは重要)
        if "close" in df.columns:
            # 簡易ボラティリティ (24時間の標準偏差)
            returns = df["close"].pct_change()
            volatility = returns.rolling(window=24).std().fillna(0)
            
            new_features["time_interaction_volatility_us"] = volatility * new_features["time_session_us"]
            new_features["time_interaction_volatility_overlap"] = volatility * new_features["time_session_overlap"]

        # 6. 適応的ボラティリティ・クラスタリング (Adaptive Volatility)
        # 「普段のこの時間はどうなのか？」との比較
        # 日次周期(24h)での平均ボラティリティとの乖離
        if "close" in df.columns:
            returns = df["close"].pct_change()
            # 現在のボラティリティ (1h)
            current_vol = returns.rolling(window=3).std().fillna(0)
            
            # 過去の同時刻のボラティリティ平均 (簡易的に24hラグの移動平均で代用)
            # 本来は groupby(hour) だが、未来のデータを使わないよう rolling で実装
            # 24h, 48h, 72h... 前のボラティリティの平均
            historical_vol = (
                current_vol.shift(24) + current_vol.shift(48) + current_vol.shift(72)
            ) / 3
            historical_vol = historical_vol.bfill().fillna(0) # ゼロ除算回避のため
            
            # 相対ボラティリティ比 (Relative Volatility Ratio)
            # 1.0 > : 普段より動いている（異常検知）
            new_features["time_adaptive_vol_ratio"] = current_vol / (historical_vol + 1e-9)

        # 7. 市場微細構造プロキシ (Microstructure Proxy)
        # Amihudの非流動性指標 (Illiquidity Ratio): |Return| / Volume
        # 時間帯ごとの「板の薄さ」を捉える
        if "close" in df.columns and "volume" in df.columns:
            abs_return = df["close"].pct_change().abs()
            volume = df["volume"].replace(0, 1) # 0除算回避
            
            amihud_illiquidity = abs_return / volume
            
            # これも対数変換等が必要だが、ここでは相対値として扱う
            # セッションごとの流動性リスクを表す
            new_features["time_micro_illiquidity"] = amihud_illiquidity * 1000000 # スケール調整

        # 8. イベント経過時間 (Time Since Session Start)
        # 開始直後か、終了間際かを連続値で表現
        hours = index.hour
        # 東京開始(0)からの経過
        new_features["time_since_tokyo"] = hours % 24
        # ロンドン開始(8)からの経過
        new_features["time_since_london"] = (hours - 8) % 24
        # NY開始(13)からの経過
        new_features["time_since_ny"] = (hours - 13) % 24

        return self.create_result_dataframe_efficient(df, new_features)
