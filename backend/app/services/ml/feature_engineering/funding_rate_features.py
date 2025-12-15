"""
ファンディングレート特徴量計算

8時間ごとのファンディングレートデータから、
1時間足の価格予測に有効な特徴量を生成します。

【加工方針 2025-11-24】
実データの46%が0.01%(ベースライン)に張り付いているため、
生の比率(ratio)ではなく、以下のように加工して学習効率を高める。
1. bps単位への変換 (x 10000)
2. ベースライン(1.0bps)からの乖離 (deviation) を中心化
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FundingRateFeatureCalculator:
    """
    ファンディングレート特徴量計算クラス
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初期化
        """
        config = config or {}
        self.settlement_interval = config.get("settlement_interval", 8)
        # ベースラインは通常0.01% (0.0001)
        self.baseline_rate = config.get("baseline_rate", 0.0001)

        logger.info(
            f"FundingRateFeatureCalculator初期化: "
            f"決済間隔={self.settlement_interval}h, "
            f"ベースライン金利={self.baseline_rate}"
        )

    def calculate_features(
        self, ohlcv_df: pd.DataFrame, funding_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        全特徴量を計算
        """
        logger.info("ファンディングレート特徴量の計算を開始...")

        if funding_df.empty:
            logger.warning("ファンディングレートデータが空です")
            return ohlcv_df.copy()

        # 1. データの前処理とマージ
        df = self._prepare_data(ohlcv_df, funding_df)

        # 2. 欠損値処理
        df = self._handle_missing_values(df)

        # 3. 数値加工 (bps変換とベースライン乖離) - ここが重要
        df = self._transform_data(df)

        # 4. 特徴量の計算 (加工済みデータを使用)
        df = self._add_basic_rate_features(df)
        df = self._add_time_cycle_features(df)
        df = self._add_momentum_features(df)
        df = self._add_regime_features(df)
        df = self._add_price_interaction_features(df, ohlcv_df)
        df = self._add_market_distortion_features(df, ohlcv_df)

        added_features = [col for col in df.columns if col.startswith("fr_")]
        logger.info(f"ファンディングレート特徴量を追加: {len(added_features)}個")

        return df

    def _prepare_data(
        self, ohlcv_df: pd.DataFrame, funding_df: pd.DataFrame
    ) -> pd.DataFrame:
        """データの前処理とマージ"""
        result = ohlcv_df.copy()

        # timestampカラムの正規化
        if isinstance(result.index, pd.DatetimeIndex):
            if "timestamp" in result.columns:
                result = result.drop(columns=["timestamp"])
            result = result.reset_index()
            if "index" in result.columns and "timestamp" not in result.columns:
                result = result.rename(columns={"index": "timestamp"})
            elif result.index.name == "timestamp":
                pass
            if "timestamp" not in result.columns:
                result = result.rename(columns={result.columns[0]: "timestamp"})

        elif "timestamp" not in result.columns:
            logger.warning(
                "OHLCVデータにtimestampカラムがなく、インデックスもDatetimeIndexではありません"
            )
            return result

        if (
            "timestamp" not in funding_df.columns
            or "funding_rate" not in funding_df.columns
        ):
            logger.warning("ファンディングレートデータに必要なカラムがありません")
            return result

        funding_sorted = funding_df.sort_values("timestamp").copy()
        result["timestamp"] = pd.to_datetime(result["timestamp"])
        funding_sorted["timestamp"] = pd.to_datetime(funding_sorted["timestamp"])

        result = pd.merge_asof(
            result.sort_values("timestamp"),
            funding_sorted[["timestamp", "funding_rate"]],
            on="timestamp",
            direction="backward",
        )

        if "funding_rate" in result.columns:
            result["funding_rate_raw"] = result["funding_rate"]

        result = result.set_index("timestamp")
        return result

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """階層的欠損値処理"""
        if "funding_rate_raw" not in df.columns:
            return df

        df["fr_imputed_flag"] = 0
        missing_mask = df["funding_rate_raw"].isna()

        if missing_mask.sum() > 0:
            # FutureWarning対応: method='linear'は非推奨
            # ファンディングレートは通常8時間ごとに更新されるため、次の更新までは前回の値を維持するのが適切
            df["funding_rate_raw"] = df["funding_rate_raw"].ffill()

            # データリーク防止: bfill()は未来データを使うため使用しない
            # 最初のデータより前の欠損はベースライン値 (0.01%) で埋める
            df["funding_rate_raw"] = df["funding_rate_raw"].fillna(self.baseline_rate)

        return df

    def _transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        データを機械学習しやすい形に加工

        1. fr_bps: ベーシスポイント単位 (x10000)
           - 0.0001 -> 1.0
           - スケールが大きくなり、計算誤差が減る

        2. fr_deviation_bps: ベースライン(0.01%)からの乖離
           - 通常時(0.01%)は 0.0 になる
           - 0を中心とした分布になり、モデルが「異常」を検知しやすくなる
        """
        if "funding_rate_raw" not in df.columns:
            return df

        # bps変換
        df["fr_bps"] = df["funding_rate_raw"] * 10000

        # ベースライン乖離 (0.01% = 1.0bps)
        baseline_bps = self.baseline_rate * 10000
        df["fr_deviation_bps"] = df["fr_bps"] - baseline_bps

        return df

    def _add_basic_rate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """基本金利指標特徴量 (bpsベースに変更)"""
        if "fr_bps" not in df.columns:
            return df

        # 生のrawデータよりも、deviationの方が情報価値が高い可能性があるが、
        # 基本特徴量としてはbps値を使う
        lag_periods = [3]  # fr_lag_1p, 2p は低重要度のため削除
        for lag in lag_periods:
            shift_hours = lag * self.settlement_interval
            df[f"fr_lag_{lag}p"] = df["fr_bps"].shift(shift_hours)

        return df

    def _add_time_cycle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """時間サイクル特徴量"""
        if not isinstance(df.index, pd.DatetimeIndex):
            return df

        df["fr_hours_since_settlement"] = df.index.hour % self.settlement_interval

        cycle_phase = (
            2 * np.pi * df["fr_hours_since_settlement"] / self.settlement_interval
        )
        df["fr_cycle_sin"] = np.sin(cycle_phase)
        df["fr_cycle_cos"] = np.cos(cycle_phase)

        return df

    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """モメンタム特徴量 (deviationベースに変更)"""
        if "fr_deviation_bps" not in df.columns:
            return df

        # 変化量（差分）
        # deviationの変化を見ることで、過熱感の加速/減速を捉える

        span_3 = 3 * self.settlement_interval

        # EMAもdeviationに対してかける（通常時0への回帰を見やすくする）
        df["fr_ema_3periods"] = (
            df["fr_deviation_bps"].ewm(span=span_3, adjust=False).mean()
        )

        return df

    def _add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """レジーム特徴量 (bps基準で判定)"""
        if "fr_bps" not in df.columns:
            return df

        fr_bps = df["fr_bps"]

        # 閾値もbps換算
        # -0.01% -> -1.0
        # 0.00% -> 0.0
        # 0.05% -> 5.0
        # 0.15% -> 15.0

        conditions = [
            fr_bps < -1.0,
            (fr_bps >= -1.0) & (fr_bps < 0.0),
            (fr_bps >= 0.0) & (fr_bps <= 5.0),
            (fr_bps > 5.0) & (fr_bps <= 15.0),
            fr_bps > 15.0,
        ]
        choices = [-2, -1, 0, 1, 2]

        df["fr_regime_encoded"] = np.select(conditions, choices, default=0)

        regime_changes = df["fr_regime_encoded"] != df["fr_regime_encoded"].shift(1)
        df["regime_duration"] = regime_changes.cumsum()
        df["regime_duration"] = df.groupby("regime_duration").cumcount()

        return df

    def _add_price_interaction_features(
        self, df: pd.DataFrame, ohlcv_df: pd.DataFrame
    ) -> pd.DataFrame:
        """価格相互作用特徴量"""
        if "fr_deviation_bps" not in df.columns or "close" not in df.columns:
            return df

        window = 24
        # 乖離と価格の相関（過熱時に価格が上がっているか、逆行しているか）
        df["fr_price_corr_24h"] = (
            df["fr_deviation_bps"].rolling(window=window).corr(df["close"])
        )

        returns = df["close"].pct_change(fill_method=None)
        realized_vol = returns.rolling(window=24).std() * np.sqrt(24)

        realized_vol = realized_vol.replace(0, np.nan)
        # ボラティリティ1単位あたりのFR乖離量

        return df

    def _add_market_distortion_features(
        self, df: pd.DataFrame, ohlcv_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        市場の歪みを捉える特徴量（deviationベースで再設計）
        """
        if "fr_deviation_bps" not in df.columns:
            return df

        # deviation（乖離）を使うことで、ベースラインからの距離を直接扱う
        dev = df["fr_deviation_bps"]

        # 1. 累積乖離エネルギー (Cumulative Deviation)
        # ベースラインから離れた状態がどれだけ続いているか（＝マグマの蓄積）
        # 単純累積だと無限に増えるので、過去N期間のWindow累積にする
        for window in [24, 72]:
            if (
                window == 24
            ):  # fr_cumulative_deviation_24h は低重要度のためコメントアウト
                continue
            df[f"fr_cumulative_deviation_{window}h"] = dev.rolling(window=window).sum()

        # 2. FR移動平均からの乖離（トレンドからの乖離）
        for window in [24, 72, 168]:
            ma = dev.rolling(window=window).mean()
            df[f"fr_zscore_{window}h"] = (dev - ma) / (
                dev.rolling(window=window).std() + 1e-8
            )

        # 3. FR変化の加速度

        # 4. FR極値フラグ（絶対値ベース）
        abs_dev = dev.abs()
        rolling_percentile_95 = abs_dev.rolling(window=720).quantile(0.95)
        df["fr_extreme_flag"] = (abs_dev > rolling_percentile_95).astype(int)

        # 5. FRヒートマップ

        # 6. FR変化の方向性
        fr_diff = dev.diff()
        df["fr_trend_direction"] = np.sign(fr_diff).fillna(0)

        logger.debug("市場歪み特徴量(加工版)を追加しました")

        return df


def validate_funding_rate_data(df: pd.DataFrame) -> bool:
    """ファンディングレートデータの検証"""
    if df.empty:
        logger.warning("ファンディングレートデータが空です")
        return True

    required_cols = ["timestamp", "funding_rate"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"必須カラムが見つかりません: {missing_cols}")

    if "timestamp" in df.columns and len(df) > 1:
        if not df["timestamp"].is_monotonic_increasing:
            raise ValueError("タイムスタンプがソートされていません")

    return True



