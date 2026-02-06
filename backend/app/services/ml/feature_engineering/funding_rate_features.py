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

    def calculate_features(self, df: pd.DataFrame, f_df: pd.DataFrame) -> pd.DataFrame:
        """全特徴量を計算"""
        if f_df is None or f_df.empty:
            return df.copy()

        # 1. データ準備とマージ (pd.merge_asofを使用)
        res = df.copy()
        if not isinstance(res.index, pd.DatetimeIndex):
            if "timestamp" in res.columns:
                res = res.set_index("timestamp")
            else:
                # インデックスが日付でない場合は何もしない（通常はDatetimeIndexを想定）
                return df.copy()

        # マージ用の一時的なデータフレームを作成
        df_temp = pd.DataFrame(index=res.index)
        df_temp["__orig_index__"] = res.index
        df_temp.index = df_temp.index.tz_localize(None)

        f_temp = f_df.sort_values("timestamp").copy()
        f_temp["timestamp"] = pd.to_datetime(f_temp["timestamp"]).dt.tz_localize(None)

        # 高速マージ
        merged = pd.merge_asof(
            df_temp,
            f_temp[["timestamp", "funding_rate"]],
            left_index=True,
            right_on="timestamp",
            direction="backward",
        )
        merged.index = merged["__orig_index__"]
        res["funding_rate"] = merged["funding_rate"].fillna(self.baseline_rate)

        # 2. 数値加工 (ベクトル化)
        res["fr_bps"] = res["funding_rate"] * 10000
        res["fr_dev"] = res["fr_bps"] - (self.baseline_rate * 10000)

        # 3. 特徴量追加
        # ラグ
        res["fr_lag_3p"] = res["fr_bps"].shift(3 * self.settlement_interval)

        # 周期
        h = res.index.hour % self.settlement_interval
        res["fr_cycle_sin"] = np.sin(2 * np.pi * h / self.settlement_interval)
        res["fr_cycle_cos"] = np.cos(2 * np.pi * h / self.settlement_interval)

        # EMA (ベクトル化)
        span = 3 * self.settlement_interval
        res["fr_ema_3p"] = res["fr_dev"].ewm(span=span, adjust=False).mean()

        # レジーム判定
        res["fr_regime"] = np.select(
            [
                res["fr_bps"] < -1.0,
                res["fr_bps"] < 0.0,
                res["fr_bps"] <= 5.0,
                res["fr_bps"] <= 15.0,
            ],
            [-2, -1, 0, 1],
            default=2,
        )

        # 相関とZ-Score (ベクトル化)
        close_series = res["close"] if "close" in res.columns else None
        for w in [72, 168]:
            roll = res["fr_bps"].rolling(window=w, min_periods=1)
            # Z-Score
            m = roll.mean()
            s = roll.std(ddof=0)
            res[f"fr_zscore_{w}h"] = (
                ((res["fr_bps"] - m) / s).fillna(0.0).replace([np.inf, -np.inf], 0.0)
            )

            # 相関 (価格がある場合)
            if close_series is not None:
                res[f"fr_price_corr_{w}h"] = (
                    res["fr_bps"]
                    .rolling(window=w, min_periods=w // 2)
                    .corr(close_series)
                ).fillna(0.0)

        # 元の実装にあるカラム名との互換性
        if "fr_price_corr_72h" in res.columns:
            res["fr_price_corr"] = res["fr_price_corr_72h"]
        else:
            res["fr_price_corr"] = 0.0

        res["fr_extreme"] = np.where(res["fr_bps"].abs() > 10.0, 1, 0)
        res["fr_direction"] = np.sign(res["fr_dev"].diff()).fillna(0)

        # 不要なカラムを削除して返す
        if "funding_rate" in res.columns:
            res = res.drop(columns=["funding_rate"])

        return res


def validate_funding_rate_data(df: pd.DataFrame) -> bool:
    """ファンディングレートデータの検証"""
    if df is None:
        raise ValueError("データがNoneです")

    required_cols = ["timestamp", "funding_rate"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"必須カラムが見つかりません: {missing_cols}")

    if df.empty:
        logger.warning("ファンディングレートデータが空です")
        return True

    if "timestamp" in df.columns and len(df) > 1:
        if not df["timestamp"].is_monotonic_increasing:
            raise ValueError("タイムスタンプがソートされていません")

    return True
