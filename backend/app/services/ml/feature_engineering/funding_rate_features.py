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
        if f_df.empty:
            return df.copy()
        
        # 1. データ準備とマージ
        res = df.copy()
        if not isinstance(res.index, pd.DatetimeIndex):
            res = res.set_index("timestamp") if "timestamp" in res.columns else res
        
        f_sorted = f_df.sort_values("timestamp")
        res = pd.merge_asof(
            res.sort_index(), f_sorted[["timestamp", "funding_rate"]].sort_values("timestamp"),
            left_index=True, right_on="timestamp", direction="backward"
        ).set_index("timestamp")

        # 2. 欠損値処理 & 数値加工
        fr_s = res["funding_rate"].ffill().fillna(self.baseline_rate)
        res["fr_bps"] = fr_s * 10000
        res["fr_dev"] = res["fr_bps"] - (self.baseline_rate * 10000)

        # 3. 特徴量追加
        res["fr_lag_3p"] = res["fr_bps"].shift(3 * self.settlement_interval)
        
        h = res.index.hour % self.settlement_interval
        res["fr_cycle_sin"] = np.sin(2 * np.pi * h / self.settlement_interval)
        res["fr_cycle_cos"] = np.cos(2 * np.pi * h / self.settlement_interval)
        
        res["fr_ema_3p"] = res["fr_dev"].ewm(span=3 * self.settlement_interval).mean()
        
        # レジーム判定
        res["fr_regime"] = np.select([
            res["fr_bps"] < -1.0, res["fr_bps"] < 0.0, res["fr_bps"] <= 5.0, res["fr_bps"] <= 15.0
        ], [-2, -1, 0, 1], default=2)

        # 相関とZ-Score
        res["fr_price_corr"] = res["fr_dev"].rolling(24).corr(res["close"])
        for w in [72, 168]:
            m, s = res["fr_dev"].rolling(w).mean(), res["fr_dev"].rolling(w).std()
            res[f"fr_zscore_{w}h"] = (res["fr_dev"] - m) / (s + 1e-8)

        res["fr_extreme"] = (res["fr_dev"].abs() > res["fr_dev"].abs().rolling(720).quantile(0.95)).astype(int)
        res["fr_direction"] = np.sign(res["fr_dev"].diff()).fillna(0)

        return res.drop(columns=["funding_rate"], errors="ignore")


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



