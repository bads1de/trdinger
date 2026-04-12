"""将来ボラティリティ回帰用のターゲット生成サービス。"""

from __future__ import annotations

import logging
from typing import Any, cast, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class VolatilityTargetService:
    """forward window から future_log_realized_vol を生成する。"""

    def prepare_targets(
        self,
        features_df: pd.DataFrame,
        ohlcv_df: pd.DataFrame,
        **training_params: Any,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        市場データから回帰学習用のターゲット（将来の対数実現ボラティリティ）を生成し、特徴量と同期させます。

        このメソッドは以下のプロセスでターゲットを算出します：
        1. **対数リターンの算出**: 価格データの変化率を対数スケールで計算。
        2. **将来ボラティリティの推定**: 指定された `prediction_horizon`（ホライゾン）期間内の
           対数リターンの二乗和の平方根（実現ボラティリティ）を算出。
        3. **対数変換**: ボラティリティ分布の歪みを抑え、モデルが学習しやすくするために対数変換を適用。
           $Target_t = \\ln(\\sqrt{\\sum_{i=1}^{n} r_{t+i}^2} + \\epsilon)$
        4. **インデックス同期**: 特徴量データと計算された将来ターゲットのタイムスタンプを厳密に一致させ、
           かつ未来情報のリーク（前方参照）が発生しないようにホライゾン分だけシフトして結合。

        Args:
            features_df (pd.DataFrame): 算出済みの特徴量。
            ohlcv_df (pd.DataFrame): ターゲット算出の基準となる価格データ（DatetimeIndex必須）。
            **training_params: 
                - `prediction_horizon` (int): 予測対象とする将来のバー数。
                - `price_column` (str): 計算に使用する価格カラム名（デフォルト: "close"）。

        Returns:
            Tuple[pd.DataFrame, pd.Series]: (クリーニング済みの特徴量, 同期されたターゲットラベル) のタプル。

        Raises:
            ValueError: データが不足している場合、またはホライゾン設定が不正な場合。
        """
        if features_df is None or features_df.empty:
            raise ValueError("特徴量データが空です")
        if ohlcv_df is None or ohlcv_df.empty:
            raise ValueError("OHLCVデータが空です")

        horizon = int(
            training_params.get(
                "prediction_horizon",
                training_params.get("horizon_n", 1),
            )
        )
        if horizon < 1:
            raise ValueError("prediction_horizon は 1 以上である必要があります")

        price_col = str(training_params.get("price_column", "close")).lower()
        ohlcv_norm = ohlcv_df.copy()
        ohlcv_norm.columns = [str(column).lower() for column in ohlcv_norm.columns]
        if price_col not in ohlcv_norm.columns:
            raise ValueError(f"価格カラムが存在しません: {price_col}")

        prices: pd.Series = cast(pd.Series, pd.to_numeric(ohlcv_norm[price_col], errors="coerce"))
        # np.log(prices) の結果が Series であることを明示して .diff() を使用
        log_returns = pd.Series(np.log(prices), index=prices.index).diff()

        forward_variance = pd.Series(0.0, index=log_returns.index, dtype=float)
        valid_forward_window = pd.Series(True, index=log_returns.index, dtype=bool)
        for offset in range(1, horizon + 1):
            shifted_returns = log_returns.shift(-offset)
            forward_variance = forward_variance.add(
                shifted_returns.pow(2),
                fill_value=0.0,
            )
            valid_forward_window &= shifted_returns.notna()

        future_rv = np.sqrt(forward_variance)
        future_log_rv = pd.Series(np.log(future_rv + 1e-8), index=future_rv.index)  # type: ignore[reportAttributeAccessIssue]
        future_log_rv = future_log_rv.where(valid_forward_window)

        aligned_features = features_df.copy()
        aligned_features = aligned_features.replace([np.inf, -np.inf], np.nan)
        # 先頭の欠損を未来値で埋めるとリークになるため、過去方向のみで補完する。
        aligned_features = aligned_features.ffill()

        common_index = aligned_features.index.intersection(future_log_rv.index)
        aligned_features = aligned_features.loc[common_index]
        aligned_targets = future_log_rv.loc[common_index]

        valid_mask = aligned_targets.notna()
        valid_mask &= np.isfinite(aligned_targets.to_numpy())
        valid_mask &= aligned_features.notna().all(axis=1)

        cleaned_features = aligned_features.loc[valid_mask].copy()
        cleaned_targets = aligned_targets.loc[valid_mask].astype(float)

        if cleaned_features.empty or cleaned_targets.empty:
            raise ValueError("future_log_realized_vol の生成結果が空です")

        logger.info(
            "将来ボラティリティターゲット生成完了: samples=%s horizon=%s",
            len(cleaned_targets),
            horizon,
        )
        cleaned_targets.name = "future_log_realized_vol"
        return cleaned_features, cleaned_targets
