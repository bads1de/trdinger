"""将来ボラティリティ回帰用のターゲット生成サービス。"""

from __future__ import annotations

import logging
from typing import Any, Tuple

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

        prices = pd.to_numeric(ohlcv_norm[price_col], errors="coerce")
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
        aligned_features = aligned_features.ffill().bfill()

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
