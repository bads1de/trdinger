"""DRLポリシー連携アダプタ"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DRLPolicyAdapter:
    """強化学習ポリシーとハイブリッドGAを接続するためのシンプルなアダプタ"""

    def __init__(
        self,
        policy_type: str = "ppo",
        policy_config: Optional[Dict[str, Any]] = None,
        predict_fn: Optional[Callable[[pd.DataFrame], Dict[str, float]]] = None,
    ) -> None:
        self.policy_type = policy_type
        self.policy_config = policy_config or {}
        self._custom_predict = predict_fn

        window = int(self.policy_config.get("momentum_window", 12))
        self._momentum_window = max(4, window)
        self._volatility_window = int(self.policy_config.get("volatility_window", 12))
        self._volatility_window = max(4, self._volatility_window)

    def predict_signals(self, features_df: pd.DataFrame) -> Dict[str, float]:
        """特徴量から up / down / range の確率を推定"""

        if self._custom_predict is not None:
            try:
                return self._normalise(self._custom_predict(features_df))
            except Exception as exc:
                logger.warning(
                    "カスタムDRL予測が失敗したためデフォルト推論を使用します: %s", exc
                )

        if features_df is None or features_df.empty:
            return {"up": 1 / 3, "down": 1 / 3, "range": 1 / 3}

        close_series = self._extract_series(features_df, "close")
        if close_series is None:
            return {"up": 1 / 3, "down": 1 / 3, "range": 1 / 3}

        returns = close_series.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        if returns.empty:
            return {"up": 1 / 3, "down": 1 / 3, "range": 1 / 3}

        momentum_window = min(self._momentum_window, len(returns))
        volatility_window = min(self._volatility_window, len(returns))

        momentum = returns.tail(momentum_window).mean()
        volatility = returns.tail(volatility_window).std(ddof=0)

        scale = float(self.policy_config.get("momentum_scale", 50.0))
        vol_scale = float(self.policy_config.get("volatility_scale", 20.0))

        up_score = np.exp(np.clip(momentum * scale, -50.0, 50.0))
        down_score = np.exp(np.clip(-momentum * scale, -50.0, 50.0))
        range_score = np.exp(np.clip(-volatility * vol_scale, -50.0, 50.0))

        scores = {"up": up_score, "down": down_score, "range": range_score}
        return self._normalise(scores)

    @staticmethod
    def _extract_series(features_df: pd.DataFrame, column: str) -> Optional[pd.Series]:
        if column in features_df.columns:
            series = features_df[column]
            if np.issubdtype(series.dtype, np.number):
                return series.astype(float)
        if column in features_df.index.names:
            try:
                return features_df.index.get_level_values(column).astype(float)
            except Exception:
                return None
        return None

    @staticmethod
    def _normalise(signals: Dict[str, float]) -> Dict[str, float]:
        up = float(signals.get("up", 0.0))
        down = float(signals.get("down", 0.0))
        range_score = float(signals.get("range", 0.0))

        total = up + down + range_score
        if total <= 0.0 or not np.isfinite(total):
            return {"up": 1 / 3, "down": 1 / 3, "range": 1 / 3}

        normalised = {
            "up": up / total,
            "down": down / total,
            "range": range_score / total,
        }
        return normalised
