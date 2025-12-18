"""イベント駆動型ラベル生成

イベント駆動型（トリプルバリア）のラベル生成を実装します。
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BarrierProfile:
    """バリアプロファイル設定"""

    base_tp: float
    base_sl: float
    holding_period: int


class EventDrivenLabelGenerator:
    """
    マーケットイベント（価格変動）に基づくラベル生成器

    トリプルバリア法を拡張し、市場のレジーム（ボラティリティ状態）ごとに
    利確・損切・保持期間の係数を動的に適用します。
    HRHP (High Reward High Probability) や LRLP (Low Risk Low Probability)
    といった戦略的ラベルを生成し、学習効率の高いデータセットを作成します。
    """

    REGIME_FACTORS: Dict[Union[int, str], dict] = {
        0: {"tp": 1.15, "sl": 1.0, "holding": 1.1},
        1: {"tp": 0.85, "sl": 0.9, "holding": 0.8},
        2: {"tp": 1.4, "sl": 1.25, "holding": 0.7},
        "default": {"tp": 1.0, "sl": 1.0, "holding": 1.0},
    }

    DEFAULT_PROFILES: Dict[str, BarrierProfile] = {
        "hrhp": BarrierProfile(base_tp=0.035, base_sl=0.02, holding_period=24),
        "lrlp": BarrierProfile(base_tp=0.015, base_sl=0.01, holding_period=12),
    }

    def generate_hrhp_lrlp_labels(
        self,
        market_data: pd.DataFrame,
        regime_labels: Optional[Sequence[int]] = None,
        profile_overrides: Optional[Dict[str, dict]] = None,
    ) -> Tuple[pd.DataFrame, dict]:
        """
        市場レジームに適応したトリプルバリアラベルを生成

        現在の市場のボラティリティやトレンド方向（レジーム）に基づき、
        動的に調整されたバリア（利確/損切幅）を適用して、
        理想的なトレード結果（ラベル）を各データポイントに付与します。

        Args:
            market_data: OHLCV を含むマーケットデータ
            regime_labels: 各足に対応する市場レジーム（0: 低、1: 中、2: 高ボラ等）
            profile_overrides: デフォルトのバリア設定を上書きする辞書

        Returns:
            (生成されたラベルを含む DataFrame, 処理統計や適用プロファイルを含む辞書)
        """

        self._ensure_required_columns(market_data)
        profiles = self._resolve_profiles(profile_overrides)

        regime_array: Optional[np.ndarray] = None
        if regime_labels is not None:
            regime_array = np.asarray(regime_labels, dtype=int)
            if regime_array.size < len(market_data):
                pad_value = regime_array[-1] if regime_array.size > 0 else 0
                pad_count = len(market_data) - regime_array.size
                regime_array = np.concatenate(
                    [regime_array, np.full(pad_count, pad_value, dtype=int)]
                )

        hrhp_labels = self._apply_profile(market_data, profiles["hrhp"], regime_array)
        lrlp_labels = self._apply_profile(market_data, profiles["lrlp"], regime_array)

        index = market_data.index[: len(hrhp_labels)]
        labels_df = pd.DataFrame(
            {"label_hrhp": hrhp_labels, "label_lrlp": lrlp_labels}, index=index
        )

        if regime_array is None:
            labels_df["market_regime"] = 0
            active_regime = 0
        else:
            regime_series = pd.Series(regime_array[: len(hrhp_labels)], index=index)
            labels_df["market_regime"] = regime_series
            active_regime = int(regime_series.iloc[-1])

        regime_profiles = self._summarize_regime_profiles(profiles)
        label_distribution = {
            "hrhp": self._distribution_summary(labels_df["label_hrhp"]),
            "lrlp": self._distribution_summary(labels_df["label_lrlp"]),
        }

        info = {
            "regime_profiles": regime_profiles,
            "label_distribution": label_distribution,
            "active_regime": active_regime,
            "active_threshold_profile": regime_profiles.get(active_regime)
            or regime_profiles.get("default"),
        }

        return labels_df, info

    def _ensure_required_columns(self, market_data: pd.DataFrame) -> None:
        required = {"open", "high", "low", "close"}
        normalized = {col.lower(): col for col in market_data.columns}
        missing = required - set(normalized.keys())
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        for lower_name, original in normalized.items():
            if lower_name in required and lower_name != original:
                market_data[lower_name] = market_data[original]

    def _resolve_profiles(
        self, overrides: Optional[Dict[str, dict]]
    ) -> Dict[str, BarrierProfile]:
        overrides = overrides or {}
        resolved: Dict[str, BarrierProfile] = {}
        for name, base_profile in self.DEFAULT_PROFILES.items():
            override = overrides.get(name, {})
            resolved[name] = BarrierProfile(
                base_tp=float(override.get("base_tp", base_profile.base_tp)),
                base_sl=float(override.get("base_sl", base_profile.base_sl)),
                holding_period=int(
                    override.get("holding_period", base_profile.holding_period)
                ),
            )
        return resolved

    def _apply_profile(
        self,
        market_data: pd.DataFrame,
        profile: BarrierProfile,
        regime_array: Optional[np.ndarray],
    ) -> np.ndarray:
        n = len(market_data) - 1
        if n <= 0:
            return np.array([], dtype=int)

        close, high, low = (
            market_data["close"].values,
            market_data["high"].values,
            market_data["low"].values,
        )
        labels = np.zeros(n, dtype=int)

        # デフォルトファクターを取得
        def_factors = self.REGIME_FACTORS["default"]

        for idx in range(n):
            reg = regime_array[idx] if regime_array is not None else "default"
            f = self.REGIME_FACTORS.get(reg, def_factors)

            tp = profile.base_tp * f["tp"]
            sl = profile.base_sl * f["sl"]
            hld = max(1, int(round(profile.holding_period * f["holding"])))

            labels[idx] = self._first_touch_label(close, high, low, idx, tp, sl, hld)

        return labels

    def _resolve_regime_factors(self, regime_value: Optional[int]) -> Dict[str, float]:
        return self.REGIME_FACTORS.get(regime_value, self.REGIME_FACTORS["default"])

    def _first_touch_label(
        self,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        start_idx: int,
        tp_mult: float,
        sl_mult: float,
        holding: int,
    ) -> int:
        """
        トリプルバリアの「最初の接触（First Touch）」を判定

        水平バリア（利確/損切）または垂直バリア（時間制限）のいずれかに
        価格が最初に到達した時点の結果を正規化して返します。

        Args:
            close: 終値配列
            high: 高値配列
            low: 安値配列
            start_idx: エントリー（基準）のインデックス
            tp_mult: 利確幅（価格に対する割合）
            sl_mult: 損切幅（価格に対する割合）
            holding: 最大保持期間（バー数）

        Returns:
            1: 利確バリアに接触、-1: 損切バリアに接触、0: タイムアウト（時間切れ）
        """
        entry_price = close[start_idx]
        if entry_price <= 0:
            return 0

        up_bar, dn_bar = entry_price * (1 + tp_mult), entry_price * (1 - sl_mult)
        end_idx = min(len(close) - 1, start_idx + holding)

        for i in range(start_idx + 1, end_idx + 1):
            hit_up, hit_dn = high[i] >= up_bar, low[i] <= dn_bar
            if hit_up and hit_dn:
                return -1 if (dn_bar - low[i]) >= (high[i] - up_bar) else 1
            if hit_up:
                return 1
            if hit_dn:
                return -1
        return 0

    def _summarize_regime_profiles(
        self, profiles: Dict[str, BarrierProfile]
    ) -> Dict[Any, dict]:
        summary: Dict[Any, dict] = {}
        for regime_key in [0, 1, 2, "default"]:
            factors = self._resolve_regime_factors(
                regime_key if isinstance(regime_key, int) else None
            )
            summary[regime_key] = {
                "hrhp": self._profile_stats(profiles["hrhp"], factors),
                "lrlp": self._profile_stats(profiles["lrlp"], factors),
            }
        return summary

    def _profile_stats(
        self, profile: BarrierProfile, factors: Dict[str, float]
    ) -> dict:
        return {
            "take_profit": round(profile.base_tp * factors["tp"], 6),
            "stop_loss": round(profile.base_sl * factors["sl"], 6),
            "holding_period": max(
                1, int(round(profile.holding_period * factors["holding"]))
            ),
        }

    def _distribution_summary(self, labels: pd.Series) -> dict:
        total = len(labels)
        if total == 0:
            return {"positive_ratio": 0.0, "negative_ratio": 0.0, "neutral_ratio": 0.0}
        pos_ratio = float((labels == 1).sum()) / total
        neg_ratio = float((labels == -1).sum()) / total
        neu_ratio = float((labels == 0).sum()) / total
        return {
            "positive_ratio": pos_ratio,
            "negative_ratio": neg_ratio,
            "neutral_ratio": neu_ratio,
        }
