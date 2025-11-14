"""
ファンディングレート特徴量計算

8時間ごとのファンディングレートデータから、
1時間足の価格予測に有効なTier 1特徴量（15個）を生成します。

特徴量設計:
- 基本金利指標（4個）: raw, lag_1p, lag_2p, lag_3p
- 時間サイクル（3個）: hours_since_settlement, cycle_sin, cycle_cos
- モメンタム（3個）: velocity, ema_3periods, ema_7periods
- レジーム（2個）: regime_encoded, regime_duration
- 価格相互作用（2個）: price_corr_24h, volatility_adjusted
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FundingRateFeatureCalculator:
    """
    ファンディングレート特徴量計算クラス

    8時間ごとのファンディングレートデータから、
    1時間足の価格予測に有効な特徴量を生成します。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初期化

        Args:
            config: 特徴量計算の設定
                - settlement_interval: 決済間隔（デフォルト: 8時間）
                - baseline_rate: ベースライン金利（デフォルト: 0.0001 = 0.01%）
        """
        config = config or {}
        self.settlement_interval = config.get("settlement_interval", 8)
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

        Args:
            ohlcv_df: OHLCVデータ（1時間足）
            funding_df: ファンディングレートデータ（8時間ごと）

        Returns:
            特徴量を追加したDataFrame
        """
        logger.info("ファンディングレート特徴量の計算を開始...")

        if funding_df.empty:
            logger.warning("ファンディングレートデータが空です")
            return ohlcv_df.copy()

        # 1. データの前処理とマージ
        df = self._prepare_data(ohlcv_df, funding_df)

        # 2. 欠損値処理
        df = self._handle_missing_values(df)

        # 3. Tier 1特徴量の計算
        df = self._add_basic_rate_features(df)
        df = self._add_time_cycle_features(df)
        df = self._add_momentum_features(df)
        df = self._add_regime_features(df)
        df = self._add_price_interaction_features(df, ohlcv_df)

        added_features = [col for col in df.columns if col.startswith("fr_")]
        logger.info(f"ファンディングレート特徴量を追加: {len(added_features)}個")

        return df

    def _prepare_data(
        self, ohlcv_df: pd.DataFrame, funding_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        データの前処理とマージ

        Args:
            ohlcv_df: OHLCVデータ（1時間足）
            funding_df: ファンディングレートデータ（8時間ごと）

        Returns:
            マージされたDataFrame
        """
        # OHLCVデータをコピー
        result = ohlcv_df.copy()

        # timestampカラムの存在確認と自動補完
        if "timestamp" not in result.columns:
            # インデックスがDatetimeIndexの場合はカラムに変換
            if isinstance(result.index, pd.DatetimeIndex):
                result["timestamp"] = result.index
                logger.debug("インデックスからtimestampカラムを生成しました")
            else:
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

        # ファンディングレートデータを前方補完でマージ
        # 8時間ごとの値を1時間足に展開
        funding_sorted = funding_df.sort_values("timestamp").copy()

        # merge_asofで最も近い過去の値をマージ
        result = pd.merge_asof(
            result.sort_values("timestamp"),
            funding_sorted[["timestamp", "funding_rate"]],
            on="timestamp",
            direction="backward",
        )

        # funding_rate_rawとして保存
        if "funding_rate" in result.columns:
            result["funding_rate_raw"] = result["funding_rate"]

        return result

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        階層的欠損値処理

        1. 決済時刻の真の欠損を検出
        2. 決済時刻の欠損のみ線形補間（最大2期間）
        3. 非決済時刻は前方補完済み（merge_asofで処理済み）
        4. 補完フラグを追加

        Args:
            df: 入力DataFrame

        Returns:
            欠損値処理後のDataFrame
        """
        if "funding_rate_raw" not in df.columns:
            return df

        # 補完フラグを初期化
        df["fr_imputed_flag"] = 0

        # 欠損値の位置を記録
        missing_mask = df["funding_rate_raw"].isna()

        if missing_mask.sum() > 0:
            # 線形補間（最大2期間）
            df["funding_rate_raw"] = df["funding_rate_raw"].interpolate(
                method="linear", limit=2, limit_direction="both"
            )

            # 補間されたデータにフラグを設定
            df.loc[missing_mask & ~df["funding_rate_raw"].isna(), "fr_imputed_flag"] = 1

            # 残った欠損値は前方補完
            df["funding_rate_raw"] = df["funding_rate_raw"].ffill()

            # それでも残る欠損値は後方補完
            df["funding_rate_raw"] = df["funding_rate_raw"].bfill()

            logger.debug(f"欠損値処理: {missing_mask.sum()}個の欠損値を補完")

        return df

    def _add_basic_rate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        基本金利指標特徴量

        - funding_rate_raw: Raw値（既に追加済み）
        - fr_lag_1p: 1期間前（8時間前）
        - fr_lag_2p: 2期間前（16時間前）
        - fr_lag_3p: 3期間前（24時間前）

        Args:
            df: 入力DataFrame

        Returns:
            特徴量追加後のDataFrame
        """
        if "funding_rate_raw" not in df.columns:
            return df

        # ラグ特徴量（期間 = settlement_interval）
        lag_periods = [1, 2, 3]
        for lag in lag_periods:
            shift_hours = lag * self.settlement_interval
            df[f"fr_lag_{lag}p"] = df["funding_rate_raw"].shift(shift_hours)

        return df

    def _add_time_cycle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        時間サイクル特徴量

        - fr_hours_since_settlement: 決済からの経過時間（0-7）
        - fr_cycle_sin: サイクルのsin成分
        - fr_cycle_cos: サイクルのcos成分

        Args:
            df: 入力DataFrame

        Returns:
            特徴量追加後のDataFrame
        """
        if "timestamp" not in df.columns:
            # インデックスがDatetimeIndexの場合はカラムに変換
            if isinstance(df.index, pd.DatetimeIndex):
                df["timestamp"] = df.index
                logger.debug(
                    "時間サイクル特徴量: インデックスからtimestampカラムを生成"
                )
            else:
                logger.warning("時間サイクル特徴量: timestampカラムがありません")
                return df

        # 決済からの経過時間を計算
        # 決済時刻は0, 8, 16時なので、時刻を8で割った余り
        df["fr_hours_since_settlement"] = (
            df["timestamp"].dt.hour % self.settlement_interval
        )

        # サイクルの三角関数表現（0-8時間の周期）
        # 2π / 8 = π/4
        cycle_phase = (
            2 * np.pi * df["fr_hours_since_settlement"] / self.settlement_interval
        )
        df["fr_cycle_sin"] = np.sin(cycle_phase)
        df["fr_cycle_cos"] = np.cos(cycle_phase)

        return df

    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        モメンタム特徴量

        - fr_velocity: 変化率（1期間の変化）
        - fr_ema_3periods: 3期間EMA（24時間）
        - fr_ema_7periods: 7期間EMA（56時間）

        Args:
            df: 入力DataFrame

        Returns:
            特徴量追加後のDataFrame
        """
        if "funding_rate_raw" not in df.columns:
            return df

        # 変化率（velocity）
        df["fr_velocity"] = df["funding_rate_raw"].pct_change(
            periods=self.settlement_interval
        )

        # EMA（期間はsettlement_interval単位）
        # 3期間 = 24時間、7期間 = 56時間
        span_3 = 3 * self.settlement_interval
        span_7 = 7 * self.settlement_interval

        df["fr_ema_3periods"] = (
            df["funding_rate_raw"].ewm(span=span_3, adjust=False).mean()
        )
        df["fr_ema_7periods"] = (
            df["funding_rate_raw"].ewm(span=span_7, adjust=False).mean()
        )

        return df

    def _add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        レジーム特徴量

        レジーム分類:
        - -2: 超低金利（FR < -0.01%）
        - -1: 低金利（-0.01% ≤ FR < 0%）
        -  0: 通常（0% ≤ FR ≤ 0.05%）
        -  1: 過熱（0.05% < FR ≤ 0.15%）
        -  2: 極端過熱（FR > 0.15%）

        特徴量:
        - fr_regime_encoded: レジームコード
        - regime_duration: 現在のレジーム継続期間

        Args:
            df: 入力DataFrame

        Returns:
            特徴量追加後のDataFrame
        """
        if "funding_rate_raw" not in df.columns:
            return df

        # レジーム分類
        fr = df["funding_rate_raw"]

        conditions = [
            fr < -0.0001,  # 超低金利
            (fr >= -0.0001) & (fr < 0),  # 低金利
            (fr >= 0) & (fr <= 0.0005),  # 通常
            (fr > 0.0005) & (fr <= 0.0015),  # 過熱
            fr > 0.0015,  # 極端過熱
        ]
        choices = [-2, -1, 0, 1, 2]

        df["fr_regime_encoded"] = np.select(conditions, choices, default=0)

        # レジーム継続期間を計算
        # レジームが変わるたびにリセット
        regime_changes = df["fr_regime_encoded"] != df["fr_regime_encoded"].shift(1)
        df["regime_duration"] = regime_changes.cumsum()
        df["regime_duration"] = df.groupby("regime_duration").cumcount()

        return df

    def _add_price_interaction_features(
        self, df: pd.DataFrame, ohlcv_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        価格相互作用特徴量

        - fr_price_corr_24h: 24時間の価格との相関
        - fr_volatility_adjusted: ボラティリティ調整済みFR

        Args:
            df: 入力DataFrame（FR特徴量を含む）
            ohlcv_df: 元のOHLCVデータ

        Returns:
            特徴量追加後のDataFrame
        """
        if "funding_rate_raw" not in df.columns or "close" not in df.columns:
            return df

        # 24時間の価格との相関
        window = 24
        # ローリング相関を計算（両方のSeriesを同時にローリング）
        df["fr_price_corr_24h"] = (
            df["funding_rate_raw"].rolling(window=window).corr(df["close"])
        )

        # ボラティリティ調整済みFR
        # FR / realized_volatility
        returns = df["close"].pct_change()
        realized_vol = returns.rolling(window=24).std() * np.sqrt(24)

        # ゼロ除算を避ける
        realized_vol = realized_vol.replace(0, np.nan)
        df["fr_volatility_adjusted"] = df["funding_rate_raw"] / (realized_vol + 1e-8)

        return df


def validate_funding_rate_data(df: pd.DataFrame) -> bool:
    """
    ファンディングレートデータの検証

    Args:
        df: 検証するDataFrame

    Returns:
        検証成功: True、失敗: False

    Raises:
        ValueError: データが無効な場合
    """
    if df.empty:
        logger.warning("ファンディングレートデータが空です")
        return True

    required_cols = ["timestamp", "funding_rate"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"必須カラムが見つかりません: {missing_cols}")

    # ファンディングレートの範囲チェック（-100% ~ 100%）
    if "funding_rate" in df.columns:
        fr = df["funding_rate"].dropna()
        if len(fr) > 0:
            if (fr < -1.0).any() or (fr > 1.0).any():
                raise ValueError(
                    f"ファンディングレートが範囲外です: min={fr.min()}, max={fr.max()}"
                )

    # タイムスタンプのソート確認
    if "timestamp" in df.columns and len(df) > 1:
        if not df["timestamp"].is_monotonic_increasing:
            raise ValueError("タイムスタンプがソートされていません")

    logger.debug("ファンディングレートデータの検証成功")
    return True
