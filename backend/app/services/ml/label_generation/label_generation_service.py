import logging
from typing import Any, Dict, Optional, Tuple, cast

import pandas as pd

from app.services.ml.common.config import ml_config_manager
from app.services.ml.label_generation.signal_generator import SignalGenerator
from app.utils.error_handler import DataError

from .label_cache import LabelCache

logger = logging.getLogger(__name__)


class LabelGenerationService:
    """
    機械学習用ラベルの生成とデータクリーニングを担うサービス

    トリプルバリア法（Triple Barrier Method）をベースに、CUSUM フィルターや
    SignalGenerator によるイベント駆動型のラベリングをサポートします。
    特徴量データと OHLCV データを時間足で整列させ、NaN を排除した
    学習可能（Ready-to-train）なデータセットを構築します。
    """

    def prepare_labels(
        self,
        features_df: pd.DataFrame,
        ohlcv_df: pd.DataFrame,
        use_signal_generator: bool = False,
        signal_config: Optional[Dict[str, Any]] = None,
        use_cusum: bool = False,
        cusum_threshold: Optional[float] = None,
        cusum_vol_multiplier: float = 1.0,
        **training_params,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        高度な統計手法を用いて、機械学習モデルの学習に最適な「意味のある」正解ラベルを生成します。

        このメソッドは、単純な「n足後のリターン」ではなく、実際のトレーディングを模した
        **イベント駆動型ラベリング**（トリプルバリア法）を採用しています。

        主なプロセス：
        1. **サンプリング（イベント検出）**: CUSUMフィルターやSignalGeneratorにより、
           市場に統計的な変化が生じた重要なタイミングのみを学習対象として抽出します（ノイズの低減）。
        2. **バリアの設定**: 各サンプリング点において、未来の「利食い」「損切り」「時間制限」の3つの境界線を引きます。
        3. **ラベル確定**: 最初にどのバリアに価格が接触したかに基づいてラベル（1 or 0等）を決定します。

        Args:
            features_df (pd.DataFrame): 計算済みの特徴量集合。
            ohlcv_df (pd.DataFrame): ラベリングの基準となる市場価格データ。
            use_signal_generator (bool): 戦略的なシグナル（ボリンジャーバンド等）発生時のみを抽出するか。
            use_cusum (bool): 累積偏差（CUSUM）が閾値を超えた「変化点」のみを抽出するか。
            **training_params:
                - `horizon_n` (int): 決済までの最大保持期間（バー数）。
                - `pt_factor` (float): ボラティリティに対する利確幅の倍率。
                - `sl_factor` (float): ボラティリティに対する損切幅の倍率。
                - `threshold_method` (str): バリア幅の決定手法（"volatility", "fixed"等）。

        Returns:
            Tuple[pd.DataFrame, pd.Series]: 特徴量と同期され、未来リークのないクリーンな学習用データセット。
        """
        label_config = ml_config_manager.config.training.label_generation
        label_cache = LabelCache(ohlcv_df)

        try:
            logger.info("🎯 トリプルバリア法でラベル生成を開始します。")

            # イベント検出
            t_events = self._detect_events(
                ohlcv_df,
                use_cusum,
                cusum_threshold,
                cusum_vol_multiplier,
                use_signal_generator,
                signal_config,
            )

            if t_events is not None and len(t_events) == 0:
                logger.warning("⚠️ イベントが検出されませんでした。")
                return (
                    pd.DataFrame(columns=features_df.columns),
                    pd.Series(dtype=int, name="label"),
                )

            # ラベル生成
            timeframe = training_params.get(
                "timeframe", label_config.timeframe
            )
            labels = label_cache.get_labels(
                horizon_n=training_params.get(
                    "horizon_n", label_config.horizon_n
                ),
                threshold_method=training_params.get(
                    "threshold_method", label_config.threshold_method
                ),
                threshold=training_params.get(
                    "threshold", label_config.threshold
                ),
                timeframe=timeframe,
                price_column=label_config.price_column,
                pt_factor=training_params.get("pt_factor", 1.0),
                sl_factor=training_params.get("sl_factor", 1.0),
                use_atr=training_params.get("use_atr", True),
                atr_period=training_params.get("atr_period", 14),
                binary_label=True,
                t_events=t_events,
                min_window=training_params.get("min_window", 5),
                window_step=training_params.get("window_step", 1),
            )

            # データクリーニング
            common_idx = features_df.index.intersection(labels.index)
            labels_clean = labels.loc[common_idx].dropna().astype(int)
            features_clean = features_df.loc[labels_clean.index]

            logger.info(f"✅ ラベル生成完了: {len(features_clean)}サンプル")
            return features_clean, labels_clean

        except Exception as e:
            logger.error(f"❌ ラベル生成エラー: {e}", exc_info=True)
            raise DataError(f"ラベル生成に失敗しました: {e}")

    def _detect_events(
        self,
        ohlcv_df: pd.DataFrame,
        use_cusum: bool,
        cusum_threshold: Optional[float],
        cusum_vol_multiplier: float,
        use_signal_generator: bool,
        signal_config: Optional[Dict[str, Any]],
    ) -> Optional[pd.DatetimeIndex]:
        """
        ラベリングの起点となるイベント時刻（サンプリングポイント）を検出

        Args:
            ohlcv_df: 元の価格データ
            use_cusum: CUSUM フィルタリングを有効にするか
            cusum_threshold: CUSUM の感度（固定）
            cusum_vol_multiplier: ボラティリティ倍率
            use_signal_generator: SignalGenerator モードか
            signal_config: SignalGenerator 用の設定

        Returns:
            イベントが検出された時刻の DatetimeIndex（フィルタリングなしの場合は None）
        """
        if use_cusum:
            from app.services.ml.label_generation.cusum_generator import (
                CusumSignalGenerator,
            )

            logger.info("🔍 CUSUMフィルターでイベントを検出します。")
            cusum_gen = CusumSignalGenerator()
            volatility = cusum_gen.get_daily_volatility(
                cast(pd.Series, ohlcv_df["close"])
            )
            return cusum_gen.get_events(
                df=ohlcv_df,
                threshold=cusum_threshold,
                volatility=volatility,
                vol_multiplier=cusum_vol_multiplier,
            )

        if use_signal_generator:
            logger.info("🔍 SignalGenerator でイベントを検出します。")
            config = signal_config or {
                "use_bb": True,
                "use_donchian": False,
                "use_volume": False,
                "bb_window": 20,
                "bb_dev": 2.0,
            }
            return SignalGenerator().get_combined_events(df=ohlcv_df, **config)

        return None
