import logging
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from app.services.ml.label_cache import LabelCache
from app.services.ml.label_generation.signal_generator import SignalGenerator
from app.utils.error_handler import DataError

logger = logging.getLogger(__name__)


class LabelGenerationService:
    """
    ラベル生成サービス

    学習用データのラベル生成、クリーニング、マッピングを担当します。
    メタラベリング対応: SignalGenerator でイベントを検出し、
    そのイベントが発生した足のみをラベリング対象とします。
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
        学習用データ（特徴量とラベル）を準備します。

        Args:
            features_df: 特徴量DataFrame
            ohlcv_df: OHLCV DataFrame (LabelCache初期化用)
            use_signal_generator: SignalGeneratorを使用するか
            signal_config: SignalGeneratorの設定
            use_cusum: CUSUMフィルターを使用するか
            cusum_threshold: CUSUMフィルターの閾値
            cusum_vol_multiplier: CUSUMフィルターの動的閾値乗数
            **training_params: 学習パラメータ

        Returns:
            Tuple[pd.DataFrame, pd.Series]: (クリーニング済み特徴量, 数値化済みラベル)
        """
        from app.config.unified_config import unified_config
        label_config = unified_config.ml.training.label_generation
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
            labels = label_cache.get_labels(
                horizon_n=training_params.get("horizon_n", label_config.horizon_n),
                threshold_method=training_params.get(
                    "threshold_method", label_config.threshold_method
                ),
                threshold=training_params.get("threshold", label_config.threshold),
                timeframe=label_config.timeframe,
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
    ) -> Optional[pd.Index]:
        """イベント時刻を検出"""
        if use_cusum:
            from app.services.ml.label_generation.cusum_generator import (
                CusumSignalGenerator,
            )

            logger.info("🔍 CUSUMフィルターでイベントを検出します。")
            cusum_gen = CusumSignalGenerator()
            volatility = cusum_gen.get_daily_volatility(ohlcv_df["close"])
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



