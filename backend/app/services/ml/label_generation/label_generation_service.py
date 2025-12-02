import logging
from typing import Tuple, Optional, Dict, Any
import pandas as pd

from app.utils.error_handler import DataError
from app.services.ml.label_cache import LabelCache
from app.services.ml.label_generation.signal_generator import SignalGenerator

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
            use_signal_generator: SignalGeneratorを使用するか（デフォルト: False）
            signal_config: SignalGeneratorの設定（デフォルト: None）
            use_cusum: CUSUMフィルターを使用するか（デフォルト: False）
            cusum_threshold: CUSUMフィルターの閾値（Noneの場合はボラティリティを使用）
            cusum_vol_multiplier: CUSUMフィルターの動的閾値乗数（デフォルト: 1.0）
            **training_params: 学習パラメータ

        Returns:
            Tuple[pd.DataFrame, pd.Series]: (クリーニング済み特徴量, 数値化済みラベル)

        Notes:
            - use_cusum=True の場合、CusumSignalGenerator でイベントを検出します（Scientific Meta-Labeling）。
            - use_signal_generator=True の場合、SignalGenerator でイベントを検出します（従来のMeta-Labeling）。
            - 両方Falseの場合、全ての足をラベリング対象とします。
        """
        # ラベル生成設定を取得（遅延インポートで循環インポート回避）
        from app.config.unified_config import unified_config
        from app.services.ml.label_generation.cusum_generator import (
            CusumSignalGenerator,
        )

        label_config = unified_config.ml.training.label_generation

        # LabelCache を使用してラベルを生成
        label_cache = LabelCache(ohlcv_df)

        try:
            logger.info("🎯 トリプルバリア法でラベル生成を開始します。")

            t_events = None

            # 1. CUSUMフィルターによるイベント検出 (Scientific Approach)
            if use_cusum:
                logger.info("🔍 CUSUMフィルターでイベントを検出します。")
                cusum_gen = CusumSignalGenerator()

                # ボラティリティ計算
                volatility = cusum_gen.get_daily_volatility(ohlcv_df["close"])

                t_events = cusum_gen.get_events(
                    df=ohlcv_df,
                    threshold=cusum_threshold,
                    volatility=volatility,
                    vol_multiplier=cusum_vol_multiplier,
                )
                logger.info(f"✅ CUSUMイベント検出: {len(t_events)}件")

            # 2. 従来のSignalGeneratorによるイベント検出 (Heuristic Approach)
            elif use_signal_generator:
                logger.info("🔍 SignalGenerator でイベントを検出します。")

                # signal_config のデフォルト値を設定
                if signal_config is None:
                    signal_config = {
                        "use_bb": True,
                        "use_donchian": False,
                        "use_volume": False,
                        "bb_window": 20,
                        "bb_dev": 2.0,
                    }

                # SignalGeneratorでイベント検出
                signal_gen = SignalGenerator()
                t_events = signal_gen.get_combined_events(df=ohlcv_df, **signal_config)

                logger.info(f"✅ SignalGeneratorイベント検出: {len(t_events)}件")

            # イベントが0件の場合の処理
            if t_events is not None and len(t_events) == 0:
                logger.warning(
                    "⚠️ イベントが検出されませんでした。空のデータセットを返します。"
                )
                return (
                    pd.DataFrame(columns=features_df.columns),
                    pd.Series(dtype=int, name="label"),
                )

            # ラベル生成 (t_eventsを指定)
            labels = label_cache.get_labels(
                horizon_n=label_config.horizon_n,
                threshold_method=label_config.threshold_method,
                threshold=label_config.threshold,
                timeframe=label_config.timeframe,
                price_column=label_config.price_column,
                pt_factor=training_params.get("pt_factor", 1.0),
                sl_factor=training_params.get("sl_factor", 1.0),
                use_atr=training_params.get("use_atr", True),
                atr_period=training_params.get("atr_period", 14),
                binary_label=True,
                t_events=t_events,  # イベント時刻を渡す
            )

            # NaNを削除してクリーンなデータを作成
            common_index = features_df.index.intersection(labels.index)
            features_clean = features_df.loc[common_index].copy()
            labels_clean = labels.loc[common_index].copy()

            valid_idx = labels_clean.notna()
            features_clean = features_clean[valid_idx]
            labels_clean = labels_clean[valid_idx].astype(int)

            logger.info(f"✅ ラベル生成完了: {len(features_clean)}サンプル")

            return features_clean, labels_clean

        except Exception as e:
            logger.error(f"❌ ラベル生成設定でエラー発生: {e}", exc_info=True)
            raise DataError(f"ラベル生成に失敗しました: {e}")
