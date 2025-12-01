import logging
from typing import Tuple
import pandas as pd

from app.config.unified_config import unified_config
from app.utils.error_handler import DataError
from app.services.ml.label_cache import LabelCache # LabelCacheを使用

logger = logging.getLogger(__name__)


class LabelGenerationService:
    """
    ラベル生成サービス

    学習用データのラベル生成、クリーニング、マッピングを担当します。
    """

    def prepare_labels(
        self, features_df: pd.DataFrame, ohlcv_df: pd.DataFrame, **training_params
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        学習用データ（特徴量とラベル）を準備します。

        Args:
            features_df: 特徴量DataFrame
            ohlcv_df: OHLCV DataFrame (LabelCache初期化用)
            **training_params: 学習パラメータ

        Returns:
            Tuple[pd.DataFrame, pd.Series]: (クリーニング済み特徴量, 数値化済みラベル)
        """
        # ラベル生成設定を取得
        label_config = unified_config.ml.training.label_generation

        # LabelCache を使用してラベルを生成
        label_cache = LabelCache(ohlcv_df)

        try:
            logger.info("🎯 トリプルバリア法でラベル生成を開始します。")

            labels = label_cache.get_labels(
                horizon_n=label_config.horizon_n,
                threshold_method=label_config.threshold_method, # "TRIPLE_BARRIER" に固定されている
                threshold=label_config.threshold, # TBMではダミー値
                timeframe=label_config.timeframe,
                price_column=label_config.price_column,
                pt_factor=training_params.get("pt_factor", 1.0), # GAConfig等から取得できる想定
                sl_factor=training_params.get("sl_factor", 1.0),
                use_atr=training_params.get("use_atr", True),
                atr_period=training_params.get("atr_period", 14),
                binary_label=True, # メタラベリングのためにバイナリラベルに固定
            )

            # NaNを削除してクリーンなデータを作成
            common_index = features_df.index.intersection(labels.index)
            features_clean = features_df.loc[common_index].copy()
            labels_clean = labels.loc[common_index].copy()

            valid_idx = labels_clean.notna()
            features_clean = features_clean[valid_idx]
            labels_clean = labels_clean[valid_idx].astype(int) # TBMの出力はすでに0/1

            logger.info(f"✅ ラベル生成完了: {len(features_clean)}サンプル")

            return features_clean, labels_clean

        except Exception as e:
            logger.error(f"❌ ラベル生成設定でエラー発生: {e}", exc_info=True)
            raise DataError(f"ラベル生成に失敗しました: {e}")
