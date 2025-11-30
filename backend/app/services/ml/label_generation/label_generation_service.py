import logging
from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np

from app.config.unified_config import unified_config
from app.utils.data_processing import data_processor as data_preprocessor
from app.utils.error_handler import DataError
from .presets import (
    apply_preset_by_name,
    forward_classification_preset,
)
from .main import LabelGenerator

logger = logging.getLogger(__name__)


class LabelGenerationService:
    """
    ラベル生成サービス

    学習用データのラベル生成、クリーニング、マッピングを担当します。
    """

    def prepare_labels(
        self, features_df: pd.DataFrame, **training_params
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        学習用データ（特徴量とラベル）を準備します。

        Args:
            features_df: 特徴量DataFrame
            **training_params: 学習パラメータ (target_columnなど)

        Returns:
            Tuple[pd.DataFrame, pd.Series]: (クリーニング済み特徴量, 数値化済みラベル)
        """
        # ラベル生成設定を取得
        label_config = unified_config.ml.training.label_generation

        # デフォルトのプリセット/カスタム設定ロジック
        try:
            logger.info("🎯 新しいラベル生成設定を使用")

            if label_config.use_preset:
                try:
                    labels, _ = apply_preset_by_name(
                        features_df, label_config.default_preset
                    )
                except ValueError:
                    logger.warning("⚠️ フォールバック: カスタム設定を使用します")
                    labels = forward_classification_preset(
                        df=features_df,
                        timeframe=label_config.timeframe,
                        horizon_n=label_config.horizon_n,
                        threshold=label_config.threshold,
                        price_column=label_config.price_column,
                        threshold_method=label_config.get_threshold_method_enum(),
                    )
            else:
                labels = forward_classification_preset(
                    df=features_df,
                    timeframe=label_config.timeframe,
                    horizon_n=label_config.horizon_n,
                    threshold=label_config.threshold,
                    price_column=label_config.price_column,
                    threshold_method=label_config.get_threshold_method_enum(),
                )

            # NaNを削除してクリーンなデータを作成
            valid_idx = labels.notna()
            features_clean = features_df[valid_idx].copy()
            labels_clean = labels[valid_idx].copy()

            # 文字列ラベルを数値に変換
            unique_labels = set(labels_clean.unique())
            if "TREND" in unique_labels:
                label_mapping = {"RANGE": 0, "TREND": 1}
            else:
                label_mapping = {"DOWN": 0, "RANGE": 1, "UP": 2}

            labels_numeric = labels_clean.map(label_mapping)

            if labels_numeric.isna().any():
                valid_numeric_idx = labels_numeric.notna()
                features_clean = features_clean[valid_numeric_idx]
                labels_numeric = labels_numeric[valid_numeric_idx]

            logger.info(f"✅ ラベル生成完了: {len(features_clean)}サンプル")

            return features_clean, labels_numeric

        except Exception as e:
            logger.error(f"❌ 新しいラベル生成設定でエラー発生: {e}")
            raise DataError(f"ラベル生成に失敗しました: {e}")
