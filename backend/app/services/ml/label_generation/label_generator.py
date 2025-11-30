"""
ラベル生成の主要クラス

価格変化率から3クラス分類（上昇・下落・レンジ）のラベルを生成するためのLabelGeneratorクラスを提供します。
"""

import logging
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from app.services.ml.common.ml_utils import calculate_price_change
from .enums import ThresholdMethod
from .threshold import ThresholdCalculator

logger = logging.getLogger(__name__)


class LabelGenerator:
    """
    ラベル生成クラス

    価格変化率から3クラス分類（上昇・下落・レンジ）のラベルを生成します。
    データの特性に応じて動的に閾値を調整する機能を提供します。
    """

    def __init__(self):
        """初期化"""
        self.threshold_calculator = ThresholdCalculator()

    def generate_labels(
        self,
        price_data: pd.Series | pd.DataFrame,
        method: ThresholdMethod = ThresholdMethod.STD_DEVIATION,
        target_distribution: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        価格データからラベルを生成

        Args:
            price_data: 価格データ（Close価格）またはDataFrame
            method: 閾値計算方法
            target_distribution: 目標分布（例: {"up": 0.33, "down": 0.33, "range": 0.34}）
            **kwargs: 各方法固有のパラメータ
                target_column: DataFrame入力時のターゲットカラム名

        Returns:
            Tuple[pd.Series, Dict[str, Any]]: (ラベルSeries, 閾値情報の辞書)
        """
        try:
            # DataFrame入力 + target_column指定のサポート
            target_column = kwargs.pop("target_column", None)
            series_input: pd.Series
            if isinstance(price_data, pd.DataFrame):
                if not target_column or target_column not in price_data.columns:
                    raise ValueError("target_column が必要です")
                series_input = price_data[target_column]
            else:
                series_input = price_data  # type: ignore

            # 価格変化率を計算
            # - Series入力: 次期変化率（forward）で学習用に整合 (shift=-1)
            # - DataFrame+target_column入力: 過去→現在（backward）でテスト互換 (shift=0)
            shift = 0 if target_column is not None else -1

            try:
                price_change = calculate_price_change(
                    series_input, shift=shift, fill_na=True
                )
            except Exception as e:
                logger.error(f"価格変化率計算エラー: {e}")
                raise ValueError(f"価格変化率計算に失敗しました: {e}") from e

            # NaNを除去
            valid_mask = price_change.notna()
            price_change_clean = price_change[valid_mask]

            if len(price_change_clean) == 0:
                raise ValueError("有効な価格変化率データがありません")

            # 閾値を計算
            threshold_info = self.threshold_calculator.calculate_thresholds(
                price_change_clean,
                method,
                target_distribution,
                **kwargs,
            )

            threshold_up = threshold_info["threshold_up"]
            threshold_down = threshold_info["threshold_down"]

            # ラベルを生成
            labels = pd.Series(
                1, index=price_change.index, dtype=int
            )  # デフォルト：レンジ
            labels[price_change > threshold_up] = 2  # 上昇
            labels[price_change < threshold_down] = 0  # 下落

            # 最後の行は予測できないので除外（forward shiftの場合）
            if shift < 0:
                labels = labels.iloc[:shift]

            # 分布情報を追加
            label_counts = labels.value_counts().sort_index()
            total_count = len(labels)

            distribution_info = {
                "down_count": label_counts.get(0, 0),
                "range_count": label_counts.get(1, 0),
                "up_count": label_counts.get(2, 0),
                "down_ratio": (
                    label_counts.get(0, 0) / total_count if total_count > 0 else 0
                ),
                "range_ratio": (
                    label_counts.get(1, 0) / total_count if total_count > 0 else 0
                ),
                "up_ratio": (
                    label_counts.get(2, 0) / total_count if total_count > 0 else 0
                ),
                "total_count": total_count,
            }

            threshold_info.update(distribution_info)

            logger.info(
                f"ラベル生成完了: 上昇={distribution_info['up_count']}({distribution_info['up_ratio'] * 100:.1f}%), "
                f"下落={distribution_info['down_count']}({distribution_info['down_ratio'] * 100:.1f}%), "
                f"レンジ={distribution_info['range_count']}({distribution_info['range_ratio'] * 100:.1f}%)"
            )

            # 常にタプルを返すように統一
            return labels, threshold_info

        except Exception as e:
            logger.error(f"ラベル生成エラー: {e}")
            raise

    @staticmethod
    def validate_label_distribution(
        labels: pd.Series, min_class_ratio: float = 0.05, max_class_ratio: float = 0.95
    ) -> Dict[str, Any]:
        """
        ラベル分布を検証

        Args:
            labels: ラベルSeries
            min_class_ratio: 最小クラス比率
            max_class_ratio: 最大クラス比率

        Returns:
            検証結果の辞書
        """
        label_counts = labels.value_counts().sort_index()
        total_count = len(labels)

        validation_result = {
            "is_valid": True,
            "warnings": [],
            "errors": [],
            "distribution": {},
        }

        for label in [0, 1, 2]:  # 下落、レンジ、上昇
            count = label_counts.get(label, 0)
            ratio = count / total_count if total_count > 0 and count is not None else 0

            class_name = ["down", "range", "up"][label]
            validation_result["distribution"][class_name] = {
                "count": count,
                "ratio": ratio,
            }

            if ratio < min_class_ratio:
                validation_result["warnings"].append(
                    f"{class_name}クラスの比率が低すぎます: {ratio * 100:.2f}% < {min_class_ratio * 100:.2f}%"
                )

            if ratio > max_class_ratio:
                validation_result["errors"].append(
                    f"{class_name}クラスの比率が高すぎます: {ratio * 100:.2f}% > {max_class_ratio * 100:.2f}%"
                )
                validation_result["is_valid"] = False

        # 1クラスしかない場合の特別チェック
        unique_labels = labels.nunique()
        if unique_labels <= 1:
            validation_result["errors"].append(
                f"ラベルが{unique_labels}種類しかありません。機械学習には最低2種類必要です。"
            )
            validation_result["is_valid"] = False

        return validation_result
