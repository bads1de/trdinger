"""
拡張評価指標システム

分析報告書で提案された包括的な評価指標を実装。
不均衡データに対する適切な評価指標を提供します。
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from dataclasses import dataclass
import warnings

logger = logging.getLogger(__name__)


@dataclass
class MetricsConfig:
    """評価指標設定"""

    include_balanced_accuracy: bool = True
    include_pr_auc: bool = True
    include_roc_auc: bool = True
    include_confusion_matrix: bool = True
    include_classification_report: bool = True
    average_method: str = "weighted"  # 'macro', 'micro', 'weighted'
    zero_division: int = 0


class EnhancedMetricsCalculator:
    """
    拡張評価指標計算器

    不均衡データに適した包括的な評価指標を提供し、
    モデルの性能を多角的に評価します。
    """

    def __init__(self, config: MetricsConfig = None):
        """
        初期化

        Args:
            config: 評価指標設定
        """
        self.config = config or MetricsConfig()

    def calculate_comprehensive_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        包括的な評価指標を計算

        Args:
            y_true: 真のラベル
            y_pred: 予測ラベル
            y_proba: 予測確率（オプション）
            class_names: クラス名のリスト

        Returns:
            評価指標の辞書
        """
        logger.info("📊 包括的な評価指標を計算中...")

        metrics = {}

        try:
            # 基本的な精度指標
            metrics.update(self._calculate_basic_metrics(y_true, y_pred))

            # 不均衡データ対応指標
            if self.config.include_balanced_accuracy:
                metrics.update(self._calculate_balanced_metrics(y_true, y_pred))

            # 確率ベース指標
            if y_proba is not None:
                metrics.update(self._calculate_probability_metrics(y_true, y_proba))

            # 混同行列
            if self.config.include_confusion_matrix:
                metrics.update(
                    self._calculate_confusion_matrix_metrics(
                        y_true, y_pred, class_names
                    )
                )

            # 分類レポート
            if self.config.include_classification_report:
                metrics.update(
                    self._calculate_classification_report(y_true, y_pred, class_names)
                )

            # クラス別詳細指標
            metrics.update(
                self._calculate_per_class_metrics(y_true, y_pred, class_names)
            )

            # データ分布情報
            metrics.update(self._calculate_distribution_metrics(y_true, y_pred))

            logger.info("✅ 評価指標計算完了")

        except Exception as e:
            logger.error(f"評価指標計算エラー: {e}")
            metrics["error"] = str(e)

        return metrics

    def _calculate_basic_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """基本的な精度指標を計算"""
        metrics = {}

        try:
            # 標準精度
            metrics["accuracy"] = accuracy_score(y_true, y_pred)

            # 精密度、再現率、F1スコア
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true,
                y_pred,
                average=self.config.average_method,
                zero_division=self.config.zero_division,
            )

            metrics["precision"] = precision
            metrics["recall"] = recall
            metrics["f1_score"] = f1

            # マクロ平均も計算
            precision_macro, recall_macro, f1_macro, _ = (
                precision_recall_fscore_support(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=self.config.zero_division,
                )
            )

            metrics["precision_macro"] = precision_macro
            metrics["recall_macro"] = recall_macro
            metrics["f1_score_macro"] = f1_macro

        except Exception as e:
            logger.warning(f"基本指標計算エラー: {e}")

        return metrics

    def _calculate_balanced_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """不均衡データ対応指標を計算"""
        metrics = {}

        try:
            # バランス精度（分析報告書で推奨）
            metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)

            # クラス重み付き精度
            sample_weight = self._calculate_class_weights(y_true)
            metrics["weighted_accuracy"] = accuracy_score(
                y_true, y_pred, sample_weight=sample_weight
            )

        except Exception as e:
            logger.warning(f"バランス指標計算エラー: {e}")

        return metrics

    def _calculate_probability_metrics(
        self, y_true: np.ndarray, y_proba: np.ndarray
    ) -> Dict[str, float]:
        """確率ベース指標を計算"""
        metrics = {}

        try:
            n_classes = len(np.unique(y_true))

            if self.config.include_roc_auc:
                # ROC-AUC
                if n_classes == 2:
                    # 二値分類
                    metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    # 多クラス分類
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        metrics["roc_auc_ovr"] = roc_auc_score(
                            y_true, y_proba, multi_class="ovr", average="weighted"
                        )
                        metrics["roc_auc_ovo"] = roc_auc_score(
                            y_true, y_proba, multi_class="ovo", average="weighted"
                        )

            if self.config.include_pr_auc:
                # PR-AUC（分析報告書で推奨）
                if n_classes == 2:
                    # 二値分類
                    metrics["pr_auc"] = average_precision_score(y_true, y_proba[:, 1])
                else:
                    # 多クラス分類：各クラスのPR-AUCを計算
                    pr_aucs = []
                    for i in range(n_classes):
                        y_true_binary = (y_true == i).astype(int)
                        pr_auc = average_precision_score(y_true_binary, y_proba[:, i])
                        pr_aucs.append(pr_auc)
                        metrics[f"pr_auc_class_{i}"] = pr_auc

                    metrics["pr_auc_macro"] = np.mean(pr_aucs)
                    metrics["pr_auc_weighted"] = np.average(
                        pr_aucs, weights=np.bincount(y_true)
                    )

        except Exception as e:
            logger.warning(f"確率指標計算エラー: {e}")

        return metrics

    def _calculate_confusion_matrix_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """混同行列関連指標を計算"""
        metrics = {}

        try:
            # 混同行列
            cm = confusion_matrix(y_true, y_pred)
            metrics["confusion_matrix"] = cm.tolist()

            # 正規化された混同行列
            cm_normalized = confusion_matrix(y_true, y_pred, normalize="true")
            metrics["confusion_matrix_normalized"] = cm_normalized.tolist()

            # クラス名があれば追加
            if class_names:
                metrics["class_names"] = class_names

            # 混同行列から派生する指標
            if cm.shape[0] == 2:  # 二値分類の場合
                tn, fp, fn, tp = cm.ravel()
                metrics["true_negatives"] = int(tn)
                metrics["false_positives"] = int(fp)
                metrics["false_negatives"] = int(fn)
                metrics["true_positives"] = int(tp)

                # 特異度
                metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                # 感度（再現率と同じ）
                metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        except Exception as e:
            logger.warning(f"混同行列計算エラー: {e}")

        return metrics

    def _calculate_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """分類レポートを計算"""
        metrics = {}

        try:
            # 分類レポート（辞書形式）
            report = classification_report(
                y_true,
                y_pred,
                target_names=class_names,
                output_dict=True,
                zero_division=self.config.zero_division,
            )
            metrics["classification_report"] = report

        except Exception as e:
            logger.warning(f"分類レポート計算エラー: {e}")

        return metrics

    def _calculate_per_class_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """クラス別詳細指標を計算"""
        metrics = {}

        try:
            # クラス別の精密度、再現率、F1スコア
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true, y_pred, average=None, zero_division=self.config.zero_division
            )

            classes = np.unique(y_true)
            per_class_metrics = {}

            for i, class_label in enumerate(classes):
                class_name = (
                    class_names[i]
                    if class_names and i < len(class_names)
                    else f"class_{class_label}"
                )
                per_class_metrics[class_name] = {
                    "precision": float(precision[i]),
                    "recall": float(recall[i]),
                    "f1_score": float(f1[i]),
                    "support": int(support[i]),
                }

            metrics["per_class_metrics"] = per_class_metrics

        except Exception as e:
            logger.warning(f"クラス別指標計算エラー: {e}")

        return metrics

    def _calculate_distribution_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """データ分布関連指標を計算"""
        metrics = {}

        try:
            # 真のラベル分布
            true_counts = np.bincount(y_true)
            true_distribution = true_counts / len(y_true)
            metrics["true_label_distribution"] = true_distribution.tolist()

            # 予測ラベル分布
            pred_counts = np.bincount(y_pred, minlength=len(true_counts))
            pred_distribution = pred_counts / len(y_pred)
            metrics["predicted_label_distribution"] = pred_distribution.tolist()

            # 不均衡比率
            max_class_ratio = np.max(true_distribution) / np.min(
                true_distribution[true_distribution > 0]
            )
            metrics["class_imbalance_ratio"] = float(max_class_ratio)

            # サンプル数
            metrics["total_samples"] = len(y_true)
            metrics["n_classes"] = len(true_counts)

        except Exception as e:
            logger.warning(f"分布指標計算エラー: {e}")

        return metrics

    def _calculate_class_weights(self, y_true: np.ndarray) -> np.ndarray:
        """クラス重みを計算"""
        classes, counts = np.unique(y_true, return_counts=True)
        total_samples = len(y_true)
        n_classes = len(classes)

        # バランス重み計算
        weights = total_samples / (n_classes * counts)

        # 各サンプルの重みを計算
        sample_weights = np.zeros(len(y_true))
        for i, class_label in enumerate(classes):
            mask = y_true == class_label
            sample_weights[mask] = weights[i]

        return sample_weights

    def generate_metrics_summary(self, metrics: Dict[str, Any]) -> str:
        """評価指標のサマリーを生成"""
        summary_lines = []
        summary_lines.append("📊 評価指標サマリー")
        summary_lines.append("=" * 50)

        # 主要指標
        if "accuracy" in metrics:
            summary_lines.append(f"精度 (Accuracy): {metrics['accuracy']:.4f}")

        if "balanced_accuracy" in metrics:
            summary_lines.append(f"バランス精度: {metrics['balanced_accuracy']:.4f}")

        if "f1_score" in metrics:
            summary_lines.append(f"F1スコア: {metrics['f1_score']:.4f}")

        if "roc_auc" in metrics:
            summary_lines.append(f"ROC-AUC: {metrics['roc_auc']:.4f}")

        if "pr_auc" in metrics:
            summary_lines.append(f"PR-AUC: {metrics['pr_auc']:.4f}")

        # クラス不均衡情報
        if "class_imbalance_ratio" in metrics:
            summary_lines.append(
                f"クラス不均衡比率: {metrics['class_imbalance_ratio']:.2f}"
            )

        return "\n".join(summary_lines)

    def save_metrics_report(self, metrics: Dict[str, Any], filepath: str):
        """評価指標レポートをファイルに保存"""
        try:
            import json

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"評価指標レポートを保存: {filepath}")
        except Exception as e:
            logger.error(f"レポート保存エラー: {e}")


# グローバルインスタンス
enhanced_metrics_calculator = EnhancedMetricsCalculator()
