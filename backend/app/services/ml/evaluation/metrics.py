"""
統合評価指標システム

分析報告書で提案された包括的な評価指標を実装。
不均衡データに対する適切な評価指標を提供します。
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    log_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    multilabel_confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from app.utils.error_handler import safe_operation

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
    zero_division: str = "warn"


class MetricsCalculator:
    """
    統合評価指標計算器

    不均衡データに適した包括的な評価指標を提供し、
    モデルの性能を多角的に評価します。
    """

    def __init__(self, config: MetricsConfig | None = None):
        """
        初期化

        Args:
            config: 評価指標設定
        """
        self.config = config or MetricsConfig()

    @safe_operation(
        context="包括的な評価指標計算", is_api_call=False, default_return={}
    )
    def calculate_comprehensive_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None,
        level: str = "full",
    ) -> Dict[str, Any]:
        """
        包括的な評価指標を計算

        Args:
            y_true: 真のラベル (np.ndarray または pd.Series)
            y_pred: 予測ラベル
            y_proba: 予測確率（オプション）
            class_names: クラス名のリスト
            level: 計算レベル ('full' または 'basic')
                   - 'basic': 基本的な精度指標のみ（高速）
                   - 'full': 全ての指標（AUC、分布など含む）

        Returns:
            評価指標の辞書
        """
        # pd.Series の場合は numpy 配列に変換
        if hasattr(y_true, "values"):
            y_true = y_true.values  # type: ignore[reportAttributeAccessIssue]

        if level == "full":
            logger.info("📊 包括的な評価指標を計算中(Full)...")

        metrics = {}

        # 基本的な精度指標（常に計算）
        metrics.update(self._calculate_basic_metrics(y_true, y_pred))

        # Basicモードの場合はここで終了
        if level == "basic":
            # バランス精度は重要かつ軽量なので計算
            if self.config.include_balanced_accuracy:
                metrics.update(self._calculate_balanced_metrics(y_true, y_pred))
            return metrics

        # 以下はFullモードのみ実行

        # 不均衡データ対応指標
        if self.config.include_balanced_accuracy:
            metrics.update(self._calculate_balanced_metrics(y_true, y_pred))

        # 確率ベース指標
        if y_proba is not None:
            metrics.update(self._calculate_probability_metrics(y_true, y_proba))

        # 混同行列
        if self.config.include_confusion_matrix:
            metrics.update(
                self._calculate_confusion_matrix_metrics(y_true, y_pred, class_names)
            )

        # 分類レポート
        if self.config.include_classification_report:
            metrics.update(
                self._calculate_classification_report(y_true, y_pred, class_names)
            )

        # クラス別詳細指標
        metrics.update(self._calculate_per_class_metrics(y_true, y_pred, class_names))

        # データ分布情報
        metrics.update(self._calculate_distribution_metrics(y_true, y_pred))

        if level == "full":
            logger.info("✅ 評価指標計算完了")

        return metrics

    def _calculate_basic_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """基本的な精度指標を計算"""
        try:
            # 標準メトリクス
            p, r, f, _ = precision_recall_fscore_support(
                y_true,
                y_pred,
                average=self.config.average_method,
                zero_division=self.config.zero_division,
            )
            pm, rm, fm, _ = precision_recall_fscore_support(
                y_true, y_pred, average="macro", zero_division=self.config.zero_division
            )

            return {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": float(p),
                "recall": float(r),
                "f1_score": float(f),
                "precision_macro": float(pm),
                "recall_macro": float(rm),
                "f1_score_macro": float(fm),
                "matthews_corrcoef": matthews_corrcoef(y_true, y_pred),
                "cohen_kappa": cohen_kappa_score(y_true, y_pred),
            }
        except Exception as e:
            logger.warning(f"基本指標計算エラー: {e}")
            return {}

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
            y_pos = (
                y_proba[:, 1]
                if y_proba.ndim > 1 and y_proba.shape[1] > 1
                else y_proba.ravel()
            )

            if self.config.include_roc_auc:
                if n_classes == 2:
                    metrics["roc_auc"] = roc_auc_score(y_true, y_pos)
                else:
                    metrics["roc_auc"] = roc_auc_score(
                        y_true, y_proba, multi_class="ovr", average="weighted"
                    )

            if self.config.include_pr_auc:
                if n_classes == 2:
                    metrics["pr_auc"] = average_precision_score(y_true, y_pos)
                else:
                    pr_aucs = [
                        average_precision_score(
                            (y_true == i).astype(int), y_proba[:, i]
                        )
                        for i in range(n_classes)
                        if np.sum(y_true == i) > 0
                    ]
                    metrics["pr_auc"] = np.mean(np.asarray(pr_aucs)) if pr_aucs else 0.0

            metrics["log_loss"] = log_loss(y_true, y_proba)
            if n_classes == 2:
                metrics["brier_score"] = brier_score_loss(y_true, y_pos)
            else:
                metrics["brier_score"] = np.mean(
                    np.asarray(
                        [
                            brier_score_loss((y_true == i).astype(int), y_proba[:, i])
                            for i in range(n_classes)
                        ]
                    )
                )
        except Exception as e:
            logger.warning(f"確率指標計算エラー: {e}")
            metrics.update({"log_loss": 0.0, "brier_score": 0.0})
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
            cm = confusion_matrix(y_true, y_pred)
            metrics["confusion_matrix"] = cm.tolist()
            metrics["confusion_matrix_normalized"] = confusion_matrix(
                y_true, y_pred, normalize="true"
            ).tolist()
            if class_names:
                metrics["class_names"] = class_names

            labels = np.unique(y_true)
            mcm = multilabel_confusion_matrix(y_true, y_pred, labels=labels)
            tn, fp, fn, tp = mcm[:, 0, 0], mcm[:, 0, 1], mcm[:, 1, 0], mcm[:, 1, 1]

            eps = 1e-12
            spec, sens, npv, ppv = (
                tn / (tn + fp + eps),
                tp / (tp + fn + eps),
                tn / (tn + fn + eps),
                tp / (tp + fp + eps),
            )

            if len(labels) == 2:
                metrics.update(
                    {
                        "true_negatives": int(tn[1]),
                        "false_positives": int(fp[1]),
                        "false_negatives": int(fn[1]),
                        "true_positives": int(tp[1]),
                        "specificity": float(spec[1]),
                        "sensitivity": float(sens[1]),
                        "npv": float(npv[1]),
                        "ppv": float(ppv[1]),
                    }
                )
            else:
                counts = np.bincount(y_true)
                weights = counts / (counts.sum() + eps)
                metrics.update(
                    {
                        "specificity": float(np.average(spec, weights=weights)),
                        "sensitivity": float(np.average(sens, weights=weights)),
                        "npv": float(np.average(npv, weights=weights)),
                        "ppv": float(np.average(ppv, weights=weights)),
                    }
                )
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
        try:
            # average=None の場合、各指標はクラスごとの配列として返される
            res = precision_recall_fscore_support(
                y_true, y_pred, average=None, zero_division=self.config.zero_division
            )
            p, r, f, s = map(np.asanyarray, res)
            labels = np.unique(y_true)

            return {
                "per_class_metrics": {
                    (
                        class_names[i]
                        if class_names and i < len(class_names)
                        else f"class_{lab}"
                    ): {
                        "precision": float(p[i]),
                        "recall": float(r[i]),
                        "f1_score": float(f[i]),
                        "support": int(s[i]),
                    }
                    for i, lab in enumerate(labels)
                }
            }
        except Exception as e:
            logger.warning(f"クラス別指標計算エラー: {e}")
            return {}

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

    def calculate_volatility_regression_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """future_log_realized_vol 回帰用の評価指標を計算。"""
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        if y_true.size == 0 or y_pred.size == 0:
            return {
                "qlike": 0.0,
                "rmse_log_rv": 0.0,
                "mae_log_rv": 0.0,
            }

        eps = 1e-12
        true_var = np.exp(2.0 * y_true)
        pred_var = np.maximum(np.exp(2.0 * y_pred), eps)
        qlike = float(np.mean(np.log(pred_var + eps) + true_var / pred_var))

        return {
            "qlike": qlike,
            "rmse_log_rv": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae_log_rv": float(mean_absolute_error(y_true, y_pred)),
        }


def get_default_metrics() -> Dict[str, float]:
    """デフォルトの評価メトリクス辞書を返す（全て0.0初期化）"""
    keys = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "auc_score",
        "auc_roc",
        "auc_pr",
        "balanced_accuracy",
        "matthews_corrcoef",
        "cohen_kappa",
        "specificity",
        "sensitivity",
        "npv",
        "ppv",
        "log_loss",
        "brier_score",
        "qlike",
        "rmse_log_rv",
        "mae_log_rv",
        "loss",
        "val_accuracy",
        "val_loss",
        "training_time",
    ]
    return {k: 0.0 for k in keys}


# グローバルインスタンス
metrics_collector = MetricsCalculator()
