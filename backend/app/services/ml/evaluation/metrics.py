"""
統合評価指標システム

分析報告書で提案された包括的な評価指標を実装。
不均衡データに対する適切な評価指標を提供し、
システム全体のメトリクス収集・管理機能も統合します。
"""

import logging
import threading
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
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


@dataclass
class MetricData:
    """メトリクスデータ"""

    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """パフォーマンスメトリクス"""

    operation: str
    duration_ms: float
    memory_mb: float
    cpu_percent: float
    success: bool
    timestamp: datetime
    error_message: Optional[str] = None


@dataclass
class ModelEvaluationMetrics:
    """モデル評価メトリクス"""

    model_name: str
    model_type: str
    metrics: Dict[str, Any]
    timestamp: datetime
    dataset_info: Dict[str, Any] = field(default_factory=dict)
    training_params: Dict[str, Any] = field(default_factory=dict)


class MetricsCalculator:
    """
    統合評価指標計算器・収集器

    不均衡データに適した包括的な評価指標を提供し、
    モデルの性能を多角的に評価します。
    また、システム全体のメトリクス収集・管理機能も統合しています。
    """

    def __init__(self, config: MetricsConfig | None = None, max_history: int = 1000):
        """
        初期化

        Args:
            config: 評価指標設定
            max_history: メトリクス履歴の最大保持数
        """
        self.config = config or MetricsConfig()

        # メトリクス収集機能の初期化
        self.max_history = max_history
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self._performance_metrics: deque = deque(maxlen=max_history)
        self._model_evaluation_metrics: deque = deque(maxlen=max_history)
        self._error_counts: Dict[str, int] = defaultdict(int)
        self._operation_counts: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    @safe_operation(
        context="包括的な評価指標計算", is_api_call=False, default_return={}
    )
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

        logger.info("✅ 評価指標計算完了")

        return metrics

    def record_metric(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """
        メトリクスを記録

        Args:
            name: メトリクス名
            value: 値
            tags: タグ
            context: コンテキスト情報
        """
        with self._lock:
            metric = MetricData(
                name=name,
                value=value,
                timestamp=datetime.now(),
                tags=tags or {},
                context=context or {},
            )
            self._metrics[name].append(metric)

    def record_performance(
        self,
        operation: str,
        duration_ms: float,
        memory_mb: float = 0.0,
        cpu_percent: float = 0.0,
        success: bool = True,
        error_message: Optional[str] = None,
    ):
        """
        パフォーマンスメトリクスを記録

        Args:
            operation: 操作名
            duration_ms: 処理時間（ミリ秒）
            memory_mb: メモリ使用量（MB）
            cpu_percent: CPU使用率（%）
            success: 成功フラグ
            error_message: エラーメッセージ
        """
        with self._lock:
            perf_metric = PerformanceMetrics(
                operation=operation,
                duration_ms=duration_ms,
                memory_mb=memory_mb,
                cpu_percent=cpu_percent,
                success=success,
                timestamp=datetime.now(),
                error_message=error_message,
            )
            self._performance_metrics.append(perf_metric)

            # 操作カウント
            self._operation_counts[operation] += 1

            # エラーカウント
            if not success:
                self._error_counts[operation] += 1

    def record_error(self, operation: str, error_type: str, error_message: str):
        """
        エラーを記録

        Args:
            operation: 操作名
            error_type: エラータイプ
            error_message: エラーメッセージ
        """
        with self._lock:
            self._error_counts[f"{operation}_{error_type}"] += 1

        # ログにも出力
        logger.error(
            f"Operation: {operation}, Error: {error_type}, Message: {error_message}"
        )

    def record_model_evaluation_metrics(
        self,
        model_name: str,
        model_type: str,
        evaluation_metrics: Dict[str, Any],
        dataset_info: Optional[Dict[str, Any]] = None,
        training_params: Optional[Dict[str, Any]] = None,
    ):
        """
        モデル評価メトリクスを記録

        Args:
            model_name: モデル名
            model_type: モデルタイプ
            evaluation_metrics: 評価メトリクス
            dataset_info: データセット情報
            training_params: 学習パラメータ
        """
        with self._lock:
            # 個別メトリクスも記録
            for metric_name, value in evaluation_metrics.items():
                if isinstance(value, (int, float)):
                    self.record_metric(
                        name=metric_name,
                        value=float(value),
                        tags={
                            "model_name": model_name,
                            "model_type": model_type,
                            "metric_category": "model_evaluation",
                        },
                        context={
                            "dataset_info": dataset_info or {},
                            "training_params": training_params or {},
                        },
                    )

            # モデル評価メトリクス全体を記録
            model_eval_metric = ModelEvaluationMetrics(
                model_name=model_name,
                model_type=model_type,
                metrics=evaluation_metrics,
                timestamp=datetime.now(),
                dataset_info=dataset_info or {},
                training_params=training_params or {},
            )
            self._model_evaluation_metrics.append(model_eval_metric)

    @safe_operation(context="統合メトリクス評価", is_api_call=False, default_return={})
    def evaluate_and_record_model(
        self,
        model_name: str,
        model_type: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None,
        dataset_info: Optional[Dict[str, Any]] = None,
        training_params: Optional[Dict[str, Any]] = None,
        training_time: Optional[float] = None,
        memory_usage: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        モデルを評価し、結果を記録

        Args:
            model_name: モデル名
            model_type: モデルタイプ
            y_true: 真のラベル
            y_pred: 予測ラベル
            y_proba: 予測確率（オプション）
            class_names: クラス名のリスト（オプション）
            dataset_info: データセット情報（オプション）
            training_params: 学習パラメータ（オプション）
            training_time: 学習時間（秒）（オプション）
            memory_usage: メモリ使用量（MB）（オプション）

        Returns:
            評価結果の辞書
        """
        logger.info(f"📊 統合メトリクス評価開始: {model_name} ({model_type})")

        # 包括的な評価を実行
        evaluation_metrics = self.calculate_comprehensive_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
            class_names=class_names,
        )

        # パフォーマンス情報を追加
        if training_time is not None:
            evaluation_metrics["training_time"] = training_time
            # パフォーマンスメトリクスとしても記録
            self.record_performance(
                operation=f"model_training_{model_type}",
                duration_ms=training_time * 1000,  # 秒をミリ秒に変換
                memory_mb=memory_usage or 0.0,
                success=True,
            )

        if memory_usage is not None:
            evaluation_metrics["memory_usage"] = memory_usage

        # モデル評価結果を記録
        self.record_model_evaluation_metrics(
            model_name=model_name,
            model_type=model_type,
            evaluation_metrics=evaluation_metrics,
            dataset_info=dataset_info,
            training_params=training_params,
        )

        # 主要メトリクスをログ出力
        accuracy = evaluation_metrics.get("accuracy", 0.0)
        f1_score = evaluation_metrics.get("f1_score", 0.0)
        balanced_accuracy = evaluation_metrics.get("balanced_accuracy", 0.0)

        logger.info(
            f"✅ モデル評価完了: {model_name} - "
            f"精度={accuracy:.4f}, F1={f1_score:.4f}, バランス精度={balanced_accuracy:.4f}"
        )

        return evaluation_metrics

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

            # 追加の基本指標
            metrics["matthews_corrcoef"] = matthews_corrcoef(y_true, y_pred)
            metrics["cohen_kappa"] = cohen_kappa_score(y_true, y_pred)

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
                    try:
                        y_prob_positive = (
                            y_proba[:, 1]
                            if y_proba.ndim > 1 and y_proba.shape[1] > 1
                            else y_proba.ravel()
                        )
                        metrics["roc_auc"] = roc_auc_score(y_true, y_prob_positive)
                    except ValueError as e:
                        logger.warning(f"二値分類ROC-AUC計算エラー: {e}")
                        metrics["roc_auc"] = 0.0
                else:
                    # 多クラス分類
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        try:
                            metrics["roc_auc"] = roc_auc_score(
                                y_true, y_proba, multi_class="ovr", average="weighted"
                            )
                            # 追加の詳細指標
                            metrics["roc_auc_ovr"] = metrics["roc_auc"]
                            metrics["roc_auc_ovo"] = roc_auc_score(
                                y_true, y_proba, multi_class="ovo", average="weighted"
                            )
                        except ValueError as e:
                            logger.warning(f"多クラスROC-AUC計算エラー: {e}")
                            metrics["roc_auc"] = 0.0
                            metrics["roc_auc_ovr"] = 0.0
                            metrics["roc_auc_ovo"] = 0.0

            if self.config.include_pr_auc:
                # PR-AUC（分析報告書で推奨）
                if n_classes == 2:
                    # 二値分類
                    try:
                        y_prob_positive = (
                            y_proba[:, 1]
                            if y_proba.ndim > 1 and y_proba.shape[1] > 1
                            else y_proba.ravel()
                        )
                        metrics["pr_auc"] = average_precision_score(
                            y_true, y_prob_positive
                        )
                    except ValueError as e:
                        logger.warning(f"二値分類PR-AUC計算エラー: {e}")
                        metrics["pr_auc"] = 0.0
                else:
                    # 多クラス分類：各クラスのPR-AUCを計算
                    try:
                        pr_aucs = []
                        for i in range(n_classes):
                            y_true_binary = (y_true == i).astype(int)
                            if (
                                np.sum(y_true_binary) > 0
                            ):  # クラスが存在する場合のみ計算
                                pr_auc = average_precision_score(
                                    y_true_binary, y_proba[:, i]
                                )
                                pr_aucs.append(pr_auc)
                                metrics[f"pr_auc_class_{i}"] = pr_auc
                            else:
                                metrics[f"pr_auc_class_{i}"] = 0.0

                        if pr_aucs:
                            metrics["pr_auc"] = np.mean(pr_aucs)  # メイン指標として使用
                            metrics["pr_auc_macro"] = np.mean(pr_aucs)
                            metrics["pr_auc_weighted"] = np.average(
                                pr_aucs,
                                weights=np.bincount(y_true)[np.bincount(y_true) > 0],
                            )
                        else:
                            metrics["pr_auc"] = 0.0
                            metrics["pr_auc_macro"] = 0.0
                            metrics["pr_auc_weighted"] = 0.0
                    except Exception as e:
                        logger.warning(f"多クラスPR-AUC計算エラー: {e}")
                        metrics["pr_auc"] = 0.0
                        metrics["pr_auc_macro"] = 0.0
                        metrics["pr_auc_weighted"] = 0.0

            # Log Loss
            try:
                metrics["log_loss"] = log_loss(y_true, y_proba)
            except ValueError as e:
                logger.warning(f"Log Loss計算エラー: {e}")
                metrics["log_loss"] = 0.0

            # Brier Score
            if n_classes == 2:
                # 二値分類
                try:
                    y_prob_positive = (
                        y_proba[:, 1]
                        if y_proba.ndim > 1 and y_proba.shape[1] > 1
                        else y_proba.ravel()
                    )
                    metrics["brier_score"] = brier_score_loss(y_true, y_prob_positive)
                except ValueError as e:
                    logger.warning(f"二値分類Brier Score計算エラー: {e}")
                    metrics["brier_score"] = 0.0
            else:
                # 多クラス分類では各クラスのBrier Scoreを計算して平均
                try:
                    brier_scores = []
                    for i in range(n_classes):
                        y_true_binary = (y_true == i).astype(int)
                        brier_score = brier_score_loss(y_true_binary, y_proba[:, i])
                        brier_scores.append(brier_score)
                    metrics["brier_score"] = np.mean(brier_scores)
                except Exception as e:
                    logger.warning(f"多クラスBrier Score計算エラー: {e}")
                    metrics["brier_score"] = 0.0

        except Exception as e:
            logger.warning(f"確率指標計算エラー: {e}")
            # エラー時はデフォルト値を設定
            metrics.update(
                {
                    "log_loss": 0.0,
                    "brier_score": 0.0,
                }
            )

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
                # NPV, PPV
                metrics["npv"] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
                metrics["ppv"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            else:  # 多クラス分類の場合（multilabel_confusion_matrixでベクトル化）
                labels = np.unique(y_true)
                mcm = multilabel_confusion_matrix(y_true, y_pred, labels=labels)
                tn = mcm[:, 0, 0].astype(float)
                fp = mcm[:, 0, 1].astype(float)
                fn = mcm[:, 1, 0].astype(float)
                tp = mcm[:, 1, 1].astype(float)

                eps = 1e-12
                specificity_vec = tn / (tn + fp + eps)
                sensitivity_vec = tp / (tp + fn + eps)
                npv_vec = tn / (tn + fn + eps)
                ppv_vec = tp / (tp + fp + eps)

                # 重み付き平均（クラス頻度で重み付け）
                counts = np.array(
                    [(y_true == lab).sum() for lab in labels], dtype=float
                )
                weights = counts / (counts.sum() + eps)

                metrics["specificity"] = float(
                    np.average(specificity_vec, weights=weights)
                )
                metrics["sensitivity"] = float(
                    np.average(sensitivity_vec, weights=weights)
                )
                metrics["npv"] = float(np.average(npv_vec, weights=weights))
                metrics["ppv"] = float(np.average(ppv_vec, weights=weights))

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
            # typing.castを使って型を明示的に指定し、型チェッカーのエラーを解決
            from typing import cast

            precision, recall, f1_score, support = cast(
                tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                precision_recall_fscore_support(
                    y_true,
                    y_pred,
                    average=None,
                    zero_division=self.config.zero_division,
                ),
            )

            # 各変数を明示的にnumpy配列として扱う
            precision_array: np.ndarray = precision
            recall_array: np.ndarray = recall
            f1_array: np.ndarray = f1_score  # f1 -> f1_score でスペルチェック問題を解決
            support_array: np.ndarray = support

            classes = np.unique(y_true)
            per_class_metrics = {}

            for i, class_label in enumerate(classes):
                # class_namesの型を明示的にチェック
                if class_names is not None and i < len(class_names):
                    class_name = class_names[i]
                else:
                    class_name = f"class_{class_label}"

                per_class_metrics[class_name] = {
                    "precision": float(precision_array[i]),
                    "recall": float(recall_array[i]),
                    "f1_score": float(f1_array[i]),  # f1 -> f1_array に修正
                    "support": int(support_array[i]),
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


# グローバルインスタンス（統合されたメトリクス計算器・収集器）
metrics_collector = MetricsCalculator()
