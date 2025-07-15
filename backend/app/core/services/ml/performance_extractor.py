"""
性能指標抽出サービス

モデルメタデータから実際の性能指標を抽出する機能を提供
"""

import logging
from typing import Dict, Any, Optional
import joblib
import os

logger = logging.getLogger(__name__)


class PerformanceExtractor:
    """モデルメタデータから性能指標を抽出するクラス"""

    @staticmethod
    def extract_performance_metrics(model_path: str) -> Dict[str, float]:
        """
        モデルファイルから実際の性能指標を抽出

        Args:
            model_path: モデルファイルのパス

        Returns:
            性能指標の辞書
        """
        print(f"=== DEBUG: extract_performance_metrics呼び出し ===")
        print(f"model_path: {model_path}")

        try:
            if not os.path.exists(model_path):
                logger.warning(f"モデルファイルが見つかりません: {model_path}")
                print(f"=== DEBUG: モデルファイルが見つかりません ===")
                return PerformanceExtractor._get_default_metrics()

            print(f"=== DEBUG: モデルファイル読み込み開始 ===")
            # モデルデータを読み込み
            model_data = joblib.load(model_path)

            if not isinstance(model_data, dict):
                logger.warning("古い形式のモデルファイル（直接モデルオブジェクト）")
                print(f"=== DEBUG: 古い形式のモデルファイル ===")
                return PerformanceExtractor._get_default_metrics()

            metadata = model_data.get("metadata", {})
            print(f"=== DEBUG: メタデータ取得完了 ===")
            print(f"メタデータキー: {list(metadata.keys())}")

            # 新しい形式の性能指標を確認
            if PerformanceExtractor._has_new_format_metrics(metadata):
                print(f"=== DEBUG: 新しい形式の性能指標を使用 ===")
                return PerformanceExtractor._extract_new_format_metrics(metadata)

            print(f"=== DEBUG: classification_reportから抽出を開始 ===")
            # 古い形式の場合、classification_reportから抽出
            return PerformanceExtractor._extract_from_classification_report(metadata)

        except Exception as e:
            logger.error(f"性能指標抽出エラー: {e}")
            print(f"=== DEBUG: エラー発生: {e} ===")
            import traceback

            traceback.print_exc()
            return PerformanceExtractor._get_default_metrics()

    @staticmethod
    def _has_new_format_metrics(metadata: Dict[str, Any]) -> bool:
        """新しい形式の性能指標が存在するかチェック"""
        required_keys = ["precision", "recall", "f1_score"]
        return all(key in metadata and metadata[key] > 0 for key in required_keys)

    @staticmethod
    def _extract_new_format_metrics(metadata: Dict[str, Any]) -> Dict[str, float]:
        """新しい形式のメタデータから性能指標を抽出"""
        return {
            # 基本指標
            "accuracy": metadata.get("accuracy", 0.0),
            "precision": metadata.get("precision", 0.0),
            "recall": metadata.get("recall", 0.0),
            "f1_score": metadata.get("f1_score", 0.0),
            # AUC指標
            "auc_score": metadata.get("auc_score", 0.0),  # 後方互換性
            "auc_roc": metadata.get("auc_roc", metadata.get("auc_score", 0.0)),
            "auc_pr": metadata.get("auc_pr", 0.0),
            # 高度な指標
            "balanced_accuracy": metadata.get("balanced_accuracy", 0.0),
            "matthews_corrcoef": metadata.get("matthews_corrcoef", 0.0),
            "cohen_kappa": metadata.get("cohen_kappa", 0.0),
            # 専門指標
            "specificity": metadata.get("specificity", 0.0),
            "sensitivity": metadata.get(
                "sensitivity", metadata.get("recall", 0.0)
            ),  # sensitivityはrecallと同じ
            "npv": metadata.get("npv", 0.0),
            "ppv": metadata.get(
                "ppv", metadata.get("precision", 0.0)
            ),  # ppvはprecisionと同じ
            # 確率指標
            "log_loss": metadata.get("log_loss", 0.0),
            "brier_score": metadata.get("brier_score", 0.0),
            # その他
            "loss": metadata.get("loss", 0.0),
            "val_accuracy": metadata.get("val_accuracy", 0.0),
            "val_loss": metadata.get("val_loss", 0.0),
            "training_time": metadata.get("training_time", 0.0),
        }

    @staticmethod
    def _extract_from_classification_report(
        metadata: Dict[str, Any],
    ) -> Dict[str, float]:
        """classification_reportから性能指標を抽出"""
        class_report = metadata.get("classification_report", {})

        if not class_report or "weighted avg" not in class_report:
            logger.warning(
                "classification_reportが見つからないか、weighted avgが存在しません"
            )
            return PerformanceExtractor._get_default_metrics()

        weighted_avg = class_report["weighted avg"]

        # classification_reportから実際の値を抽出
        precision = weighted_avg.get("precision", 0.0)
        recall = weighted_avg.get("recall", 0.0)
        f1_score = weighted_avg.get("f1-score", 0.0)  # ハイフン付きに注意

        logger.info(
            f"classification_reportから抽出: precision={precision}, recall={recall}, f1_score={f1_score}"
        )
        print(f"=== DEBUG: classification_reportから抽出 ===")
        print(f"precision: {precision}")
        print(f"recall: {recall}")
        print(f"f1_score: {f1_score}")
        print(f"=== END DEBUG ===")

        return {
            # 基本指標
            "accuracy": metadata.get("accuracy", 0.0),
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            # AUC指標
            "auc_score": 0.0,  # classification_reportにはAUCが含まれていない
            "auc_roc": 0.0,
            "auc_pr": 0.0,
            # 高度な指標
            "balanced_accuracy": 0.0,
            "matthews_corrcoef": 0.0,
            "cohen_kappa": 0.0,
            # 専門指標
            "specificity": 0.0,
            "sensitivity": recall,  # sensitivityはrecallと同じ
            "npv": 0.0,
            "ppv": precision,  # ppvはprecisionと同じ
            # 確率指標
            "log_loss": 0.0,
            "brier_score": 0.0,
            # その他
            "loss": 0.0,
            "val_accuracy": 0.0,
            "val_loss": 0.0,
            "training_time": 0.0,
        }

    @staticmethod
    def _get_default_metrics() -> Dict[str, float]:
        """デフォルトの性能指標を返す"""
        return {
            # 基本指標
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            # AUC指標
            "auc_score": 0.0,
            "auc_roc": 0.0,
            "auc_pr": 0.0,
            # 高度な指標
            "balanced_accuracy": 0.0,
            "matthews_corrcoef": 0.0,
            "cohen_kappa": 0.0,
            # 専門指標
            "specificity": 0.0,
            "sensitivity": 0.0,
            "npv": 0.0,
            "ppv": 0.0,
            # 確率指標
            "log_loss": 0.0,
            "brier_score": 0.0,
            # その他
            "loss": 0.0,
            "val_accuracy": 0.0,
            "val_loss": 0.0,
            "training_time": 0.0,
        }


# グローバルインスタンス
performance_extractor = PerformanceExtractor()
