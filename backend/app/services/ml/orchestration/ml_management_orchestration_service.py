"""
ML管理 オーケストレーションサービス
"""

import logging
from typing import List, Dict, Any
import os
from urllib.parse import unquote
from datetime import datetime
from fastapi import HTTPException

from app.services.ml.model_manager import model_manager
from app.services.auto_strategy.services.ml_orchestrator import MLOrchestrator

from app.services.ml.feature_engineering.automl_feature_analyzer import (
    AutoMLFeatureAnalyzer,
)

from app.services.ml.config.ml_config_manager import ml_config_manager


logger = logging.getLogger(__name__)


class MLManagementOrchestrationService:
    """
    ML管理機能のビジネスロジックを集約したサービスクラス
    """

    def __init__(self):
        self.ml_orchestrator = MLOrchestrator()

    async def get_formatted_models(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        学習済みモデルの一覧を取得し、フロントエンド表示用に整形する
        """
        models = model_manager.list_models("*")

        # モデル情報を整形
        formatted_models = []
        for model in models:
            # 基本情報
            model_info = {
                "id": model["name"],
                "name": model["name"],
                "path": model["path"],
                "size_mb": model["size_mb"],
                "modified_at": model["modified_at"].isoformat(),
                "directory": model["directory"],
                "is_active": self._is_active_model(model),  # アクティブモデルの判定
            }

            # モデルの詳細情報を取得
            try:
                model_data = model_manager.load_model(model["path"])
                if model_data and "metadata" in model_data:
                    metadata = model_data["metadata"]

                    # 性能指標を追加
                    model_info.update(
                        {
                            "accuracy": metadata.get("accuracy", 0.0),
                            "precision": metadata.get("precision", 0.0),
                            "recall": metadata.get("recall", 0.0),
                            "f1_score": metadata.get("f1_score", 0.0),
                            "feature_count": metadata.get("feature_count", 0),
                            "model_type": metadata.get("model_type", "LightGBM"),
                            "training_samples": metadata.get("training_samples", 0),
                        }
                    )

                    # classification_reportから詳細指標を抽出
                    if "classification_report" in metadata:
                        report = metadata["classification_report"]
                        if isinstance(report, dict) and "macro avg" in report:
                            macro_avg = report["macro avg"]
                            model_info.update(
                                {
                                    "precision": macro_avg.get(
                                        "precision", model_info.get("precision", 0.0)
                                    ),
                                    "recall": macro_avg.get(
                                        "recall", model_info.get("recall", 0.0)
                                    ),
                                    "f1_score": macro_avg.get(
                                        "f1-score", model_info.get("f1_score", 0.0)
                                    ),
                                }
                            )

                    logger.info(
                        f"✅ モデル詳細情報を取得: {model['name']} - 精度: {model_info.get('accuracy', 0.0):.3f}, F1: {model_info.get('f1_score', 0.0):.3f}, 特徴量: {model_info.get('feature_count', 0)}個"
                    )

            except Exception as e:
                logger.warning(f"モデル詳細情報取得エラー {model['name']}: {e}")
                # エラーの場合はデフォルト値を設定
                model_info.update(
                    {
                        "accuracy": 0.0,
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1_score": 0.0,
                        "feature_count": 0,
                        "model_type": "Unknown",
                        "training_samples": 0,
                    }
                )

            formatted_models.append(model_info)

        return {"models": formatted_models}

    async def delete_model(self, model_id: str) -> Dict[str, str]:
        """
        指定されたモデルを削除
        """
        logger.info(f"モデル削除要求: {model_id}")

        decoded_model_id = unquote(model_id)

        models = model_manager.list_models("*")
        target_model = None

        for model in models:
            if model["name"] == decoded_model_id or model["name"] == model_id:
                target_model = model
                break

        if not target_model:
            logger.warning(f"モデルが見つかりません: {decoded_model_id}")
            logger.info(f"利用可能なモデル: {[m['name'] for m in models]}")
            raise HTTPException(
                status_code=404, detail=f"モデルが見つかりません: {decoded_model_id}"
            )

        if not os.path.exists(target_model["path"]):
            logger.warning(f"モデルファイルが存在しません: {target_model['path']}")
            raise HTTPException(status_code=404, detail="モデルファイルが存在しません")

        try:
            os.remove(target_model["path"])
            logger.info(f"モデル削除完了: {decoded_model_id} -> {target_model['path']}")
            return {"message": "モデルが削除されました"}
        except Exception as e:
            logger.error(f"モデルファイル削除エラー: {e}")
            raise HTTPException(
                status_code=500, detail="モデルファイルの削除に失敗しました"
            )

    async def get_ml_status(self) -> Dict[str, Any]:
        """
        MLモデルの現在の状態を取得
        """
        status = self.ml_orchestrator.get_model_status()

        latest_model = model_manager.get_latest_model("*")

        if latest_model and os.path.exists(latest_model):
            try:
                model_data = model_manager.load_model(latest_model)
                if model_data and "metadata" in model_data:
                    metadata = model_data["metadata"]
                    model_info = {
                        "accuracy": metadata.get("accuracy", 0.0),
                        "precision": metadata.get("precision", 0.0),
                        "recall": metadata.get("recall", 0.0),
                        "f1_score": metadata.get("f1_score", 0.0),
                        "auc_score": metadata.get("auc_score", 0.0),
                        "auc_roc": metadata.get("auc_roc", 0.0),
                        "auc_pr": metadata.get("auc_pr", 0.0),
                        "balanced_accuracy": metadata.get("balanced_accuracy", 0.0),
                        "matthews_corrcoef": metadata.get("matthews_corrcoef", 0.0),
                        "cohen_kappa": metadata.get("cohen_kappa", 0.0),
                        "specificity": metadata.get("specificity", 0.0),
                        "sensitivity": metadata.get("sensitivity", 0.0),
                        "npv": metadata.get("npv", 0.0),
                        "ppv": metadata.get("ppv", 0.0),
                        "log_loss": metadata.get("log_loss", 0.0),
                        "brier_score": metadata.get("brier_score", 0.0),
                        "model_type": metadata.get("model_type", "LightGBM"),
                        "last_updated": datetime.fromtimestamp(
                            os.path.getmtime(latest_model)
                        ).isoformat(),
                        "training_samples": metadata.get("training_samples", 0),
                        "test_samples": metadata.get("test_samples", 0),
                        "file_size_mb": os.path.getsize(latest_model) / (1024 * 1024),
                        "feature_count": metadata.get("feature_count", 0),
                        "num_classes": metadata.get("num_classes", 2),
                        "best_iteration": metadata.get("best_iteration", 0),
                        "train_test_split": metadata.get("train_test_split", 0.8),
                        "random_state": metadata.get("random_state", 42),
                        "feature_importance": metadata.get("feature_importance", {}),
                        "classification_report": metadata.get(
                            "classification_report", {}
                        ),
                    }

                    if "classification_report" in metadata:
                        report = metadata["classification_report"]
                        if isinstance(report, dict) and "macro avg" in report:
                            macro_avg = report["macro avg"]
                            if model_info["precision"] == 0.0:
                                model_info["precision"] = macro_avg.get(
                                    "precision", 0.0
                                )
                            if model_info["recall"] == 0.0:
                                model_info["recall"] = macro_avg.get("recall", 0.0)
                            if model_info["f1_score"] == 0.0:
                                model_info["f1_score"] = macro_avg.get("f1-score", 0.0)

                    logger.info(
                        f"📊 ML Status API - モデル詳細情報を取得: 精度={model_info['accuracy']:.4f}, F1={model_info['f1_score']:.4f}, 特徴量={model_info['feature_count']}個"
                    )
                else:
                    model_info = {
                        "accuracy": 0.0,
                        "model_type": "LightGBM",
                        "last_updated": datetime.fromtimestamp(
                            os.path.getmtime(latest_model)
                        ).isoformat(),
                        "training_samples": 0,
                        "file_size_mb": os.path.getsize(latest_model) / (1024 * 1024),
                        "feature_count": 0,
                    }
                status["model_info"] = model_info

                # ModelManagerから直接メタデータを取得
                model_data = model_manager.load_model(latest_model)
                if model_data and "metadata" in model_data:
                    metadata = model_data["metadata"]
                    # 新しい形式の性能指標を抽出（全ての評価指標を含む）
                    performance_metrics = {
                        # 基本指標
                        "accuracy": metadata.get("accuracy", 0.0),
                        "precision": metadata.get("precision", 0.0),
                        "recall": metadata.get("recall", 0.0),
                        "f1_score": metadata.get("f1_score", 0.0),
                        # AUC指標
                        "auc_roc": metadata.get("auc_roc", 0.0),
                        "auc_pr": metadata.get("auc_pr", 0.0),
                        # 高度な指標
                        "balanced_accuracy": metadata.get("balanced_accuracy", 0.0),
                        "matthews_corrcoef": metadata.get("matthews_corrcoef", 0.0),
                        "cohen_kappa": metadata.get("cohen_kappa", 0.0),
                        # 専門指標
                        "specificity": metadata.get("specificity", 0.0),
                        "sensitivity": metadata.get("sensitivity", 0.0),
                        "npv": metadata.get("npv", 0.0),
                        "ppv": metadata.get("ppv", 0.0),
                        # 確率指標
                        "log_loss": metadata.get("log_loss", 0.0),
                        "brier_score": metadata.get("brier_score", 0.0),
                        # その他
                        "loss": metadata.get("loss", 0.0),
                        "val_accuracy": metadata.get("val_accuracy", 0.0),
                        "val_loss": metadata.get("val_loss", 0.0),
                        "training_time": metadata.get("training_time", 0.0),
                    }
                    status["performance_metrics"] = performance_metrics
                else:
                    # デフォルト値を設定
                    status["performance_metrics"] = {
                        "accuracy": 0.0,
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1_score": 0.0,
                        "auc_roc": 0.0,
                        "auc_pr": 0.0,
                        "balanced_accuracy": 0.0,
                        "matthews_corrcoef": 0.0,
                        "cohen_kappa": 0.0,
                    }
            except Exception as e:
                logger.warning(f"モデル情報取得エラー: {e}")
                status["model_info"] = {
                    "accuracy": 0.0,
                    "model_type": "Unknown",
                    "last_updated": datetime.fromtimestamp(
                        os.path.getmtime(latest_model)
                    ).isoformat(),
                    "training_samples": 0,
                    "file_size_mb": os.path.getsize(latest_model) / (1024 * 1024),
                    "feature_count": 0,
                }
                status["performance_metrics"] = {
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1_score": 0.0,
                    "auc_score": 0.0,
                    "auc_roc": 0.0,
                    "auc_pr": 0.0,
                    "balanced_accuracy": 0.0,
                    "matthews_corrcoef": 0.0,
                    "cohen_kappa": 0.0,
                    "specificity": 0.0,
                    "sensitivity": 0.0,
                    "npv": 0.0,
                    "ppv": 0.0,
                    "log_loss": 0.0,
                    "brier_score": 0.0,
                    "loss": 0.0,
                    "val_accuracy": 0.0,
                    "val_loss": 0.0,
                    "training_time": 0.0,
                }

        return status

    async def get_feature_importance(self, top_n: int = 10) -> Dict[str, Any]:
        """
        特徴量重要度を取得
        """
        feature_importance = self.ml_orchestrator.get_feature_importance(top_n)
        return {"feature_importance": feature_importance}

    async def get_automl_feature_analysis(self, top_n: int = 20) -> Dict[str, Any]:
        """
        AutoML特徴量分析結果を取得
        """
        feature_importance = self.ml_orchestrator.get_feature_importance(100)
        if not feature_importance:
            return {"error": "特徴量重要度データがありません"}

        analyzer = AutoMLFeatureAnalyzer()
        analysis_result = analyzer.analyze_feature_importance(feature_importance, top_n)
        return analysis_result

    async def cleanup_old_models(self) -> Dict[str, str]:
        """
        古いモデルファイルをクリーンアップ
        """
        model_manager.cleanup_expired_models()
        return {"message": "古いモデルファイルが削除されました"}

    def get_ml_config_dict(self) -> Dict[str, Any]:
        """
        ML設定を辞書形式で取得

        Returns:
            ML設定辞書
        """
        return ml_config_manager.get_config_dict()

    async def update_ml_config(self, config_updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        ML設定を更新

        Args:
            config_updates: 更新する設定項目

        Returns:
            更新結果
        """
        try:
            success = ml_config_manager.update_config(config_updates)

            if success:
                return {
                    "success": True,
                    "message": "ML設定が正常に更新されました",
                    "updated_config": ml_config_manager.get_config_dict(),
                }
            else:
                return {"success": False, "message": "ML設定の更新に失敗しました"}

        except Exception as e:
            logger.error(f"ML設定更新エラー: {e}")
            return {
                "success": False,
                "message": f"ML設定の更新中にエラーが発生しました: {e}",
            }

    async def reset_ml_config(self) -> Dict[str, Any]:
        """
        ML設定をデフォルト値にリセット

        Returns:
            リセット結果
        """
        try:
            success = ml_config_manager.reset_config()

            if success:
                return {
                    "success": True,
                    "message": "ML設定がデフォルト値にリセットされました",
                    "config": ml_config_manager.get_config_dict(),
                }
            else:
                return {"success": False, "message": "ML設定のリセットに失敗しました"}

        except Exception as e:
            logger.error(f"ML設定リセットエラー: {e}")
            return {
                "success": False,
                "message": f"ML設定のリセット中にエラーが発生しました: {e}",
            }

    def _is_active_model(self, model: Dict[str, Any]) -> bool:
        """
        モデルがアクティブかどうかを判定

        Args:
            model: モデル情報辞書

        Returns:
            アクティブの場合はTrue、そうでない場合はFalse
        """
        try:
            # 最新のモデルファイルと比較してアクティブか判定
            latest_model = model_manager.get_latest_model("*")
            if latest_model:
                # パスが一致する場合はアクティブと判定
                return model["path"] == latest_model

            # モデルが1つだけの場合はアクティブと判定
            all_models = model_manager.list_models("*")
            return len(all_models) == 1 and all_models[0]["path"] == model["path"]

        except Exception as e:
            logger.warning(f"アクティブモデル判定エラー: {e}")
            return False
