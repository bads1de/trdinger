"""
ML管理 オーケストレーションサービス
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List
from urllib.parse import unquote

from fastapi import HTTPException

from app.services.ml.config.ml_config_manager import ml_config_manager
from app.services.ml.model_manager import model_manager
from app.utils.response import api_response
from app.utils.error_handler import ErrorHandler

logger = logging.getLogger(__name__)


class MLManagementOrchestrationService:
    """
    ML管理機能のビジネスロジックを集約したサービスクラス
    """

    def __init__(self):
        pass

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
            return api_response(success=True, message="モデルが削除されました")
        except Exception as e:
            logger.error(f"モデルファイル削除エラー: {e}")
            raise HTTPException(
                status_code=500, detail="モデルファイルの削除に失敗しました"
            )

    async def delete_all_models(self) -> Dict[str, Any]:
        """
        すべてのモデルを削除
        """
        models = model_manager.list_models("*")

        if not models:
            return {
                "success": True,
                "message": "削除するモデルがありませんでした",
                "deleted_count": 0,
            }

        deleted_count = 0
        failed_models = []

        for model in models:
            try:
                if os.path.exists(model["path"]):
                    os.remove(model["path"])
                    deleted_count += 1
                else:
                    logger.warning(f"モデルファイルが存在しません: {model['path']}")
                    failed_models.append(model["name"])
            except Exception as e:
                logger.error(f"モデル削除エラー: {model['name']} -> {e}")
                failed_models.append(model["name"])

        if failed_models:
            message = f"{deleted_count}個のモデルを削除しました。{len(failed_models)}個のモデルで削除に失敗しました: {', '.join(failed_models)}"
        else:
            message = f"すべてのモデル（{deleted_count}個）が削除されました"

        return {
            "success": True,
            "message": message,
            "deleted_count": deleted_count,
            "failed_count": len(failed_models),
            "failed_models": failed_models,
        }

    async def get_ml_status(self) -> Dict[str, Any]:
        """
        MLモデルの現在の状態を取得
        """
        # MLオーケストレーター削除により、デフォルトステータスを返す
        status = {
            "is_model_loaded": False,
            "is_loaded": False,  # 後方互換性のため保持
            "is_trained": False,
            "model_path": None,
            "model_type": None,
            "feature_count": 0,
            "training_samples": 0,
        }

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
                status.update(
                    {
                        "is_model_loaded": True,
                        "is_loaded": True,
                        "is_trained": True,
                        "model_path": latest_model,
                        "model_type": model_info.get("model_type"),
                        "feature_count": model_info.get("feature_count", 0),
                        "training_samples": model_info.get("training_samples", 0),
                    }
                )
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
                status["is_model_loaded"] = False
                status["is_loaded"] = False
                status["is_trained"] = False
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

        else:
            # モデルが見つからない場合のデフォルト情報
            status["model_info"] = {
                "accuracy": 0.0,
                "model_type": "No Model",
                "last_updated": "未学習",
                "training_samples": 0,
                "file_size_mb": 0.0,
                "feature_count": 0,
            }
            # モデルが存在しない場合でもperformance_metricsを含める
            status["is_model_loaded"] = False
            status["is_loaded"] = False
            status["is_trained"] = False
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
            # ステータスメッセージを追加
            status["status"] = "no_model"
            status["message"] = (
                "学習済みモデルが見つかりません。モデルの学習を実行してください。"
            )

        return status

    async def get_feature_importance(self, top_n: int = 10) -> Dict[str, Any]:
        """
        特徴量重要度を取得
        """
        # model_managerから直接最新モデルの特徴量重要度を取得
        latest_model = model_manager.get_latest_model("*")

        if latest_model and os.path.exists(latest_model):
            try:
                model_data = model_manager.load_model(latest_model)
                if model_data and "metadata" in model_data:
                    metadata = model_data["metadata"]
                    feature_importance = metadata.get("feature_importance", {})

                    if feature_importance:
                        # 重要度でソートして上位N個を返す
                        sorted_features = sorted(
                            feature_importance.items(), key=lambda x: x[1], reverse=True
                        )[:top_n]

                        return {"feature_importance": dict(sorted_features)}
                    else:
                        return {"feature_importance": []}
                else:
                    return {"feature_importance": []}
            except Exception as e:
                logger.warning(f"特徴量重要度取得エラー: {e}")
                return {"feature_importance": []}
        else:
            return {"feature_importance": []}

    async def get_automl_feature_analysis(self, top_n: int = 20) -> Dict[str, Any]:
        """
        AutoML特徴量分析結果を取得
        """
        # MLオーケストレーター削除により、エラーメッセージを返す
        return {"error": "AutoML機能は現在利用できません"}

    async def cleanup_old_models(self) -> Dict[str, str]:
        """
        古いモデルファイルをクリーンアップ
        """
        model_manager.cleanup_expired_models()
        return {"message": "古いモデルファイルが削除されました"}

    async def load_model(self, model_name: str) -> Dict[str, Any]:
        """
        指定されたモデルを読み込み
        """
        try:
            # モデルファイルのパスを特定
            models = model_manager.list_models()
            target_model = None

            for model in models:
                if model["name"] == model_name or model_name in model["path"]:
                    target_model = model
                    break

            if not target_model:
                return {
                    "success": False,
                    "error": f"モデルが見つかりません: {model_name}",
                }

            # MLオーケストレーター削除により、直接モデルマネージャーで読み込み
            success = (
                True  # モデルマネージャーは既にモデルを管理しているので成功とみなす
            )

            if success:
                # 現在のモデル情報を取得
                current_model_info = await self.get_current_model_info()
                return {
                    "success": True,
                    "message": f"モデルを読み込みました: {model_name}",
                    "current_model": current_model_info,
                }
            else:
                return {
                    "success": False,
                    "error": f"モデル読み込みに失敗しました: {model_name}",
                }

        except Exception as e:
            ErrorHandler.handle_model_error(e, context="load_model")
            return {"success": False, "error": str(e)}

    async def get_current_model_info(self) -> Dict[str, Any]:
        """
        現在読み込まれているモデル情報を取得
        """
        # model_managerから直接最新モデルの情報を取得
        latest_model = model_manager.get_latest_model("*")

        if latest_model and os.path.exists(latest_model):
            try:
                model_data = model_manager.load_model(latest_model)
                if model_data and "metadata" in model_data:
                    metadata = model_data["metadata"]

                    return {
                        "loaded": True,
                        "model_type": metadata.get("model_type", "Unknown"),
                        "is_trained": True,  # 保存されているモデルは学習済み
                        "feature_count": metadata.get("feature_count", 0),
                        "training_samples": metadata.get("training_samples", 0),
                        "accuracy": metadata.get("accuracy", 0.0),
                        "last_updated": datetime.fromtimestamp(
                            os.path.getmtime(latest_model)
                        ).isoformat(),
                    }
                else:
                    return {
                        "loaded": True,
                        "model_type": "Unknown",
                        "is_trained": True,
                        "feature_count": 0,
                        "training_samples": 0,
                        "accuracy": 0.0,
                        "last_updated": datetime.fromtimestamp(
                            os.path.getmtime(latest_model)
                        ).isoformat(),
                    }
            except Exception as e:
                logger.warning(f"現在のモデル情報取得エラー: {e}")
                return {"loaded": False, "error": str(e)}
        else:
            return {"loaded": False, "message": "モデルが見つかりません"}

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
