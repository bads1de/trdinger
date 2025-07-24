"""
ML管理 オーケストレーションサービス
"""

import logging
from typing import List, Dict, Any

from app.core.services.ml.model_manager import model_manager

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
                "is_active": False,  # TODO: アクティブモデルの判定ロジック
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
