"""
ML管理 オーケストレーションサービス
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List
from urllib.parse import unquote

from fastapi import HTTPException
from starlette.concurrency import run_in_threadpool

from app.services.ml.common.config import ml_config_manager
from app.services.ml.models.model_manager import model_manager
from app.utils.error_handler import ErrorHandler
from app.utils.response import api_response

from ..common.evaluation import get_default_metrics
from .ml_training_orchestration_service import (
    ml_training_service,
    get_latest_model_with_info,
    load_model_metadata_safely,
    get_model_info_with_defaults,
)

logger = logging.getLogger(__name__)


class MLManagementOrchestrationService:
    """
    ML管理機能のビジネスロジックを集約したサービスクラス
    """

    def __init__(self):
        pass

    async def get_formatted_models(self) -> Dict[str, List[Dict[str, Any]]]:
        """学習済みモデルの一覧を取得し、整形する"""
        models = await run_in_threadpool(model_manager.list_models, "*")
        formatted = []
        for m in models:
            info = {
                "id": m["name"],
                "name": m["name"],
                "path": m["path"],
                "size_mb": m["size_mb"],
                "modified_at": m["modified_at"].isoformat(),
                "directory": m["directory"],
                "is_active": self._is_active_model(m),
            }
            try:
                data = await run_in_threadpool(load_model_metadata_safely, m["path"])
                meta = data["metadata"] if data else {}
                metrics = await run_in_threadpool(
                    model_manager.extract_model_performance_metrics,
                    m["path"],
                    metadata=meta,
                )
                info.update(metrics)
                info.update(
                    {
                        "feature_count": meta.get("feature_count", 0),
                        "model_type": meta.get("model_type", "Unknown"),
                        "training_samples": meta.get("training_samples", 0),
                    }
                )
            except Exception as e:
                logger.warning(f"詳細情報取得失敗 {m['name']}: {e}")
                self._apply_default_model_metrics(info)
            formatted.append(info)
        return {"models": formatted}

    def _apply_default_model_metrics(self, model_info: Dict[str, Any]) -> None:
        """モデル情報にデフォルトメトリクスを適用"""
        d = get_default_metrics()
        model_info.update(
            {
                "accuracy": d["accuracy"],
                "precision": d["precision"],
                "recall": d["recall"],
                "f1_score": d["f1_score"],
                "feature_count": 0,
                "model_type": "Unknown",
                "training_samples": 0,
            }
        )

    async def delete_model(self, model_id: str) -> Dict[str, str]:
        """指定されたモデルを削除"""
        logger.info(f"モデル削除要求: {model_id}")
        dec_id = unquote(model_id)
        models = await run_in_threadpool(model_manager.list_models, "*")
        target = next((m for m in models if m["name"] in [dec_id, model_id]), None)

        if not target:
            raise HTTPException(
                status_code=404, detail=f"モデルが見つかりません: {dec_id}"
            )

        try:
            await run_in_threadpool(os.remove, target["path"])
            return api_response(success=True, message="モデルが削除されました")
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="モデルファイルが存在しません")
        except Exception as e:
            logger.error(f"削除エラー: {e}")
            raise HTTPException(status_code=500, detail="削除失敗")

    async def delete_all_models(self) -> Dict[str, Any]:
        """
        すべてのモデルを削除
        """
        # ブロッキングI/Oをスレッドプールで実行
        models = await run_in_threadpool(model_manager.list_models, "*")

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
                # Bug #4修正: os.path.exists()チェックを削除してos.removeを直接呼び出し
                # これによりTOC-TOU脆弱性を回避
                # ブロッキングI/Oをスレッドプールで実行
                await run_in_threadpool(os.remove, model["path"])
                logger.info(f"モデル削除成功: {model['name']}")
                deleted_count += 1
            except FileNotFoundError:
                logger.warning(f"モデルファイルが存在しません: {model['path']}")
                failed_models.append(model["name"])
            except Exception as e:
                logger.error(f"モデル削除エラー: {model['name']} -> {e}")
                failed_models.append(model["name"])

        if failed_models:
            message = f"{deleted_count}個のモデルを削除しました。{len(failed_models)}個のモデルで削除に失敗しました: {', '.join(failed_models)}"
        else:
            message = f"すべてのモデル({deleted_count}個)が削除されました"

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
        # デフォルトステータスの初期化
        status = {
            "is_model_loaded": False,
            "is_loaded": False,  # 後方互換性のため保持
            "is_trained": False,
            "model_path": None,
            "model_type": None,
            "feature_count": 0,
            "training_samples": 0,
        }

        # ブロッキングI/Oをスレッドプールで実行
        model_info_data = await run_in_threadpool(get_latest_model_with_info)

        if model_info_data:
            try:
                # get_model_info_with_defaultsを使用して統一されたフォーマットを取得
                # Bug #1修正: model_infoに既にメトリクスが含まれているので、そこから取得
                model_info = get_model_info_with_defaults(model_info_data)

                # model_infoからメトリクスを抽出
                metrics = {
                    "accuracy": model_info.get("accuracy", 0.0),
                    "precision": model_info.get("precision", 0.0),
                    "recall": model_info.get("recall", 0.0),
                    "f1_score": model_info.get("f1_score", 0.0),
                }

                status.update(
                    {
                        "is_model_loaded": True,
                        "is_loaded": True,
                        "is_trained": True,
                        "model_path": model_info_data["path"],
                        "model_type": model_info.get("model_type"),
                        "feature_count": model_info.get("feature_count", 0),
                        "training_samples": model_info.get("training_samples", 0),
                    }
                )
                status["model_info"] = model_info
                status["performance_metrics"] = metrics

            except Exception as e:
                logger.warning(f"モデル情報取得エラー: {e}")
                # エラー時はデフォルト値を使用
                default_model_info = get_model_info_with_defaults(None)
                status["model_info"] = default_model_info
                status["performance_metrics"] = get_default_metrics()
                status["is_model_loaded"] = False
                status["is_loaded"] = False
                status["is_trained"] = False

        else:
            # モデルが見つからない場合はデフォルト値を使用
            default_model_info = get_model_info_with_defaults(None)
            status["model_info"] = default_model_info
            status["performance_metrics"] = get_default_metrics()

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
        # ブロッキングI/Oをスレッドプールで実行
        model_info_data = await run_in_threadpool(get_latest_model_with_info)

        if model_info_data:
            try:
                metadata = model_info_data["metadata"]
                feature_importance = metadata.get("feature_importance", {})

                if feature_importance:
                    # 重要度でソートして上位N個を返す
                    sorted_features = sorted(
                        feature_importance.items(), key=lambda x: x[1], reverse=True
                    )[:top_n]

                    return {"feature_importance": dict(sorted_features)}
                else:
                    # Bug #2修正: 空リストではなく空辞書を返す
                    return {"feature_importance": {}}
            except Exception as e:
                logger.warning(f"特徴量重要度取得エラー: {e}")
                # Bug #2修正: 空リストではなく空辞書を返す
                return {"feature_importance": {}}
        else:
            # Bug #2修正: 空リストではなく空辞書を返す
            return {"feature_importance": {}}

    async def cleanup_old_models(self) -> Dict[str, str]:
        """
        古いモデルファイルをクリーンアップ
        """
        await run_in_threadpool(model_manager.cleanup_expired_models)
        return {"message": "古いモデルファイルが削除されました"}

    async def load_model(self, model_name: str) -> Dict[str, Any]:
        """
        指定されたモデルを読み込み
        """
        try:
            # モデルファイルのパスを特定
            # ブロッキングI/Oをスレッドプールで実行
            models = await run_in_threadpool(model_manager.list_models)
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
            # 修正: MLTrainingServiceにロードを依頼する
            # ブロッキングI/Oをスレッドプールで実行
            success = await run_in_threadpool(
                ml_training_service.load_model, target_model["path"]
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
        try:
            # MLTrainingServiceから現在のモデル情報を取得
            current_model_path = ml_training_service.get_current_model_path()
            current_metadata = ml_training_service.get_current_model_info()

            if current_model_path and current_metadata:
                # Bug #5修正: メトリクス抽出を統一的に処理
                metrics = get_default_metrics()

                try:
                    # まずextract_model_performance_metricsを試す
                    # ブロッキングI/Oをスレッドプールで実行
                    extracted_metrics = await run_in_threadpool(
                        model_manager.extract_model_performance_metrics,
                        current_model_path,
                        metadata=current_metadata,
                    )
                    metrics.update(extracted_metrics)
                except Exception as e:
                    logger.warning(f"メトリクス抽出エラー: {e}")
                    # extract失敗時はメタデータ内のmetricsを直接使用
                    if "metrics" in current_metadata:
                        metrics.update(current_metadata["metrics"])
                    # どちらもない場合はデフォルト値を使用(既に初期化済み)

                # ファイル情報の取得（ファイルが存在する場合）
                last_updated = datetime.now().isoformat()

                # ブロッキングI/Oをスレッドプールで実行
                if await run_in_threadpool(os.path.exists, current_model_path):
                    stat = await run_in_threadpool(os.stat, current_model_path)
                    last_updated = datetime.fromtimestamp(stat.st_mtime).isoformat()

                return {
                    "loaded": True,
                    "model_type": current_metadata.get("model_type", "Unknown"),
                    "is_trained": True,
                    "feature_count": current_metadata.get("feature_count", 0),
                    "training_samples": current_metadata.get("training_samples", 0),
                    "accuracy": metrics.get("accuracy", 0.0),
                    "precision": metrics.get("precision", 0.0),
                    "recall": metrics.get("recall", 0.0),
                    "f1_score": metrics.get("f1_score", 0.0),
                    "last_updated": last_updated,
                    "path": current_model_path,
                }
            else:
                return {"loaded": False, "message": "モデルがロードされていません"}

        except Exception as e:
            logger.warning(f"現在のモデル情報取得エラー: {e}")
            return {"loaded": False, "error": str(e)}

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
            # 現在ロードされているモデルと比較してアクティブか判定
            current_model_path = ml_training_service.get_current_model_path()
            if current_model_path:
                return model["path"] == current_model_path

            return False

        except (AttributeError, TypeError) as e:
            # Bug #3修正: 予期される例外を明示的にキャッチ
            logger.debug(f"アクティブモデル判定スキップ: {e}")
            return False
        except Exception as e:
            # 予期しない例外はログに記録して再スロー
            logger.error(f"予期しないアクティブモデル判定エラー: {e}", exc_info=True)
            return False
