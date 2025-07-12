"""
ML管理API

フロントエンド用のML管理機能を提供するAPIエンドポイント
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

from app.core.services.ml.model_manager import model_manager
from app.core.services.ml.ml_training_service import ml_training_service
from app.core.services.auto_strategy.services.ml_orchestrator import MLOrchestrator
from app.core.config.ml_config import ml_config
from app.core.utils.ml_error_handler import MLErrorHandler

router = APIRouter()
logger = logging.getLogger(__name__)

# グローバルインスタンス
ml_orchestrator = MLOrchestrator()

# トレーニング状態管理
training_status = {
    "is_training": False,
    "progress": 0,
    "status": "idle",
    "message": "待機中",
    "start_time": None,
    "end_time": None,
    "error": None,
    "model_info": None
}


@router.get("/models")
async def get_models():
    """
    学習済みモデルの一覧を取得
    
    Returns:
        モデル一覧
    """
    try:
        models = model_manager.list_models("*")
        
        # モデル情報を整形
        formatted_models = []
        for model in models:
            formatted_models.append({
                "id": model["name"],
                "name": model["name"],
                "path": model["path"],
                "size_mb": model["size_mb"],
                "modified_at": model["modified_at"].isoformat(),
                "directory": model["directory"],
                "is_active": False  # TODO: アクティブモデルの判定ロジック
            })
        
        return {"models": formatted_models}
        
    except Exception as e:
        logger.error(f"モデル一覧取得エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """
    指定されたモデルを削除
    
    Args:
        model_id: モデルID
    """
    try:
        # モデルファイルを検索
        models = model_manager.list_models("*")
        target_model = None
        
        for model in models:
            if model["name"] == model_id:
                target_model = model
                break
        
        if not target_model:
            raise HTTPException(status_code=404, detail="モデルが見つかりません")
        
        # バックアップしてから削除
        backup_path = model_manager.backup_model(target_model["path"])
        if backup_path:
            import os
            os.remove(target_model["path"])
            logger.info(f"モデル削除完了: {model_id}")
            return {"message": "モデルが削除されました", "backup_path": backup_path}
        else:
            raise HTTPException(status_code=500, detail="モデルのバックアップに失敗しました")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"モデル削除エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_id}/backup")
async def backup_model(model_id: str):
    """
    指定されたモデルをバックアップ
    
    Args:
        model_id: モデルID
    """
    try:
        # モデルファイルを検索
        models = model_manager.list_models("*")
        target_model = None
        
        for model in models:
            if model["name"] == model_id:
                target_model = model
                break
        
        if not target_model:
            raise HTTPException(status_code=404, detail="モデルが見つかりません")
        
        backup_path = model_manager.backup_model(target_model["path"])
        if backup_path:
            return {"message": "バックアップが完了しました", "backup_path": backup_path}
        else:
            raise HTTPException(status_code=500, detail="バックアップに失敗しました")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"モデルバックアップエラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_ml_status():
    """
    MLモデルの現在の状態を取得
    
    Returns:
        モデル状態情報
    """
    try:
        status = ml_orchestrator.get_model_status()
        
        # 追加情報を取得
        latest_model = model_manager.get_latest_model("*")
        if latest_model:
            import os
            model_info = {
                "accuracy": 0.85,  # TODO: 実際の精度を取得
                "model_type": "LightGBM",
                "last_updated": datetime.fromtimestamp(os.path.getmtime(latest_model)).isoformat(),
                "training_samples": 10000  # TODO: 実際のサンプル数を取得
            }
            status["model_info"] = model_info
        
        return status
        
    except Exception as e:
        logger.error(f"ML状態取得エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/feature-importance")
async def get_feature_importance(top_n: int = 10):
    """
    特徴量重要度を取得
    
    Args:
        top_n: 上位N個の特徴量
    
    Returns:
        特徴量重要度
    """
    try:
        feature_importance = ml_orchestrator.get_feature_importance(top_n)
        return {"feature_importance": feature_importance}
        
    except Exception as e:
        logger.error(f"特徴量重要度取得エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config")
async def get_ml_config():
    """
    ML設定を取得
    
    Returns:
        ML設定
    """
    try:
        config_dict = {
            "data_processing": {
                "max_ohlcv_rows": ml_config.data_processing.MAX_OHLCV_ROWS,
                "max_feature_rows": ml_config.data_processing.MAX_FEATURE_ROWS,
                "feature_calculation_timeout": ml_config.data_processing.FEATURE_CALCULATION_TIMEOUT,
                "model_training_timeout": ml_config.data_processing.MODEL_TRAINING_TIMEOUT,
            },
            "model": {
                "model_save_path": ml_config.model.MODEL_SAVE_PATH,
                "max_model_versions": ml_config.model.MAX_MODEL_VERSIONS,
                "model_retention_days": ml_config.model.MODEL_RETENTION_DAYS,
            },
            "lightgbm": {
                "learning_rate": ml_config.lightgbm.LEARNING_RATE,
                "num_leaves": ml_config.lightgbm.NUM_LEAVES,
                "feature_fraction": ml_config.lightgbm.FEATURE_FRACTION,
                "bagging_fraction": ml_config.lightgbm.BAGGING_FRACTION,
                "num_boost_round": ml_config.lightgbm.NUM_BOOST_ROUND,
                "early_stopping_rounds": ml_config.lightgbm.EARLY_STOPPING_ROUNDS,
            },
            "training": {
                "train_test_split": ml_config.training.TRAIN_TEST_SPLIT,
                "prediction_horizon": ml_config.training.PREDICTION_HORIZON,
                "threshold_up": ml_config.training.THRESHOLD_UP,
                "threshold_down": ml_config.training.THRESHOLD_DOWN,
                "min_training_samples": ml_config.training.MIN_TRAINING_SAMPLES,
            },
            "prediction": {
                "default_up_prob": ml_config.prediction.DEFAULT_UP_PROB,
                "default_down_prob": ml_config.prediction.DEFAULT_DOWN_PROB,
                "default_range_prob": ml_config.prediction.DEFAULT_RANGE_PROB,
            }
        }
        
        return config_dict
        
    except Exception as e:
        logger.error(f"ML設定取得エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/config")
async def update_ml_config(config_data: Dict[str, Any]):
    """
    ML設定を更新
    
    Args:
        config_data: 更新する設定データ
    """
    try:
        # TODO: 設定の更新ロジックを実装
        # 現在は読み取り専用として扱う
        logger.info(f"ML設定更新要求: {config_data}")
        return {"message": "設定が更新されました（現在は読み取り専用）"}
        
    except Exception as e:
        logger.error(f"ML設定更新エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/config/reset")
async def reset_ml_config():
    """
    ML設定をデフォルト値にリセット
    """
    try:
        # TODO: 設定のリセットロジックを実装
        logger.info("ML設定リセット要求")
        return {"message": "設定がデフォルト値にリセットされました（現在は読み取り専用）"}
        
    except Exception as e:
        logger.error(f"ML設定リセットエラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/training/status")
async def get_training_status():
    """
    トレーニング状態を取得
    
    Returns:
        トレーニング状態
    """
    return training_status


@router.post("/training/start")
async def start_training(background_tasks: BackgroundTasks, config_data: Dict[str, Any]):
    """
    モデルトレーニングを開始
    
    Args:
        background_tasks: バックグラウンドタスク
        config_data: トレーニング設定
    """
    try:
        if training_status["is_training"]:
            raise HTTPException(status_code=400, detail="既にトレーニングが実行中です")
        
        # トレーニング状態を更新
        training_status.update({
            "is_training": True,
            "progress": 0,
            "status": "starting",
            "message": "トレーニングを開始しています...",
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "error": None,
            "model_info": None
        })
        
        # バックグラウンドでトレーニングを実行
        background_tasks.add_task(run_training_task, config_data)
        
        return {"message": "トレーニングが開始されました"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"トレーニング開始エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/training/stop")
async def stop_training():
    """
    トレーニングを停止
    """
    try:
        if not training_status["is_training"]:
            raise HTTPException(status_code=400, detail="トレーニングが実行されていません")
        
        # トレーニング状態を更新
        training_status.update({
            "is_training": False,
            "status": "stopped",
            "message": "トレーニングが停止されました",
            "end_time": datetime.now().isoformat()
        })
        
        return {"message": "トレーニングが停止されました"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"トレーニング停止エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/cleanup")
async def cleanup_old_models():
    """
    古いモデルファイルをクリーンアップ
    """
    try:
        model_manager.cleanup_expired_models()
        return {"message": "古いモデルファイルが削除されました"}
        
    except Exception as e:
        logger.error(f"モデルクリーンアップエラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def run_training_task(config_data: Dict[str, Any]):
    """
    バックグラウンドでトレーニングを実行
    
    Args:
        config_data: トレーニング設定
    """
    try:
        # TODO: 実際のトレーニングロジックを実装
        # 現在はダミーの進捗更新
        
        training_status.update({
            "status": "loading_data",
            "message": "データを読み込んでいます...",
            "progress": 10
        })
        
        import asyncio
        await asyncio.sleep(2)
        
        training_status.update({
            "status": "training",
            "message": "モデルを学習しています...",
            "progress": 50
        })
        
        await asyncio.sleep(5)
        
        training_status.update({
            "status": "completed",
            "message": "トレーニングが完了しました",
            "progress": 100,
            "is_training": False,
            "end_time": datetime.now().isoformat(),
            "model_info": {
                "accuracy": 0.87,
                "feature_count": 45,
                "training_samples": 8000,
                "test_samples": 2000
            }
        })
        
    except Exception as e:
        logger.error(f"トレーニングタスクエラー: {e}")
        training_status.update({
            "is_training": False,
            "status": "error",
            "message": f"トレーニングエラー: {str(e)}",
            "end_time": datetime.now().isoformat(),
            "error": str(e)
        })
