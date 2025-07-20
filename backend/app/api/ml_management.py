"""
ML管理API

フロントエンド用のML管理機能を提供するAPIエンドポイント
"""

from fastapi import APIRouter, BackgroundTasks
from typing import Dict, Any
import logging
from datetime import datetime
import os

from app.core.services.ml.model_manager import model_manager
from app.core.services.ml.ml_training_service import ml_training_service
from app.core.services.auto_strategy.services.ml_orchestrator import MLOrchestrator
from app.core.services.ml.config import ml_config
from app.core.services.ml.performance_extractor import performance_extractor
from app.core.utils.unified_error_handler import UnifiedErrorHandler

from app.core.services.backtest_data_service import BacktestDataService
from database.repositories.ohlcv_repository import OHLCVRepository
from database.repositories.open_interest_repository import OpenInterestRepository
from database.repositories.funding_rate_repository import FundingRateRepository
from database.connection import get_db
from app.config.unified_config import unified_config

router = APIRouter(prefix="/api/ml", tags=["ml_management"])
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
    "model_info": None,
}


@router.get("/models")
async def get_models():
    """
    学習済みモデルの一覧を取得

    Returns:
        モデル一覧
    """

    async def _get_models():
        models = model_manager.list_models("*")

        # モデル情報を整形
        formatted_models = []
        for model in models:
            formatted_models.append(
                {
                    "id": model["name"],
                    "name": model["name"],
                    "path": model["path"],
                    "size_mb": model["size_mb"],
                    "modified_at": model["modified_at"].isoformat(),
                    "directory": model["directory"],
                    "is_active": False,  # TODO: アクティブモデルの判定ロジック
                }
            )

        return {"models": formatted_models}

    return await UnifiedErrorHandler.safe_execute_async(_get_models)


@router.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """
    指定されたモデルを削除

    Args:
        model_id: モデルID（ファイル名）
    """

    async def _delete_model():
        logger.info(f"モデル削除要求: {model_id}")

        # モデルIDをデコード（URLエンコードされている場合）
        from urllib.parse import unquote

        decoded_model_id = unquote(model_id)

        # モデルファイルを検索
        models = model_manager.list_models("*")
        target_model = None

        for model in models:
            # ファイル名で比較（拡張子も含む）
            if model["name"] == decoded_model_id or model["name"] == model_id:
                target_model = model
                break

        if not target_model:
            logger.warning(f"モデルが見つかりません: {decoded_model_id}")
            logger.info(f"利用可能なモデル: {[m['name'] for m in models]}")
            from fastapi import HTTPException

            raise HTTPException(
                status_code=404, detail=f"モデルが見つかりません: {decoded_model_id}"
            )

        # ファイルの存在確認
        import os

        if not os.path.exists(target_model["path"]):
            logger.warning(f"モデルファイルが存在しません: {target_model['path']}")
            from fastapi import HTTPException

            raise HTTPException(status_code=404, detail="モデルファイルが存在しません")

        # モデルを削除
        try:
            os.remove(target_model["path"])
            logger.info(f"モデル削除完了: {decoded_model_id} -> {target_model['path']}")
            return {"message": "モデルが削除されました"}
        except Exception as e:
            logger.error(f"モデルファイル削除エラー: {e}")
            raise HTTPException(
                status_code=500, detail="モデルファイルの削除に失敗しました"
            )

    return await UnifiedErrorHandler.safe_execute_async(_delete_model)


@router.get("/status")
async def get_ml_status():
    """
    MLモデルの現在の状態を取得

    Returns:
        モデル状態情報
    """

    async def _get_ml_status():
        status = ml_orchestrator.get_model_status()

        latest_model = model_manager.get_latest_model("*")

        if latest_model and os.path.exists(latest_model):
            try:
                model_data = model_manager.load_model(latest_model)
                if model_data and "metadata" in model_data:
                    metadata = model_data["metadata"]
                    model_info = {
                        "accuracy": metadata.get("accuracy", 0.0),
                        "model_type": metadata.get("model_type", "LightGBM"),
                        "last_updated": datetime.fromtimestamp(
                            os.path.getmtime(latest_model)
                        ).isoformat(),
                        "training_samples": metadata.get("training_samples", 0),
                        "file_size_mb": os.path.getsize(latest_model) / (1024 * 1024),
                        "feature_count": metadata.get("feature_count", 0),
                    }
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

                performance_metrics = performance_extractor.extract_performance_metrics(
                    latest_model
                )
                status["performance_metrics"] = performance_metrics
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

        # トレーニング状態情報を追加
        if training_status.get("is_training"):
            status["is_training"] = True
            status["training_progress"] = training_status.get("progress", 0)
            status["status"] = training_status.get("status", "unknown")

        return status

    return await UnifiedErrorHandler.safe_execute_async(_get_ml_status)


@router.get("/training/status")
async def get_training_status():
    """
    MLトレーニングの現在の状態を取得

    Returns:
        トレーニング状態情報
    """

    async def _get_training_status():
        return training_status

    return await UnifiedErrorHandler.safe_execute_async(_get_training_status)


@router.get("/feature-importance")
async def get_feature_importance(top_n: int = 10):
    """
    特徴量重要度を取得

    Args:
        top_n: 上位N個の特徴量

    Returns:
        特徴量重要度
    """

    async def _get_feature_importance():
        feature_importance = ml_orchestrator.get_feature_importance(top_n)
        return {"feature_importance": feature_importance}

    return await UnifiedErrorHandler.safe_execute_async(_get_feature_importance)


@router.get("/config")
async def get_ml_config():
    """
    ML設定を取得

    Returns:
        ML設定
    """

    async def _get_ml_config():
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
            },
        }

        return config_dict

    return await UnifiedErrorHandler.safe_execute_async(_get_ml_config)


@router.put("/config")
async def update_ml_config(config_data: Dict[str, Any]):
    """
    ML設定を更新

    Args:
        config_data: 更新する設定データ
    """

    async def _update_ml_config():
        # TODO: 設定の更新ロジックを実装
        # 現在は読み取り専用として扱う
        logger.info(f"ML設定更新要求: {config_data}")
        return {"message": "設定が更新されました（現在は読み取り専用）"}

    return await UnifiedErrorHandler.safe_execute_async(_update_ml_config)


@router.post("/config/reset")
async def reset_ml_config():
    """
    ML設定をデフォルト値にリセット
    """

    async def _reset_ml_config():
        # TODO: 設定のリセットロジックを実装
        logger.info("ML設定リセット要求")
        return {
            "message": "設定がデフォルト値にリセットされました（現在は読み取り専用）"
        }

    return await UnifiedErrorHandler.safe_execute_async(_reset_ml_config)


@router.post("/training/start")
async def start_training(
    background_tasks: BackgroundTasks, config_data: Dict[str, Any]
):
    """
    モデルトレーニングを開始

    Args:
        background_tasks: バックグラウンドタスク
        config_data: トレーニング設定
    """

    async def _start_training():
        if training_status["is_training"]:
            from fastapi import HTTPException

            raise HTTPException(status_code=400, detail="既にトレーニングが実行中です")

        # トレーニング状態を更新
        training_status.update(
            {
                "is_training": True,
                "progress": 0,
                "status": "starting",
                "message": "トレーニングを開始しています...",
                "start_time": datetime.now().isoformat(),
                "end_time": None,
                "error": None,
                "model_info": None,
            }
        )

        # バックグラウンドでトレーニングを実行
        background_tasks.add_task(run_training_task, config_data)

        return {"message": "トレーニングが開始されました"}

    return await UnifiedErrorHandler.safe_execute_async(_start_training)


@router.post("/training/stop")
async def stop_training():
    """
    トレーニングを停止
    """

    async def _stop_training():
        if not training_status["is_training"]:
            from fastapi import HTTPException

            raise HTTPException(
                status_code=400, detail="トレーニングが実行されていません"
            )

        # トレーニング状態を更新
        training_status.update(
            {
                "is_training": False,
                "status": "stopped",
                "message": "トレーニングが停止されました",
                "end_time": datetime.now().isoformat(),
            }
        )

        return {"message": "トレーニングが停止されました"}

    return await UnifiedErrorHandler.safe_execute_async(_stop_training)


@router.post("/models/cleanup")
async def cleanup_old_models():
    """
    古いモデルファイルをクリーンアップ
    """

    async def _cleanup_old_models():
        model_manager.cleanup_expired_models()
        return {"message": "古いモデルファイルが削除されました"}

    return await UnifiedErrorHandler.safe_execute_async(_cleanup_old_models)


def get_data_service():
    """データサービスの依存性注入"""
    db = next(get_db())
    try:
        ohlcv_repo = OHLCVRepository(db)
        oi_repo = OpenInterestRepository(db)
        fr_repo = FundingRateRepository(db)
        return BacktestDataService(ohlcv_repo, oi_repo, fr_repo)
    finally:
        db.close()


async def run_training_task(config_data: Dict[str, Any]):
    """
    バックグラウンドでトレーニングを実行

    Args:
        config_data: トレーニング設定
    """
    try:
        # 設定データから必要な情報を取得
        raw_symbol = config_data.get("symbol", "BTCUSDT")
        # シンボルを正規化（BTCUSDT -> BTC/USDT:USDT）
        symbol = unified_config.market.symbol_mapping.get(raw_symbol, raw_symbol)
        timeframe = config_data.get("timeframe", "1h")
        start_date_str = config_data.get("start_date")
        end_date_str = config_data.get("end_date")

        if not start_date_str or not end_date_str:
            raise ValueError("start_date and end_date are required")

        start_date = datetime.fromisoformat(start_date_str)
        end_date = datetime.fromisoformat(end_date_str)
        save_model = config_data.get("save_model", True)
        train_test_split = config_data.get("train_test_split", 0.8)
        random_state = config_data.get("random_state", 42)

        logger.info(
            f"トレーニング設定: symbol={symbol}, timeframe={timeframe}, period={start_date} to {end_date}"
        )

        # トレーニング開始
        training_status.update(
            {
                "is_training": True,
                "progress": 0,
                "status": "starting",
                "message": "トレーニングを開始しています...",
                "start_time": datetime.now().isoformat(),
                "end_time": None,
                "error": None,
            }
        )

        # データサービスを初期化
        data_service = get_data_service()

        # データ取得
        training_status.update(
            {
                "progress": 10,
                "status": "loading_data",
                "message": "トレーニングデータを読み込んでいます...",
            }
        )

        training_data = data_service.get_data_for_backtest(
            symbol=symbol, timeframe=timeframe, start_date=start_date, end_date=end_date
        )

        if training_data.empty:
            raise ValueError(f"指定された期間のデータが見つかりません: {symbol}")

        logger.info(f"トレーニングデータ取得完了: {len(training_data)}行")

        # 追加データ取得（オプション）
        training_status.update(
            {
                "progress": 20,
                "status": "loading_data",
                "message": "追加データを読み込んでいます...",
            }
        )

        # ファンディングレートとオープンインタレストデータを分離
        funding_rate_data = None
        open_interest_data = None

        if (
            "funding_rate" in training_data.columns
            and training_data["funding_rate"].notna().any()
        ):
            funding_rate_data = training_data[["funding_rate"]].copy()
            logger.info("ファンディングレートデータを使用します")

        if (
            "open_interest" in training_data.columns
            and training_data["open_interest"].notna().any()
        ):
            open_interest_data = training_data[["open_interest"]].copy()
            logger.info("建玉残高データを使用します")

        # OHLCVデータのみを抽出
        ohlcv_data = training_data[["Open", "High", "Low", "Close", "Volume"]].copy()

        # トレーニング実行
        training_status.update(
            {
                "progress": 30,
                "status": "training",
                "message": "特徴量を計算しています...",
            }
        )

        training_result = ml_training_service.train_model(
            training_data=ohlcv_data,
            funding_rate_data=funding_rate_data,
            open_interest_data=open_interest_data,
            save_model=save_model,
            model_name="ml_training_model",
            test_size=1 - train_test_split,
            random_state=random_state,
        )

        # モデル学習完了後の進捗更新
        training_status.update(
            {"progress": 90, "status": "saving", "message": "モデルを保存しています..."}
        )

        # トレーニング完了
        training_status.update(
            {
                "progress": 100,
                "status": "completed",
                "message": "トレーニングが完了しました",
                "end_time": datetime.now().isoformat(),
                "is_training": False,
                "model_info": {
                    "accuracy": training_result.get("accuracy", 0.0),
                    "feature_count": training_result.get("feature_count", 0),
                    "training_samples": len(training_data),
                    "test_samples": training_result.get("test_samples", 0),
                    "model_path": training_result.get("model_path", ""),
                    "model_type": training_result.get("model_type", "LightGBM"),
                },
            }
        )

        logger.info(f"MLモデルトレーニング完了: {symbol}")

    except Exception as e:
        logger.error(f"トレーニングタスクエラー: {e}")

        training_status.update(
            {
                "is_training": False,
                "status": "error",
                "message": f"トレーニングエラー: {str(e)}",
                "end_time": datetime.now().isoformat(),
                "error": str(e),
                "progress": 0,
            }
        )
