"""
ML管理API

フロントエンド用のML管理機能を提供するAPIエンドポイント
"""

from fastapi import APIRouter, Depends
from typing import Dict, Any
import logging
from sqlalchemy.orm import Session
from datetime import datetime
import os

from app.core.services.ml.model_manager import model_manager
from app.core.services.auto_strategy.services.ml_orchestrator import MLOrchestrator
from app.core.services.ml.config import ml_config
from app.core.services.ml.performance_extractor import performance_extractor
from app.core.utils.unified_error_handler import UnifiedErrorHandler

from app.core.services.backtest_data_service import BacktestDataService
from backend.app.core.utils.api_utils import APIResponseHelper
from database.repositories.ohlcv_repository import OHLCVRepository
from database.repositories.open_interest_repository import OpenInterestRepository
from database.repositories.funding_rate_repository import FundingRateRepository
from database.connection import get_db


router = APIRouter(prefix="/api/ml", tags=["ml_management"])
logger = logging.getLogger(__name__)

# グローバルインスタンス
ml_orchestrator = MLOrchestrator()


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
                        # 基本性能指標
                        "accuracy": metadata.get("accuracy", 0.0),
                        "precision": metadata.get("precision", 0.0),
                        "recall": metadata.get("recall", 0.0),
                        "f1_score": metadata.get("f1_score", 0.0),
                        # AUC指標
                        "auc_score": metadata.get("auc_score", 0.0),
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
                        # モデル情報
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
                        # 学習パラメータ
                        "train_test_split": metadata.get("train_test_split", 0.8),
                        "random_state": metadata.get("random_state", 42),
                        # 特徴量重要度とレポート
                        "feature_importance": metadata.get("feature_importance", {}),
                        "classification_report": metadata.get(
                            "classification_report", {}
                        ),
                    }

                    # classification_reportから詳細指標を抽出（メタデータに直接ない場合）
                    if "classification_report" in metadata:
                        report = metadata["classification_report"]
                        if isinstance(report, dict) and "macro avg" in report:
                            macro_avg = report["macro avg"]
                            # メタデータに直接値がない場合のみ上書き
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
        # トレーニング状態は ml_training.py のエンドポイントから取得
        # ここでは状態を返さない

        return status

    return await UnifiedErrorHandler.safe_execute_async(_get_ml_status)


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


@router.get("/automl-feature-analysis")
async def get_automl_feature_analysis(top_n: int = 20):
    """
    AutoML特徴量分析結果を取得

    Args:
        top_n: 分析する上位特徴量数

    Returns:
        AutoML特徴量分析結果
    """

    async def _get_automl_feature_analysis():
        # 特徴量重要度を取得
        feature_importance = ml_orchestrator.get_feature_importance(
            100
        )  # より多くの特徴量を取得

        if not feature_importance:
            return {"error": "特徴量重要度データがありません"}

        # AutoML特徴量分析を実行
        from app.core.services.ml.feature_engineering.automl_feature_analyzer import (
            AutoMLFeatureAnalyzer,
        )

        analyzer = AutoMLFeatureAnalyzer()
        analysis_result = analyzer.analyze_feature_importance(feature_importance, top_n)

        return analysis_result

    return await UnifiedErrorHandler.safe_execute_async(_get_automl_feature_analysis)


@router.get("/automl-presets")
async def get_automl_presets():
    """
    AutoML設定プリセット一覧を取得

    Returns:
        AutoML設定プリセット一覧
    """

    async def _get_automl_presets():
        from app.core.services.ml.feature_engineering.automl_preset_service import (
            AutoMLPresetService,
        )

        preset_service = AutoMLPresetService()
        presets = preset_service.get_all_presets()

        return {
            "presets": [
                {
                    "name": preset.name,
                    "description": preset.description,
                    "market_condition": preset.market_condition.value,
                    "trading_strategy": preset.trading_strategy.value,
                    "data_size": preset.data_size.value,
                    "config": preset.config,
                    "performance_notes": preset.performance_notes,
                }
                for preset in presets
            ],
            "summary": preset_service.get_preset_summary(),
        }

    return await UnifiedErrorHandler.safe_execute_async(_get_automl_presets)


@router.get("/automl-presets/{preset_name}")
async def get_automl_preset(preset_name: str):
    """
    特定のAutoML設定プリセットを取得

    Args:
        preset_name: プリセット名

    Returns:
        AutoML設定プリセット
    """

    async def _get_automl_preset():
        from app.core.services.ml.feature_engineering.automl_preset_service import (
            AutoMLPresetService,
        )

        try:
            preset_service = AutoMLPresetService()
            preset = preset_service.get_preset_by_name(preset_name)

            return {
                "name": preset.name,
                "description": preset.description,
                "market_condition": preset.market_condition.value,
                "trading_strategy": preset.trading_strategy.value,
                "data_size": preset.data_size.value,
                "config": preset.config,
                "performance_notes": preset.performance_notes,
            }
        except ValueError as e:
            return {"error": str(e)}

    return await UnifiedErrorHandler.safe_execute_async(_get_automl_preset)


@router.post("/automl-presets/recommend")
async def recommend_automl_preset(
    market_condition: str = None, trading_strategy: str = None, data_size: str = None
):
    """
    条件に基づいてAutoML設定プリセットを推奨

    Args:
        market_condition: 市場条件
        trading_strategy: 取引戦略
        data_size: データサイズ

    Returns:
        推奨AutoML設定プリセット
    """

    async def _recommend_automl_preset():
        from app.core.services.ml.feature_engineering.automl_preset_service import (
            AutoMLPresetService,
            MarketCondition,
            TradingStrategy,
            DataSize,
        )

        preset_service = AutoMLPresetService()

        # 文字列をEnumに変換
        market_cond = None
        if market_condition:
            try:
                market_cond = MarketCondition(market_condition)
            except ValueError:
                pass

        trading_strat = None
        if trading_strategy:
            try:
                trading_strat = TradingStrategy(trading_strategy)
            except ValueError:
                pass

        data_sz = None
        if data_size:
            try:
                data_sz = DataSize(data_size)
            except ValueError:
                pass

        # プリセットを推奨
        preset = preset_service.recommend_preset(
            market_condition=market_cond,
            trading_strategy=trading_strat,
            data_size=data_sz,
        )

        return {
            "recommended_preset": {
                "name": preset.name,
                "description": preset.description,
                "market_condition": preset.market_condition.value,
                "trading_strategy": preset.trading_strategy.value,
                "data_size": preset.data_size.value,
                "config": preset.config,
                "performance_notes": preset.performance_notes,
            },
            "recommendation_criteria": {
                "market_condition": market_condition,
                "trading_strategy": trading_strategy,
                "data_size": data_size,
            },
        }

    return await UnifiedErrorHandler.safe_execute_async(_recommend_automl_preset)


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
        return APIResponseHelper.api_response(
            success=True, message="設定が更新されました（現在は読み取り専用）"
        )

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


@router.post("/models/cleanup")
async def cleanup_old_models():
    """
    古いモデルファイルをクリーンアップ
    """

    async def _cleanup_old_models():
        model_manager.cleanup_expired_models()
        return {"message": "古いモデルファイルが削除されました"}

    return await UnifiedErrorHandler.safe_execute_async(_cleanup_old_models)


def get_data_service(db: Session = Depends(get_db)) -> BacktestDataService:
    """データサービスの依存性注入"""
    ohlcv_repo = OHLCVRepository(db)
    oi_repo = OpenInterestRepository(db)
    fr_repo = FundingRateRepository(db)
    return BacktestDataService(ohlcv_repo, oi_repo, fr_repo)
