"""
MLトレーニングAPI

機械学習モデルのトレーニングと管理を行うAPIエンドポイント
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from app.core.services.ml.ml_training_service import MLTrainingService
from app.core.services.auto_strategy.services.ml_orchestrator import MLOrchestrator
from app.core.services.backtest_data_service import BacktestDataService
from database.repositories.ohlcv_repository import OHLCVRepository
from database.repositories.open_interest_repository import OpenInterestRepository
from database.repositories.funding_rate_repository import FundingRateRepository
from database.repositories.bayesian_optimization_repository import (
    BayesianOptimizationRepository,
)
from database.connection import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ml-training", tags=["ML Training"])

# グローバルなトレーニング状態管理
training_status = {
    "is_training": False,
    "progress": 0,
    "status": "idle",
    "message": "",
    "start_time": None,
    "end_time": None,
    "model_info": None,
    "error": None,
}


class MLTrainingConfig(BaseModel):
    """MLトレーニング設定"""

    symbol: str = Field(..., description="取引ペア（例: BTC/USDT:USDT）")
    timeframe: str = Field(default="1h", description="時間軸")
    start_date: str = Field(..., description="開始日（YYYY-MM-DD）")
    end_date: str = Field(..., description="終了日（YYYY-MM-DD）")
    validation_split: float = Field(default=0.2, description="検証データ分割比率")
    prediction_horizon: int = Field(default=24, description="予測期間（時間）")
    threshold_up: float = Field(default=0.02, description="上昇判定閾値")
    threshold_down: float = Field(default=-0.02, description="下落判定閾値")
    save_model: bool = Field(default=True, description="モデルを保存するか")
    # 新しい設定項目
    train_test_split: float = Field(
        default=0.8, description="トレーニング/テスト分割比率"
    )
    cross_validation_folds: int = Field(
        default=5, description="クロスバリデーション分割数"
    )
    random_state: int = Field(default=42, description="ランダムシード")
    early_stopping_rounds: int = Field(default=100, description="早期停止ラウンド数")
    max_depth: int = Field(default=10, description="最大深度")
    n_estimators: int = Field(default=100, description="推定器数")
    learning_rate: float = Field(default=0.1, description="学習率")

    # プロファイル適用設定
    use_profile: bool = Field(default=False, description="プロファイルを使用するか")
    profile_id: Optional[int] = Field(None, description="使用するプロファイルID")
    profile_name: Optional[str] = Field(None, description="使用するプロファイル名")


class MLTrainingResponse(BaseModel):
    """MLトレーニング応答"""

    success: bool
    message: str
    training_id: Optional[str] = None


class MLStatusResponse(BaseModel):
    """MLステータス応答"""

    is_training: bool
    progress: int
    status: str
    message: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    model_info: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


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


async def train_ml_model_background(config: MLTrainingConfig):
    """バックグラウンドでMLモデルをトレーニング"""
    global training_status

    try:
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

        start_date = datetime.fromisoformat(config.start_date)
        end_date = datetime.fromisoformat(config.end_date)

        training_data = data_service.get_data_for_backtest(
            symbol=config.symbol,
            timeframe=config.timeframe,
            start_date=start_date,
            end_date=end_date,
        )

        if training_data.empty:
            raise ValueError(f"指定された期間のデータが見つかりません: {config.symbol}")

        # MLサービスを初期化
        training_status.update(
            {
                "progress": 20,
                "status": "initializing",
                "message": "MLサービスを初期化しています...",
            }
        )

        ml_service = MLTrainingService()

        # プロファイルからハイパーパラメータを適用
        training_params = {}
        if config.use_profile and (config.profile_id or config.profile_name):
            training_status.update(
                {
                    "progress": 25,
                    "status": "loading_profile",
                    "message": "プロファイルからハイパーパラメータを読み込んでいます...",
                }
            )

            db = next(get_db())
            try:
                bayesian_repo = BayesianOptimizationRepository(db)
                profile = None

                if config.profile_id:
                    profile = bayesian_repo.get_by_id(config.profile_id)
                elif config.profile_name:
                    profile = bayesian_repo.get_by_profile_name(config.profile_name)

                if profile:
                    # プロファイルからハイパーパラメータを取得
                    best_params = profile.best_params

                    # 設定を上書き
                    for param_name, param_value in best_params.items():
                        if hasattr(config, param_name):
                            setattr(config, param_name, param_value)
                            training_params[param_name] = param_value

                    logger.info(
                        f"プロファイル '{profile.profile_name}' からハイパーパラメータを適用: {training_params}"
                    )
                    training_status.update(
                        {
                            "message": f"プロファイル '{profile.profile_name}' からハイパーパラメータを適用しました"
                        }
                    )
                else:
                    logger.warning(
                        f"指定されたプロファイルが見つかりません: ID={config.profile_id}, Name={config.profile_name}"
                    )

            except Exception as e:
                logger.warning(f"プロファイル読み込みエラー: {e}")
            finally:
                db.close()

        # モデルトレーニング
        training_status.update(
            {
                "progress": 30,
                "status": "training",
                "message": "モデルをトレーニングしています...",
            }
        )

        # ファンディングレートとオープンインタレストデータを分離
        funding_rate_data = None
        open_interest_data = None

        if "FundingRate" in training_data.columns:
            funding_rate_data = training_data[["FundingRate"]].copy()

        if "OpenInterest" in training_data.columns:
            open_interest_data = training_data[["OpenInterest"]].copy()

        # OHLCVデータのみを抽出
        ohlcv_data = training_data[["Open", "High", "Low", "Close", "Volume"]].copy()

        # トレーニング実行
        training_result = ml_service.train_model(
            training_data=ohlcv_data,
            funding_rate_data=funding_rate_data,
            open_interest_data=open_interest_data,
            save_model=config.save_model,
            # 新しいMLTrainingServiceは設定から自動的にパラメータを取得
            test_size=1 - config.train_test_split,
            random_state=config.random_state,
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
                    "loss": training_result.get("loss", 0.0),
                    "model_path": training_result.get("model_path", ""),
                    "feature_count": training_result.get("feature_count", 0),
                    "training_samples": len(training_data),
                    "validation_split": config.validation_split,
                },
            }
        )

        logger.info(f"MLモデルトレーニング完了: {config.symbol}")

    except Exception as e:
        logger.error(f"MLトレーニングエラー: {e}")
        training_status.update(
            {
                "is_training": False,
                "status": "error",
                "message": f"トレーニングエラー: {str(e)}",
                "end_time": datetime.now().isoformat(),
                "error": str(e),
            }
        )


@router.post("/train", response_model=MLTrainingResponse)
async def start_ml_training(
    config: MLTrainingConfig, background_tasks: BackgroundTasks
):
    """
    MLモデルのトレーニングを開始
    """
    global training_status

    try:
        # 既にトレーニング中の場合はエラー
        if training_status["is_training"]:
            raise HTTPException(status_code=400, detail="既にトレーニングが実行中です")

        # 設定の検証
        try:
            start_date = datetime.fromisoformat(config.start_date)
            end_date = datetime.fromisoformat(config.end_date)

            if start_date >= end_date:
                raise ValueError("開始日は終了日より前である必要があります")

            if (end_date - start_date).days < 7:
                raise ValueError("トレーニング期間は最低7日間必要です")

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # バックグラウンドタスクでトレーニング開始
        background_tasks.add_task(train_ml_model_background, config)

        return MLTrainingResponse(
            success=True,
            message="MLトレーニングを開始しました",
            training_id=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MLトレーニング開始エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/training/status", response_model=MLStatusResponse)
async def get_ml_training_status():
    """
    MLトレーニングの状態を取得
    """
    return MLStatusResponse(**training_status)


@router.get("/model-info")
async def get_ml_model_info():
    """
    現在のMLモデル情報を取得
    """
    try:
        ml_orchestrator = MLOrchestrator()
        model_status = ml_orchestrator.get_model_status()

        return {
            "success": True,
            "model_status": model_status,
            "last_training": training_status.get("model_info"),
        }

    except Exception as e:
        logger.error(f"MLモデル情報取得エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
async def stop_ml_training():
    """
    MLトレーニングを停止
    """
    global training_status

    try:
        if not training_status["is_training"]:
            raise HTTPException(
                status_code=400, detail="実行中のトレーニングがありません"
            )

        # トレーニング停止（実際の実装では、トレーニングプロセスを停止する必要があります）
        training_status.update(
            {
                "is_training": False,
                "status": "stopped",
                "message": "トレーニングが停止されました",
                "end_time": datetime.now().isoformat(),
            }
        )

        return {"success": True, "message": "トレーニングを停止しました"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MLトレーニング停止エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))
