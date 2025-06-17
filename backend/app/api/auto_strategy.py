"""
自動戦略生成API

遺伝的アルゴリズムによる戦略自動生成のAPIエンドポイントを提供します。
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import logging

from app.core.services.auto_strategy import AutoStrategyService
from app.core.services.auto_strategy.models.ga_config import GAConfig
from app.core.services.auto_strategy.models.strategy_gene import StrategyGene
from database.connection import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auto-strategy", tags=["auto-strategy"])

# グローバルサービスインスタンス
try:
    auto_strategy_service = AutoStrategyService()
    logger.info("AutoStrategyService初期化成功")
except Exception as e:
    logger.error(f"AutoStrategyService初期化エラー: {e}", exc_info=True)
    auto_strategy_service = None


# リクエスト・レスポンスモデル


class GAGenerationRequest(BaseModel):
    """GA戦略生成リクエスト"""

    experiment_name: str = Field(..., description="実験名")
    base_config: Dict[str, Any] = Field(..., description="基本バックテスト設定")
    ga_config: Dict[str, Any] = Field(..., description="GA設定")

    class Config:
        schema_extra = {
            "example": {
                "experiment_name": "BTC_Strategy_Gen_001",
                "base_config": {
                    "symbol": "BTC/USDT",
                    "timeframe": "1h",
                    "start_date": "2024-01-01",
                    "end_date": "2024-12-19",
                    "initial_capital": 100000,
                    "commission_rate": 0.00055,
                },
                "ga_config": {
                    "population_size": 50,
                    "generations": 30,
                    "crossover_rate": 0.8,
                    "mutation_rate": 0.1,
                    "elite_size": 5,
                    "max_indicators": 5,
                    "allowed_indicators": ["SMA", "EMA", "RSI", "MACD", "BB"],
                },
            }
        }


class GAGenerationResponse(BaseModel):
    """GA戦略生成レスポンス"""

    success: bool
    experiment_id: str
    message: str


class GAProgressResponse(BaseModel):
    """GA進捗レスポンス"""

    success: bool
    progress: Optional[Dict[str, Any]] = None
    message: str


class GAResultResponse(BaseModel):
    """GA結果レスポンス"""

    success: bool
    result: Optional[Dict[str, Any]] = None
    message: str


class StrategyTestRequest(BaseModel):
    """戦略テストリクエスト"""

    strategy_gene: Dict[str, Any] = Field(..., description="戦略遺伝子")
    backtest_config: Dict[str, Any] = Field(..., description="バックテスト設定")


class StrategyTestResponse(BaseModel):
    """戦略テストレスポンス"""

    success: bool
    result: Optional[Dict[str, Any]] = None
    errors: Optional[List[str]] = None
    message: str


# APIエンドポイント


@router.post("/generate", response_model=GAGenerationResponse)
async def generate_strategy(
    request: GAGenerationRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    GA戦略生成を開始

    遺伝的アルゴリズムを使用して取引戦略を自動生成します。
    バックグラウンドで実行され、進捗は別のエンドポイントで確認できます。
    """
    try:
        logger.info(f"=== GA戦略生成API呼び出し開始 ===")
        logger.info(f"実験名: {request.experiment_name}")
        logger.info(f"base_config: {request.base_config}")
        logger.info(f"ga_config: {request.ga_config}")

        # サービス初期化チェック
        if auto_strategy_service is None:
            logger.error("AutoStrategyService is None")
            raise HTTPException(
                status_code=503,
                detail="AutoStrategyService is not available. Please check server logs.",
            )

        logger.info(f"GA戦略生成開始: {request.experiment_name}")

        # GA設定の構築
        logger.info("GA設定を構築中...")
        logger.info(
            f"リクエストのallowed_indicators: {request.ga_config.get('allowed_indicators', [])}"
        )
        try:
            ga_config = GAConfig.from_dict(request.ga_config)
            logger.info(
                f"GA設定構築完了: {len(ga_config.allowed_indicators)} indicators"
            )
            logger.info(
                f"構築後のallowed_indicators: {ga_config.allowed_indicators[:5]}..."
            )
        except Exception as e:
            logger.error(f"GA設定構築エラー: {type(e).__name__}: {e}")
            import traceback

            logger.error(f"GA設定構築トレースバック:\n{traceback.format_exc()}")
            raise HTTPException(
                status_code=400, detail=f"GA config creation failed: {str(e)}"
            )

        # 設定の検証
        logger.info("GA設定を検証中...")
        try:
            is_valid, errors = ga_config.validate()
            if not is_valid:
                logger.error(f"GA設定検証失敗: {errors}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid GA configuration: {', '.join(errors)}",
                )
            logger.info("GA設定検証成功")
        except Exception as e:
            logger.error(f"GA設定検証エラー: {type(e).__name__}: {e}")
            import traceback

            logger.error(f"GA設定検証トレースバック:\n{traceback.format_exc()}")
            raise

        # バックテスト設定のシンボルを正規化
        logger.info("バックテスト設定のシンボルを正規化中...")
        backtest_config = request.base_config.copy()
        original_symbol = backtest_config.get("symbol", "BTC/USDT")

        # シンボルの正規化（BTC/USDT -> BTC/USDT:USDT）
        if original_symbol == "BTC/USDT":
            normalized_symbol = "BTC/USDT:USDT"
            backtest_config["symbol"] = normalized_symbol
            logger.info(f"シンボル正規化: {original_symbol} -> {normalized_symbol}")
        else:
            logger.info(f"シンボル正規化不要: {original_symbol}")

        # 戦略生成を開始（バックグラウンド実行）
        logger.info("戦略生成を開始中...")
        experiment_id = auto_strategy_service.start_strategy_generation(
            experiment_name=request.experiment_name,
            ga_config=ga_config,
            backtest_config=backtest_config,
        )
        logger.info(f"戦略生成開始成功: {experiment_id}")

        return GAGenerationResponse(
            success=True,
            experiment_id=experiment_id,
            message=f"戦略生成を開始しました。実験ID: {experiment_id}",
        )

    except ValueError as e:
        logger.error(f"設定エラー: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        # AutoStrategyServiceから詳細なエラー情報が含まれたRuntimeErrorが発生した場合
        logger.error(f"サービス実行時エラー: {e}")
        raise HTTPException(status_code=500, detail=f"Service error: {str(e)}")
    except Exception as e:
        import traceback

        error_msg = str(e)
        error_type = type(e).__name__
        traceback_str = traceback.format_exc()

        logger.error(f"戦略生成開始エラー - 例外型: {error_type}")
        logger.error(f"戦略生成開始エラー - メッセージ: {error_msg}")
        logger.error(f"戦略生成開始エラー - トレースバック:\n{traceback_str}")

        # エラーメッセージが空の場合の対処
        if not error_msg:
            error_msg = f"Unknown {error_type} error occurred"

        detailed_error = f"{error_type}: {error_msg}"

        raise HTTPException(
            status_code=500, detail=f"Internal server error: {detailed_error}"
        )


@router.get("/experiments/{experiment_id}/progress", response_model=GAProgressResponse)
async def get_experiment_progress(experiment_id: str):
    """
    実験の進捗を取得

    指定された実験IDの進捗状況をリアルタイムで取得します。
    """
    try:
        progress = auto_strategy_service.get_experiment_progress(experiment_id)

        if progress is None:
            raise HTTPException(
                status_code=404, detail=f"Experiment not found: {experiment_id}"
            )

        return GAProgressResponse(
            success=True, progress=progress.to_dict(), message="進捗情報を取得しました"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"進捗取得エラー: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/experiments/{experiment_id}/results", response_model=GAResultResponse)
async def get_experiment_results(experiment_id: str):
    """
    実験結果を取得

    完了した実験の結果を取得します。
    """
    try:
        result = auto_strategy_service.get_experiment_result(experiment_id)

        if result is None:
            # 実験が存在するか確認
            progress = auto_strategy_service.get_experiment_progress(experiment_id)
            if progress is None:
                raise HTTPException(
                    status_code=404, detail=f"Experiment not found: {experiment_id}"
                )
            else:
                raise HTTPException(
                    status_code=202, detail="実験はまだ完了していません"
                )

        # 結果を整形
        formatted_result = {
            "experiment_id": experiment_id,
            "best_strategy": result["best_strategy"].to_dict(),
            "best_fitness": result["best_fitness"],
            "execution_time": result["execution_time"],
            "generations_completed": result["generations_completed"],
            "final_population_size": result["final_population_size"],
        }

        return GAResultResponse(
            success=True, result=formatted_result, message="実験結果を取得しました"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"結果取得エラー: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/experiments", response_model=List[Dict[str, Any]])
async def list_experiments():
    """
    実験一覧を取得

    実行中・完了済みの全実験の一覧を取得します。
    """
    try:
        experiments = auto_strategy_service.list_experiments()
        return experiments

    except Exception as e:
        logger.error(f"実験一覧取得エラー: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/experiments/{experiment_id}/stop")
async def stop_experiment(experiment_id: str):
    """
    実験を停止

    実行中の実験を停止します。
    """
    try:
        success = auto_strategy_service.stop_experiment(experiment_id)

        if not success:
            raise HTTPException(
                status_code=400,
                detail="実験を停止できませんでした（存在しないか、既に完了している可能性があります）",
            )

        return {"success": True, "message": "実験を停止しました"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"実験停止エラー: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/test-strategy", response_model=StrategyTestResponse)
async def test_strategy(request: StrategyTestRequest):
    """
    単一戦略のテスト実行

    指定された戦略遺伝子から戦略を生成し、バックテストを実行します。
    GA実行前の戦略検証に使用できます。
    """
    try:
        # 戦略遺伝子の復元
        strategy_gene = StrategyGene.from_dict(request.strategy_gene)

        # テスト実行
        result = auto_strategy_service.test_strategy_generation(
            strategy_gene, request.backtest_config
        )

        if result["success"]:
            return StrategyTestResponse(
                success=True, result=result, message="戦略テストが完了しました"
            )
        else:
            return StrategyTestResponse(
                success=False,
                errors=result.get("errors", [result.get("error", "Unknown error")]),
                message="戦略テストに失敗しました",
            )

    except Exception as e:
        logger.error(f"戦略テストエラー: {e}")
        return StrategyTestResponse(
            success=False,
            errors=[str(e)],
            message="戦略テスト実行中にエラーが発生しました",
        )


@router.get("/config/default", response_model=Dict[str, Any])
async def get_default_config():
    """
    デフォルトGA設定を取得

    推奨されるGA設定のデフォルト値を返します。
    """
    try:
        default_config = GAConfig.create_default()
        return {
            "success": True,
            "config": default_config.to_dict(),
            "message": "デフォルト設定を取得しました",
        }

    except Exception as e:
        logger.error(f"デフォルト設定取得エラー: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/config/presets", response_model=Dict[str, Any])
async def get_config_presets():
    """
    GA設定プリセットを取得

    用途別のGA設定プリセット（高速、標準、徹底）を返します。
    """
    try:
        presets = {
            "fast": GAConfig.create_fast().to_dict(),
            "default": GAConfig.create_default().to_dict(),
            "thorough": GAConfig.create_thorough().to_dict(),
        }

        return {
            "success": True,
            "presets": presets,
            "message": "設定プリセットを取得しました",
        }

    except Exception as e:
        logger.error(f"プリセット取得エラー: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
