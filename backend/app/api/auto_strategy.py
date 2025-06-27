"""
自動戦略生成API

遺伝的アルゴリズムによる戦略自動生成のAPIエンドポイントを提供します。
"""

from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import logging

from app.core.services.auto_strategy import AutoStrategyService
from app.core.services.auto_strategy.models.ga_config import GAConfig
from app.core.services.auto_strategy.models.strategy_gene import StrategyGene
from database.connection import get_db
from app.core.utils.api_utils import APIResponseHelper

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
                    "population_size": 10,
                    "generations": 5,
                    "crossover_rate": 0.8,
                    "mutation_rate": 0.1,
                    "elite_size": 2,
                    "max_indicators": 3,
                    "allowed_indicators": ["SMA", "EMA", "RSI", "MACD", "BB", "ATR"],
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
        logger.info("=== GA戦略生成API呼び出し開始 ===")
        logger.info(f"実験名: {request.experiment_name}")
        logger.info(f"base_config: {request.base_config}")
        logger.info(f"ga_config: {request.ga_config}")

        # サービス初期化チェック
        if auto_strategy_service is None:
            logger.error("AutoStrategyServiceがNoneです")
            logger.error(
                "AutoStrategyServiceが利用できません。サーバーログを確認してください。",
                exc_info=True,
            )
            raise HTTPException(
                status_code=503,
                detail="AutoStrategyServiceが利用できません。サーバーログを確認してください。",
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
            logger.error(f"GA設定の作成に失敗しました: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=400, detail=f"GA設定の作成に失敗しました: {str(e)}"
            )

        # 設定の検証
        logger.info("GA設定を検証中...")
        try:
            is_valid, errors = ga_config.validate()
            if not is_valid:
                logger.error(f"GA設定検証失敗: {errors}")
                logger.error(
                    f"Invalid GA configuration: {', '.join(errors)}", exc_info=True
                )
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid GA configuration: {', '.join(errors)}",
                )
            logger.info("GA設定検証成功")
        except Exception as e:
            logger.error(f"GA設定検証エラー: {type(e).__name__}: {e}")
            import traceback

            logger.error(f"GA設定検証トレースバック:\n{traceback.format_exc()}")
            logger.error(f"GA configuration validation failed: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=400, detail=f"GA configuration validation failed: {str(e)}"
            )

        # バックテスト設定のシンボルを正規化
        logger.info("バックテスト設定のシンボルを正規化中...")
        backtest_config = request.base_config.copy()
        original_symbol = backtest_config.get("symbol", "BTC/USDT")

        # シンボルの正規化（例: BTC/USDT -> BTC/USDT:USDT）
        if original_symbol and ":" not in original_symbol:
            # Bybitの線形永久契約を想定し、:USDT を付与
            normalized_symbol = f"{original_symbol}:USDT"
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
            message="GA戦略生成を開始しました",
        )

    except Exception as e:
        logger.error("GA戦略生成エラー", exc_info=True)
        raise HTTPException(status_code=500, detail="GA戦略生成エラー") from e


@router.get("/experiments/{experiment_id}/progress", response_model=GAProgressResponse)
async def get_experiment_progress(experiment_id: str):
    """
    実験の進捗を取得

    指定された実験IDの進捗状況をリアルタイムで取得します。
    """
    try:
        if auto_strategy_service is None:
            logger.error(
                "AutoStrategyServiceが利用できません。サーバーログを確認してください。",
                exc_info=True,
            )
            raise HTTPException(
                status_code=503,
                detail="AutoStrategyServiceが利用できません。サーバーログを確認してください。",
            )

        progress = auto_strategy_service.get_experiment_progress(experiment_id)

        if progress is None:
            logger.error(f"実験が見つかりません: {experiment_id}", exc_info=True)
            raise HTTPException(
                status_code=404, detail=f"実験が見つかりません: {experiment_id}"
            )

        return APIResponseHelper.api_response(
            success=True, data=progress.to_dict(), message="進捗情報を取得しました"
        )

    except Exception as e:
        logger.error("進捗取得エラー", exc_info=True)
        raise HTTPException(status_code=500, detail="進捗取得エラー") from e


@router.get("/experiments/{experiment_id}/results", response_model=GAResultResponse)
async def get_experiment_results(experiment_id: str):
    """
    実験結果を取得

    完了した実験の結果を取得します。
    """
    try:
        if auto_strategy_service is None:
            logger.error(
                "AutoStrategyServiceが利用できません。サーバーログを確認してください。",
                exc_info=True,
            )
            raise HTTPException(
                status_code=503,
                detail="AutoStrategyServiceが利用できません。サーバーログを確認してください。",
            )

        result = auto_strategy_service.get_experiment_result(experiment_id)

        if result is None:
            # 実験が存在するか確認
            progress = auto_strategy_service.get_experiment_progress(experiment_id)
            if progress is None:
                logger.error(f"実験が見つかりません: {experiment_id}", exc_info=True)
                raise HTTPException(
                    status_code=404, detail=f"実験が見つかりません: {experiment_id}"
                )
            else:
                logger.error("実験はまだ完了していません", exc_info=True)
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

        return APIResponseHelper.api_response(
            success=True, data=formatted_result, message="実験結果を取得しました"
        )

    except Exception as e:
        logger.error("結果取得エラー", exc_info=True)
        raise HTTPException(status_code=500, detail="結果取得エラー") from e


@router.get("/experiments", response_model=List[Dict[str, Any]])
async def list_experiments():
    """
    実験一覧を取得

    実行中・完了済みの全実験の一覧を取得します。
    """
    try:
        if auto_strategy_service is None:
            logger.error(
                "AutoStrategyServiceが利用できません。サーバーログを確認してください。",
                exc_info=True,
            )
            raise HTTPException(
                status_code=503,
                detail="AutoStrategyServiceが利用できません。サーバーログを確認してください。",
            )

        experiments = auto_strategy_service.list_experiments()
        return APIResponseHelper.api_response(
            success=True,
            data={"experiments": experiments},
            message="実験一覧を取得しました",
        )

    except Exception as e:
        logger.error("実験一覧取得エラー", exc_info=True)
        raise HTTPException(status_code=500, detail="実験一覧取得エラー") from e


@router.post("/experiments/{experiment_id}/stop")
async def stop_experiment(experiment_id: str):
    """
    実験を停止

    実行中の実験を停止します。
    """
    try:
        if auto_strategy_service is None:
            logger.error(
                "AutoStrategyServiceが利用できません。サーバーログを確認してください。",
                exc_info=True,
            )
            raise HTTPException(
                status_code=503,
                detail="AutoStrategyServiceが利用できません。サーバーログを確認してください。",
            )

        success = auto_strategy_service.stop_experiment(experiment_id)

        if not success:
            logger.error(
                "実験を停止できませんでした（存在しないか、既に完了している可能性があります）",
                exc_info=True,
            )
            raise HTTPException(
                status_code=400,
                detail="実験を停止できませんでした（存在しないか、既に完了している可能性があります）",
            )

        return APIResponseHelper.api_response(
            success=True, message="実験を停止しました"
        )

    except Exception as e:
        logger.error("実験停止エラー", exc_info=True)
        raise HTTPException(status_code=500, detail="実験停止エラー") from e


@router.post("/test-strategy", response_model=StrategyTestResponse)
async def test_strategy(request: StrategyTestRequest):
    """
    単一戦略のテスト実行

    指定された戦略遺伝子から戦略を生成し、バックテストを実行します。
    GA実行前の戦略検証に使用できます。
    """
    try:
        if auto_strategy_service is None:
            logger.error(
                "AutoStrategyServiceが利用できません。サーバーログを確認してください。",
                exc_info=True,
            )
            raise HTTPException(
                status_code=503,
                detail="AutoStrategyServiceが利用できません。サーバーログを確認してください。",
            )

        # 戦略遺伝子の復元
        strategy_gene = StrategyGene.from_dict(request.strategy_gene)

        # テスト実行
        result = auto_strategy_service.test_strategy_generation(
            strategy_gene, request.backtest_config
        )

        if result["success"]:
            return APIResponseHelper.api_response(
                success=True, data=result, message="戦略テストが完了しました"
            )
        else:
            return APIResponseHelper.api_response(
                success=False,
                data=result,
                message="戦略テストに失敗しました",
            )

    except Exception as e:
        logger.error("戦略テスト実行中にエラーが発生しました", exc_info=True)
        raise HTTPException(
            status_code=500, detail="戦略テスト実行中にエラーが発生しました"
        ) from e


@router.get("/config/default", response_model=Dict[str, Any])
async def get_default_config():
    """
    デフォルトGA設定を取得

    推奨されるGA設定のデフォルト値を返します。
    """
    try:
        if auto_strategy_service is None:
            logger.error(
                "AutoStrategyServiceが利用できません。サーバーログを確認してください。",
                exc_info=True,
            )
            raise HTTPException(
                status_code=503,
                detail="AutoStrategyServiceが利用できません。サーバーログを確認してください。",
            )

        default_config = GAConfig.create_default()
        return APIResponseHelper.api_response(
            success=True,
            message="デフォルト設定を取得しました",
            data={"config": default_config.to_dict()},
        )

    except Exception as e:
        logger.error("デフォルト設定取得エラー", exc_info=True)
        raise HTTPException(status_code=500, detail="デフォルト設定取得エラー") from e


@router.get("/config/presets", response_model=Dict[str, Any])
async def get_config_presets():
    """
    GA設定プリセットを取得

    用途別のGA設定プリセット（高速、標準、徹底）を返します。
    """
    try:
        if auto_strategy_service is None:
            logger.error(
                "AutoStrategyServiceが利用できません。サーバーログを確認してください。",
                exc_info=True,
            )
            raise HTTPException(
                status_code=503,
                detail="AutoStrategyServiceが利用できません。サーバーログを確認してください。",
            )

        presets = {
            "fast": GAConfig.create_fast().to_dict(),
            "default": GAConfig.create_default().to_dict(),
            "thorough": GAConfig.create_thorough().to_dict(),  # type: ignore # NOTE: basedpyrightの誤検知のため無視
        }

        return APIResponseHelper.api_response(
            success=True,
            message="設定プリセットを取得しました",
            data={"presets": presets},
        )

    except Exception as e:
        logger.error("プリセット取得エラー", exc_info=True)
        raise HTTPException(status_code=500, detail="プリセット取得エラー") from e
