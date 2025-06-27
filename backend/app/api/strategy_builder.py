"""
ストラテジービルダーAPIエンドポイント

ユーザー定義戦略の作成・管理機能を提供するAPIエンドポイント
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from database.connection import get_db
from app.core.services.strategy_builder_service import StrategyBuilderService
from app.core.utils.api_utils import APIErrorHandler, APIResponseHelper

# ログ設定
logger = logging.getLogger(__name__)

# ルーター作成
router = APIRouter(prefix="/api/strategy-builder", tags=["strategy-builder"])


# リクエストモデル
class ValidateStrategyRequest(BaseModel):
    """戦略検証リクエストモデル"""

    strategy_config: Dict[str, Any] = Field(..., description="戦略設定")


# リクエスト/レスポンスモデル
class StrategyCreateRequest(BaseModel):
    """戦略作成リクエスト"""

    name: str = Field(..., min_length=1, max_length=255, description="戦略名")
    description: Optional[str] = Field(None, max_length=1000, description="戦略の説明")
    strategy_config: Dict[str, Any] = Field(
        ..., description="戦略設定（StrategyGene形式）"
    )


class StrategyUpdateRequest(BaseModel):
    """戦略更新リクエスト"""

    name: Optional[str] = Field(
        None, min_length=1, max_length=255, description="戦略名"
    )
    description: Optional[str] = Field(None, max_length=1000, description="戦略の説明")
    strategy_config: Optional[Dict[str, Any]] = Field(
        None, description="戦略設定（StrategyGene形式）"
    )


class StrategyValidateRequest(BaseModel):
    """戦略検証リクエスト"""

    strategy_config: Dict[str, Any] = Field(
        ..., description="戦略設定（StrategyGene形式）"
    )


@router.get("/indicators")
async def get_available_indicators(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    利用可能なテクニカル指標の一覧を取得

    Returns:
        指標情報の辞書（カテゴリ別）
    """

    async def _get_indicators():
        try:
            service = StrategyBuilderService(db)
            indicators = service.get_available_indicators()

            return APIResponseHelper.success_response(
                data={"categories": indicators},
                message="利用可能なテクニカル指標の一覧を取得しました",
            )

        except Exception as e:
            logger.error(f"指標一覧取得エラー: {e}")
            raise HTTPException(status_code=500, detail="指標一覧の取得に失敗しました")

    return await APIErrorHandler.handle_api_exception(_get_indicators)


@router.post("/validate")
async def validate_strategy(
    request: StrategyValidateRequest, db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    戦略設定の妥当性を検証

    Args:
        request: 戦略検証リクエスト

    Returns:
        検証結果
    """

    async def _validate():
        try:
            service = StrategyBuilderService(db)
            is_valid, errors = service.validate_strategy_config(request.strategy_config)

            if is_valid:
                return APIResponseHelper.success_response(
                    data={"is_valid": True, "errors": []}, message="戦略設定は有効です"
                )
            else:
                return APIResponseHelper.success_response(
                    data={"is_valid": False, "errors": errors},
                    message="戦略設定に問題があります",
                )

        except Exception as e:
            logger.error(f"戦略検証エラー: {e}")
            raise HTTPException(status_code=500, detail="戦略検証に失敗しました")

    return await APIErrorHandler.handle_api_exception(_validate)


@router.post("/preview")
async def preview_strategy(
    request: ValidateStrategyRequest, db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    戦略設定のプレビューを生成

    Args:
        request: 戦略プレビューリクエスト

    Returns:
        戦略プレビューデータ
    """

    async def _preview():
        try:
            service = StrategyBuilderService(db)

            # まず戦略設定を検証
            is_valid, errors = service.validate_strategy_config(request.strategy_config)

            if not is_valid:
                raise HTTPException(
                    status_code=400, detail=f"戦略設定が無効です: {', '.join(errors)}"
                )

            # プレビューデータを生成
            preview_data = service.generate_strategy_preview(request.strategy_config)

            return APIResponseHelper.success_response(
                data=preview_data, message="戦略プレビューが生成されました"
            )

        except Exception as e:
            logger.error(f"戦略プレビューエラー: {e}")
            raise HTTPException(
                status_code=500, detail="戦略プレビューの生成に失敗しました"
            )

    return await APIErrorHandler.handle_api_exception(_preview)


@router.post("/save")
async def save_strategy(
    request: StrategyCreateRequest, db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    戦略を保存

    Args:
        request: 戦略作成リクエスト

    Returns:
        保存された戦略情報
    """

    async def _save():
        try:
            service = StrategyBuilderService(db)
            user_strategy = service.save_strategy(
                name=request.name,
                description=request.description,
                strategy_config=request.strategy_config,
            )

            return APIResponseHelper.success_response(
                data=user_strategy.to_dict(),
                message=f"戦略 '{request.name}' を保存しました",
            )

        except ValueError as e:
            logger.warning(f"戦略保存バリデーションエラー: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"戦略保存エラー: {e}")
            raise HTTPException(status_code=500, detail="戦略の保存に失敗しました")

    return await APIErrorHandler.handle_api_exception(_save)


@router.get("/strategies")
async def get_strategies(
    active_only: bool = Query(True, description="アクティブな戦略のみを取得"),
    limit: Optional[int] = Query(None, ge=1, le=100, description="取得件数制限"),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    保存済み戦略の一覧を取得

    Args:
        active_only: アクティブな戦略のみを取得するか
        limit: 取得件数制限

    Returns:
        戦略一覧
    """

    async def _get_strategies():
        try:
            service = StrategyBuilderService(db)
            strategies = service.get_strategies(active_only=active_only, limit=limit)

            strategies_data = [strategy.to_dict() for strategy in strategies]

            return APIResponseHelper.success_response(
                data={"strategies": strategies_data, "count": len(strategies_data)},
                message=f"戦略一覧を取得しました（{len(strategies_data)}件）",
            )

        except Exception as e:
            logger.error(f"戦略一覧取得エラー: {e}")
            raise HTTPException(status_code=500, detail="戦略一覧の取得に失敗しました")

    return await APIErrorHandler.handle_api_exception(_get_strategies)


@router.get("/strategies/{strategy_id}")
async def get_strategy(
    strategy_id: int, db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    戦略の詳細を取得

    Args:
        strategy_id: 戦略ID

    Returns:
        戦略詳細情報
    """

    async def _get_strategy():
        try:
            service = StrategyBuilderService(db)
            strategy = service.get_strategy_by_id(strategy_id)

            if not strategy:
                raise HTTPException(
                    status_code=404, detail=f"戦略が見つかりません（ID: {strategy_id}）"
                )

            return APIResponseHelper.success_response(
                data=strategy.to_dict(),
                message=f"戦略詳細を取得しました（ID: {strategy_id}）",
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"戦略詳細取得エラー (ID={strategy_id}): {e}")
            raise HTTPException(status_code=500, detail="戦略詳細の取得に失敗しました")

    return await APIErrorHandler.handle_api_exception(_get_strategy)


@router.put("/strategies/{strategy_id}")
async def update_strategy(
    strategy_id: int, request: StrategyUpdateRequest, db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    戦略を更新

    Args:
        strategy_id: 戦略ID
        request: 戦略更新リクエスト

    Returns:
        更新された戦略情報
    """

    async def _update():
        try:
            service = StrategyBuilderService(db)
            updated_strategy = service.update_strategy(
                strategy_id=strategy_id,
                name=request.name,
                description=request.description,
                strategy_config=request.strategy_config,
            )

            if not updated_strategy:
                raise HTTPException(
                    status_code=404, detail=f"戦略が見つかりません（ID: {strategy_id}）"
                )

            return APIResponseHelper.success_response(
                data=updated_strategy.to_dict(),
                message=f"戦略を更新しました（ID: {strategy_id}）",
            )

        except ValueError as e:
            logger.warning(f"戦略更新バリデーションエラー: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"戦略更新エラー (ID={strategy_id}): {e}")
            raise HTTPException(status_code=500, detail="戦略の更新に失敗しました")

    return await APIErrorHandler.handle_api_exception(_update)


@router.delete("/strategies/{strategy_id}")
async def delete_strategy(
    strategy_id: int, db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    戦略を削除（論理削除）

    Args:
        strategy_id: 戦略ID

    Returns:
        削除結果
    """

    async def _delete():
        try:
            service = StrategyBuilderService(db)
            result = service.delete_strategy(strategy_id)

            if not result:
                raise HTTPException(
                    status_code=404, detail=f"戦略が見つかりません（ID: {strategy_id}）"
                )

            return APIResponseHelper.success_response(
                data={"deleted": True, "strategy_id": strategy_id},
                message=f"戦略を削除しました（ID: {strategy_id}）",
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"戦略削除エラー (ID={strategy_id}): {e}")
            raise HTTPException(status_code=500, detail="戦略の削除に失敗しました")

    return await APIErrorHandler.handle_api_exception(_delete)
