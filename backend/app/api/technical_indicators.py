"""
テクニカル指標API

テクニカル指標データの取得・計算機能を提供するAPIエンドポイント
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any
import logging

from database.connection import get_db, ensure_db_initialized
from database.repositories.technical_indicator_repository import (
    TechnicalIndicatorRepository,
)
from app.core.services.technical_indicator_service import TechnicalIndicatorService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["technical-indicators"])


@router.get("/technical-indicators")
async def get_technical_indicator_data(
    symbol: str = Query(..., description="取引ペアシンボル（例: 'BTC/USDT'）"),
    timeframe: str = Query("1h", description="時間枠（例: '1h', '4h', '1d'）"),
    indicator_type: Optional[str] = Query(
        None, description="指標タイプ（例: 'SMA', 'EMA', 'RSI'）"
    ),
    period: Optional[int] = Query(None, description="期間（例: 14, 20, 50）"),
    limit: Optional[int] = Query(100, description="取得するデータ数（1-1000）"),
    start_date: Optional[str] = Query(None, description="開始日（YYYY-MM-DD形式）"),
    end_date: Optional[str] = Query(None, description="終了日（YYYY-MM-DD形式）"),
    db: Session = Depends(get_db),
):
    """
    テクニカル指標データを取得します

    指定された条件に基づいてテクニカル指標データを取得します。

    Args:
        symbol: 取引ペアシンボル（例: 'BTC/USDT'）
        timeframe: 時間枠（例: '1h', '4h', '1d'）
        indicator_type: 指標タイプ（例: 'SMA', 'EMA', 'RSI'）
        period: 期間（例: 14, 20, 50）
        limit: 取得するデータ数（1-1000）
        start_date: 開始日（YYYY-MM-DD形式）
        end_date: 終了日（YYYY-MM-DD形式）
        db: データベースセッション

    Returns:
        テクニカル指標データを含むJSONレスポンス

    Raises:
        HTTPException: パラメータが無効な場合やデータベースエラーが発生した場合
    """
    try:
        logger.info(
            f"テクニカル指標データ取得開始: symbol={symbol}, timeframe={timeframe}, "
            f"indicator_type={indicator_type}, period={period}"
        )

        # パラメータの検証
        if limit and (limit < 1 or limit > 1000):
            raise HTTPException(
                status_code=400, detail="limitは1から1000の間で指定してください"
            )

        # 日付パラメータの変換
        start_time = None
        end_time = None
        if start_date:
            try:
                start_time = datetime.strptime(start_date, "%Y-%m-%d").replace(
                    tzinfo=timezone.utc
                )
            except ValueError:
                raise HTTPException(
                    status_code=400, detail="start_dateの形式が無効です（YYYY-MM-DD）"
                )

        if end_date:
            try:
                end_time = datetime.strptime(end_date, "%Y-%m-%d").replace(
                    tzinfo=timezone.utc
                ) + timedelta(days=1)
            except ValueError:
                raise HTTPException(
                    status_code=400, detail="end_dateの形式が無効です（YYYY-MM-DD）"
                )

        # データベース初期化確認
        if not ensure_db_initialized():
            logger.error("データベースの初期化に失敗しました")
            raise HTTPException(
                status_code=500, detail="データベースの初期化に失敗しました"
            )

        # リポジトリを作成
        repository = TechnicalIndicatorRepository(db)

        # テクニカル指標データを取得
        technical_indicators = repository.get_technical_indicator_data(
            symbol=symbol,
            indicator_type=indicator_type,
            period=period,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
        )

        # レスポンスデータを作成
        indicator_data = [indicator.to_dict() for indicator in technical_indicators]

        logger.info(f"テクニカル指標データ取得完了: {len(indicator_data)}件")

        return {
            "success": True,
            "data": {
                "symbol": symbol,
                "timeframe": timeframe,
                "indicator_type": indicator_type,
                "period": period,
                "count": len(indicator_data),
                "technical_indicators": indicator_data,
            },
            "message": f"{symbol} のテクニカル指標データを取得しました（{len(indicator_data)}件）",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"テクニカル指標データ取得エラー: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"テクニカル指標データの取得中にエラーが発生しました: {str(e)}",
        )


@router.post("/technical-indicators/calculate")
async def calculate_technical_indicator(
    symbol: str = Query(..., description="取引ペアシンボル（例: 'BTC/USDT'）"),
    timeframe: str = Query("1h", description="時間枠（例: '1h', '4h', '1d'）"),
    indicator_type: str = Query(
        ..., description="指標タイプ（例: 'SMA', 'EMA', 'RSI'）"
    ),
    period: int = Query(..., description="期間（例: 14, 20, 50）"),
    limit: Optional[int] = Query(
        None, description="OHLCVデータの取得件数制限（計算用）"
    ),
    db: Session = Depends(get_db),
):
    """
    テクニカル指標を計算してデータベースに保存します

    指定された条件でテクニカル指標を計算し、データベースに保存します。

    Args:
        symbol: 取引ペアシンボル（例: 'BTC/USDT'）
        timeframe: 時間枠（例: '1h', '4h', '1d'）
        indicator_type: 指標タイプ（例: 'SMA', 'EMA', 'RSI'）
        period: 期間（例: 14, 20, 50）
        limit: OHLCVデータの取得件数制限
        db: データベースセッション

    Returns:
        計算結果を含むJSONレスポンス

    Raises:
        HTTPException: パラメータが無効な場合やAPI/データベースエラーが発生した場合
    """
    try:
        logger.info(
            f"テクニカル指標計算開始: symbol={symbol}, timeframe={timeframe}, "
            f"indicator_type={indicator_type}, period={period}"
        )

        # データベース初期化確認
        if not ensure_db_initialized():
            logger.error("データベースの初期化に失敗しました")
            raise HTTPException(
                status_code=500, detail="データベースの初期化に失敗しました"
            )

        # テクニカル指標サービスを作成
        service = TechnicalIndicatorService()

        # データベースリポジトリを作成
        repository = TechnicalIndicatorRepository(db)

        # テクニカル指標を計算・保存
        result = await service.calculate_and_save_technical_indicator(
            symbol=symbol,
            timeframe=timeframe,
            indicator_type=indicator_type,
            period=period,
            repository=repository,
            limit=limit,
        )

        logger.info(f"テクニカル指標計算完了: {result}")

        return {
            "success": True,
            "data": result,
            "message": f"{result['saved_count']}件のテクニカル指標データを保存しました",
        }

    except ValueError as e:
        logger.error(f"パラメータエラー: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"テクニカル指標計算エラー: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"テクニカル指標の計算中にエラーが発生しました: {str(e)}",
        )


@router.get("/technical-indicators/current")
async def get_current_technical_indicator_values(
    symbol: str = Query(..., description="取引ペアシンボル（例: 'BTC/USDT'）"),
    timeframe: str = Query("1h", description="時間枠（例: '1h', '4h', '1d'）"),
    db: Session = Depends(get_db),
):
    """
    現在のテクニカル指標値を取得します

    指定されたシンボルと時間枠の最新テクニカル指標値を取得します。

    Args:
        symbol: 取引ペアシンボル（例: 'BTC/USDT'）
        timeframe: 時間枠（例: '1h', '4h', '1d'）
        db: データベースセッション

    Returns:
        現在のテクニカル指標値を含むJSONレスポンス

    Raises:
        HTTPException: パラメータが無効な場合やデータベースエラーが発生した場合
    """
    try:
        logger.info(
            f"現在のテクニカル指標値取得開始: symbol={symbol}, timeframe={timeframe}"
        )

        # データベース初期化確認
        if not ensure_db_initialized():
            logger.error("データベースの初期化に失敗しました")
            raise HTTPException(
                status_code=500, detail="データベースの初期化に失敗しました"
            )

        # リポジトリを作成
        repository = TechnicalIndicatorRepository(db)

        # 現在のテクニカル指標値を取得
        current_values = repository.get_current_technical_indicator_values(
            symbol=symbol, timeframe=timeframe
        )

        logger.info(f"現在のテクニカル指標値取得完了: {len(current_values)}件")

        return {
            "success": True,
            "data": {
                "symbol": symbol,
                "timeframe": timeframe,
                "count": len(current_values),
                "current_values": current_values,
            },
            "message": f"{symbol} の現在のテクニカル指標値を取得しました（{len(current_values)}件）",
        }

    except Exception as e:
        logger.error(f"現在のテクニカル指標値取得エラー: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"現在のテクニカル指標値の取得中にエラーが発生しました: {str(e)}",
        )


@router.post("/technical-indicators/bulk-calculate")
async def bulk_calculate_technical_indicators(
    symbol: str = Query(..., description="取引ペアシンボル（例: 'BTC/USDT'）"),
    timeframe: str = Query("1h", description="時間枠（例: '1h', '4h', '1d'）"),
    use_default: bool = Query(
        True, description="デフォルト指標セットを使用するかどうか"
    ),
    limit: Optional[int] = Query(
        None, description="OHLCVデータの取得件数制限（計算用）"
    ),
    db: Session = Depends(get_db),
):
    """
    複数のテクニカル指標を一括計算してデータベースに保存します

    デフォルトの指標セット（SMA20/50, EMA20/50, RSI14）を一括計算します。

    Args:
        symbol: 取引ペアシンボル（例: 'BTC/USDT'）
        timeframe: 時間枠（例: '1h', '4h', '1d'）
        use_default: デフォルト指標セットを使用するかどうか
        limit: OHLCVデータの取得件数制限
        db: データベースセッション

    Returns:
        一括計算結果を含むJSONレスポンス

    Raises:
        HTTPException: パラメータが無効な場合やAPI/データベースエラーが発生した場合
    """
    try:
        logger.info(
            f"テクニカル指標一括計算開始: symbol={symbol}, timeframe={timeframe}"
        )

        # データベース初期化確認
        if not ensure_db_initialized():
            logger.error("データベースの初期化に失敗しました")
            raise HTTPException(
                status_code=500, detail="データベースの初期化に失敗しました"
            )

        # テクニカル指標サービスを作成
        service = TechnicalIndicatorService()

        # データベースリポジトリを作成
        repository = TechnicalIndicatorRepository(db)

        # 指標設定を取得
        if use_default:
            indicators = service.get_default_indicators()
        else:
            # カスタム指標設定（将来の拡張用）
            indicators = service.get_default_indicators()

        # 複数のテクニカル指標を一括計算・保存
        result = await service.calculate_and_save_multiple_indicators(
            symbol=symbol,
            timeframe=timeframe,
            indicators=indicators,
            repository=repository,
            limit=limit,
        )

        logger.info(f"テクニカル指標一括計算完了: {result}")

        return {
            "success": True,
            "data": result,
            "message": f"{result['total_saved']}件のテクニカル指標データを保存しました "
            f"({result['successful_indicators']}/{result['total_indicators']}指標成功)",
        }

    except ValueError as e:
        logger.error(f"パラメータエラー: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"テクニカル指標一括計算エラー: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"テクニカル指標の一括計算中にエラーが発生しました: {str(e)}",
        )


@router.get("/technical-indicators/supported")
async def get_supported_indicators():
    """
    サポートされているテクニカル指標の情報を取得します

    Returns:
        サポート指標の情報を含むJSONレスポンス
    """
    try:
        logger.info("サポート指標情報取得開始")

        # テクニカル指標サービスを作成
        service = TechnicalIndicatorService()

        # サポート指標情報を取得
        supported_indicators = service.get_supported_indicators()
        default_indicators = service.get_default_indicators()

        logger.info("サポート指標情報取得完了")

        return {
            "success": True,
            "data": {
                "supported_indicators": supported_indicators,
                "default_indicators": default_indicators,
            },
            "message": "サポートされているテクニカル指標の情報を取得しました",
        }

    except Exception as e:
        logger.error(f"サポート指標情報取得エラー: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"サポート指標情報の取得中にエラーが発生しました: {str(e)}",
        )


@router.get("/technical-indicators/available")
async def get_available_indicators(
    symbol: str = Query(..., description="取引ペアシンボル（例: 'BTC/USDT'）"),
    db: Session = Depends(get_db),
):
    """
    指定されたシンボルで利用可能なテクニカル指標の一覧を取得します

    Args:
        symbol: 取引ペアシンボル（例: 'BTC/USDT'）
        db: データベースセッション

    Returns:
        利用可能な指標の一覧を含むJSONレスポンス

    Raises:
        HTTPException: パラメータが無効な場合やデータベースエラーが発生した場合
    """
    try:
        logger.info(f"利用可能な指標一覧取得開始: symbol={symbol}")

        # データベース初期化確認
        if not ensure_db_initialized():
            logger.error("データベースの初期化に失敗しました")
            raise HTTPException(
                status_code=500, detail="データベースの初期化に失敗しました"
            )

        # リポジトリを作成
        repository = TechnicalIndicatorRepository(db)

        # 利用可能な指標を取得
        available_indicators = repository.get_available_indicators(symbol=symbol)

        logger.info(f"利用可能な指標一覧取得完了: {len(available_indicators)}件")

        return {
            "success": True,
            "data": {
                "symbol": symbol,
                "count": len(available_indicators),
                "available_indicators": available_indicators,
            },
            "message": f"{symbol} で利用可能なテクニカル指標の一覧を取得しました（{len(available_indicators)}件）",
        }

    except Exception as e:
        logger.error(f"利用可能な指標一覧取得エラー: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"利用可能な指標一覧の取得中にエラーが発生しました: {str(e)}",
        )
