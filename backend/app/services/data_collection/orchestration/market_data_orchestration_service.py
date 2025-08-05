"""
市場データオーケストレーションサービス
"""

import logging
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from app.utils.api_utils import APIResponseHelper, DateTimeHelper
from app.utils.data_conversion import OHLCVDataConverter
from database.repositories.ohlcv_repository import OHLCVRepository

logger = logging.getLogger(__name__)


class MarketDataOrchestrationService:
    def __init__(self, db_session: Session):
        self.db_session = db_session
        self.repository = OHLCVRepository(db_session)

    async def get_ohlcv_data(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        logger.info(
            f"OHLCVデータ取得リクエスト: symbol={symbol}, timeframe={timeframe}, limit={limit}"
        )

        start_time = None
        end_time = None

        if start_date:
            start_time = DateTimeHelper.parse_iso_datetime(start_date)
        if end_date:
            end_time = DateTimeHelper.parse_iso_datetime(end_date)

        if start_time is None and end_time is None:
            ohlcv_records = self.repository.get_latest_ohlcv_data(
                symbol=symbol,
                timeframe=timeframe,
                limit=limit,
            )
        else:
            ohlcv_records = self.repository.get_ohlcv_data(
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time,
                limit=limit,
            )

        ohlcv_data = OHLCVDataConverter.db_to_api_format(ohlcv_records)

        logger.info(f"OHLCVデータ取得成功: {len(ohlcv_data)}件")

        return APIResponseHelper.api_response(
            success=True,
            data={"ohlcv_data": ohlcv_data, "symbol": symbol, "timeframe": timeframe},
            message=f"{symbol} の {timeframe} OHLCVデータを取得しました",
        )
