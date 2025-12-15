"""
市場データオーケストレーションサービス
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from app.utils.data_conversion import OHLCVDataConverter
from app.utils.response import api_response
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
        start_time = None
        end_time = None

        if start_date:
            start_time = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
        if end_date:
            end_time = datetime.fromisoformat(end_date.replace("Z", "+00:00"))

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

        logger.debug(f"OHLCVデータ取得成功: {len(ohlcv_data)}件")

        return api_response(
            success=True,
            data={"ohlcv_data": ohlcv_data, "symbol": symbol, "timeframe": timeframe},
            message=f"{symbol} の {timeframe} OHLCVデータを取得しました",
        )


