"""
CCXT ライブラリを使用してBybit取引所からOHLCVデータを取得するサービスです。
リアルタイムの市場データを提供し、エラーハンドリングとデータ検証を含みます。
"""

import logging
from typing import Any, Dict, List, Optional

from app.config.unified_config import unified_config
from app.utils.data_conversion import OHLCVDataConverter
from database.repositories.ohlcv_repository import OHLCVRepository
from app.utils.normalization_service import SymbolNormalizationService

from .bybit_service import BybitService

logger = logging.getLogger(__name__)


class BybitMarketDataService(BybitService):
    """
    Bybit取引所からの市場データ取得サービス

    CCXT ライブラリを使用してBybit取引所からOHLCVデータを取得し、
    適切なエラーハンドリングとデータ検証を提供します。
    """

    def __init__(self):
        """
        サービスを初期化します
        """
        super().__init__()

    async def fetch_ohlcv_data(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 100,
        since: Optional[int] = None,
        params: Dict[str, Any] = {},
    ) -> List[List]:
        """
        OHLCVデータを取得します

        Args:
            symbol: 取引ペアシンボル（例: 'BTC/USD:BTC'）
            timeframe: 時間軸（例: '1h', '1d'）
            limit: 取得するデータ数（1-1000）
            since: 開始タイムスタンプ（ミリ秒）
            params: CCXT追加パラメータ

        Returns:
            OHLCVデータのリスト。各要素は [timestamp, open, high, low, close, volume] の形式

        Raises:
            ValueError: パラメータが無効な場合
            ccxt.NetworkError: ネットワークエラーの場合
            ccxt.ExchangeError: 取引所エラーの場合
        """
        # パラメータの検証
        self._validate_parameters(symbol, limit=limit)
        self._validate_timeframe(timeframe)

        # シンボルの正規化
        normalized_symbol = SymbolNormalizationService.normalize_symbol(symbol, "bybit")

        # 基底クラスの共通エラーハンドリングを使用
        ohlcv_data = await self._handle_ccxt_errors(
            f"OHLCVデータ取得: {normalized_symbol}, {timeframe}, limit={limit}, since={since}",
            self.exchange.fetch_ohlcv,
            normalized_symbol,
            timeframe,
            since,
            limit,
            params,
        )

        # データの検証
        self._validate_ohlcv_data(ohlcv_data)

        return ohlcv_data

    async def _save_ohlcv_to_database(
        self,
        ohlcv_data: List[List],
        symbol: str,
        timeframe: str,
        repository: OHLCVRepository,
    ) -> int:
        """
        OHLCVデータをデータベースに保存します（内部メソッド）

        Args:
            ohlcv_data: OHLCVデータのリスト
            symbol: 取引ペアシンボル
            timeframe: 時間軸
            repository: OHLCVリポジトリ

        Returns:
            保存された件数
        """
        # OHLCVデータを辞書形式に変換
        records = OHLCVDataConverter.ccxt_to_db_format(ohlcv_data, symbol, timeframe)

        # データベースに挿入
        return repository.insert_ohlcv_data(records)

    def _validate_timeframe(self, timeframe: str) -> None:
        """
        時間軸が有効かどうかを検証します

        Args:
            timeframe: 検証する時間軸

        Raises:
            ValueError: 時間軸が無効な場合
        """
        if timeframe not in unified_config.market.supported_timeframes:
            raise ValueError(
                f"無効な時間軸: {timeframe}. "
                f"サポート対象: {', '.join(unified_config.market.supported_timeframes)}"
            )

    def _validate_ohlcv_data(self, data: List[List]) -> None:
        """
        OHLCVデータの形式を検証します

        Args:
            data: 検証するOHLCVデータ

        Raises:
            ValueError: データ形式が無効な場合
        """
        if not isinstance(data, list):
            raise ValueError("OHLCVデータはリストである必要があります")

        if len(data) == 0:
            logger.warning("取得したOHLCVデータが空です。")
            return

        for i, candle in enumerate(data):
            if not isinstance(candle, list) or len(candle) != 6:
                raise ValueError(f"ローソク足データ[{i}]の形式が無効です: {candle}")

            timestamp, open_price, high, low, close, volume = candle

            # 数値型の検証
            if not all(isinstance(x, (int, float)) for x in candle):
                raise ValueError(
                    f"ローソク足データ[{i}]に非数値が含まれています: {candle}"
                )

            # 価格関係の検証
            if high < max(open_price, close) or low > min(open_price, close):
                raise ValueError(f"ローソク足データ[{i}]の価格関係が無効です: {candle}")

            if high < low:
                raise ValueError(
                    f"ローソク足データ[{i}]で高値が安値より小さいです: {candle}"
                )

            # 正の値の検証
            if any(x < 0 for x in [open_price, high, low, close]):
                raise ValueError(
                    f"ローソク足データ[{i}]に負の価格が含まれています: {candle}"
                )

            if volume < 0:
                raise ValueError(
                    f"ローソク足データ[{i}]に負の出来高が含まれています: {candle}"
                )
