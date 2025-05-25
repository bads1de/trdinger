"""
市場データサービス

CCXT ライブラリを使用してBybit取引所からOHLCVデータを取得するサービスです。
リアルタイムの市場データを提供し、エラーハンドリングとデータ検証を含みます。

@author Trdinger Development Team
@version 1.0.0
"""

import asyncio
import ccxt
from typing import List, Optional
from datetime import datetime, timezone
import logging

from config import MarketDataConfig


# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BybitMarketDataService:
    """
    Bybit取引所からの市場データ取得サービス

    CCXT ライブラリを使用してBybit取引所からOHLCVデータを取得し、
    適切なエラーハンドリングとデータ検証を提供します。
    """

    def __init__(self):
        """
        サービスを初期化します

        Bybit取引所のインスタンスを作成し、レート制限を有効化します。
        """
        try:
            self.exchange = ccxt.bybit(MarketDataConfig.BYBIT_CONFIG)
            logger.info("Bybit取引所インスタンスを初期化しました")
        except Exception as e:
            logger.error(f"Bybit取引所の初期化に失敗しました: {e}")
            raise

    async def fetch_ohlcv_data(
        self, symbol: str, timeframe: str = "1h", limit: int = 100
    ) -> List[List]:
        """
        OHLCVデータを取得します

        Args:
            symbol: 取引ペアシンボル（例: 'BTC/USD:BTC'）
            timeframe: 時間軸（例: '1h', '1d'）
            limit: 取得するデータ数（1-1000）

        Returns:
            OHLCVデータのリスト。各要素は [timestamp, open, high, low, close, volume] の形式

        Raises:
            ValueError: パラメータが無効な場合
            ccxt.NetworkError: ネットワークエラーの場合
            ccxt.ExchangeError: 取引所エラーの場合
        """
        # パラメータの検証
        self._validate_parameters(symbol, timeframe, limit)

        # シンボルの正規化
        normalized_symbol = self.normalize_symbol(symbol)

        try:
            logger.info(
                f"OHLCVデータを取得中: {normalized_symbol}, {timeframe}, limit={limit}"
            )

            # 非同期でOHLCVデータを取得
            ohlcv_data = await asyncio.get_event_loop().run_in_executor(
                None,
                self.exchange.fetch_ohlcv,
                normalized_symbol,
                timeframe,
                None,  # since
                limit,
            )

            logger.info(f"OHLCVデータを取得しました: {len(ohlcv_data)}件")

            # データの検証
            self._validate_ohlcv_data(ohlcv_data)

            return ohlcv_data

        except ccxt.BadSymbol as e:
            logger.error(f"無効なシンボル: {normalized_symbol}")
            raise ccxt.BadSymbol(f"無効なシンボル: {normalized_symbol}") from e
        except ccxt.NetworkError as e:
            logger.error(f"ネットワークエラー: {e}")
            raise
        except ccxt.ExchangeError as e:
            logger.error(f"取引所エラー: {e}")
            raise
        except Exception as e:
            logger.error(f"予期しないエラー: {e}")
            raise ccxt.ExchangeError(f"データ取得中にエラーが発生しました: {e}") from e

    def normalize_symbol(self, symbol: str) -> str:
        """
        シンボルを正規化します

        Args:
            symbol: 正規化するシンボル

        Returns:
            正規化されたシンボル

        Raises:
            ValueError: サポートされていないシンボルの場合
        """
        return MarketDataConfig.normalize_symbol(symbol)

    def _validate_parameters(self, symbol: str, timeframe: str, limit: int) -> None:
        """
        パラメータを検証します

        Args:
            symbol: シンボル
            timeframe: 時間軸
            limit: 制限値

        Raises:
            ValueError: パラメータが無効な場合
        """
        if not symbol or not isinstance(symbol, str):
            raise ValueError("シンボルは空でない文字列である必要があります")

        if not MarketDataConfig.validate_timeframe(timeframe):
            raise ValueError(
                f"無効な時間軸: {timeframe}. "
                f"サポート対象: {', '.join(MarketDataConfig.SUPPORTED_TIMEFRAMES)}"
            )

        if not MarketDataConfig.validate_limit(limit):
            raise ValueError(
                f"制限値は{MarketDataConfig.MIN_LIMIT}から"
                f"{MarketDataConfig.MAX_LIMIT}の間である必要があります: {limit}"
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
            logger.warning("OHLCVデータが空です")
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


# サービスのシングルトンインスタンス
_service_instance: Optional[BybitMarketDataService] = None


def get_market_data_service() -> BybitMarketDataService:
    """
    市場データサービスのインスタンスを取得します

    Returns:
        BybitMarketDataServiceのインスタンス
    """
    global _service_instance
    if _service_instance is None:
        _service_instance = BybitMarketDataService()
    return _service_instance
