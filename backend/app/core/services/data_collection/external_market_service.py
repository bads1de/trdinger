"""
外部市場データ収集サービス

yfinance APIを使用してSP500、NASDAQ、DXY、VIXなどの
外部市場データを取得・処理するサービスです。
"""

import logging
import asyncio
from typing import List, Dict, Optional
from datetime import datetime, timezone

import yfinance as yf
import pandas as pd

from app.core.utils.data_converter import DataValidator
from database.repositories.external_market_repository import ExternalMarketRepository

logger = logging.getLogger(__name__)


class ExternalMarketService:
    """外部市場データ収集サービス"""

    # 対象シンボルの定義（計画書に基づく）
    SYMBOLS = {
        "^GSPC": "S&P 500",
        "^IXIC": "NASDAQ Composite",
        "DX-Y.NYB": "US Dollar Index",
        "^VIX": "CBOE Volatility Index",
    }

    def __init__(self):
        """サービスを初期化"""
        self.session = None

    async def __aenter__(self):
        """非同期コンテキストマネージャーの開始"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """非同期コンテキストマネージャーの終了"""
        pass

    def fetch_external_market_data(
        self,
        symbols: Optional[List[str]] = None,
        period: str = "1mo",
        interval: str = "1d",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[dict]:
        """
        yfinance APIから外部市場データを取得

        Args:
            symbols: 取得するシンボルのリスト（デフォルト: 全シンボル）
            period: 取得期間（1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max）
            interval: データ間隔（1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo）
            start_date: 開始日（YYYY-MM-DD形式、periodより優先）
            end_date: 終了日（YYYY-MM-DD形式、periodより優先）

        Returns:
            外部市場データのリスト

        Raises:
            Exception: データ取得エラーの場合
        """
        try:
            target_symbols = symbols or list(self.SYMBOLS.keys())
            logger.info(
                f"外部市場データを取得中: symbols={target_symbols}, period={period}"
            )

            all_data = []

            for symbol in target_symbols:
                try:
                    # yfinanceでデータ取得
                    ticker = yf.Ticker(symbol)

                    # 日付指定がある場合は期間指定より優先
                    if start_date and end_date:
                        hist_data = ticker.history(
                            start=start_date, end=end_date, interval=interval
                        )
                    elif start_date:
                        hist_data = ticker.history(start=start_date, interval=interval)
                    else:
                        hist_data = ticker.history(period=period, interval=interval)

                    if hist_data.empty:
                        logger.warning(
                            f"シンボル {symbol} のデータが取得できませんでした"
                        )
                        continue

                    # データ変換
                    symbol_data = self._convert_yfinance_data_to_db_format(
                        symbol, hist_data
                    )
                    all_data.extend(symbol_data)

                    logger.info(
                        f"シンボル {symbol} のデータを {len(symbol_data)} 件取得しました"
                    )

                except Exception as e:
                    logger.error(f"シンボル {symbol} のデータ取得エラー: {e}")
                    continue

            logger.info(f"外部市場データを合計 {len(all_data)} 件取得しました")
            return all_data

        except Exception as e:
            logger.error(f"外部市場データ取得エラー: {e}")
            raise

    def _convert_yfinance_data_to_db_format(
        self, symbol: str, yfinance_data: pd.DataFrame
    ) -> List[dict]:
        """
        yfinanceデータをデータベース形式に変換

        Args:
            symbol: シンボル
            yfinance_data: yfinanceから取得したデータ

        Returns:
            データベース挿入用のデータリスト
        """
        db_records = []
        current_time = datetime.now(timezone.utc)

        for index, row in yfinance_data.iterrows():
            try:
                # タイムスタンプの処理
                if hasattr(index, "tz_localize"):
                    # タイムゾーン情報がない場合はUTCとして扱う
                    if index.tz is None:
                        data_timestamp = index.tz_localize("UTC")
                    else:
                        data_timestamp = index.tz_convert("UTC")
                else:
                    # 通常のdatetimeの場合
                    data_timestamp = index.replace(tzinfo=timezone.utc)

                # NaN値の処理
                open_price = row["Open"] if pd.notna(row["Open"]) else 0.0
                high_price = row["High"] if pd.notna(row["High"]) else 0.0
                low_price = row["Low"] if pd.notna(row["Low"]) else 0.0
                close_price = row["Close"] if pd.notna(row["Close"]) else 0.0
                volume = row["Volume"] if pd.notna(row["Volume"]) else None

                # 価格データの妥当性チェック
                if all(
                    x <= 0 for x in [open_price, high_price, low_price, close_price]
                ):
                    logger.warning(
                        f"シンボル {symbol} の {data_timestamp} のデータが無効です（全て0以下）"
                    )
                    continue

                db_record = {
                    "symbol": symbol,
                    "open": float(open_price),
                    "high": float(high_price),
                    "low": float(low_price),
                    "close": float(close_price),
                    "volume": int(volume) if volume is not None else None,
                    "data_timestamp": data_timestamp,
                    "timestamp": current_time,
                }

                db_records.append(db_record)

            except Exception as e:
                logger.warning(
                    f"データ変換エラー、スキップします: {index}, エラー: {e}"
                )
                continue

        return db_records

    async def fetch_and_save_external_market_data(
        self,
        symbols: Optional[List[str]] = None,
        period: str = "1mo",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        repository: Optional[ExternalMarketRepository] = None,
    ) -> Dict:
        """
        外部市場データを取得してデータベースに保存

        Args:
            symbols: 取得するシンボルのリスト
            period: 取得期間
            start_date: 開始日（YYYY-MM-DD形式、periodより優先）
            end_date: 終了日（YYYY-MM-DD形式、periodより優先）
            repository: データベースリポジトリ

        Returns:
            処理結果を含む辞書
        """
        try:
            # データ取得
            external_market_data = self.fetch_external_market_data(
                symbols=symbols, period=period, start_date=start_date, end_date=end_date
            )

            if not external_market_data:
                logger.warning("取得した外部市場データが空です")
                return {
                    "success": True,
                    "fetched_count": 0,
                    "inserted_count": 0,
                    "message": "取得データが空でした",
                }

            # データ検証
            if not DataValidator.validate_external_market_data(external_market_data):
                raise ValueError("取得した外部市場データが無効です")

            # データベースに保存
            if repository:
                inserted_count = repository.insert_external_market_data(
                    external_market_data
                )
            else:
                logger.warning(
                    "リポジトリが提供されていません。データは保存されませんでした。"
                )
                inserted_count = 0

            result = {
                "success": True,
                "fetched_count": len(external_market_data),
                "inserted_count": inserted_count,
                "message": f"外部市場データを {inserted_count} 件保存しました",
            }

            logger.info(result["message"])
            return result

        except Exception as e:
            logger.error(f"外部市場データの取得・保存エラー: {e}")
            return {
                "success": False,
                "error": str(e),
                "fetched_count": 0,
                "inserted_count": 0,
            }

    def get_available_symbols(self) -> Dict[str, str]:
        """
        利用可能なシンボルの一覧を取得

        Returns:
            シンボルと説明の辞書
        """
        return self.SYMBOLS.copy()

    async def fetch_latest_data(
        self, symbols: Optional[List[str]] = None
    ) -> List[dict]:
        """
        最新データのみを取得（1日分）

        Args:
            symbols: 取得するシンボルのリスト

        Returns:
            最新の外部市場データのリスト
        """
        return self.fetch_external_market_data(
            symbols=symbols,
            period="2d",  # 最新データを確実に取得するため2日分取得
            interval="1d",
        )

    async def fetch_historical_data(
        self,
        symbols: Optional[List[str]] = None,
        period: str = "1y",
    ) -> List[dict]:
        """
        履歴データを取得

        Args:
            symbols: 取得するシンボルのリスト
            period: 取得期間

        Returns:
            履歴の外部市場データのリスト
        """
        return self.fetch_external_market_data(
            symbols=symbols, period=period, interval="1d"
        )


# スタンドアロン実行用の関数
async def main():
    """メイン関数（テスト用）"""
    logging.basicConfig(level=logging.INFO)

    async with ExternalMarketService() as service:
        # 利用可能なシンボルの確認
        symbols = service.get_available_symbols()
        print(f"利用可能なシンボル: {symbols}")

        # 最新データの取得テスト
        latest_data = await service.fetch_latest_data()
        print(f"取得データ数: {len(latest_data)}")

        if latest_data:
            print(f"サンプルデータ: {latest_data[0]}")


if __name__ == "__main__":
    asyncio.run(main())
