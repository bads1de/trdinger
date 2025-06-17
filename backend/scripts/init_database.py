#!/usr/bin/env python3
"""
データベース初期化スクリプト
"""
import asyncio
import logging
from database.connection import init_db, test_connection, SessionLocal
from data_collector.collector import DataCollector


# ログ設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def main():
    """
    メイン処理
    1. データベース接続テスト
    2. テーブル作成
    3. BTC/USD日足データの収集
    """
    try:
        logger.info("=== データベース初期化開始 ===")

        # 1. データベース接続テスト
        logger.info("データベース接続をテスト中...")
        if not test_connection():
            logger.error("データベース接続に失敗しました")
            return

        # 2. データベース初期化（テーブル作成）
        logger.info("データベースを初期化中...")
        init_db()

        # 3. BTC/USDT日足データを収集
        logger.info("BTC/USDT日足データを収集中...")
        try:
            with SessionLocal() as db:
                collector = DataCollector(db)
                collected_count = await collector.collect_historical_data(
                    symbol="BTC/USDT", timeframe="1d", days_back=365
                )
        except Exception as e:
            logger.error(f"BTC日足データ収集エラー: {e}", exc_info=True)
            collected_count = 0

        logger.info("=== 初期化完了 ===")
        logger.info(f"収集されたデータ件数: {collected_count}")

    except Exception as e:
        logger.error(f"初期化エラー: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
