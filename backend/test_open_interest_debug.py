"""
オープンインタレスト機能のデバッグテスト
"""

import asyncio
import logging
from app.core.services.open_interest_service import BybitOpenInterestService
from database.connection import get_db, init_db
from database.repository import OpenInterestRepository

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_open_interest_debug():
    """オープンインタレスト機能のデバッグテスト"""
    
    # データベース初期化
    init_db()
    
    # サービスとリポジトリの作成
    service = BybitOpenInterestService()
    db = next(get_db())
    repository = OpenInterestRepository(db)
    
    try:
        # 1. 現在のオープンインタレスト取得テスト
        logger.info("=== 現在のオープンインタレスト取得テスト ===")
        current_oi = await service.fetch_current_open_interest("BTC/USDT")
        logger.info(f"現在のオープンインタレスト: {current_oi}")
        
        # 2. オープンインタレスト履歴取得テスト
        logger.info("=== オープンインタレスト履歴取得テスト ===")
        history_oi = await service.fetch_open_interest_history("BTC/USDT", limit=3)
        logger.info(f"履歴データ件数: {len(history_oi)}")
        for i, item in enumerate(history_oi):
            logger.info(f"履歴データ {i+1}: {item}")
        
        # 3. データベース保存テスト
        logger.info("=== データベース保存テスト ===")
        result = await service.fetch_and_save_open_interest_data(
            symbol="BTC/USDT",
            limit=3,
            repository=repository,
            fetch_all=False,
        )
        logger.info(f"保存結果: {result}")
        
        # 4. データベースからの取得テスト
        logger.info("=== データベースからの取得テスト ===")
        saved_data = repository.get_open_interest_data("BTC/USDT:USDT", limit=10)
        logger.info(f"保存されたデータ件数: {len(saved_data)}")
        for i, item in enumerate(saved_data):
            logger.info(f"保存データ {i+1}: {item}")
        
    except Exception as e:
        logger.error(f"テストエラー: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        db.close()

if __name__ == "__main__":
    asyncio.run(test_open_interest_debug())
