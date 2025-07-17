#!/usr/bin/env python3
"""
Fear & Greed Index 実装テストスクリプト

実装したFear & Greed Index機能の包括的なテストを実行します。
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timezone

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connection import SessionLocal, init_db, test_connection
from database.repositories.fear_greed_repository import FearGreedIndexRepository
from app.core.services.data_collection.fear_greed_service import FearGreedIndexService
from data_collector.external_market_collector import ExternalMarketDataCollector

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_database_setup():
    """データベースセットアップのテスト"""
    logger.info("=== データベースセットアップテスト ===")
    
    # 1. データベース接続テスト
    logger.info("データベース接続をテスト中...")
    if not test_connection():
        logger.error("データベース接続に失敗しました")
        return False
    
    # 2. テーブル作成
    logger.info("テーブルを作成中...")
    try:
        init_db()
        logger.info("テーブル作成完了")
    except Exception as e:
        logger.error(f"テーブル作成エラー: {e}")
        return False
    
    # 3. リポジトリテスト
    logger.info("リポジトリをテスト中...")
    try:
        with SessionLocal() as db:
            repository = FearGreedIndexRepository(db)
            count = repository.get_data_count()
            logger.info(f"現在のデータ件数: {count}")
    except Exception as e:
        logger.error(f"リポジトリテストエラー: {e}")
        return False
    
    logger.info("✅ データベースセットアップテスト完了")
    return True


async def test_api_service():
    """APIサービスのテスト"""
    logger.info("=== APIサービステスト ===")
    
    try:
        async with FearGreedIndexService() as service:
            # 1. データ取得テスト
            logger.info("Fear & Greed Index データを取得中...")
            data = await service.fetch_fear_greed_data(limit=5)
            
            if data:
                logger.info(f"✅ データ取得成功: {len(data)} 件")
                logger.info(f"最新データ例: {data[0] if data else 'なし'}")
            else:
                logger.warning("⚠️ データが取得できませんでした")
                return False
                
    except Exception as e:
        logger.error(f"APIサービステストエラー: {e}")
        return False
    
    logger.info("✅ APIサービステスト完了")
    return True


async def test_data_collection():
    """データ収集のテスト"""
    logger.info("=== データ収集テスト ===")
    
    try:
        with SessionLocal() as db:
            repository = FearGreedIndexRepository(db)
            
            # 収集前のデータ件数
            before_count = repository.get_data_count()
            logger.info(f"収集前のデータ件数: {before_count}")
            
            # データ収集実行
            async with ExternalMarketDataCollector() as collector:
                result = await collector.collect_fear_greed_data(
                    limit=10,
                    db_session=db
                )
            
            if result["success"]:
                logger.info(f"✅ データ収集成功: {result['message']}")
                logger.info(f"取得件数: {result['fetched_count']}")
                logger.info(f"挿入件数: {result['inserted_count']}")
                
                # 収集後のデータ件数
                after_count = repository.get_data_count()
                logger.info(f"収集後のデータ件数: {after_count}")
                
                return True
            else:
                logger.error(f"❌ データ収集失敗: {result.get('error', 'Unknown error')}")
                return False
                
    except Exception as e:
        logger.error(f"データ収集テストエラー: {e}")
        return False


async def test_repository_operations():
    """リポジトリ操作のテスト"""
    logger.info("=== リポジトリ操作テスト ===")
    
    try:
        with SessionLocal() as db:
            repository = FearGreedIndexRepository(db)
            
            # 1. データ範囲取得
            data_range = repository.get_data_range()
            logger.info(f"データ範囲: {data_range}")
            
            # 2. 最新データ取得
            latest_data = repository.get_latest_fear_greed_data(limit=3)
            logger.info(f"最新データ件数: {len(latest_data)}")
            
            if latest_data:
                latest = latest_data[0]
                logger.info(f"最新データ例: value={latest.value}, classification={latest.value_classification}")
            
            # 3. 最新タイムスタンプ取得
            latest_timestamp = repository.get_latest_data_timestamp()
            logger.info(f"最新タイムスタンプ: {latest_timestamp}")
            
    except Exception as e:
        logger.error(f"リポジトリ操作テストエラー: {e}")
        return False
    
    logger.info("✅ リポジトリ操作テスト完了")
    return True


async def test_incremental_collection():
    """差分収集のテスト"""
    logger.info("=== 差分収集テスト ===")
    
    try:
        async with ExternalMarketDataCollector() as collector:
            result = await collector.collect_incremental_fear_greed_data()
            
            if result["success"]:
                logger.info(f"✅ 差分収集成功: {result['message']}")
                logger.info(f"収集タイプ: {result.get('collection_type', 'unknown')}")
                logger.info(f"取得件数: {result['fetched_count']}")
                logger.info(f"挿入件数: {result['inserted_count']}")
                return True
            else:
                logger.error(f"❌ 差分収集失敗: {result.get('error', 'Unknown error')}")
                return False
                
    except Exception as e:
        logger.error(f"差分収集テストエラー: {e}")
        return False


async def test_data_status():
    """データ状態確認のテスト"""
    logger.info("=== データ状態確認テスト ===")
    
    try:
        async with ExternalMarketDataCollector() as collector:
            status = await collector.get_data_status()
            
            if status["success"]:
                logger.info("✅ データ状態取得成功")
                logger.info(f"データ範囲: {status['data_range']}")
                logger.info(f"最新タイムスタンプ: {status['latest_timestamp']}")
                logger.info(f"現在時刻: {status['current_time']}")
                return True
            else:
                logger.error(f"❌ データ状態取得失敗: {status.get('error', 'Unknown error')}")
                return False
                
    except Exception as e:
        logger.error(f"データ状態確認テストエラー: {e}")
        return False


async def main():
    """メイン関数"""
    logger.info("🚀 Fear & Greed Index 実装テスト開始")
    logger.info("=" * 80)
    
    test_results = []
    
    # 各テストを実行
    tests = [
        ("データベースセットアップ", test_database_setup),
        ("APIサービス", test_api_service),
        ("データ収集", test_data_collection),
        ("リポジトリ操作", test_repository_operations),
        ("差分収集", test_incremental_collection),
        ("データ状態確認", test_data_status),
    ]
    
    for test_name, test_func in tests:
        try:
            logger.info(f"\n🔍 {test_name}テスト開始...")
            result = await test_func()
            test_results.append((test_name, result))
            
            if result:
                logger.info(f"✅ {test_name}テスト成功")
            else:
                logger.error(f"❌ {test_name}テスト失敗")
                
        except Exception as e:
            logger.error(f"❌ {test_name}テスト例外: {e}")
            test_results.append((test_name, False))
    
    # 結果サマリー
    logger.info("\n" + "=" * 80)
    logger.info("📊 テスト結果サマリー")
    logger.info("=" * 80)
    
    passed = 0
    failed = 0
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status} {test_name}")
        
        if result:
            passed += 1
        else:
            failed += 1
    
    logger.info(f"\n合計: {len(test_results)} テスト")
    logger.info(f"成功: {passed}")
    logger.info(f"失敗: {failed}")
    
    if failed == 0:
        logger.info("🎉 全てのテストが成功しました！")
        logger.info("Fear & Greed Index 実装は正常に動作しています。")
    else:
        logger.error(f"⚠️ {failed} 個のテストが失敗しました。")
        logger.error("実装に問題がある可能性があります。")
    
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
