#!/usr/bin/env python3
"""
外部市場データ実装テストスクリプト

実装した外部市場データ機能の包括的なテストを実行します。
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timezone

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connection import SessionLocal, init_db, test_connection
from database.repositories.external_market_repository import ExternalMarketRepository
from app.core.services.data_collection.external_market_service import ExternalMarketService
from data_collector.external_market_collector import ExternalMarketDataCollector

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_database_connection():
    """データベース接続テスト"""
    logger.info("=== データベース接続テスト ===")
    
    try:
        # データベース接続テスト
        if test_connection():
            logger.info("✓ データベース接続成功")
        else:
            logger.error("✗ データベース接続失敗")
            return False
            
        # データベース初期化
        init_db()
        logger.info("✓ データベース初期化完了")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ データベース接続テストエラー: {e}")
        return False


async def test_external_market_service():
    """外部市場データサービステスト"""
    logger.info("=== 外部市場データサービステスト ===")
    
    try:
        async with ExternalMarketService() as service:
            # 利用可能なシンボルの確認
            symbols = service.get_available_symbols()
            logger.info(f"✓ 利用可能なシンボル: {symbols}")
            
            # 最新データの取得テスト（少量）
            logger.info("最新データ取得テスト中...")
            latest_data = await service.fetch_latest_data(symbols=["^GSPC"])  # S&P500のみテスト
            
            if latest_data:
                logger.info(f"✓ 最新データ取得成功: {len(latest_data)} 件")
                logger.info(f"サンプルデータ: {latest_data[0]}")
            else:
                logger.warning("⚠ 最新データが空です")
            
            return True
            
    except Exception as e:
        logger.error(f"✗ 外部市場データサービステストエラー: {e}")
        return False


async def test_external_market_repository():
    """外部市場データリポジトリテスト"""
    logger.info("=== 外部市場データリポジトリテスト ===")
    
    try:
        session = SessionLocal()
        repository = ExternalMarketRepository(session)
        
        # データ統計の取得
        statistics = repository.get_data_statistics()
        logger.info(f"✓ データ統計: {statistics}")
        
        # シンボル一覧の取得
        symbols = repository.get_symbols()
        logger.info(f"✓ データベース内シンボル: {symbols}")
        
        # 最新データタイムスタンプの取得
        latest_timestamp = repository.get_latest_data_timestamp()
        logger.info(f"✓ 最新データタイムスタンプ: {latest_timestamp}")
        
        session.close()
        return True
        
    except Exception as e:
        logger.error(f"✗ 外部市場データリポジトリテストエラー: {e}")
        return False


async def test_data_collection():
    """データ収集テスト"""
    logger.info("=== データ収集テスト ===")
    
    try:
        async with ExternalMarketDataCollector() as collector:
            # データ状態の確認
            status = await collector.get_external_market_data_status()
            logger.info(f"✓ データ状態: {status}")
            
            # 少量のデータ収集テスト
            logger.info("少量データ収集テスト中...")
            result = await collector.collect_external_market_data(
                symbols=["^GSPC"],  # S&P500のみ
                period="5d"  # 5日分のみ
            )
            
            if result["success"]:
                logger.info(f"✓ データ収集成功: {result}")
            else:
                logger.error(f"✗ データ収集失敗: {result}")
                return False
            
            return True
            
    except Exception as e:
        logger.error(f"✗ データ収集テストエラー: {e}")
        return False


async def test_data_validation():
    """データ検証テスト"""
    logger.info("=== データ検証テスト ===")
    
    try:
        from app.core.utils.data_converter import DataValidator
        
        # 有効なテストデータ
        valid_data = [
            {
                "symbol": "^GSPC",
                "open": 4500.0,
                "high": 4550.0,
                "low": 4480.0,
                "close": 4520.0,
                "volume": 1000000,
                "data_timestamp": datetime.now(timezone.utc),
                "timestamp": datetime.now(timezone.utc),
            }
        ]
        
        # 無効なテストデータ
        invalid_data = [
            {
                "symbol": "^GSPC",
                "open": -100.0,  # 負の値（無効）
                "high": 4550.0,
                "low": 4480.0,
                "close": 4520.0,
                "volume": 1000000,
                "data_timestamp": datetime.now(timezone.utc),
                "timestamp": datetime.now(timezone.utc),
            }
        ]
        
        # 有効データの検証
        if DataValidator.validate_external_market_data(valid_data):
            logger.info("✓ 有効データの検証成功")
        else:
            logger.error("✗ 有効データの検証失敗")
            return False
        
        # 無効データの検証
        if not DataValidator.validate_external_market_data(invalid_data):
            logger.info("✓ 無効データの検証成功（正しく無効と判定）")
        else:
            logger.error("✗ 無効データの検証失敗（無効データが有効と判定された）")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"✗ データ検証テストエラー: {e}")
        return False


async def test_incremental_collection():
    """差分収集テスト"""
    logger.info("=== 差分収集テスト ===")
    
    try:
        async with ExternalMarketDataCollector() as collector:
            # 差分収集テスト
            logger.info("差分収集テスト中...")
            result = await collector.collect_incremental_external_market_data(
                symbols=["^GSPC"]  # S&P500のみ
            )
            
            if result["success"]:
                logger.info(f"✓ 差分収集成功: {result}")
            else:
                logger.error(f"✗ 差分収集失敗: {result}")
                return False
            
            return True
            
    except Exception as e:
        logger.error(f"✗ 差分収集テストエラー: {e}")
        return False


async def run_all_tests():
    """全テストを実行"""
    logger.info("外部市場データ実装テストを開始します...")
    
    tests = [
        ("データベース接続", test_database_connection),
        ("外部市場データサービス", test_external_market_service),
        ("外部市場データリポジトリ", test_external_market_repository),
        ("データ検証", test_data_validation),
        ("データ収集", test_data_collection),
        ("差分収集", test_incremental_collection),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name}テスト開始 ---")
        try:
            result = await test_func()
            results[test_name] = result
            if result:
                logger.info(f"✓ {test_name}テスト成功")
            else:
                logger.error(f"✗ {test_name}テスト失敗")
        except Exception as e:
            logger.error(f"✗ {test_name}テスト例外: {e}")
            results[test_name] = False
    
    # 結果サマリー
    logger.info("\n=== テスト結果サマリー ===")
    success_count = sum(1 for result in results.values() if result)
    total_count = len(results)
    
    for test_name, result in results.items():
        status = "✓ 成功" if result else "✗ 失敗"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\n総合結果: {success_count}/{total_count} テスト成功")
    
    if success_count == total_count:
        logger.info("🎉 全テスト成功！外部市場データ機能は正常に動作しています。")
        return True
    else:
        logger.error("❌ 一部テストが失敗しました。問題を修正してください。")
        return False


if __name__ == "__main__":
    asyncio.run(run_all_tests())
