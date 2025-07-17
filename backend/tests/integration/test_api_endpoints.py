#!/usr/bin/env python3
"""
Fear & Greed Index API エンドポイントテストスクリプト

実装したAPIエンドポイントの動作確認を行います。
"""

import asyncio
import aiohttp
import json
import logging
import sys
import os
from datetime import datetime, timezone

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class APITester:
    """APIテスタークラス"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_health_check(self):
        """ヘルスチェックエンドポイントのテスト"""
        logger.info("=== ヘルスチェックテスト ===")
        
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"✅ ヘルスチェック成功: {data}")
                    return True
                else:
                    logger.error(f"❌ ヘルスチェック失敗: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"❌ ヘルスチェック例外: {e}")
            return False
    
    async def test_fear_greed_status(self):
        """Fear & Greed Index データ状態取得テスト"""
        logger.info("=== データ状態取得テスト ===")
        
        try:
            async with self.session.get(f"{self.base_url}/api/fear-greed/status") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"✅ データ状態取得成功")
                    logger.info(f"レスポンス: {json.dumps(data, indent=2, ensure_ascii=False)}")
                    return True
                else:
                    text = await response.text()
                    logger.error(f"❌ データ状態取得失敗: {response.status} - {text}")
                    return False
        except Exception as e:
            logger.error(f"❌ データ状態取得例外: {e}")
            return False
    
    async def test_fear_greed_collect(self):
        """Fear & Greed Index データ収集テスト"""
        logger.info("=== データ収集テスト ===")
        
        try:
            async with self.session.post(f"{self.base_url}/api/fear-greed/collect?limit=5") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"✅ データ収集成功")
                    logger.info(f"レスポンス: {json.dumps(data, indent=2, ensure_ascii=False)}")
                    return True
                else:
                    text = await response.text()
                    logger.error(f"❌ データ収集失敗: {response.status} - {text}")
                    return False
        except Exception as e:
            logger.error(f"❌ データ収集例外: {e}")
            return False
    
    async def test_fear_greed_latest(self):
        """最新Fear & Greed Index データ取得テスト"""
        logger.info("=== 最新データ取得テスト ===")
        
        try:
            async with self.session.get(f"{self.base_url}/api/fear-greed/latest?limit=3") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"✅ 最新データ取得成功")
                    logger.info(f"データ件数: {len(data.get('data', []))}")
                    if data.get('data'):
                        latest = data['data'][0]
                        logger.info(f"最新データ例: value={latest.get('value')}, classification={latest.get('value_classification')}")
                    return True
                else:
                    text = await response.text()
                    logger.error(f"❌ 最新データ取得失敗: {response.status} - {text}")
                    return False
        except Exception as e:
            logger.error(f"❌ 最新データ取得例外: {e}")
            return False
    
    async def test_fear_greed_data(self):
        """Fear & Greed Index データ取得テスト"""
        logger.info("=== データ取得テスト ===")
        
        try:
            async with self.session.get(f"{self.base_url}/api/fear-greed/data?limit=5") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"✅ データ取得成功")
                    logger.info(f"データ件数: {len(data.get('data', []))}")
                    return True
                else:
                    text = await response.text()
                    logger.error(f"❌ データ取得失敗: {response.status} - {text}")
                    return False
        except Exception as e:
            logger.error(f"❌ データ取得例外: {e}")
            return False
    
    async def test_fear_greed_incremental(self):
        """Fear & Greed Index 差分収集テスト"""
        logger.info("=== 差分収集テスト ===")
        
        try:
            async with self.session.post(f"{self.base_url}/api/fear-greed/collect-incremental") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"✅ 差分収集成功")
                    logger.info(f"レスポンス: {json.dumps(data, indent=2, ensure_ascii=False)}")
                    return True
                else:
                    text = await response.text()
                    logger.error(f"❌ 差分収集失敗: {response.status} - {text}")
                    return False
        except Exception as e:
            logger.error(f"❌ 差分収集例外: {e}")
            return False


async def main():
    """メイン関数"""
    logger.info("🚀 Fear & Greed Index API エンドポイントテスト開始")
    logger.info("=" * 80)
    
    test_results = []
    
    async with APITester() as tester:
        # 各テストを実行
        tests = [
            ("ヘルスチェック", tester.test_health_check),
            ("データ状態取得", tester.test_fear_greed_status),
            ("データ収集", tester.test_fear_greed_collect),
            ("最新データ取得", tester.test_fear_greed_latest),
            ("データ取得", tester.test_fear_greed_data),
            ("差分収集", tester.test_fear_greed_incremental),
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
    logger.info("📊 APIテスト結果サマリー")
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
        logger.info("🎉 全てのAPIテストが成功しました！")
        logger.info("Fear & Greed Index APIエンドポイントは正常に動作しています。")
    else:
        logger.error(f"⚠️ {failed} 個のAPIテストが失敗しました。")
        logger.error("サーバーが起動していることを確認してください。")
    
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
