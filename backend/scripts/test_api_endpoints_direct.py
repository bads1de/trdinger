#!/usr/bin/env python3
"""
Fear & Greed Index API エンドポイント直接テストスクリプト

修正したAPIエンドポイントの動作確認を行います。
"""

import asyncio
import aiohttp
import json
import logging
import sys
import os

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_endpoints():
    """APIエンドポイントをテスト"""
    logger.info("🚀 Fear & Greed Index API エンドポイント直接テスト開始")
    logger.info("=" * 80)
    
    base_url = "http://localhost:8000"
    timeout = aiohttp.ClientTimeout(total=30)
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        
        # 1. ヘルスチェック
        logger.info("📡 ヘルスチェック")
        try:
            async with session.get(f"{base_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"✅ ヘルスチェック成功: {data}")
                else:
                    logger.error(f"❌ ヘルスチェック失敗: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"❌ ヘルスチェック例外: {e}")
            return False
        
        # 2. Fear & Greed Index データ状態取得
        logger.info("\n📊 データ状態取得テスト")
        try:
            async with session.get(f"{base_url}/api/fear-greed/status") as response:
                logger.info(f"ステータスコード: {response.status}")
                text = await response.text()
                logger.info(f"レスポンステキスト: {text[:500]}...")
                
                if response.status == 200:
                    data = json.loads(text)
                    logger.info(f"✅ データ状態取得成功")
                    logger.info(f"レスポンス構造: {list(data.keys())}")
                    if 'data' in data:
                        logger.info(f"データ内容: {data['data']}")
                else:
                    logger.error(f"❌ データ状態取得失敗: {response.status}")
                    logger.error(f"エラー内容: {text}")
                    
        except Exception as e:
            logger.error(f"❌ データ状態取得例外: {e}")
        
        # 3. Fear & Greed Index 最新データ取得
        logger.info("\n📈 最新データ取得テスト")
        try:
            async with session.get(f"{base_url}/api/fear-greed/latest?limit=5") as response:
                logger.info(f"ステータスコード: {response.status}")
                text = await response.text()
                logger.info(f"レスポンステキスト: {text[:500]}...")
                
                if response.status == 200:
                    data = json.loads(text)
                    logger.info(f"✅ 最新データ取得成功")
                    logger.info(f"レスポンス構造: {list(data.keys())}")
                    if 'data' in data:
                        if isinstance(data['data'], dict) and 'data' in data['data']:
                            actual_data = data['data']['data']
                            logger.info(f"データ件数: {len(actual_data)}")
                            if actual_data:
                                logger.info(f"最新データ例: {actual_data[0]}")
                        else:
                            logger.info(f"データ: {data['data']}")
                else:
                    logger.error(f"❌ 最新データ取得失敗: {response.status}")
                    logger.error(f"エラー内容: {text}")
                    
        except Exception as e:
            logger.error(f"❌ 最新データ取得例外: {e}")
        
        # 4. Fear & Greed Index データ取得
        logger.info("\n📋 データ取得テスト")
        try:
            async with session.get(f"{base_url}/api/fear-greed/data?limit=3") as response:
                logger.info(f"ステータスコード: {response.status}")
                text = await response.text()
                logger.info(f"レスポンステキスト: {text[:500]}...")
                
                if response.status == 200:
                    data = json.loads(text)
                    logger.info(f"✅ データ取得成功")
                    logger.info(f"レスポンス構造: {list(data.keys())}")
                else:
                    logger.error(f"❌ データ取得失敗: {response.status}")
                    logger.error(f"エラー内容: {text}")
                    
        except Exception as e:
            logger.error(f"❌ データ取得例外: {e}")
        
        # 5. Fear & Greed Index データ収集
        logger.info("\n📥 データ収集テスト")
        try:
            async with session.post(f"{base_url}/api/fear-greed/collect?limit=5") as response:
                logger.info(f"ステータスコード: {response.status}")
                text = await response.text()
                logger.info(f"レスポンステキスト: {text[:500]}...")
                
                if response.status == 200:
                    data = json.loads(text)
                    logger.info(f"✅ データ収集成功")
                    logger.info(f"レスポンス構造: {list(data.keys())}")
                else:
                    logger.error(f"❌ データ収集失敗: {response.status}")
                    logger.error(f"エラー内容: {text}")
                    
        except Exception as e:
            logger.error(f"❌ データ収集例外: {e}")
        
        logger.info("\n" + "=" * 80)
        logger.info("📊 API エンドポイント直接テスト完了")
        logger.info("=" * 80)
        
        return True


async def main():
    """メイン関数"""
    try:
        success = await test_endpoints()
        return success
        
    except Exception as e:
        logger.error(f"❌ テスト実行中にエラーが発生しました: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
