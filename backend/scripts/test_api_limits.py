#!/usr/bin/env python3
"""
Alternative.me API制限テストスクリプト

Alternative.me APIの制限を調査し、最適な設定を決定します。
"""

import asyncio
import aiohttp
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


async def test_api_limits():
    """Alternative.me APIの制限をテスト"""
    logger.info("🔍 Alternative.me API制限調査開始")
    logger.info("=" * 60)
    
    api_url = "https://api.alternative.me/fng/"
    timeout = aiohttp.ClientTimeout(total=30)
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # 1. 基本的なAPIレスポンステスト
        logger.info("📡 基本APIレスポンステスト")
        try:
            async with session.get(api_url, params={"limit": 10, "format": "json"}) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"✅ 基本API成功: {response.status}")
                    logger.info(f"データ件数: {len(data.get('data', []))}")
                    
                    # レスポンス構造の確認
                    if 'metadata' in data:
                        logger.info(f"メタデータ: {data['metadata']}")
                else:
                    logger.error(f"❌ 基本API失敗: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"❌ 基本APIエラー: {e}")
            return False
        
        # 2. 様々なlimit値でのテスト
        logger.info("\n📊 limit値テスト")
        test_limits = [1, 10, 30, 50, 100, 200, 500, 1000, 2000]
        
        for limit in test_limits:
            try:
                async with session.get(api_url, params={"limit": limit, "format": "json"}) as response:
                    if response.status == 200:
                        data = await response.json()
                        actual_count = len(data.get('data', []))
                        logger.info(f"limit={limit:4d}: 実際取得={actual_count:4d}件 ({'✅' if actual_count > 0 else '❌'})")
                        
                        if actual_count < limit and actual_count > 0:
                            logger.info(f"  ⚠️ 最大取得可能件数: {actual_count}件")
                            break
                    else:
                        logger.error(f"limit={limit:4d}: エラー {response.status}")
                        
                # レート制限を避けるため少し待機
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"limit={limit:4d}: 例外 {e}")
        
        # 3. データの日付範囲確認
        logger.info("\n📅 データ範囲確認")
        try:
            async with session.get(api_url, params={"limit": 1000, "format": "json"}) as response:
                if response.status == 200:
                    data = await response.json()
                    fear_greed_data = data.get('data', [])
                    
                    if fear_greed_data:
                        # 最新と最古のデータ
                        newest = fear_greed_data[0]
                        oldest = fear_greed_data[-1]
                        
                        newest_date = datetime.fromtimestamp(int(newest['timestamp']), tz=timezone.utc)
                        oldest_date = datetime.fromtimestamp(int(oldest['timestamp']), tz=timezone.utc)
                        
                        date_range = (newest_date - oldest_date).days
                        
                        logger.info(f"最新データ: {newest_date.strftime('%Y-%m-%d')} (値: {newest['value']}, 分類: {newest['value_classification']})")
                        logger.info(f"最古データ: {oldest_date.strftime('%Y-%m-%d')} (値: {oldest['value']}, 分類: {oldest['value_classification']})")
                        logger.info(f"データ期間: {date_range}日間")
                        logger.info(f"データ密度: {len(fear_greed_data) / max(date_range, 1):.2f}件/日")
                        
                        # データの統計
                        values = [int(item['value']) for item in fear_greed_data]
                        logger.info(f"値の範囲: {min(values)} ～ {max(values)}")
                        logger.info(f"平均値: {sum(values) / len(values):.1f}")
                        
                        # 分類の分布
                        classifications = {}
                        for item in fear_greed_data:
                            cls = item['value_classification']
                            classifications[cls] = classifications.get(cls, 0) + 1
                        
                        logger.info("分類分布:")
                        for cls, count in sorted(classifications.items()):
                            percentage = (count / len(fear_greed_data)) * 100
                            logger.info(f"  {cls}: {count}件 ({percentage:.1f}%)")
                    
        except Exception as e:
            logger.error(f"❌ データ範囲確認エラー: {e}")
        
        # 4. レート制限テスト
        logger.info("\n⏱️ レート制限テスト")
        start_time = datetime.now()
        request_count = 0
        
        try:
            for i in range(10):  # 10回連続リクエスト
                async with session.get(api_url, params={"limit": 1, "format": "json"}) as response:
                    request_count += 1
                    if response.status != 200:
                        logger.warning(f"リクエスト{i+1}: ステータス {response.status}")
                    else:
                        logger.info(f"リクエスト{i+1}: 成功")
                
                # 短い間隔でリクエスト
                await asyncio.sleep(0.1)
                
        except Exception as e:
            logger.error(f"レート制限テストエラー: {e}")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"10回リクエスト完了: {duration:.2f}秒 (平均: {duration/request_count:.2f}秒/リクエスト)")
        
        # 5. 推奨設定の提案
        logger.info("\n" + "=" * 60)
        logger.info("📋 推奨設定")
        logger.info("=" * 60)
        logger.info("✅ 最大取得件数: 1000件程度")
        logger.info("✅ 通常取得件数: 30-100件")
        logger.info("✅ リクエスト間隔: 1秒以上")
        logger.info("✅ タイムアウト: 30秒")
        logger.info("✅ データ更新頻度: 1日1回")
        logger.info("✅ エラー時リトライ: 3回まで")
        
        return True


async def main():
    """メイン関数"""
    try:
        success = await test_api_limits()
        
        if success:
            logger.info("\n🎯 API制限調査完了")
            logger.info("上記の推奨設定を参考に、システムを最適化してください。")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ API制限調査中にエラーが発生しました: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
