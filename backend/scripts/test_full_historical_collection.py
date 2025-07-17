#!/usr/bin/env python3
"""
Fear & Greed Index 全期間データ収集テストスクリプト

全期間の履歴データ収集機能をテストし、データの整合性を確認します。
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timezone

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connection import SessionLocal, init_db
from database.repositories.fear_greed_repository import FearGreedIndexRepository
from data_collector.external_market_collector import ExternalMarketDataCollector

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_full_historical_collection():
    """全期間データ収集のテスト"""
    logger.info("🚀 Fear & Greed Index 全期間データ収集テスト開始")
    logger.info("=" * 80)
    
    # データベース初期化
    init_db()
    
    with SessionLocal() as db:
        repository = FearGreedIndexRepository(db)
        
        # 収集前の状態確認
        logger.info("📊 収集前のデータ状態確認")
        before_status = repository.get_data_range()
        logger.info(f"収集前データ件数: {before_status['total_count']}")
        logger.info(f"収集前データ範囲: {before_status['oldest_data']} ～ {before_status['newest_data']}")
        
        # 全期間データ収集実行
        logger.info("\n📥 全期間データ収集実行中...")
        async with ExternalMarketDataCollector() as collector:
            result = await collector.collect_historical_fear_greed_data(
                limit=1000,  # 最大1000件
                db_session=db
            )
        
        if not result["success"]:
            logger.error(f"❌ 全期間データ収集失敗: {result.get('error', 'Unknown error')}")
            return False
        
        logger.info(f"✅ 全期間データ収集成功!")
        logger.info(f"取得件数: {result['fetched_count']}")
        logger.info(f"挿入件数: {result['inserted_count']}")
        logger.info(f"収集タイプ: {result.get('collection_type', 'unknown')}")
        
        # 収集後の状態確認
        logger.info("\n📊 収集後のデータ状態確認")
        after_status = repository.get_data_range()
        logger.info(f"収集後データ件数: {after_status['total_count']}")
        logger.info(f"収集後データ範囲: {after_status['oldest_data']} ～ {after_status['newest_data']}")
        
        # データ増加量の確認
        data_increase = after_status['total_count'] - before_status['total_count']
        logger.info(f"データ増加量: {data_increase}件")
        
        # 最新データの詳細確認
        logger.info("\n📈 最新データの詳細確認")
        latest_data = repository.get_latest_fear_greed_data(limit=5)
        
        if latest_data:
            logger.info(f"最新データ件数: {len(latest_data)}")
            for i, data in enumerate(latest_data[:3]):  # 最新3件を表示
                logger.info(f"  {i+1}. 日付: {data.data_timestamp.strftime('%Y-%m-%d')}, "
                           f"値: {data.value}, 分類: {data.value_classification}")
        else:
            logger.warning("⚠️ 最新データが取得できませんでした")
        
        # データ整合性チェック
        logger.info("\n🔍 データ整合性チェック")
        
        # 1. 値の範囲チェック
        invalid_values = db.query(repository.model_class).filter(
            (repository.model_class.value < 0) | (repository.model_class.value > 100)
        ).count()
        
        if invalid_values > 0:
            logger.error(f"❌ 無効な値のデータが {invalid_values} 件見つかりました")
            return False
        else:
            logger.info("✅ 全ての値が有効範囲（0-100）内です")
        
        # 2. 分類の妥当性チェック
        valid_classifications = [
            "Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"
        ]
        invalid_classifications = db.query(repository.model_class).filter(
            ~repository.model_class.value_classification.in_(valid_classifications)
        ).count()
        
        if invalid_classifications > 0:
            logger.error(f"❌ 無効な分類のデータが {invalid_classifications} 件見つかりました")
            return False
        else:
            logger.info("✅ 全ての分類が有効です")
        
        # 3. 重複データチェック
        from sqlalchemy import func
        duplicate_count = db.query(
            repository.model_class.data_timestamp,
            func.count(repository.model_class.id).label('count')
        ).group_by(
            repository.model_class.data_timestamp
        ).having(
            func.count(repository.model_class.id) > 1
        ).count()
        
        if duplicate_count > 0:
            logger.warning(f"⚠️ 重複データが {duplicate_count} 件見つかりました")
        else:
            logger.info("✅ 重複データはありません")
        
        # 4. データ連続性チェック（簡易版）
        if after_status['total_count'] >= 2:
            # 最古と最新の日付差を計算
            try:
                oldest = datetime.fromisoformat(after_status['oldest_data'].replace('Z', '+00:00'))
                newest = datetime.fromisoformat(after_status['newest_data'].replace('Z', '+00:00'))
                date_diff = (newest - oldest).days
                
                logger.info(f"データ期間: {date_diff}日間")
                logger.info(f"データ密度: {after_status['total_count'] / max(date_diff, 1):.2f}件/日")
                
                # 期待される最小データ数（1日1件として）
                expected_min_count = max(date_diff * 0.8, 1)  # 80%のカバレッジを期待
                
                if after_status['total_count'] >= expected_min_count:
                    logger.info("✅ データ密度は適切です")
                else:
                    logger.warning(f"⚠️ データ密度が低い可能性があります（期待値: {expected_min_count:.0f}件以上）")
                    
            except Exception as e:
                logger.warning(f"⚠️ データ連続性チェック中にエラー: {e}")
        
        # 5. 最新データの新しさチェック
        if latest_data:
            latest_timestamp = latest_data[0].data_timestamp
            now = datetime.now(timezone.utc)
            hours_diff = (now - latest_timestamp.replace(tzinfo=timezone.utc)).total_seconds() / 3600
            
            if hours_diff <= 48:  # 48時間以内
                logger.info(f"✅ 最新データは新しいです（{hours_diff:.1f}時間前）")
            else:
                logger.warning(f"⚠️ 最新データが古い可能性があります（{hours_diff:.1f}時間前）")
        
        # テスト結果サマリー
        logger.info("\n" + "=" * 80)
        logger.info("📊 全期間データ収集テスト結果サマリー")
        logger.info("=" * 80)
        logger.info(f"✅ 収集成功: {result['success']}")
        logger.info(f"📥 取得件数: {result['fetched_count']}")
        logger.info(f"💾 挿入件数: {result['inserted_count']}")
        logger.info(f"📈 総データ件数: {after_status['total_count']}")
        logger.info(f"📅 データ範囲: {after_status['oldest_data']} ～ {after_status['newest_data']}")
        logger.info(f"🔢 データ増加: +{data_increase}件")
        
        if after_status['total_count'] > 0:
            logger.info("🎉 全期間データ収集テストが成功しました！")
            logger.info("Fear & Greed Index データが正常に収集・保存されています。")
            return True
        else:
            logger.error("❌ データが収集されませんでした。")
            return False


async def main():
    """メイン関数"""
    try:
        success = await test_full_historical_collection()
        
        if success:
            logger.info("\n🎯 次のステップ:")
            logger.info("1. フロントエンドでデータ表示を確認")
            logger.info("2. 定期的な差分収集の設定")
            logger.info("3. データ可視化機能の実装")
            logger.info("4. 機械学習モデルへの統合")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ テスト実行中にエラーが発生しました: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
