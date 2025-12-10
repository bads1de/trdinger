#!/usr/bin/env python3
"""
全時間足OHLCVデータ全期間収集スクリプト (カスタムロジック版)
サポートされている全ての時間足について、DBに存在する最古のデータよりさらに過去へ遡って、
APIで取得可能な全期間のデータを取得します。
"""

import asyncio
import logging
import os
import sys
import traceback
from datetime import datetime, timezone, timedelta

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# ロギング設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

async def collect_all_historical_data(symbol: str = "BTC/USDT:USDT"):
    logger.info(f"全時間足データ全期間収集開始 (過去方向へスクロール): {symbol}")
    
    try:
        from app.services.data_collection.historical.historical_data_service import (
            HistoricalDataService,
        )
        from app.config.unified_config import unified_config
        from database.connection import SessionLocal
        from database.repositories.ohlcv_repository import OHLCVRepository
        from app.services.data_collection.bybit.market_data_service import BybitMarketDataService
        
        db = SessionLocal()
        
        # 1. 1m足については過去データ取得を制限（データ量が多すぎるため）
        #    過去1年間分くらいにとどめるのが現実的
        ONE_YEAR_MS = 365 * 24 * 60 * 60 * 1000
        
        try:
            repository = OHLCVRepository(db)
            market_service = BybitMarketDataService()
            
            # サポートされている全ての時間足を取得
            timeframes = unified_config.market.supported_timeframes
            logger.info(f"対象時間足: {timeframes}")
            
            total_saved_all = 0
            results = {}
            
            for tf in timeframes:
                try:
                    logger.info("-" * 40)
                    logger.info(f"時間足 {tf} の処理を開始")
                    
                    saved_count_tf = 0
                    
                    # この時間足の最大ページ数（設定またはデフォルト）
                    # 1分足で6年分取得する場合: 6 * 365 * 24 * 60 = 3,153,600件 / 1000 = 3154ページ
                    # 余裕を持って10000ページに設定
                    max_pages = 10000
                    
                    # 1m足の場合の警告ログ（制限は撤廃）
                    if tf == "1m":
                         logger.info("1m足の全期間データ取得を開始します（数分かかる場合があります）")
                    
                    # DBから最古のタイムスタンプを取得
                    oldest_ts = repository.get_oldest_timestamp(
                        timestamp_column="timestamp",
                        filter_conditions={"symbol": symbol, "timeframe": tf}
                    )
                    
                    # 次のリクエストの終了時間（end）を決定
                    # Bybit API の 'end' は指定されたタイムスタンプ *以前* のデータを返す
                    if oldest_ts:
                        # DBにある最古データの1ミリ秒前
                        end_ms = int(oldest_ts.timestamp() * 1000) - 1
                        logger.info(f"  既存データあり。最古: {oldest_ts}。続きから過去へ取得します (end_ms={end_ms})")
                    else:
                        # データがない場合は現在時刻から
                        end_ms = None
                        logger.info("  既存データなし。現在時刻から過去へ取得します")

                    limit = 1000 # 最大取得数
                    
                    for i in range(max_pages):
                        params = {}
                        if end_ms:
                            params["end"] = end_ms
                            
                        # データ取得
                        # fetch_ohlcv_data は [timestamp, open, high, low, close, volume] のリストを返す
                        # デフォルトでタイムスタンプ昇順
                        ohlcv_data = await market_service.fetch_ohlcv_data(
                            symbol, tf, limit=limit, params=params
                        )
                        
                        if not ohlcv_data:
                            logger.info(f"  {tf}: データ取得終了 (これ以上過去のデータはありません)")
                            break
                            
                        # DBに保存
                        saved = await market_service._save_ohlcv_to_database(
                            ohlcv_data, symbol, tf, repository
                        )
                        saved_count_tf += saved
                        
                        # 次の end_ms を更新（今回取得したデータの中で最も古いもの - 1ms）
                        # ohlcv_data は昇順なので、最初の要素が最も古い
                        oldest_in_batch = ohlcv_data[0][0]
                        first_dt = datetime.fromtimestamp(oldest_in_batch/1000, tz=timezone.utc)
                        last_dt = datetime.fromtimestamp(ohlcv_data[-1][0]/1000, tz=timezone.utc)
                        
                        logger.info(f"  {tf} バッチ {i+1}/{max_pages}: {len(ohlcv_data)}件取得 ({first_dt} ~ {last_dt}) -> {saved}件保存")
                        
                        end_ms = oldest_in_batch - 1
                        
                        # APIレート制限のための待機
                        await asyncio.sleep(0.1) # 必要に応じて調整
                        
                    logger.info(f"  {tf} 完了: 合計 {saved_count_tf}件 追加保存")
                    results[tf] = {"status": "success", "saved": saved_count_tf}
                    total_saved_all += saved_count_tf
                    
                except Exception as e:
                    logger.error(f"  {tf} の処理中にエラー: {e}")
                    results[tf] = {"status": "error", "error": str(e)}
                    # 次の時間足へ進む
                    
            logger.info("=" * 60)
            logger.info(f"全処理完了。総追加保存件数: {total_saved_all}")
            
            db.commit()
            return True
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"致命的なエラー: {e}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(collect_all_historical_data())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n処理が中断されました")
        sys.exit(130)
