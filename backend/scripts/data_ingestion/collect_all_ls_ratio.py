"""
全通貨ペアのLong/Short Ratioを一括収集するスクリプト

BybitのUSDT無期限契約（Linear Perpetual）の全ペアを対象に、
Long/Short Ratioの履歴データを取得・保存します。

使用方法:
    python backend/scripts/data_ingestion/collect_all_ls_ratio.py [period]

    period: 収集する時間足 (default: 1h)
            対応値: 5min, 15min, 30min, 1h, 4h, 1d
"""

import asyncio
import logging
import sys
import os
from datetime import datetime

# プロジェクトルートへのパスを追加してモジュールをインポート可能にする
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import ccxt.async_support as ccxt
from app.services.data_collection.bybit.long_short_ratio_service import BybitLongShortRatioService
from database.repositories.long_short_ratio_repository import LongShortRatioRepository
from database.connection import get_db

# ロガー設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("collect_ls_ratio.log")
    ]
)
logger = logging.getLogger(__name__)

async def get_linear_symbols():
    """Bybitから有効なUSDT無期限契約のシンボルリストを取得（BTC/USDT:USDTのみ）"""
    logger.info("対象シンボルをフィルタリング中 (BTC/USDT:USDTのみ)...")
    return ["BTC/USDT:USDT"]

async def collect_symbol_data(service, repository, symbol, period):
    """1つのシンボルのデータを収集"""
    try:
        logger.info(f"[{symbol}] 収集開始 (period: {period})")
        
        # 履歴データの収集（start_dateを指定しないことで、可能な限り過去から取得）
        # ※ サービスの collect_historical_long_short_ratio_data は内部でページネーションを行う
        saved_count = await service.collect_historical_long_short_ratio_data(
            symbol=symbol,
            period=period,
            repository=repository
        )
        
        logger.info(f"[{symbol}] 収集完了: {saved_count}件 保存")
        return saved_count
        
    except Exception as e:
        logger.error(f"[{symbol}] エラー発生: {e}")
        return 0

async def main():
    # 引数から期間を取得 (デフォルト: 1h)
    period = sys.argv[1] if len(sys.argv) > 1 else "1h"
    allowed_periods = ["5min", "15min", "30min", "1h", "4h", "1d"]
    
    if period not in allowed_periods:
        logger.error(f"無効な期間です: {period}. 指定可能: {allowed_periods}")
        return

    logger.info(f"=== 全ペア Long/Short Ratio 収集開始 (Period: {period}) ===")

    # 1. シンボルリストの取得
    logger.info("市場データを取得中...")
    symbols = await get_linear_symbols()
    logger.info(f"対象通貨ペア数: {len(symbols)}")

    # 2. サービスとリポジトリの初期化
    # DBセッションの取得
    db_gen = get_db()
    db = next(db_gen)
    
    try:
        repository = LongShortRatioRepository(db)
        # サービス内でCCXTインスタンスを生成
        service = BybitLongShortRatioService() 
        
        total_saved = 0
        
        # 3. 順次収集実行
        # 並列実行しすぎるとAPIレート制限にかかるため、逐次または少数の並列で実行
        # ここでは安全のため逐次実行します
        for i, symbol in enumerate(symbols):
            logger.info(f"--- 処理中 ({i+1}/{len(symbols)}): {symbol} ---")
            
            count = await collect_symbol_data(service, repository, symbol, period)
            total_saved += count
            
            # レート制限への配慮（少し待機）
            await asyncio.sleep(0.5)

        logger.info(f"=== 全処理完了 ===")
        logger.info(f"総保存件数: {total_saved}")

    except Exception as e:
        logger.error(f"予期せぬエラー: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("処理が中断されました")
