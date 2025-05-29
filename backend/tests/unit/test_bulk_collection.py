#!/usr/bin/env python3
"""
一括収集機能のテストスクリプト

修正されたコードが正しく動作するかを確認します。
"""

import sys
import asyncio
from datetime import datetime, timezone

sys.path.append('.')

from database.connection import SessionLocal
from database.repositories.ohlcv_repository import OHLCVRepository
from app.config.market_config import MarketDataConfig


async def test_bulk_collection_logic():
    """修正された一括収集ロジックをテスト"""
    
    # サポートされている取引ペアと時間軸
    symbols = [
        "BTC/USDT", "BTC/USDT:USDT", "BTCUSD",
        "ETH/USDT", "ETH/BTC", "ETH/USDT:USDT", "ETHUSD",
        "XRP/USDT", "XRP/USDT:USDT",
        "BNB/USDT", "BNB/USDT:USDT",
        "SOL/USDT", "SOL/USDT:USDT"
    ]
    timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]

    total_combinations = len(symbols) * len(timeframes)
    started_at = datetime.now(timezone.utc).isoformat()
    
    db = SessionLocal()
    try:
        repository = OHLCVRepository(db)
        
        # 既存データをチェックして、実際に収集が必要なタスクを特定
        tasks_to_execute = []
        skipped_tasks = []
        failed_tasks = []

        print(f"一括データ収集開始: {len(symbols)}個のシンボル × {len(timeframes)}個の時間軸 = {total_combinations}組み合わせを確認")

        for symbol in symbols:
            for timeframe in timeframes:
                try:
                    # シンボルの正規化
                    normalized_symbol = MarketDataConfig.normalize_symbol(symbol)
                    
                    # 既存データをチェック
                    data_exists = repository.get_data_count(normalized_symbol, timeframe) > 0
                    
                    if data_exists:
                        skipped_tasks.append({
                            "symbol": normalized_symbol,
                            "original_symbol": symbol,
                            "timeframe": timeframe,
                            "reason": "data_exists"
                        })
                        print(f"スキップ: {normalized_symbol} {timeframe} - データが既に存在")
                    else:
                        tasks_to_execute.append({
                            "symbol": normalized_symbol,
                            "original_symbol": symbol,
                            "timeframe": timeframe
                        })
                        print(f"タスク追加: {normalized_symbol} {timeframe}")

                except Exception as task_error:
                    print(f"タスク処理エラー {symbol} {timeframe}: {task_error}")
                    failed_tasks.append({
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "error": str(task_error)
                    })
                    continue

        actual_tasks = len(tasks_to_execute)
        skipped_count = len(skipped_tasks)
        failed_count = len(failed_tasks)

        print(f"\n一括データ収集タスク分析完了:")
        print(f"  - 総組み合わせ数: {total_combinations}")
        print(f"  - 実行タスク数: {actual_tasks}")
        print(f"  - スキップ数: {skipped_count} (既存データ)")
        print(f"  - 失敗数: {failed_count}")

        result = {
            "success": True,
            "message": f"一括データ収集を開始しました（{actual_tasks}タスク実行、{skipped_count}タスクスキップ）",
            "status": "started",
            "total_combinations": total_combinations,
            "actual_tasks": actual_tasks,
            "skipped_tasks": skipped_count,
            "failed_tasks": failed_count,
            "started_at": started_at,
            "symbols": symbols,
            "timeframes": timeframes,
            "task_details": {
                "executing": tasks_to_execute,
                "skipped": skipped_tasks,
                "failed": failed_tasks
            }
        }
        
        print(f"\n期待されるAPIレスポンス:")
        print(f"  success: {result['success']}")
        print(f"  message: {result['message']}")
        print(f"  total_combinations: {result['total_combinations']}")
        print(f"  actual_tasks: {result['actual_tasks']}")
        print(f"  skipped_tasks: {result['skipped_tasks']}")
        
        return result
        
    finally:
        db.close()


if __name__ == "__main__":
    asyncio.run(test_bulk_collection_logic())
