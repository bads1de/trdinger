#!/usr/bin/env python3
"""
OI（Open Interest）・FR（Funding Rate）データ収集スクリプト

既存のAPIエンドポイントを使用して、OIとFRデータを一括収集します。
"""

import argparse
import asyncio
import logging
import os
import sys


from app.services.data_collection.orchestration.data_collection_orchestration_service import (
    DataCollectionOrchestrationService,
)
from database.connection import SessionLocal

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def collect_oi_fr_data(symbol: str = "BTC/USDT:USDT"):
    """
    OI・FRデータを収集

    Args:
        symbol: 取引ペアシンボル
    """
    logger.info(f"OI・FRデータ収集開始: {symbol}")

    db = SessionLocal()
    try:
        service = DataCollectionOrchestrationService()

        # 差分更新を実行（OHLCV、FR、OIを一括で更新）
        logger.info("差分データ更新を実行中...")
        result = await service.execute_bulk_incremental_update(symbol, db)

        logger.info(f"収集結果: {result}")

        # 結果の詳細を表示
        if "ohlcv" in result:
            logger.info(f"OHLCV: {result['ohlcv']}")
        if "funding_rate" in result:
            logger.info(f"FR: {result['funding_rate']}")
        if "open_interest" in result:
            logger.info(f"OI: {result['open_interest']}")

        db.commit()
        logger.info("SUCCESS: データ収集完了")
        return True

    except Exception as e:
        logger.error(f"FAILED: データ収集失敗: {e}")
        import traceback

        logger.error(f"エラー詳細:\n{traceback.format_exc()}")
        db.rollback()
        return False
    finally:
        db.close()


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description="OI（Open Interest）・FR（Funding Rate）データ収集スクリプト",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        使用例:
        python collect_oi_fr_data.py
        python collect_oi_fr_data.py --symbol BTC/USDT:USDT
        """,
    )

    parser.add_argument(
        "-s",
        "--symbol",
        type=str,
        default="BTC/USDT:USDT",
        help="取引ペアシンボル (デフォルト: BTC/USDT:USDT)",
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="詳細なログ出力")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print("=" * 60)
    print("OI・FRデータ収集スクリプト")
    print(f"シンボル: {args.symbol}")
    print("=" * 60)

    try:
        success = asyncio.run(collect_oi_fr_data(args.symbol))
        if success:
            print("\n[SUCCESS] データ収集完了")
            print("\n次のステップ:")
            print("  - データ確認: python tests/scripts/check_oi_data.py")
            print("  - ML評価実行: python -m scripts.ml_optimization.run_ml_pipeline")
            return 0
        else:
            print("\n[FAILED] データ収集失敗")
            return 1
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] 処理が中断されました")
        return 130
    except Exception as e:
        print(f"\n[FATAL] 予期しないエラー: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
