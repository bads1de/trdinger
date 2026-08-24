"""
全通貨ペアのLong/Short Ratioを一括収集するスクリプト

BybitのUSDT無期限契約（Linear Perpetual）の全ペアを対象に、
Long/Short Ratioの履歴データを取得・保存します。

使用方法:
    python backend/scripts/data_ingestion/collect_all_ls_ratio.py [period] [start_date]

    period: 収集する時間足 (default: 1h)
            対応値: 5min, 15min, 30min, 1h, 4h, 1d
    start_date: 収集開始日 (default: なし = 2020-10-01から、YYYY-MM-DD)
"""

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timezone

# プロジェクトルートへのパスを追加してモジュールをインポート可能にする
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from app.services.data_collection.bybit.long_short_ratio_service import (
    BybitLongShortRatioService,
)
from database.connection import get_db
from database.repositories.long_short_ratio_repository import LongShortRatioRepository

# ロガー設定（プロジェクトルートの logs/ 配下へ出力）
_log_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "logs",
)
os.makedirs(_log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(_log_dir, "collect_ls_ratio.log")),
    ],
)
logger = logging.getLogger(__name__)


async def get_linear_symbols(symbol_filter: str | None = None):
    """収集対象シンボルのリストを返す（既定: BTC/USDT:USDT）。

    symbol_filter はカンマ区切りで複数指定可能（例: "BTC/USDT:USDT,ETH/USDT:USDT"）。
    指定シンボルが設定の supported_symbols ホワイトリストに含まれるか検証する。
    """
    from app.config.unified_config import unified_config

    if symbol_filter:
        requested = [s.strip() for s in symbol_filter.split(",") if s.strip()]
    else:
        requested = ["BTC/USDT:USDT"]

    supported = unified_config.market.supported_symbols
    unknown = [s for s in requested if s not in supported]
    if unknown:
        raise ValueError(f"未対応のシンボル: {unknown} (対応: {', '.join(supported)})")
    logger.info(f"対象シンボル: {requested}")
    return requested


async def collect_symbol_data(service, repository, symbol, period, start_date=None):
    """1つのシンボルのデータを収集"""
    try:
        logger.info(f"[{symbol}] 収集開始 (period: {period}, start_date: {start_date})")

        # 履歴データの収集（start_date を指定しないことで、可能な限り過去から取得）
        # ※ サービスの collect_historical_long_short_ratio_data は内部でページネーションを行う
        saved_count = await service.collect_historical_long_short_ratio_data(
            symbol=symbol,
            period=period,
            repository=repository,
            start_date=start_date,
        )

        logger.info(f"[{symbol}] 収集完了: {saved_count}件 保存")
        return saved_count

    except Exception as e:
        logger.error(f"[{symbol}] エラー発生: {e}")
        return 0


async def main():
    parser = argparse.ArgumentParser(
        description="Long/Short Ratio 履歴データ収集スクリプト",
        epilog=(
            "使用例: python collect_all_ls_ratio.py --period 4h "
            "--start-date 2020-10-01 --symbols BTC/USDT:USDT,ETH/USDT:USDT"
        ),
    )
    parser.add_argument(
        "-p",
        "--period",
        default="1h",
        help="収集する時間足 (5min, 15min, 30min, 1h, 4h, 1d)",
    )
    parser.add_argument(
        "-s",
        "--start-date",
        default=None,
        help="収集開始日 YYYY-MM-DD（未指定時は 2020-10-01）",
    )
    parser.add_argument(
        "--symbols",
        default=None,
        help="対象シンボル（カンマ区切り・未指定時は BTC/USDT:USDT）",
    )
    args = parser.parse_args()

    period = args.period
    allowed_periods = ["5min", "15min", "30min", "1h", "4h", "1d"]
    if period not in allowed_periods:
        logger.error(f"無効な期間です: {period}. 指定可能: {allowed_periods}")
        return

    start_date = None
    if args.start_date:
        try:
            start_date = datetime.strptime(args.start_date, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            logger.error(f"無効な開始日です: {args.start_date} (YYYY-MM-DD 形式)")
            return

    symbol_filter = args.symbols

    logger.info(
        f"=== Long/Short Ratio 収集開始 (Period: {period}, "
        f"Start: {start_date or '2020-10-01'}, Symbol: {symbol_filter or 'BTC'}) ==="
    )

    # 1. シンボルリストの取得
    try:
        symbols = await get_linear_symbols(symbol_filter)
    except ValueError as e:
        logger.error(str(e))
        return
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
            logger.info(f"--- 処理中 ({i + 1}/{len(symbols)}): {symbol} ---")

            count = await collect_symbol_data(
                service, repository, symbol, period, start_date=start_date
            )
            total_saved += count

            # レート制限への配慮（少し待機）
            await asyncio.sleep(0.5)

        logger.info("=== 全処理完了 ===")
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
