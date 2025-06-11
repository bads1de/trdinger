#!/usr/bin/env python3
"""
OHLCVデータクリアスクリプト

既存のOHLCVデータを削除し、新しいBTC/USDT:USDT無期限先物データの収集に備えます。
"""

import logging
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connection import SessionLocal, init_db
from database.repositories.ohlcv_repository import OHLCVRepository

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def clear_all_ohlcv_data() -> int:
    """
    全てのOHLCVデータを削除
    
    Returns:
        削除された件数
    """
    db = SessionLocal()
    try:
        repo = OHLCVRepository(db)
        
        # 削除前の件数を確認
        from database.models import OHLCVData
        count_before = db.query(OHLCVData).count()
        logger.info(f"削除前のOHLCVデータ件数: {count_before}")
        
        if count_before == 0:
            logger.info("削除対象のデータがありません")
            return 0
        
        # ユーザー確認
        response = input(f"本当に全ての{count_before}件のOHLCVデータを削除しますか？ (yes/no): ")
        if response.lower() != 'yes':
            logger.info("削除をキャンセルしました")
            return 0
        
        # データ削除実行
        deleted_count = repo.clear_all_ohlcv_data()
        logger.info(f"✅ {deleted_count}件のOHLCVデータを削除しました")
        
        return deleted_count
        
    except Exception as e:
        logger.error(f"❌ データ削除エラー: {e}")
        raise
    finally:
        db.close()


def clear_ohlcv_data_by_symbol(symbol: str) -> int:
    """
    指定されたシンボルのOHLCVデータを削除
    
    Args:
        symbol: 削除対象のシンボル
        
    Returns:
        削除された件数
    """
    db = SessionLocal()
    try:
        repo = OHLCVRepository(db)
        
        # 削除前の件数を確認
        from database.models import OHLCVData
        count_before = db.query(OHLCVData).filter(OHLCVData.symbol == symbol).count()
        logger.info(f"削除前の{symbol}データ件数: {count_before}")
        
        if count_before == 0:
            logger.info(f"シンボル '{symbol}' のデータがありません")
            return 0
        
        # ユーザー確認
        response = input(f"本当にシンボル '{symbol}' の{count_before}件のデータを削除しますか？ (yes/no): ")
        if response.lower() != 'yes':
            logger.info("削除をキャンセルしました")
            return 0
        
        # データ削除実行
        deleted_count = repo.clear_ohlcv_data_by_symbol(symbol)
        logger.info(f"✅ シンボル '{symbol}' の{deleted_count}件のデータを削除しました")
        
        return deleted_count
        
    except Exception as e:
        logger.error(f"❌ シンボル '{symbol}' のデータ削除エラー: {e}")
        raise
    finally:
        db.close()


def clear_unsupported_symbols() -> int:
    """
    サポートされていないシンボルのデータを削除
    
    Returns:
        削除された件数
    """
    from app.config.market_config import MarketDataConfig
    
    db = SessionLocal()
    try:
        repo = OHLCVRepository(db)
        
        # 利用可能なシンボルを取得
        available_symbols = repo.get_available_symbols()
        logger.info(f"データベース内のシンボル: {available_symbols}")
        
        # サポートされていないシンボルを特定
        unsupported_symbols = [
            symbol for symbol in available_symbols 
            if symbol not in MarketDataConfig.SUPPORTED_SYMBOLS
        ]
        
        if not unsupported_symbols:
            logger.info("削除対象のサポートされていないシンボルがありません")
            return 0
        
        logger.info(f"削除対象のサポートされていないシンボル: {unsupported_symbols}")
        
        total_deleted = 0
        for symbol in unsupported_symbols:
            # 各シンボルの件数を確認
            from database.models import OHLCVData
            count = db.query(OHLCVData).filter(OHLCVData.symbol == symbol).count()
            logger.info(f"  {symbol}: {count}件")
            total_deleted += count
        
        # ユーザー確認
        response = input(f"本当にサポートされていない{len(unsupported_symbols)}個のシンボル（計{total_deleted}件）を削除しますか？ (yes/no): ")
        if response.lower() != 'yes':
            logger.info("削除をキャンセルしました")
            return 0
        
        # データ削除実行
        actual_deleted = 0
        for symbol in unsupported_symbols:
            deleted = repo.clear_ohlcv_data_by_symbol(symbol)
            actual_deleted += deleted
            logger.info(f"  ✅ {symbol}: {deleted}件削除")
        
        logger.info(f"✅ 合計{actual_deleted}件のサポートされていないシンボルデータを削除しました")
        return actual_deleted
        
    except Exception as e:
        logger.error(f"❌ サポートされていないシンボルの削除エラー: {e}")
        raise
    finally:
        db.close()


def show_current_data_status():
    """
    現在のデータ状況を表示
    """
    db = SessionLocal()
    try:
        repo = OHLCVRepository(db)
        
        # 全体の件数
        from database.models import OHLCVData
        total_count = db.query(OHLCVData).count()
        logger.info(f"総OHLCVデータ件数: {total_count}")
        
        if total_count == 0:
            logger.info("データベースにOHLCVデータがありません")
            return
        
        # シンボル別の件数
        available_symbols = repo.get_available_symbols()
        logger.info("シンボル別データ件数:")
        for symbol in available_symbols:
            count = repo.get_data_count(symbol, "1d")  # 日足での件数を表示
            logger.info(f"  {symbol}: {count}件（日足）")
        
        # サポート状況
        from app.config.market_config import MarketDataConfig
        supported_symbols = [s for s in available_symbols if s in MarketDataConfig.SUPPORTED_SYMBOLS]
        unsupported_symbols = [s for s in available_symbols if s not in MarketDataConfig.SUPPORTED_SYMBOLS]
        
        logger.info(f"サポートされているシンボル: {supported_symbols}")
        logger.info(f"サポートされていないシンボル: {unsupported_symbols}")
        
    except Exception as e:
        logger.error(f"❌ データ状況確認エラー: {e}")
        raise
    finally:
        db.close()


def main():
    """
    メイン処理
    """
    logger.info("=== OHLCVデータクリアスクリプト ===")
    
    try:
        # データベース初期化
        init_db()
        
        # 現在のデータ状況を表示
        logger.info("現在のデータ状況:")
        show_current_data_status()
        
        print("\n選択してください:")
        print("1. 全てのOHLCVデータを削除")
        print("2. サポートされていないシンボルのデータを削除")
        print("3. 特定のシンボルのデータを削除")
        print("4. データ状況のみ表示")
        print("5. 終了")
        
        choice = input("\n選択 (1-5): ").strip()
        
        if choice == "1":
            clear_all_ohlcv_data()
        elif choice == "2":
            clear_unsupported_symbols()
        elif choice == "3":
            symbol = input("削除するシンボルを入力してください: ").strip()
            if symbol:
                clear_ohlcv_data_by_symbol(symbol)
            else:
                logger.error("シンボルが入力されていません")
        elif choice == "4":
            # 既に表示済み
            pass
        elif choice == "5":
            logger.info("終了します")
        else:
            logger.error("無効な選択です")
        
        # 処理後のデータ状況を表示
        if choice in ["1", "2", "3"]:
            logger.info("\n処理後のデータ状況:")
            show_current_data_status()
        
    except Exception as e:
        logger.error(f"❌ スクリプト実行エラー: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
