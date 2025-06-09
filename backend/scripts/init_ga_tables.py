"""
GA関連テーブルの初期化スクリプト

新しく追加されたGAExperimentとGeneratedStrategyテーブルを作成します。
"""

import sys
import os
import logging

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connection import init_db, SessionLocal
from database.models import GAExperiment, GeneratedStrategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_tables_exist():
    """テーブルの存在確認"""
    try:
        db = SessionLocal()
        try:
            # GAExperimentテーブルの確認
            ga_count = db.query(GAExperiment).count()
            logger.info(f"GAExperimentテーブル: {ga_count} 件のレコード")
            
            # GeneratedStrategyテーブルの確認
            strategy_count = db.query(GeneratedStrategy).count()
            logger.info(f"GeneratedStrategyテーブル: {strategy_count} 件のレコード")
            
            return True
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"テーブル確認エラー: {e}")
        return False


def main():
    """メイン処理"""
    try:
        logger.info("=== GA関連テーブル初期化開始 ===")
        
        # データベース初期化
        logger.info("データベースを初期化中...")
        init_db()
        logger.info("データベース初期化完了")
        
        # テーブル存在確認
        logger.info("テーブル存在確認中...")
        if check_tables_exist():
            logger.info("✅ GA関連テーブルが正常に作成されました")
        else:
            logger.error("❌ テーブル作成に失敗しました")
            return False
            
        logger.info("=== GA関連テーブル初期化完了 ===")
        return True
        
    except Exception as e:
        logger.error(f"初期化エラー: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
