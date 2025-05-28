#!/usr/bin/env python3
"""
ファンディングレートデータベース調査スクリプト
"""

from database.connection import SessionLocal
from database.models import FundingRateData
from sqlalchemy import text

def main():
    db = SessionLocal()
    
    try:
        # テーブルの存在確認
        result = db.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='funding_rate_data'"))
        table_exists = result.fetchone()
        print(f'funding_rate_dataテーブル存在: {table_exists is not None}')
        
        if table_exists:
            # データ件数確認
            count = db.query(FundingRateData).count()
            print(f'ファンディングレートデータ件数: {count}')
            
            # 最新5件を表示
            latest = db.query(FundingRateData).order_by(FundingRateData.created_at.desc()).limit(5).all()
            print(f'最新データ: {len(latest)}件')
            for data in latest:
                print(f'  {data.symbol}: {data.funding_rate} at {data.funding_timestamp}')
        else:
            print('テーブルが存在しません')
            
    except Exception as e:
        print(f'エラー: {e}')
    finally:
        db.close()

if __name__ == "__main__":
    main()
