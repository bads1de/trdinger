"""
オープンインタレストテーブルを手動で作成するスクリプト
"""

from database.connection import engine
from database.models import OpenInterestData, Base

def create_open_interest_table():
    """オープンインタレストテーブルを作成"""
    try:
        # OpenInterestDataテーブルのみを作成
        OpenInterestData.__table__.create(engine, checkfirst=True)
        print("オープンインタレストテーブルを作成しました")
        
        # テーブル構造を確認
        from sqlalchemy import inspect
        inspector = inspect(engine)
        columns = inspector.get_columns('open_interest_data')
        
        print("テーブル構造:")
        for column in columns:
            print(f"  {column['name']}: {column['type']}")
            
    except Exception as e:
        print(f"テーブル作成エラー: {e}")

if __name__ == "__main__":
    create_open_interest_table()
