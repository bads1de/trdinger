"""
オープンインタレストテーブルを再作成するスクリプト
"""

from database.connection import engine
from database.models import OpenInterestData
from sqlalchemy import text

def recreate_open_interest_table():
    """オープンインタレストテーブルを再作成"""
    try:
        # 既存のテーブルを削除
        with engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS open_interest_data"))
            conn.commit()
            print("既存のオープンインタレストテーブルを削除しました")
        
        # 新しいテーブルを作成
        OpenInterestData.__table__.create(engine, checkfirst=True)
        print("新しいオープンインタレストテーブルを作成しました")
        
        # テーブル構造を確認
        from sqlalchemy import inspect
        inspector = inspect(engine)
        columns = inspector.get_columns('open_interest_data')
        
        print("新しいテーブル構造:")
        for column in columns:
            print(f"  {column['name']}: {column['type']}")
            
    except Exception as e:
        print(f"テーブル再作成エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    recreate_open_interest_table()
