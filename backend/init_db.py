#!/usr/bin/env python3
"""
データベース初期化スクリプト
"""

from database.connection import init_db, get_db
from database.models import UserStrategy
from sqlalchemy import text


def main():
    print("データベースを初期化中...")
    init_db()
    print("データベース初期化完了")

    # テーブルが作成されたか確認
    db = next(get_db())
    try:
        # テーブルの存在確認
        result = db.execute(
            text(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='user_strategies'"
            )
        ).fetchone()
        if result:
            print("user_strategiesテーブルが作成されました")
        else:
            print("user_strategiesテーブルが見つかりません")

        # 全テーブル一覧を表示
        tables = db.execute(
            text("SELECT name FROM sqlite_master WHERE type='table'")
        ).fetchall()
        print("データベース内のテーブル:")
        for table in tables:
            print(f"  - {table[0]}")

    finally:
        db.close()


if __name__ == "__main__":
    main()
