"""
SQLiteデータベースの内容を直接確認するスクリプト
"""

import sqlite3
import sys
from pathlib import Path

# データベースファイルのパス
db_path = Path(__file__).parent.parent.parent / "trdinger.db"

print(f"=== SQLiteデータベース確認 ===")
print(f"データベースファイル: {db_path}")

if not db_path.exists():
    print("❌ データベースファイルが見つかりません")
    sys.exit(1)

try:
    # SQLite接続
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # 1. テーブル一覧を取得
    print("\n1. テーブル一覧:")
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    if tables:
        for table in tables:
            print(f"   - {table[0]}")
    else:
        print("   テーブルが見つかりません")
        conn.close()
        sys.exit(1)
    
    # 2. ohlcv_dataテーブルの確認
    print("\n2. ohlcv_dataテーブルの確認:")
    try:
        cursor.execute("SELECT COUNT(*) FROM ohlcv_data;")
        count = cursor.fetchone()[0]
        print(f"   総レコード数: {count}")
        
        if count > 0:
            # シンボルと時間軸の種類を確認
            cursor.execute("SELECT DISTINCT symbol, timeframe FROM ohlcv_data;")
            distinct_data = cursor.fetchall()
            print(f"   利用可能なデータ種類: {len(distinct_data)}")
            
            for symbol, timeframe in distinct_data[:10]:  # 最初の10件
                cursor.execute(
                    "SELECT COUNT(*) FROM ohlcv_data WHERE symbol=? AND timeframe=?",
                    (symbol, timeframe)
                )
                data_count = cursor.fetchone()[0]
                print(f"   - {symbol} ({timeframe}): {data_count}件")
            
            if len(distinct_data) > 10:
                print(f"   ... 他 {len(distinct_data) - 10}種類")
                
            # 最新データの確認
            cursor.execute(
                "SELECT symbol, timeframe, timestamp, close FROM ohlcv_data ORDER BY timestamp DESC LIMIT 5;"
            )
            latest_data = cursor.fetchall()
            print("\n   最新データ（5件）:")
            for row in latest_data:
                print(f"   - {row[0]} ({row[1]}): {row[2]} - Close: {row[3]}")
        else:
            print("   ❌ OHLCVデータが見つかりません")
            print("   データ収集を実行してください")
            
    except sqlite3.Error as e:
        print(f"   ❌ ohlcv_dataテーブルのアクセスエラー: {e}")
    
    conn.close()
    print("\n=== 確認完了 ===")
    
except sqlite3.Error as e:
    print(f"❌ SQLiteエラー: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ エラー: {e}")
    sys.exit(1)
