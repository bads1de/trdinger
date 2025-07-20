"""
データベース初期化スクリプト
"""

import sys
import os
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from database.connection import init_db, test_connection, check_db_initialized
    
    print("=== データベース初期化 ===")
    
    # 1. データベース接続テスト
    print("1. データベース接続テスト...")
    connection_ok = test_connection()
    print(f"   結果: {'✅ 成功' if connection_ok else '❌ 失敗'}")
    
    if not connection_ok:
        print("   データベースに接続できません。")
        sys.exit(1)
    
    # 2. 初期化状態チェック
    print("2. 初期化状態チェック...")
    db_initialized = check_db_initialized()
    print(f"   結果: {'✅ 初期化済み' if db_initialized else '❌ 未初期化'}")
    
    if db_initialized:
        print("   データベースは既に初期化されています。")
    else:
        # 3. データベース初期化実行
        print("3. データベース初期化実行...")
        init_db()
        print("   ✅ データベース初期化完了")
    
    print("\n=== 初期化完了 ===")
    
except ImportError as e:
    print(f"❌ インポートエラー: {e}")
    print("必要なモジュールが見つかりません。")
    sys.exit(1)
except Exception as e:
    print(f"❌ エラー: {e}")
    sys.exit(1)
