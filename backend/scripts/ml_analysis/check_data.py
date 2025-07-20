"""
データベースの状態とOHLCVデータの存在確認スクリプト
"""

import sys
import os
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from database.connection import test_connection, check_db_initialized, SessionLocal
    from database.repositories.ohlcv_repository import OHLCVRepository
    from database.models import OHLCVData
    
    print("=== データベース状態確認 ===")
    
    # 1. データベース接続テスト
    print("1. データベース接続テスト...")
    connection_ok = test_connection()
    print(f"   結果: {'✅ 成功' if connection_ok else '❌ 失敗'}")
    
    # 2. データベース初期化チェック
    print("2. データベース初期化チェック...")
    db_initialized = check_db_initialized()
    print(f"   結果: {'✅ 初期化済み' if db_initialized else '❌ 未初期化'}")
    
    if not db_initialized:
        print("   データベースが初期化されていません。")
        print("   以下のコマンドで初期化してください:")
        print("   python -c \"from database.connection import init_db; init_db()\"")
        sys.exit(1)
    
    # 3. OHLCVデータの存在確認
    print("3. OHLCVデータの存在確認...")
    
    db = SessionLocal()
    try:
        repository = OHLCVRepository(db)
        
        # 利用可能なシンボルと時間軸を確認
        query = db.query(OHLCVData.symbol, OHLCVData.timeframe).distinct()
        available_data = query.all()
        
        if available_data:
            print(f"   利用可能なデータ: {len(available_data)}種類")
            for symbol, timeframe in available_data[:10]:  # 最初の10件を表示
                count = db.query(OHLCVData).filter(
                    OHLCVData.symbol == symbol,
                    OHLCVData.timeframe == timeframe
                ).count()
                print(f"   - {symbol} ({timeframe}): {count}件")
            
            if len(available_data) > 10:
                print(f"   ... 他 {len(available_data) - 10}種類")
        else:
            print("   ❌ OHLCVデータが見つかりません")
            print("   データ収集を実行してください:")
            print("   python data_collector/collector.py")
            
    finally:
        db.close()
    
    print("\n=== 確認完了 ===")
    
except ImportError as e:
    print(f"❌ インポートエラー: {e}")
    print("必要なモジュールが見つかりません。")
    sys.exit(1)
except Exception as e:
    print(f"❌ エラー: {e}")
    sys.exit(1)
