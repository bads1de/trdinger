#!/usr/bin/env python3
"""
戦略実行エンジンのデバッグ
"""
import os
import pandas as pd
from datetime import datetime, timedelta

# SQLite用の設定
os.environ["DATABASE_URL"] = "sqlite:///./trdinger_test.db"

from database.connection import SessionLocal
from database.repository import OHLCVRepository
from backtest_engine.strategy_executor import StrategyExecutor

def debug_strategy_execution():
    """
    戦略実行のデバッグ
    """
    print("=== 戦略実行デバッグ ===")
    
    # データベースからデータを取得
    db = SessionLocal()
    try:
        ohlcv_repo = OHLCVRepository(db)
        
        # 最新30日のデータを取得
        start_time = datetime.now() - timedelta(days=30)
        df = ohlcv_repo.get_ohlcv_dataframe(
            symbol="BTC/USD:BTC",
            timeframe="1d",
            start_time=start_time,
            limit=30
        )
        
        print(f"取得したデータ: {len(df)}件")
        print(f"データの期間: {df.index[0]} ～ {df.index[-1]}")
        print(f"データの列: {list(df.columns)}")
        print(f"最初の5行:")
        print(df.head())
        
        # 戦略実行エンジンを初期化
        executor = StrategyExecutor()
        
        # 指標設定
        indicators_config = [
            {"name": "SMA", "params": {"period": 20}},
            {"name": "SMA", "params": {"period": 50}}
        ]
        
        # 指標を計算
        print("\n指標計算中...")
        indicators_cache = executor.calculate_indicators(df, indicators_config)
        
        print(f"計算された指標: {list(indicators_cache.keys())}")
        
        for key, series in indicators_cache.items():
            print(f"{key}: {len(series)}件, 最新値: {series.iloc[-1]:.2f}")
        
        # 条件評価のテスト
        print("\n条件評価テスト...")
        test_condition = "SMA(close, 20) > SMA(close, 50)"
        
        # 最新のデータで条件評価
        current_index = len(df) - 1
        current_data = df.iloc[current_index]
        
        print(f"現在のデータ: {current_data}")
        print(f"テスト条件: {test_condition}")
        
        # 条件を解析
        parsed_condition = executor._parse_condition(test_condition, current_index, current_data)
        print(f"解析後の条件: {parsed_condition}")
        
        # 条件を評価
        try:
            result = eval(parsed_condition)
            print(f"条件評価結果: {result}")
        except Exception as e:
            print(f"条件評価エラー: {e}")
        
    finally:
        db.close()

if __name__ == "__main__":
    debug_strategy_execution()
