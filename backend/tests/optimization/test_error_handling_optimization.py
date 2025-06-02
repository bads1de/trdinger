#!/usr/bin/env python3
"""
バックテスト最適化のエラーハンドリングテスト

不正なパラメータや異常な条件での動作を確認します。
"""

import sys
import os
from datetime import datetime, timezone

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.services.enhanced_backtest_service import EnhancedBacktestService
from database.connection import SessionLocal
from database.repositories.ohlcv_repository import OHLCVRepository
from app.core.services.backtest_data_service import BacktestDataService


def test_invalid_parameters():
    """不正なパラメータでのエラーハンドリングテスト"""
    print("=== 不正なパラメータでのエラーハンドリングテスト ===")
    
    db = SessionLocal()
    try:
        ohlcv_repo = OHLCVRepository(db)
        data_service = BacktestDataService(ohlcv_repo)
        enhanced_service = EnhancedBacktestService(data_service)
        
        # 1. 空のパラメータ
        print("\n1. 空のパラメータテスト")
        try:
            config = {
                "strategy_name": "TEST_EMPTY_PARAMS",
                "symbol": "BTC/USDT",
                "timeframe": "1d",
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
                "initial_capital": 1000000,
                "commission_rate": 0.001,
                "strategy_config": {"strategy_type": "SMA_CROSS", "parameters": {}},
            }
            
            optimization_params = {
                "method": "grid",
                "maximize": "Sharpe Ratio",
                "parameters": {},  # 空のパラメータ
            }
            
            result = enhanced_service.optimize_strategy_enhanced(config, optimization_params)
            print("❌ エラーが発生すべきでした")
        except ValueError as e:
            print(f"✅ 期待通りのエラー: {e}")
        except Exception as e:
            print(f"⚠️ 予期しないエラー: {e}")
        
        # 2. 不正な日付範囲
        print("\n2. 不正な日付範囲テスト")
        try:
            config = {
                "strategy_name": "TEST_INVALID_DATES",
                "symbol": "BTC/USDT",
                "timeframe": "1d",
                "start_date": "2024-12-31",  # 終了日より後
                "end_date": "2024-01-01",
                "initial_capital": 1000000,
                "commission_rate": 0.001,
                "strategy_config": {"strategy_type": "SMA_CROSS", "parameters": {}},
            }
            
            optimization_params = {
                "method": "grid",
                "maximize": "Sharpe Ratio",
                "parameters": {"n1": [10, 20], "n2": [30, 40]},
            }
            
            result = enhanced_service.optimize_strategy_enhanced(config, optimization_params)
            print("❌ エラーが発生すべきでした")
        except ValueError as e:
            print(f"✅ 期待通りのエラー: {e}")
        except Exception as e:
            print(f"⚠️ 予期しないエラー: {e}")
        
        # 3. 不正な初期資金
        print("\n3. 不正な初期資金テスト")
        try:
            config = {
                "strategy_name": "TEST_INVALID_CAPITAL",
                "symbol": "BTC/USDT",
                "timeframe": "1d",
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
                "initial_capital": -1000,  # 負の値
                "commission_rate": 0.001,
                "strategy_config": {"strategy_type": "SMA_CROSS", "parameters": {}},
            }
            
            optimization_params = {
                "method": "grid",
                "maximize": "Sharpe Ratio",
                "parameters": {"n1": [10, 20], "n2": [30, 40]},
            }
            
            result = enhanced_service.optimize_strategy_enhanced(config, optimization_params)
            print("❌ エラーが発生すべきでした")
        except ValueError as e:
            print(f"✅ 期待通りのエラー: {e}")
        except Exception as e:
            print(f"⚠️ 予期しないエラー: {e}")
        
        # 4. 存在しないシンボル
        print("\n4. 存在しないシンボルテスト")
        try:
            config = {
                "strategy_name": "TEST_INVALID_SYMBOL",
                "symbol": "INVALID/SYMBOL",
                "timeframe": "1d",
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
                "initial_capital": 1000000,
                "commission_rate": 0.001,
                "strategy_config": {"strategy_type": "SMA_CROSS", "parameters": {}},
            }
            
            optimization_params = {
                "method": "grid",
                "maximize": "Sharpe Ratio",
                "parameters": {"n1": [10, 20], "n2": [30, 40]},
            }
            
            result = enhanced_service.optimize_strategy_enhanced(config, optimization_params)
            print("❌ エラーが発生すべきでした")
        except ValueError as e:
            print(f"✅ 期待通りのエラー: {e}")
        except Exception as e:
            print(f"⚠️ 予期しないエラー: {e}")
        
        # 5. 不正な制約条件
        print("\n5. 不正な制約条件テスト")
        try:
            config = {
                "strategy_name": "TEST_INVALID_CONSTRAINT",
                "symbol": "BTC/USDT",
                "timeframe": "1d",
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
                "initial_capital": 1000000,
                "commission_rate": 0.001,
                "strategy_config": {"strategy_type": "SMA_CROSS", "parameters": {}},
            }
            
            optimization_params = {
                "method": "grid",
                "maximize": "Sharpe Ratio",
                "parameters": {"n1": [10, 20], "n2": [30, 40]},
                "constraint": "invalid_constraint_string",  # 不正な制約
            }
            
            result = enhanced_service.optimize_strategy_enhanced(config, optimization_params)
            print("❌ エラーが発生すべきでした")
        except ValueError as e:
            print(f"✅ 期待通りのエラー: {e}")
        except Exception as e:
            print(f"⚠️ 予期しないエラー: {e}")
            
    finally:
        db.close()


def test_data_insufficient_scenarios():
    """データ不足時の動作確認"""
    print("\n=== データ不足時の動作確認 ===")
    
    db = SessionLocal()
    try:
        ohlcv_repo = OHLCVRepository(db)
        data_service = BacktestDataService(ohlcv_repo)
        enhanced_service = EnhancedBacktestService(data_service)
        
        # 1. 非常に短い期間（データ不足）
        print("\n1. 非常に短い期間テスト")
        try:
            config = {
                "strategy_name": "TEST_SHORT_PERIOD",
                "symbol": "BTC/USDT",
                "timeframe": "1d",
                "start_date": "2024-01-01",
                "end_date": "2024-01-02",  # 1日だけ
                "initial_capital": 1000000,
                "commission_rate": 0.001,
                "strategy_config": {"strategy_type": "SMA_CROSS", "parameters": {}},
            }
            
            optimization_params = {
                "method": "grid",
                "maximize": "Sharpe Ratio",
                "parameters": {"n1": [10, 20], "n2": [30, 40]},
            }
            
            result = enhanced_service.optimize_strategy_enhanced(config, optimization_params)
            print("⚠️ 短期間でも実行されました（警告レベル）")
        except ValueError as e:
            print(f"✅ 期待通りのエラー: {e}")
        except Exception as e:
            print(f"⚠️ 予期しないエラー: {e}")
            
    finally:
        db.close()


def test_extreme_parameters():
    """極端なパラメータでのテスト"""
    print("\n=== 極端なパラメータでのテスト ===")
    
    db = SessionLocal()
    try:
        ohlcv_repo = OHLCVRepository(db)
        data_service = BacktestDataService(ohlcv_repo)
        enhanced_service = EnhancedBacktestService(data_service)
        
        # 1. 極端に大きなパラメータ範囲
        print("\n1. 極端に大きなパラメータ範囲テスト")
        try:
            config = {
                "strategy_name": "TEST_EXTREME_PARAMS",
                "symbol": "BTC/USDT",
                "timeframe": "1d",
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
                "initial_capital": 1000000,
                "commission_rate": 0.001,
                "strategy_config": {"strategy_type": "SMA_CROSS", "parameters": {}},
            }
            
            optimization_params = {
                "method": "grid",
                "maximize": "Sharpe Ratio",
                "parameters": {
                    "n1": list(range(1, 101)),  # 100個の値
                    "n2": list(range(101, 201)),  # 100個の値 = 10,000通り
                },
                "max_tries": 10,  # 制限を設ける
            }
            
            result = enhanced_service.optimize_strategy_enhanced(config, optimization_params)
            print("⚠️ 大きなパラメータ空間でも実行されました")
        except Exception as e:
            print(f"✅ 適切に制限されました: {e}")
            
    finally:
        db.close()


if __name__ == "__main__":
    print("バックテスト最適化エラーハンドリングテスト開始")
    print("=" * 80)
    
    test_invalid_parameters()
    test_data_insufficient_scenarios()
    test_extreme_parameters()
    
    print("\n" + "=" * 80)
    print("エラーハンドリングテスト完了")
