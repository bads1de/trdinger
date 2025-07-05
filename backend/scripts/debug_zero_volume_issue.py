#!/usr/bin/env python3
"""
取引量0問題の詳細デバッグスクリプト
実際のバックテスト実行時の詳細な動作を調査します
"""

import sys
import os
import logging
import pandas as pd
from datetime import datetime

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.services.auto_strategy.models.strategy_gene import StrategyGene, IndicatorGene, Condition
from app.core.services.auto_strategy.factories.strategy_factory import StrategyFactory
from app.core.services.backtest_service import BacktestService
from app.core.services.backtest_data_service import BacktestDataService
from database.repositories.ohlcv_repository import OHLCVRepository
from database.connection import SessionLocal

# ログ設定を詳細に
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 特定のロガーのレベルを設定
logging.getLogger('app.core.services.auto_strategy.factories.strategy_factory').setLevel(logging.DEBUG)
logging.getLogger('app.core.services.backtest_service').setLevel(logging.DEBUG)


def create_simple_test_gene():
    """シンプルなテスト用戦略遺伝子を作成"""
    indicators = [
        IndicatorGene(type="SMA", parameters={"period": 5}, enabled=True),  # 短期SMA
        IndicatorGene(type="RSI", parameters={"period": 7}, enabled=True),  # 短期RSI
    ]
    
    # 非常にシンプルな条件（満たされやすい）
    entry_conditions = [
        Condition(left_operand="close", operator=">", right_operand="SMA")  # 価格がSMAより上
    ]
    
    exit_conditions = [
        Condition(left_operand="close", operator="<", right_operand="SMA")  # 価格がSMAより下
    ]
    
    risk_management = {
        "stop_loss": 0.05,      # 5%
        "take_profit": 0.10,    # 10%
        "position_size": 0.2,   # 20% - 大きめに設定
    }
    
    return StrategyGene(
        indicators=indicators,
        entry_conditions=entry_conditions,
        exit_conditions=exit_conditions,
        risk_management=risk_management,
        metadata={"test": "debug_zero_volume"}
    )


def create_test_data():
    """テスト用のOHLCVデータを作成（トレンドのあるデータ）"""
    dates = pd.date_range(start='2024-01-01', end='2024-01-03', freq='1H')
    
    # 上昇トレンドのデータを作成
    base_price = 50000
    data = []
    
    for i, date in enumerate(dates):
        # 上昇トレンドを作成
        trend = i * 5  # 時間ごとに5ドル上昇
        noise = (i % 10 - 5) * 2  # ノイズを追加
        
        close_price = base_price + trend + noise
        open_price = close_price - 2
        high_price = close_price + 10
        low_price = close_price - 10
        volume = 1000 + i
        
        data.append({
            'Open': open_price,
            'High': high_price,
            'Low': low_price,
            'Close': close_price,
            'Volume': volume,
        })
    
    df = pd.DataFrame(data, index=dates)
    return df


def debug_strategy_execution():
    """戦略実行の詳細デバッグ"""
    print("=== 戦略実行の詳細デバッグ ===")
    
    try:
        # テスト用戦略遺伝子を作成
        test_gene = create_simple_test_gene()
        print(f"戦略遺伝子作成完了:")
        print(f"  指標数: {len(test_gene.indicators)}")
        print(f"  エントリー条件数: {len(test_gene.entry_conditions)}")
        print(f"  エグジット条件数: {len(test_gene.exit_conditions)}")
        print(f"  取引量設定: {test_gene.risk_management['position_size']}")
        
        # StrategyFactoryで戦略クラスを生成
        strategy_factory = StrategyFactory()
        strategy_class = strategy_factory.create_strategy_class(test_gene)
        print(f"戦略クラス生成完了: {strategy_class.__name__}")
        
        # テストデータを作成
        test_data = create_test_data()
        print(f"テストデータ作成完了: {len(test_data)}行")
        print(f"価格範囲: {test_data['Close'].min():.2f} - {test_data['Close'].max():.2f}")
        
        # backtesting.pyで直接実行
        from backtesting import Backtest
        
        print("\nbacktesting.pyで直接実行中...")
        bt = Backtest(
            test_data,
            strategy_class,
            cash=100000.0,
            commission=0.001,
            exclusive_orders=True,
            trade_on_close=True,
        )
        
        # パラメータを渡してバックテストを実行
        strategy_params = {"strategy_gene": test_gene.to_dict()}
        print(f"戦略パラメータ: {list(strategy_params.keys())}")
        
        stats = bt.run(**strategy_params)
        
        # 結果を詳細に確認
        print(f"\n=== バックテスト結果 ===")
        print(f"総取引回数: {stats['# Trades']}")
        print(f"総リターン: {stats['Return [%]']:.2f}%")
        print(f"最終資産: {stats['Equity Final [$']:.2f}")
        print(f"勝率: {stats['Win Rate [%]']:.2f}%")
        print(f"最大ドローダウン: {stats['Max. Drawdown [%]']:.2f}%")
        
        # 取引履歴を確認
        trades = stats._trades
        print(f"\n取引履歴の詳細:")
        print(f"取引データフレームの形状: {trades.shape if trades is not None else 'None'}")
        
        if trades is not None and not trades.empty:
            print("取引が実行されました！")
            print(trades.head())
        else:
            print("❌ 取引が実行されていません")
            
            # 戦略インスタンスを作成して条件を手動チェック
            print("\n=== 手動条件チェック ===")
            strategy_instance = strategy_class()
            
            # データを設定（簡易版）
            strategy_instance.data = test_data
            
            # 指標を初期化
            try:
                strategy_instance.init()
                print("戦略初期化完了")
                
                # 指標値を確認
                if hasattr(strategy_instance, 'SMA'):
                    sma_values = strategy_instance.SMA
                    print(f"SMA値の範囲: {sma_values.min():.2f} - {sma_values.max():.2f}")
                    
                    # 最後の数値を確認
                    last_close = test_data['Close'].iloc[-1]
                    last_sma = sma_values.iloc[-1]
                    print(f"最後の終値: {last_close:.2f}")
                    print(f"最後のSMA: {last_sma:.2f}")
                    print(f"エントリー条件 (close > SMA): {last_close > last_sma}")
                
                if hasattr(strategy_instance, 'RSI'):
                    rsi_values = strategy_instance.RSI
                    print(f"RSI値の範囲: {rsi_values.min():.2f} - {rsi_values.max():.2f}")
                    print(f"最後のRSI: {rsi_values.iloc[-1]:.2f}")
                    
            except Exception as e:
                print(f"戦略初期化エラー: {e}")
                import traceback
                traceback.print_exc()
        
        return stats['# Trades'] > 0
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def debug_backtest_service():
    """BacktestServiceを使った詳細デバッグ"""
    print("\n=== BacktestServiceを使った詳細デバッグ ===")
    
    try:
        # モックのBacktestDataServiceを作成
        class MockBacktestDataService:
            def get_data_for_backtest(self, symbol, timeframe, start_date, end_date):
                return create_test_data()
        
        # BacktestServiceを初期化
        backtest_service = BacktestService(MockBacktestDataService())
        
        # テスト用戦略遺伝子を作成
        test_gene = create_simple_test_gene()
        
        # バックテスト設定
        config = {
            "strategy_name": "DEBUG_TEST",
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-01-03",
            "initial_capital": 100000.0,
            "commission_rate": 0.001,
            "strategy_config": {
                "strategy_type": "GENERATED_TEST",
                "parameters": {"strategy_gene": test_gene.to_dict()},
            }
        }
        
        print("BacktestServiceでバックテストを実行中...")
        result = backtest_service.run_backtest(config)
        
        # 結果を詳細に確認
        performance_metrics = result.get("performance_metrics", {})
        trade_history = result.get("trade_history", [])
        
        print(f"\n=== BacktestService結果 ===")
        print(f"総取引回数: {performance_metrics.get('total_trades', 0)}")
        print(f"総リターン: {performance_metrics.get('total_return', 0):.2f}%")
        print(f"最終資産: {performance_metrics.get('equity_final', 0):.2f}")
        print(f"取引履歴の件数: {len(trade_history)}")
        
        if len(trade_history) > 0:
            print("✅ BacktestServiceで取引が実行されました！")
            print("最初の取引:", trade_history[0])
        else:
            print("❌ BacktestServiceでも取引が実行されていません")
        
        return len(trade_history) > 0
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """メイン関数"""
    print("取引量0問題の詳細デバッグを開始します\n")
    
    results = []
    
    # テスト1: 戦略実行の詳細デバッグ
    results.append(debug_strategy_execution())
    
    # テスト2: BacktestServiceを使った詳細デバッグ
    results.append(debug_backtest_service())
    
    # 結果のまとめ
    print("\n" + "="*50)
    print("デバッグ結果のまとめ:")
    print(f"成功: {sum(results)}/{len(results)}")
    
    if any(results):
        print("🎉 一部のテストで取引が実行されました")
    else:
        print("❌ すべてのテストで取引が実行されませんでした")
        print("追加の調査が必要です")
    
    return any(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
