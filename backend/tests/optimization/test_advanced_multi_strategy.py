#!/usr/bin/env python3
"""
高度な複数戦略最適化テスト

複数の市場条件（トレンド、レンジ、ボラティリティ）でのテストと
リスク調整済みリターンでの戦略評価を行います。
"""

import sys
import os
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from itertools import product

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.core.services.backtest_service import BacktestService
from database.connection import SessionLocal
from database.repositories.ohlcv_repository import OHLCVRepository
from app.core.services.backtest_data_service import BacktestDataService
from database.models import OHLCVData


def generate_market_scenario_data(scenario: str, start_date: datetime, end_date: datetime, initial_price: float = 50000) -> pd.DataFrame:
    """
    市場シナリオ別のデータを生成
    
    Args:
        scenario: 'trending_up', 'trending_down', 'sideways', 'high_volatility'
        start_date: 開始日
        end_date: 終了日
        initial_price: 初期価格
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    np.random.seed(42)  # 再現性のため
    
    if scenario == 'trending_up':
        # 上昇トレンド
        trend = np.linspace(0, 0.8, len(date_range))  # 80%上昇
        daily_returns = np.random.normal(0.002, 0.02, len(date_range)) + trend / len(date_range)
        
    elif scenario == 'trending_down':
        # 下降トレンド
        trend = np.linspace(0, -0.4, len(date_range))  # 40%下落
        daily_returns = np.random.normal(-0.001, 0.02, len(date_range)) + trend / len(date_range)
        
    elif scenario == 'sideways':
        # レンジ相場
        cycle = np.sin(np.linspace(0, 8*np.pi, len(date_range))) * 0.001
        daily_returns = np.random.normal(0, 0.015, len(date_range)) + cycle
        
    elif scenario == 'high_volatility':
        # 高ボラティリティ
        volatility_spikes = np.random.choice([1, 3], len(date_range), p=[0.8, 0.2])
        daily_returns = np.random.normal(0, 0.04, len(date_range)) * volatility_spikes
        
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    
    # 価格を計算
    prices = [initial_price]
    for ret in daily_returns[1:]:
        prices.append(max(prices[-1] * (1 + ret), 1000))  # 最低価格を設定
    
    # OHLCV データを生成
    data = []
    for i, (date, price) in enumerate(zip(date_range, prices)):
        daily_volatility = abs(daily_returns[i]) + np.random.uniform(0.005, 0.015)
        
        high = price * (1 + daily_volatility)
        low = price * (1 - daily_volatility)
        
        if i == 0:
            open_price = price
        else:
            open_price = prices[i-1]
        close_price = price
        
        base_volume = 1000000
        volume_multiplier = 1 + abs(daily_returns[i]) * 20
        volume = base_volume * volume_multiplier
        
        data.append({
            'timestamp': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    return df


def insert_scenario_data_to_db(df: pd.DataFrame, symbol: str = "BTC/USDT", timeframe: str = "1d"):
    """シナリオデータをデータベースに挿入"""
    db = SessionLocal()
    try:
        # 既存データを削除
        db.query(OHLCVData).filter(
            OHLCVData.symbol == symbol,
            OHLCVData.timeframe == timeframe
        ).delete()
        
        # 新しいデータを挿入
        for timestamp, row in df.iterrows():
            ohlcv_data = OHLCVData(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=timestamp,
                open=float(row['Open']),
                high=float(row['High']),
                low=float(row['Low']),
                close=float(row['Close']),
                volume=float(row['Volume'])
            )
            db.add(ohlcv_data)
        
        db.commit()
        
    except Exception as e:
        db.rollback()
        print(f"❌ データ挿入エラー: {e}")
        raise
    finally:
        db.close()


def test_strategy_in_scenario(strategy_type: str, scenario: str, param_ranges: dict):
    """特定のシナリオで戦略をテスト"""
    print(f"\n--- {strategy_type}戦略 in {scenario}市場 ---")
    
    # シナリオデータを生成
    start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end_date = datetime(2024, 12, 31, tzinfo=timezone.utc)
    
    scenario_data = generate_market_scenario_data(scenario, start_date, end_date)
    insert_scenario_data_to_db(scenario_data)
    
    db = SessionLocal()
    try:
        ohlcv_repo = OHLCVRepository(db)
        data_service = BacktestDataService(ohlcv_repo)
        backtest_service = BacktestService(data_service)
        
        base_config = {
            "strategy_name": f"{strategy_type}_{scenario.upper()}",
            "symbol": "BTC/USDT",
            "timeframe": "1d",
            "start_date": start_date,
            "end_date": end_date,
            "initial_capital": 1000000,
            "commission_rate": 0.001,
            "strategy_config": {
                "strategy_type": strategy_type,
                "parameters": {},
            },
        }
        
        # 簡略化されたパラメータ範囲（高速テスト用）
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        combinations = list(product(*param_values))
        
        best_result = None
        best_sharpe = -float('inf')
        
        for combination in combinations[:10]:  # 最初の10組み合わせのみテスト
            params = dict(zip(param_names, combination))
            
            config = base_config.copy()
            config["strategy_config"]["parameters"] = params
            
            try:
                result = backtest_service.run_backtest(config)
                
                if "performance_metrics" in result:
                    metrics = result["performance_metrics"]
                    sharpe = metrics.get('sharpe_ratio', -float('inf'))
                    
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_result = {
                            'parameters': params,
                            'metrics': metrics,
                            'scenario': scenario
                        }
                        
            except Exception as e:
                continue
        
        if best_result:
            metrics = best_result['metrics']
            print(f"  最適パラメータ: {best_result['parameters']}")
            print(f"  シャープレシオ: {metrics.get('sharpe_ratio', 0):.3f}")
            print(f"  総リターン: {metrics.get('total_return', 0):.2f}%")
            print(f"  最大DD: {metrics.get('max_drawdown', 0):.2f}%")
            print(f"  勝率: {metrics.get('win_rate', 0):.2f}%")
        else:
            print("  ❌ 有効な結果が得られませんでした")
        
        return best_result
        
    except Exception as e:
        print(f"  ❌ エラー: {e}")
        return None
    finally:
        db.close()


def test_strategy_robustness():
    """戦略のロバストネステスト"""
    print("=== 戦略ロバストネステスト ===")
    
    # テスト対象の戦略と簡略化されたパラメータ
    strategies = {
        "SMA_CROSS": {
            "n1": [10, 20],
            "n2": [30, 50]
        },
        "RSI": {
            "period": [14, 21],
            "oversold": [25, 35],
            "overbought": [65, 75]
        },
        "MACD": {
            "fast_period": [10, 15],
            "slow_period": [20, 30],
            "signal_period": [7, 11]
        }
    }
    
    # 市場シナリオ
    scenarios = ['trending_up', 'trending_down', 'sideways', 'high_volatility']
    
    # 結果を格納
    results = {}
    
    for strategy_type, param_ranges in strategies.items():
        print(f"\n{strategy_type}戦略のロバストネステスト:")
        strategy_results = {}
        
        for scenario in scenarios:
            result = test_strategy_in_scenario(strategy_type, scenario, param_ranges)
            strategy_results[scenario] = result
        
        results[strategy_type] = strategy_results
    
    return results


def analyze_robustness_results(results):
    """ロバストネス結果の分析"""
    print("\n=== ロバストネス分析結果 ===")
    
    # 戦略別の平均パフォーマンス
    strategy_scores = {}
    
    for strategy_type, strategy_results in results.items():
        valid_results = [r for r in strategy_results.values() if r is not None]
        
        if valid_results:
            avg_sharpe = np.mean([r['metrics']['sharpe_ratio'] for r in valid_results])
            avg_return = np.mean([r['metrics']['total_return'] for r in valid_results])
            avg_drawdown = np.mean([r['metrics']['max_drawdown'] for r in valid_results])
            success_rate = len(valid_results) / len(strategy_results) * 100
            
            strategy_scores[strategy_type] = {
                'avg_sharpe': avg_sharpe,
                'avg_return': avg_return,
                'avg_drawdown': avg_drawdown,
                'success_rate': success_rate,
                'robustness_score': avg_sharpe * (success_rate / 100)  # ロバストネススコア
            }
        else:
            strategy_scores[strategy_type] = {
                'avg_sharpe': 0,
                'avg_return': 0,
                'avg_drawdown': 0,
                'success_rate': 0,
                'robustness_score': 0
            }
    
    # 結果表示
    print("戦略\t\t平均シャープ\t平均リターン\t平均DD\t\t成功率\t\tロバストネス")
    print("-" * 90)
    
    for strategy_type, scores in strategy_scores.items():
        print(f"{strategy_type}\t\t{scores['avg_sharpe']:.3f}\t\t{scores['avg_return']:.2f}%\t\t{scores['avg_drawdown']:.2f}%\t\t{scores['success_rate']:.0f}%\t\t{scores['robustness_score']:.3f}")
    
    # 最もロバストな戦略
    best_strategy = max(strategy_scores.items(), key=lambda x: x[1]['robustness_score'])
    print(f"\n🏆 最もロバストな戦略: {best_strategy[0]} (ロバストネススコア: {best_strategy[1]['robustness_score']:.3f})")
    
    return strategy_scores


def test_market_condition_analysis():
    """市場条件別の戦略適性分析"""
    print("\n=== 市場条件別戦略適性分析 ===")
    
    scenarios = ['trending_up', 'trending_down', 'sideways', 'high_volatility']
    scenario_names = {
        'trending_up': '上昇トレンド',
        'trending_down': '下降トレンド', 
        'sideways': 'レンジ相場',
        'high_volatility': '高ボラティリティ'
    }
    
    # 各シナリオで最適な戦略を特定
    for scenario in scenarios:
        print(f"\n--- {scenario_names[scenario]}市場での戦略比較 ---")
        
        scenario_results = {}
        
        # SMA_CROSS
        result = test_strategy_in_scenario("SMA_CROSS", scenario, {"n1": [15, 25], "n2": [40, 60]})
        if result:
            scenario_results["SMA_CROSS"] = result['metrics']['sharpe_ratio']
        
        # RSI
        result = test_strategy_in_scenario("RSI", scenario, {"period": [14], "oversold": [30], "overbought": [70]})
        if result:
            scenario_results["RSI"] = result['metrics']['sharpe_ratio']
        
        # MACD
        result = test_strategy_in_scenario("MACD", scenario, {"fast_period": [12], "slow_period": [26], "signal_period": [9]})
        if result:
            scenario_results["MACD"] = result['metrics']['sharpe_ratio']
        
        if scenario_results:
            best_strategy = max(scenario_results.items(), key=lambda x: x[1])
            print(f"  最適戦略: {best_strategy[0]} (シャープレシオ: {best_strategy[1]:.3f})")
        else:
            print("  有効な結果がありません")


def main():
    """メイン関数"""
    print("高度な複数戦略最適化テスト開始")
    print("=" * 80)
    
    try:
        # ロバストネステスト
        robustness_results = test_strategy_robustness()
        
        # ロバストネス分析
        strategy_scores = analyze_robustness_results(robustness_results)
        
        # 市場条件別分析
        test_market_condition_analysis()
        
        print("\n" + "=" * 80)
        print("高度なテスト完了!")
        print("\n💡 主要成果:")
        print("- 複数市場条件での戦略テスト")
        print("- ロバストネス評価による戦略比較")
        print("- 市場条件別の戦略適性分析")
        print("- リスク調整済みリターンでの評価")
        
    except Exception as e:
        print(f"❌ テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
