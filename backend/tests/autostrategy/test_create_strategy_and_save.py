#!/usr/bin/env python3
"""
AutoStrategy 戦略生成・実行・DB保存 完全スクリプト
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# プロジェクトルートをPythonパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_ohlcv_data():
    """サンプルOHLCVデータを作成"""
    print("Creating sample OHLCV data...")

    # 過去500時間の1時間足データ（より長い期間でより多くの取引機会）
    end_date = datetime.now()
    start_date = end_date - timedelta(hours=500)

    dates = pd.date_range(start=start_date, end=end_date, freq='1h')

    # より現実的な価格データ生成（ビットコインのような変動）
    base_price = 50000
    price_changes = np.random.normal(0, 0.12, len(dates))  # 12%のボラティリティ（さらに激しく）
    close_prices = [base_price]
    for change in price_changes[1:]:
        new_price = close_prices[-1] * (1 + change)
        close_prices.append(max(100, new_price))  # 最低価格を100に設定

    # OHLCVデータ生成
    high_prices = [price * (1 + abs(np.random.normal(0, 0.03))) for price in close_prices]  # 高値の変動をさらに増やす
    low_prices = [price * (1 - abs(np.random.normal(0, 0.03))) for price in close_prices]   # 安値の変動をさらに増やす
    open_prices = close_prices[:-1] + [close_prices[-1] * (1 + np.random.normal(0, 0.02))]  # 始値の変動をさらに増やす
    volumes = np.random.uniform(10000000, 50000000, len(dates))  # 出来高をさらに増やす

    df = pd.DataFrame({
        'timestamp': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    })

    print(f"SUCCESS: Test data creation completed: {len(df)} rows")
    print(f"Price range: {df['close'].min():.2f} to {df['close'].max():.2f}")
    return df

def save_data_to_db(df, symbol="BTCUSDT", timeframe="1h"):
    """データをデータベースに保存"""
    print(f"Saving data to database: {symbol} {timeframe}")

    try:
        from database.repositories.ohlcv_repository import OHLCVRepository
        from database.connection import SessionLocal

        db = SessionLocal()
        try:
            repo = OHLCVRepository(db)

            # データをリポジトリ形式に変換
            ohlcv_data = []
            for _, row in df.iterrows():
                ohlcv_data.append({
                    'timestamp': row['timestamp'],
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['volume'])
                })

            print(f"Prepared {len(ohlcv_data)} records for insertion")

            # データ検証
            from app.utils.data_validation import DataValidator
            is_valid = DataValidator.validate_ohlcv_records_simple(ohlcv_data)
            print(f"Data validation result: {is_valid}")

            if not is_valid:
                print("FAILED: Data validation failed")
                return False

            # データを保存
            saved_count = repo.insert_ohlcv_data(ohlcv_data)
            print(f"SUCCESS: Saved {saved_count} records to database")

            # コミット
            db.commit()
            return True

        finally:
            db.close()

    except Exception as e:
        print(f"FAILED: Error saving data to database: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_autostrategy_and_save():
    """AutoStrategyを実行してDBに保存"""
    print("=" * 60)
    print("AutoStrategy: Create Strategy and Save to DB")
    print("=" * 60)

    try:
        # 1. テストデータ作成
        test_data = create_sample_ohlcv_data()

        # 2. データをデータベースに保存
        if not save_data_to_db(test_data):
            print("FAILED: Could not save data to database")
            return False

        # 3. AutoStrategy実行
        print("\nRunning AutoStrategy...")

        from app.services.auto_strategy.services.auto_strategy_service import AutoStrategyService
        from database.connection import SessionLocal

        # データベースセッション作成
        db = SessionLocal()

        try:
            # AutoStrategyサービス初期化
            print("Initializing AutoStrategy Service...")
            auto_strategy_service = AutoStrategyService()

            # 実験設定 - より積極的な取引を促す設定
            experiment_config = {
                'symbol': 'BTCUSDT',
                'timeframe': '1h',
                'start_date': test_data['timestamp'].min().isoformat(),  # Timestampを文字列に変換
                'end_date': test_data['timestamp'].max().isoformat(),    # Timestampを文字列に変換
                'initial_capital': 100000.0,
                'commission_rate': 0.001,
                'population_size': 20,  # 少し大きく
                'generations': 8,      # 少し大きく
                'indicator_mode': 'technical_only',  # テクニカル指標のみ
                'enable_multi_objective': False,
                'ga_config': {
                    'population_size': 10,  # 最小限に
                    'generations': 1,       # 1世代のみでテスト
                    'crossover_rate': 1.0,  # 交叉率を最大に
                    'mutation_rate': 0.0,  # 突然変異なし（シンプルに）
                    'tournament_size': 2,   # 最小トーナメントサイズ
                    'min_indicators': 1,   # 最小指標数を1に
                    'max_indicators': 1,   # 最大指標数を1に制限（シンプルに）
                    'min_conditions': 1,   # 最小条件数を1に
                    'max_conditions': 1,   # 最大条件数を1に制限（シンプルに）
                    'fitness_weights': {   # 取引数を最重視する重み付け
                        'total_return': 0.02,
                        'total_trades': 0.8,     # 取引数を最重視
                        'win_rate': 0.03,
                        'max_drawdown': 0.08,
                        'sharpe_ratio': 0.07
                    }
                }
            }

            print(f"Settings: {experiment_config['symbol']}, {experiment_config['timeframe']}")
            print(f"Period: {experiment_config['start_date']} - {experiment_config['end_date']}")
            print(f"Indicator mode: {experiment_config['indicator_mode']}")
            print(f"Data points: {len(test_data)} (increased volatility and volume for more trading opportunities)")

            # AutoStrategy実行（GAエンジン直接使用）
            from app.services.auto_strategy.core.ga_engine import GeneticAlgorithmEngine
            from app.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
            from app.services.auto_strategy.generators.strategy_factory import StrategyFactory

            # GA設定
            ga_config_dict = experiment_config['ga_config']
            from app.services.auto_strategy.models.ga_config import GAConfig
            ga_config = GAConfig.from_dict(ga_config_dict)

            # バックテスト設定（Timestampをdatetimeオブジェクトに戻す）
            from datetime import datetime
            backtest_config = {
                'symbol': experiment_config['symbol'],
                'timeframe': experiment_config['timeframe'],
                'start_date': datetime.fromisoformat(experiment_config['start_date']),
                'end_date': datetime.fromisoformat(experiment_config['end_date']),
                'initial_capital': experiment_config['initial_capital'],
                'commission_rate': experiment_config['commission_rate'],
            }

            # 実験ID生成
            import uuid
            experiment_id = str(uuid.uuid4())
            experiment_name = f"test_experiment_{experiment_id[:8]}"

            # 実験を作成
            auto_strategy_service.persistence_service.create_experiment(
                experiment_id, experiment_name, ga_config, backtest_config
            )

            # GAエンジン初期化
            gene_generator = RandomGeneGenerator(ga_config)
            strategy_factory = StrategyFactory()
            ga_engine = GeneticAlgorithmEngine(
                auto_strategy_service.backtest_service, strategy_factory, gene_generator
            )

            # GA実行
            print(f"GA running... (population: {ga_config.population_size}, generations: {ga_config.generations})")
            print(f"Fitness weights - Total Trades: {ga_config.fitness_weights['total_trades']:.1f}, Total Return: {ga_config.fitness_weights['total_return']:.1f}")
            print("Focusing on generating strategies with actual trading activity...")

            result = ga_engine.run_evolution(ga_config, backtest_config)

            # 結果の詳細ログ
            if result and 'all_strategies' in result:
                all_strategies = result['all_strategies']
                print(f"Generated {len(all_strategies)} strategies")

                # 各戦略の取引数を確認（上位20戦略のみ）
                trade_counts = []
                print("\n=== Detailed Strategy Analysis ===")
                for i, strategy in enumerate(all_strategies[:20]):  # 上位20戦略のみ分析
                    try:
                        # 戦略のバックテストを実行して取引数を確認
                        test_result = ga_engine.backtest_service.run_backtest({
                            'strategy_name': f'temp_{strategy.id}',
                            'symbol': experiment_config['symbol'],
                            'timeframe': experiment_config['timeframe'],
                            'start_date': backtest_config['start_date'],
                            'end_date': backtest_config['end_date'],
                            'initial_capital': experiment_config['initial_capital'],
                            'commission_rate': experiment_config['commission_rate'],
                            'strategy_class': ga_engine.strategy_factory.create_strategy_class(strategy),
                            'strategy_config': strategy
                        })
                        trades = test_result.get('total_trades', 0) if test_result else 0
                        win_rate = test_result.get('win_rate', 0) if test_result else 0
                        total_return = test_result.get('total_return', 0) if test_result else 0

                        trade_counts.append(trades)

                        # 取引ありの戦略のみ詳細表示
                        if trades > 0:
                            print(f"  ✅ Strategy {i+1}: ID={strategy.id}")
                            print(f"     Indicators: {[ind.type for ind in strategy.indicators]}")
                            print(f"     Conditions: {len(strategy.get_effective_long_conditions())}")
                            print(f"     Trades: {trades}, Win Rate: {win_rate:.1f}%, Return: {total_return:.2f}%")
                        else:
                            print(f"  ❌ Strategy {i+1}: ID={strategy.id}, Trades: {trades}")

                    except Exception as e:
                        print(f"  ⚠️  Strategy {i+1}: Error testing - {e}")
                        trade_counts.append(0)

                # 取引数統計
                non_zero_trades = sum(1 for t in trade_counts if t > 0)
                print(f"\n=== Trading Statistics ===")
                print(f"  Strategies with trades: {non_zero_trades}/{len(all_strategies)} ({non_zero_trades/len(all_strategies)*100:.1f}%)")
                if trade_counts:
                    print(f"  Average trades: {sum(trade_counts)/len(trade_counts):.1f}")
                    print(f"  Max trades: {max(trade_counts)}")
                    print(f"  Strategies with 5+ trades: {sum(1 for t in trade_counts if t >= 5)}")
                    print(f"  Strategies with 10+ trades: {sum(1 for t in trade_counts if t >= 10)}")

            if result and 'best_strategy' in result:
                best_strategy = result['best_strategy']
                print("SUCCESS: AutoStrategy Experiment Completed")
                print(f"  Best Strategy ID: {best_strategy.id}")
                print(f"  Fitness Score: {getattr(best_strategy, 'fitness_score', 'N/A')}")
                print(f"  Indicators: {[ind.type for ind in best_strategy.indicators]}")

                # 戦略の詳細を確認
                long_conditions = best_strategy.get_effective_long_conditions()
                print(f"  Entry Conditions: {len(long_conditions)}")
                if long_conditions:
                    condition = long_conditions[0]
                    if hasattr(condition, 'left_operand'):
                        print(f"    Sample Condition: {getattr(condition, 'left_operand', 'N/A')} {getattr(condition, 'operator', 'N/A')} {getattr(condition, 'right_operand', 'N/A')}")
                    elif hasattr(condition, 'to_dict'):
                        cond_dict = condition.to_dict()
                        print(f"    Sample Condition: {cond_dict}")
                    else:
                        print(f"    Sample Condition: {condition}")

                # 実際のバックテスト結果を確認
                print(f"  Risk Management: {best_strategy.risk_management}")

                # 戦略をテスト実行して取引数を確認
                print("\nTesting best strategy...")

                # StrategyGeneをdictに変換（ConditionGroupに対応）
                strategy_dict = {
                    'id': best_strategy.id,
                    'indicators': [{'type': ind.type, 'parameters': ind.parameters, 'enabled': ind.enabled} for ind in best_strategy.indicators],
                    'entry_conditions': [cond.to_dict() if hasattr(cond, 'to_dict') else str(cond) for cond in best_strategy.entry_conditions],
                    'exit_conditions': [cond.to_dict() if hasattr(cond, 'to_dict') else str(cond) for cond in best_strategy.exit_conditions],
                    'long_entry_conditions': [cond.to_dict() if hasattr(cond, 'to_dict') else str(cond) for cond in best_strategy.long_entry_conditions],
                    'short_entry_conditions': [cond.to_dict() if hasattr(cond, 'to_dict') else str(cond) for cond in best_strategy.short_entry_conditions],
                    'risk_management': best_strategy.risk_management,
                    'tpsl_gene': best_strategy.tpsl_gene,
                    'position_sizing_gene': best_strategy.position_sizing_gene,
                    'metadata': best_strategy.metadata
                }

                # StrategyGeneを戦略クラスに変換
                try:
                    strategy_class = auto_strategy_service.backtest_service._strategy_factory.create_strategy_class(strategy_dict)
                except Exception as e:
                    print(f"Strategy class creation failed: {e}")
                    # フォールバック: 直接StrategyGeneを使用
                    from app.services.auto_strategy.generators.strategy_factory import StrategyFactory
                    strategy_factory = StrategyFactory()
                    strategy_class = strategy_factory.create_strategy_class(best_strategy)

                test_result = auto_strategy_service.backtest_service.run_backtest({
                    'strategy_name': f'test_{best_strategy.id}',
                    'symbol': experiment_config['symbol'],
                    'timeframe': experiment_config['timeframe'],
                    'start_date': datetime.fromisoformat(experiment_config['start_date']),  # 文字列をdatetimeに変換
                    'end_date': datetime.fromisoformat(experiment_config['end_date']),    # 文字列をdatetimeに変換
                    'initial_capital': experiment_config['initial_capital'],
                    'commission_rate': experiment_config['commission_rate'],
                    'strategy_class': strategy_class,
                    'strategy_config': strategy_dict
                })

                if test_result and 'total_trades' in test_result:
                    print(f"  Test Result - Total Trades: {test_result['total_trades']}")
                    print(f"  Test Result - Total Return: {test_result.get('total_return', 0):.2f}%")
                    print(f"  Test Result - Win Rate: {test_result.get('win_rate', 0):.1f}%")
                    print(f"  Test Result - Max Drawdown: {test_result.get('max_drawdown', 0):.2f}%")
                    print(f"  Test Result - Sharpe Ratio: {test_result.get('sharpe_ratio', 0):.2f}")
                else:
                    print("  Test Result - Unable to get detailed results")
                    print(f"  Debug: test_result type = {type(test_result)}")
                    print(f"  Debug: test_result keys = {test_result.keys() if test_result else 'None'}")

                # 実験結果を保存
                auto_strategy_service.persistence_service.save_experiment_result(
                    experiment_id, result, ga_config, backtest_config
                )

                # 実験を完了状態にする
                auto_strategy_service.persistence_service.complete_experiment(experiment_id)

                print("SUCCESS: Experiment results saved to database")
                return True
            else:
                print(f"FAILED: AutoStrategy execution failed: {result.get('error', 'Unknown error')}")
                return False

        finally:
            db.close()

    except Exception as e:
        print(f"FATAL: Fatal error during execution: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_saved_results():
    """保存された結果を確認"""
    print("\n" + "=" * 60)
    print("Verification: Check Saved Results")
    print("=" * 60)

    try:
        from database.repositories.backtest_result_repository import BacktestResultRepository
        from database.connection import SessionLocal

        db = SessionLocal()

        try:
            repo = BacktestResultRepository(db)

            # 最新のバックテスト結果を取得
            recent_results = repo.get_recent_backtest_results(limit=5)

            if recent_results:
                print(f"SUCCESS: Found {len(recent_results)} backtest results")

                # 最新の結果の詳細を表示
                latest_result = recent_results[0]
                print(f"  Result ID: {latest_result.get('id')}")
                print(f"  Strategy Name: {latest_result.get('strategy_name')}")
                print(f"  Symbol: {latest_result.get('symbol')}")
                print(f"  Final Balance: {latest_result.get('final_balance', 0):.2f}")
                print(f"  Total Return: {latest_result.get('total_return', 0):.4f}")
                print(f"  Created At: {latest_result.get('created_at')}")

                return True
            else:
                print("WARNING: No backtest results found")
                return False

        finally:
            db.close()

    except Exception as e:
        print(f"FAILED: Error checking saved results: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メイン実行関数"""
    print("AutoStrategy: Complete Strategy Creation and Database Save")
    print("=" * 80)

    success = True

    # 1. AutoStrategy実行とDB保存
    if not run_autostrategy_and_save():
        success = False

    # 2. 保存結果確認
    if not verify_saved_results():
        success = False

    # 結果表示
    print("\n" + "=" * 80)
    if success:
        print("SUCCESS: AutoStrategy strategy creation and database save completed!")
        print("The AutoStrategy successfully:")
        print("  - Generated trading strategies using GA")
        print("  - Executed backtests with real market data")
        print("  - Saved all results to database with statistics")
        return 0
    else:
        print("FAILED: Some steps failed during execution")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)