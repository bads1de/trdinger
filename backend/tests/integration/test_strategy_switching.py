"""
戦略切り替え機能の包括的テスト

このファイルは以下をテストします：
- 複数の異なる戦略の順次実行
- 同一データセットでの異なる戦略の結果比較
- 戦略パラメータ変更時の結果変化検証
- 戦略切り替え時のメモリリークやデータ汚染の確認
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import gc
import psutil
import os
from unittest.mock import Mock

from backtest.engine.strategy_executor import StrategyExecutor
from backtest.engine.indicators import TechnicalIndicators


@pytest.mark.integration
@pytest.mark.backtest
@pytest.mark.strategy_switching
class TestStrategySwitching:
    """戦略切り替え機能テスト"""

    def create_btc_test_data(self, market_condition: str = "trending", days: int = 60):
        """
        異なる市場条件のBTCテストデータを作成
        
        Args:
            market_condition: "trending", "ranging", "volatile"
            days: データの日数
        """
        dates = pd.date_range('2024-01-01', periods=days, freq='1D')
        base_price = 50000
        
        if market_condition == "trending":
            # 上昇トレンド
            trend = np.linspace(0, 0.5, days)  # 50%上昇
            noise = np.random.normal(0, 0.02, days)  # 2%のノイズ
            returns = trend + noise
        elif market_condition == "ranging":
            # レンジ相場
            returns = np.random.normal(0, 0.015, days)  # 1.5%のボラティリティ
        elif market_condition == "volatile":
            # 高ボラティリティ
            returns = np.random.normal(0, 0.05, days)  # 5%のボラティリティ
        else:
            returns = np.random.normal(0.001, 0.02, days)  # デフォルト
        
        # 価格データの生成
        prices = [base_price]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        # OHLCV データの作成
        data = []
        for i, close_price in enumerate(prices[1:]):
            open_price = prices[i]
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.005)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.005)))
            volume = np.random.uniform(500, 1500)
            
            data.append({
                'Open': round(open_price, 2),
                'High': round(high_price, 2),
                'Low': round(low_price, 2),
                'Close': round(close_price, 2),
                'Volume': volume
            })
        
        return pd.DataFrame(data, index=dates)

    def get_strategy_configs(self):
        """テスト用の戦略設定を取得"""
        return {
            'sma_cross': {
                'name': 'SMA Cross Strategy',
                'indicators': [
                    {'name': 'SMA', 'params': {'period': 20}},
                    {'name': 'SMA', 'params': {'period': 50}}
                ],
                'entry_rules': [
                    {'condition': 'SMA(close, 20) > SMA(close, 50)'}
                ],
                'exit_rules': [
                    {'condition': 'SMA(close, 20) < SMA(close, 50)'}
                ]
            },
            'rsi_strategy': {
                'name': 'RSI Strategy',
                'indicators': [
                    {'name': 'RSI', 'params': {'period': 14}}
                ],
                'entry_rules': [
                    {'condition': 'RSI(close, 14) < 30'}
                ],
                'exit_rules': [
                    {'condition': 'RSI(close, 14) > 70'}
                ]
            },
            'macd_strategy': {
                'name': 'MACD Strategy',
                'indicators': [
                    {'name': 'MACD', 'params': {'fast_period': 12, 'slow_period': 26, 'signal_period': 9}}
                ],
                'entry_rules': [
                    {'condition': 'close > 50000'}  # 簡略化した条件
                ],
                'exit_rules': [
                    {'condition': 'close < 45000'}  # 簡略化した条件
                ]
            },
            'bollinger_strategy': {
                'name': 'Bollinger Bands Strategy',
                'indicators': [
                    {'name': 'BB', 'params': {'period': 20, 'std_dev': 2.0}}
                ],
                'entry_rules': [
                    {'condition': 'close < BB(close, 20, 2.0)[2]'}  # 下限突破で買い
                ],
                'exit_rules': [
                    {'condition': 'close > BB(close, 20, 2.0)[0]'}  # 上限到達で売り
                ]
            },
            'ema_cross': {
                'name': 'EMA Cross Strategy',
                'indicators': [
                    {'name': 'EMA', 'params': {'period': 12}},
                    {'name': 'EMA', 'params': {'period': 26}}
                ],
                'entry_rules': [
                    {'condition': 'EMA(close, 12) > EMA(close, 26)'}
                ],
                'exit_rules': [
                    {'condition': 'EMA(close, 12) < EMA(close, 26)'}
                ]
            }
        }

    def test_multiple_strategy_execution(self):
        """複数戦略の順次実行テスト"""
        data = self.create_btc_test_data("trending", days=90)
        strategies = self.get_strategy_configs()
        
        results = {}
        
        for strategy_name, strategy_config in strategies.items():
            executor = StrategyExecutor(initial_capital=100000, commission_rate=0.001)
            
            try:
                result = executor.run_backtest(data, strategy_config)
                results[strategy_name] = result
                
                # 基本的な結果検証
                assert result is not None, f"{strategy_name}: 結果がNone"
                assert 'total_return' in result, f"{strategy_name}: total_returnが存在しない"
                assert 'final_equity' in result, f"{strategy_name}: final_equityが存在しない"
                assert isinstance(result['total_return'], (int, float)), f"{strategy_name}: total_returnが数値でない"
                
            except Exception as e:
                pytest.fail(f"{strategy_name}の実行中にエラー: {e}")
        
        # 全戦略が実行されたことを確認
        assert len(results) == len(strategies), "一部の戦略が実行されていない"
        
        # 結果の妥当性チェック
        for strategy_name, result in results.items():
            final_equity = result['final_equity']
            assert final_equity > 0, f"{strategy_name}: 最終資産が0以下"
            
            total_return = result['total_return']
            expected_final_equity = 100000 * (1 + total_return)
            assert abs(final_equity - expected_final_equity) < 100, \
                f"{strategy_name}: 最終資産と総リターンの不整合"

    def test_same_data_different_strategies_comparison(self):
        """同一データセットでの異なる戦略結果比較"""
        # 3つの異なる市場条件でテスト
        market_conditions = ["trending", "ranging", "volatile"]
        strategies = self.get_strategy_configs()
        
        comparison_results = {}
        
        for condition in market_conditions:
            data = self.create_btc_test_data(condition, days=60)
            condition_results = {}
            
            for strategy_name, strategy_config in strategies.items():
                executor = StrategyExecutor(initial_capital=100000, commission_rate=0.001)
                result = executor.run_backtest(data, strategy_config)
                condition_results[strategy_name] = result
            
            comparison_results[condition] = condition_results
        
        # 市場条件ごとの戦略パフォーマンス分析
        for condition, results in comparison_results.items():
            returns = {name: result['total_return'] for name, result in results.items()}
            
            # 戦略間の結果が異なることを確認（同じ結果は異常）
            unique_returns = set(round(ret, 4) for ret in returns.values())
            assert len(unique_returns) > 1, f"{condition}: 全戦略が同じ結果（データ汚染の可能性）"
            
            # 各戦略の特性確認
            if condition == "trending":
                # トレンドフォロー戦略（SMA, EMA, MACD）が良好な結果を示すはず
                trend_strategies = ['sma_cross', 'ema_cross', 'macd_strategy']
                trend_returns = [returns[s] for s in trend_strategies if s in returns]
                if trend_returns:
                    avg_trend_return = np.mean(trend_returns)
                    assert avg_trend_return > -0.1, "トレンド相場でトレンドフォロー戦略が大幅マイナス"
            
            elif condition == "ranging":
                # 逆張り戦略（RSI, Bollinger）が相対的に良好な結果を示すはず
                mean_reversion_strategies = ['rsi_strategy', 'bollinger_strategy']
                mr_returns = [returns[s] for s in mean_reversion_strategies if s in returns]
                if mr_returns:
                    # レンジ相場では大きな損失は避けられるはず
                    for ret in mr_returns:
                        assert ret > -0.2, "レンジ相場で逆張り戦略が大幅マイナス"

    def test_strategy_parameter_variation(self):
        """戦略パラメータ変更時の結果変化検証"""
        data = self.create_btc_test_data("trending", days=60)
        
        # SMA戦略のパラメータバリエーション
        sma_variations = [
            {'short': 10, 'long': 30},
            {'short': 20, 'long': 50},
            {'short': 30, 'long': 70},
            {'short': 5, 'long': 15},   # 短期
            {'short': 50, 'long': 100}  # 長期
        ]
        
        sma_results = []
        
        for params in sma_variations:
            strategy_config = {
                'name': f'SMA Cross {params["short"]}/{params["long"]}',
                'indicators': [
                    {'name': 'SMA', 'params': {'period': params['short']}},
                    {'name': 'SMA', 'params': {'period': params['long']}}
                ],
                'entry_rules': [
                    {'condition': f'SMA(close, {params["short"]}) > SMA(close, {params["long"]})'}
                ],
                'exit_rules': [
                    {'condition': f'SMA(close, {params["short"]}) < SMA(close, {params["long"]})'}
                ]
            }
            
            executor = StrategyExecutor(initial_capital=100000, commission_rate=0.001)
            result = executor.run_backtest(data, strategy_config)
            sma_results.append((params, result))
        
        # パラメータ変更による結果の変化を確認
        returns = [result['total_return'] for _, result in sma_results]
        trade_counts = [result['total_trades'] for _, result in sma_results]
        
        # 異なるパラメータで異なる結果が得られることを確認
        assert len(set(round(ret, 4) for ret in returns)) > 1, "パラメータ変更で結果が変わらない"
        assert len(set(trade_counts)) > 1, "パラメータ変更で取引回数が変わらない"
        
        # 短期パラメータほど取引回数が多いことを確認
        short_term_trades = sma_results[3][1]['total_trades']  # 5/15
        long_term_trades = sma_results[4][1]['total_trades']   # 50/100
        
        if short_term_trades > 0 and long_term_trades >= 0:
            assert short_term_trades >= long_term_trades, "短期パラメータの方が取引回数が少ない"

    def test_memory_leak_detection(self):
        """戦略切り替え時のメモリリーク検出"""
        if not hasattr(psutil, 'Process'):
            pytest.skip("psutil not available for memory monitoring")
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        data = self.create_btc_test_data("trending", days=30)
        strategies = self.get_strategy_configs()
        
        # 複数回の戦略切り替えを実行
        for iteration in range(5):
            for strategy_name, strategy_config in strategies.items():
                executor = StrategyExecutor(initial_capital=100000, commission_rate=0.001)
                result = executor.run_backtest(data, strategy_config)
                
                # 明示的にオブジェクトを削除
                del executor
                del result
            
            # ガベージコレクションを強制実行
            gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # メモリリークの検証（200MB以下の増加を許容）
        assert memory_increase < 200, f"メモリリークの可能性: {memory_increase:.2f}MB増加"

    def test_data_contamination_detection(self):
        """データ汚染の検出テスト"""
        data = self.create_btc_test_data("trending", days=60)
        
        # 同じ戦略を複数回実行して結果が一致することを確認
        strategy_config = self.get_strategy_configs()['sma_cross']
        
        results = []
        for i in range(3):
            executor = StrategyExecutor(initial_capital=100000, commission_rate=0.001)
            result = executor.run_backtest(data, strategy_config)
            results.append(result)
        
        # 同じ条件での実行結果が一致することを確認
        base_result = results[0]
        for i, result in enumerate(results[1:], 1):
            assert abs(result['total_return'] - base_result['total_return']) < 1e-6, \
                f"実行{i+1}回目で結果が異なる（データ汚染の可能性）"
            assert result['total_trades'] == base_result['total_trades'], \
                f"実行{i+1}回目で取引回数が異なる"

    def test_strategy_isolation(self):
        """戦略間の分離テスト"""
        data = self.create_btc_test_data("trending", days=60)
        
        # 異なる初期資金で同じ戦略を実行
        strategy_config = self.get_strategy_configs()['sma_cross']
        
        capitals = [50000, 100000, 200000]
        results = []
        
        for capital in capitals:
            executor = StrategyExecutor(initial_capital=capital, commission_rate=0.001)
            result = executor.run_backtest(data, strategy_config)
            results.append((capital, result))
        
        # 初期資金が異なっても総リターン率は同じはず
        returns = [result['total_return'] for _, result in results]
        
        # 総リターン率の一致確認（小数点以下4桁まで）
        base_return = returns[0]
        for i, ret in enumerate(returns[1:], 1):
            assert abs(ret - base_return) < 1e-4, \
                f"初期資金{capitals[i]}で総リターン率が異なる: {ret} vs {base_return}"

    def test_concurrent_strategy_execution(self):
        """並行戦略実行テスト"""
        import threading
        import queue
        
        data = self.create_btc_test_data("trending", days=60)
        strategies = self.get_strategy_configs()
        
        results_queue = queue.Queue()
        threads = []
        
        def run_strategy(strategy_name, strategy_config):
            try:
                executor = StrategyExecutor(initial_capital=100000, commission_rate=0.001)
                result = executor.run_backtest(data, strategy_config)
                results_queue.put((strategy_name, result, None))
            except Exception as e:
                results_queue.put((strategy_name, None, e))
        
        # 並行実行
        for strategy_name, strategy_config in strategies.items():
            thread = threading.Thread(
                target=run_strategy,
                args=(strategy_name, strategy_config)
            )
            threads.append(thread)
            thread.start()
        
        # 全スレッドの完了を待機
        for thread in threads:
            thread.join()
        
        # 結果の収集と検証
        results = {}
        errors = {}
        
        while not results_queue.empty():
            strategy_name, result, error = results_queue.get()
            if error:
                errors[strategy_name] = error
            else:
                results[strategy_name] = result
        
        # エラーがないことを確認
        assert len(errors) == 0, f"並行実行でエラー発生: {errors}"
        
        # 全戦略が正常に実行されたことを確認
        assert len(results) == len(strategies), "一部の戦略が並行実行で失敗"
        
        # 結果の妥当性確認
        for strategy_name, result in results.items():
            assert result is not None, f"{strategy_name}: 並行実行で結果がNone"
            assert 'total_return' in result, f"{strategy_name}: 並行実行で結果が不完全"

    def test_strategy_performance_consistency(self):
        """戦略パフォーマンスの一貫性テスト"""
        # 同じ市場条件で複数回テストして結果の一貫性を確認
        np.random.seed(42)  # 再現性のためのシード固定
        
        data = self.create_btc_test_data("trending", days=60)
        strategy_config = self.get_strategy_configs()['sma_cross']
        
        # 複数回実行
        results = []
        for i in range(5):
            executor = StrategyExecutor(initial_capital=100000, commission_rate=0.001)
            result = executor.run_backtest(data, strategy_config)
            results.append(result)
        
        # 結果の一貫性確認
        returns = [result['total_return'] for result in results]
        trade_counts = [result['total_trades'] for result in results]
        
        # 全ての実行で同じ結果が得られることを確認
        assert len(set(round(ret, 6) for ret in returns)) == 1, "実行ごとに結果が異なる"
        assert len(set(trade_counts)) == 1, "実行ごとに取引回数が異なる"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
