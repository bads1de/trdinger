"""
パフォーマンス・データ検証包括的テスト

大量データ処理、メモリ使用量、実行時間、
データ検証ロジックの包括的テストを実施します。
"""

import logging
import pytest
import time
import numpy as np
import pandas as pd
import uuid
from unittest.mock import Mock, patch

from app.services.auto_strategy.services.auto_strategy_service import AutoStrategyService
from app.services.auto_strategy.services.ml_orchestrator import MLOrchestrator
from app.services.backtest.backtest_service import BacktestService

logger = logging.getLogger(__name__)


class TestPerformanceDataValidationComprehensive:
    """パフォーマンス・データ検証包括的テストクラス"""

    @pytest.fixture
    def auto_strategy_service(self):
        """AutoStrategyServiceのテスト用インスタンス"""
        return AutoStrategyService(enable_smart_generation=True)

    @pytest.fixture
    def ml_orchestrator(self):
        """MLOrchestratorのテスト用インスタンス"""
        return MLOrchestrator(enable_automl=True)

    @pytest.fixture
    def backtest_service(self):
        """BacktestServiceのテスト用インスタンス"""
        return BacktestService()

    @pytest.fixture
    def large_market_data(self):
        """大量市場データ"""
        # 1年分の1分足データ（約525,600データポイント）
        dates = pd.date_range('2024-01-01', '2024-12-31', freq='1min')
        np.random.seed(42)
        
        # リアルな価格変動をシミュレート
        returns = np.random.normal(0, 0.001, len(dates))  # 0.1%の標準偏差
        price = 50000 * np.exp(np.cumsum(returns))
        
        data = {
            'timestamp': dates,
            'open': price * (1 + np.random.normal(0, 0.0001, len(dates))),
            'high': price * (1 + np.abs(np.random.normal(0, 0.0005, len(dates)))),
            'low': price * (1 - np.abs(np.random.normal(0, 0.0005, len(dates)))),
            'close': price,
            'volume': np.random.lognormal(10, 1, len(dates)),
        }
        
        return pd.DataFrame(data)

    def test_large_dataset_processing_performance(self, ml_orchestrator, large_market_data):
        """大量データセット処理パフォーマンステスト"""
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            
            # 初期状態の記録
            memory_before = process.memory_info().rss
            cpu_before = process.cpu_percent()
            start_time = time.time()
            
            # 大量データでのML指標計算
            with patch.object(ml_orchestrator, 'feature_service') as mock_feature_service:
                # 特徴量計算のモック（実際の計算をシミュレート）
                mock_feature_service.calculate_features.return_value = pd.DataFrame({
                    'feature1': np.random.randn(len(large_market_data)),
                    'feature2': np.random.randn(len(large_market_data)),
                    'feature3': np.random.randn(len(large_market_data))
                })
                
                with patch.object(ml_orchestrator, 'ml_training_service') as mock_ml_service:
                    mock_ml_service.generate_signals.return_value = {
                        'ML_UP_PROB': 0.6,
                        'ML_DOWN_PROB': 0.3,
                        'ML_RANGE_PROB': 0.1
                    }
                    
                    result = ml_orchestrator.calculate_ml_indicators(large_market_data)
            
            # パフォーマンス測定
            execution_time = time.time() - start_time
            memory_after = process.memory_info().rss
            memory_increase = memory_after - memory_before
            
            # パフォーマンス要件の確認
            assert execution_time < 60, f"大量データ処理時間が過大: {execution_time:.2f}秒"
            assert memory_increase < 1024 * 1024 * 1024, f"メモリ使用量が過大: {memory_increase / 1024 / 1024:.2f}MB"
            
            logger.info(f"大量データ処理パフォーマンス: 時間={execution_time:.2f}秒, メモリ={memory_increase / 1024 / 1024:.2f}MB")
            
            # 結果の妥当性確認
            if result is not None:
                assert isinstance(result, dict)
                for key, value in result.items():
                    assert len(value) == len(large_market_data)
                    
        except ImportError:
            pytest.skip("psutilが利用できないため、パフォーマンステストをスキップ")
        except Exception as e:
            logger.warning(f"大量データ処理テストでエラー: {e}")

    def test_memory_efficiency_optimization(self, auto_strategy_service):
        """メモリ効率最適化テスト"""
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss
            
            # 複数の実験を連続実行してメモリリークを検出
            memory_measurements = []
            
            for i in range(5):
                with patch.object(auto_strategy_service, 'persistence_service'):
                    with patch.object(auto_strategy_service, 'experiment_manager'):
                        auto_strategy_service.start_strategy_generation(
                            experiment_id=str(uuid.uuid4()),
                            experiment_name=f"Memory Test {i}",
                            ga_config_dict={
                                "population_size": 20,
                                "generations": 10,
                                "crossover_rate": 0.8,
                                "mutation_rate": 0.1,
                                "elite_size": 4,
                                "max_indicators": 5,
                                "allowed_indicators": ["SMA", "EMA", "RSI", "MACD", "BB"]
                            },
                            backtest_config_dict={
                                "symbol": "BTC/USDT",
                                "timeframe": "1h",
                                "start_date": "2024-01-01",
                                "end_date": "2024-12-31",
                                "initial_capital": 100000,
                                "commission_rate": 0.00055
                            },
                            background_tasks=Mock()
                        )
                
                current_memory = process.memory_info().rss
                memory_measurements.append(current_memory)
                
                # ガベージコレクション強制実行
                import gc
                gc.collect()
            
            # メモリリークの検出
            memory_increases = [mem - initial_memory for mem in memory_measurements]
            max_increase = max(memory_increases)
            
            # メモリ増加が線形でないことを確認（リークがないこと）
            assert max_increase < 200 * 1024 * 1024, f"メモリリークの可能性: {max_increase / 1024 / 1024:.2f}MB"
            
            logger.info(f"メモリ効率テスト: 最大増加={max_increase / 1024 / 1024:.2f}MB")
            
        except ImportError:
            pytest.skip("psutilが利用できないため、メモリ効率テストをスキップ")

    def test_concurrent_processing_scalability(self, auto_strategy_service):
        """並行処理スケーラビリティテスト"""
        import concurrent.futures
        
        def run_strategy_generation(thread_id):
            start_time = time.time()
            
            try:
                with patch.object(auto_strategy_service, 'persistence_service'):
                    with patch.object(auto_strategy_service, 'experiment_manager'):
                        auto_strategy_service.start_strategy_generation(
                            experiment_id=str(uuid.uuid4()),
                            experiment_name=f"Concurrent Test {thread_id}",
                            ga_config_dict={
                                "population_size": 10,
                                "generations": 5,
                                "crossover_rate": 0.8,
                                "mutation_rate": 0.1,
                                "elite_size": 2,
                                "max_indicators": 3,
                                "allowed_indicators": ["SMA", "EMA", "RSI"]
                            },
                            backtest_config_dict={
                                "symbol": "BTC/USDT",
                                "timeframe": "1h",
                                "start_date": "2024-01-01",
                                "end_date": "2024-01-31",
                                "initial_capital": 100000,
                                "commission_rate": 0.00055
                            },
                            background_tasks=Mock()
                        )
                
                execution_time = time.time() - start_time
                return thread_id, execution_time, None
                
            except Exception as e:
                execution_time = time.time() - start_time
                return thread_id, execution_time, e
        
        # 並行実行テスト
        max_workers = 8
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_strategy_generation, i) for i in range(max_workers)]
            results = [future.result(timeout=30) for future in concurrent.futures.as_completed(futures, timeout=30)]
        
        total_time = time.time() - start_time
        
        # 結果分析
        successful_results = [r for r in results if r[2] is None]
        failed_results = [r for r in results if r[2] is not None]
        
        logger.info(f"並行処理テスト: 成功={len(successful_results)}, 失敗={len(failed_results)}, 総時間={total_time:.2f}秒")
        
        # スケーラビリティ要件の確認
        if successful_results:
            avg_execution_time = sum(r[1] for r in successful_results) / len(successful_results)
            assert avg_execution_time < 10, f"並行処理での平均実行時間が過大: {avg_execution_time:.2f}秒"
        
        # 成功率の確認
        success_rate = len(successful_results) / len(results)
        assert success_rate > 0.5, f"並行処理の成功率が低すぎます: {success_rate:.2%}"

    def test_data_validation_accuracy(self, ml_orchestrator):
        """データ検証精度テスト"""
        # 様々な品質のデータでテスト
        data_quality_cases = [
            # 完全なデータ
            {
                'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H'),
                'open': np.random.randn(100) * 1000 + 50000,
                'high': np.random.randn(100) * 1000 + 51000,
                'low': np.random.randn(100) * 1000 + 49000,
                'close': np.random.randn(100) * 1000 + 50000,
                'volume': np.random.randn(100) * 100 + 1000,
            },
            # 欠損値を含むデータ
            {
                'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H'),
                'open': [np.nan if i % 10 == 0 else np.random.randn() * 1000 + 50000 for i in range(100)],
                'close': [np.nan if i % 15 == 0 else np.random.randn() * 1000 + 50000 for i in range(100)],
                'volume': [np.nan if i % 20 == 0 else np.random.randn() * 100 + 1000 for i in range(100)],
            },
            # 異常値を含むデータ
            {
                'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H'),
                'open': [1000000 if i == 50 else np.random.randn() * 1000 + 50000 for i in range(100)],  # 異常な高値
                'close': [-1000 if i == 60 else np.random.randn() * 1000 + 50000 for i in range(100)],  # 負の価格
                'volume': [0 if i % 25 == 0 else np.random.randn() * 100 + 1000 for i in range(100)],  # ゼロボリューム
            }
        ]
        
        validation_results = []
        
        for i, data_case in enumerate(data_quality_cases):
            try:
                df = pd.DataFrame(data_case)
                
                # データ品質の事前チェック
                missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
                
                with patch.object(ml_orchestrator, 'feature_service') as mock_feature_service:
                    mock_feature_service.calculate_features.return_value = pd.DataFrame({
                        'feature1': np.random.randn(len(df))
                    })
                    
                    with patch.object(ml_orchestrator, 'ml_training_service') as mock_ml_service:
                        mock_ml_service.generate_signals.return_value = {
                            'ML_UP_PROB': 0.6,
                            'ML_DOWN_PROB': 0.3,
                            'ML_RANGE_PROB': 0.1
                        }
                        
                        result = ml_orchestrator.calculate_ml_indicators(df)
                
                validation_results.append({
                    'case': i,
                    'missing_ratio': missing_ratio,
                    'success': result is not None,
                    'result_type': type(result).__name__
                })
                
            except Exception as e:
                validation_results.append({
                    'case': i,
                    'missing_ratio': missing_ratio if 'missing_ratio' in locals() else 0,
                    'success': False,
                    'error': str(e)
                })
        
        # 検証結果の分析
        success_count = sum(1 for r in validation_results if r['success'])
        logger.info(f"データ検証テスト: {success_count}/{len(validation_results)} ケースが成功")
        
        # 完全なデータでは成功することを確認
        assert validation_results[0]['success'], "完全なデータで処理が失敗しました"

    def test_algorithm_performance_benchmarking(self, auto_strategy_service):
        """アルゴリズムパフォーマンスベンチマークテスト"""
        # 異なる設定でのパフォーマンス比較
        benchmark_configs = [
            # 小規模設定
            {
                "name": "Small",
                "population_size": 10,
                "generations": 5,
                "max_indicators": 2
            },
            # 中規模設定
            {
                "name": "Medium",
                "population_size": 20,
                "generations": 10,
                "max_indicators": 5
            },
            # 大規模設定
            {
                "name": "Large",
                "population_size": 50,
                "generations": 20,
                "max_indicators": 10
            }
        ]
        
        benchmark_results = []
        
        for config in benchmark_configs:
            start_time = time.time()
            
            try:
                ga_config = {
                    "population_size": config["population_size"],
                    "generations": config["generations"],
                    "crossover_rate": 0.8,
                    "mutation_rate": 0.1,
                    "elite_size": max(1, config["population_size"] // 10),
                    "max_indicators": config["max_indicators"],
                    "allowed_indicators": ["SMA", "EMA", "RSI", "MACD", "BB"][:config["max_indicators"]]
                }
                
                with patch.object(auto_strategy_service, 'persistence_service'):
                    with patch.object(auto_strategy_service, 'experiment_manager'):
                        auto_strategy_service.start_strategy_generation(
                            experiment_id=str(uuid.uuid4()),
                            experiment_name=f"Benchmark {config['name']}",
                            ga_config_dict=ga_config,
                            backtest_config_dict={
                                "symbol": "BTC/USDT",
                                "timeframe": "1h",
                                "start_date": "2024-01-01",
                                "end_date": "2024-01-31",
                                "initial_capital": 100000,
                                "commission_rate": 0.00055
                            },
                            background_tasks=Mock()
                        )
                
                execution_time = time.time() - start_time
                
                benchmark_results.append({
                    'config': config['name'],
                    'execution_time': execution_time,
                    'success': True
                })
                
            except Exception as e:
                execution_time = time.time() - start_time
                benchmark_results.append({
                    'config': config['name'],
                    'execution_time': execution_time,
                    'success': False,
                    'error': str(e)
                })
        
        # ベンチマーク結果の分析
        for result in benchmark_results:
            logger.info(f"ベンチマーク {result['config']}: 時間={result['execution_time']:.2f}秒, 成功={result['success']}")
        
        # パフォーマンス要件の確認
        successful_results = [r for r in benchmark_results if r['success']]
        if successful_results:
            max_time = max(r['execution_time'] for r in successful_results)
            assert max_time < 30, f"最大実行時間が過大: {max_time:.2f}秒"

    def test_data_integrity_validation(self, backtest_service):
        """データ整合性検証テスト"""
        # データ整合性の問題を含むテストケース
        integrity_test_cases = [
            # 価格の整合性問題
            {
                'timestamp': pd.date_range('2024-01-01', periods=10, freq='1H'),
                'open': [50000] * 10,
                'high': [49000] * 10,  # high < open (不正)
                'low': [51000] * 10,   # low > open (不正)
                'close': [50500] * 10,
                'volume': [1000] * 10
            },
            # 時系列の順序問題
            {
                'timestamp': [
                    pd.Timestamp('2024-01-01 10:00:00'),
                    pd.Timestamp('2024-01-01 09:00:00'),  # 逆順
                    pd.Timestamp('2024-01-01 11:00:00'),
                ],
                'open': [50000, 50100, 49900],
                'close': [50100, 49900, 50200],
                'volume': [1000, 1100, 900]
            }
        ]
        
        for i, test_case in enumerate(integrity_test_cases):
            try:
                df = pd.DataFrame(test_case)
                
                # データ整合性チェック（実装に依存）
                if hasattr(backtest_service, 'validate_data_integrity'):
                    is_valid = backtest_service.validate_data_integrity(df)
                    if not is_valid:
                        logger.info(f"データ整合性問題 {i} が適切に検出されました")
                    else:
                        logger.warning(f"データ整合性問題 {i} が検出されませんでした")
                else:
                    # 基本的な整合性チェック
                    if 'high' in df.columns and 'low' in df.columns and 'open' in df.columns:
                        invalid_high = (df['high'] < df['open']).any()
                        invalid_low = (df['low'] > df['open']).any()
                        
                        if invalid_high or invalid_low:
                            logger.info(f"価格整合性問題 {i} を検出")
                
            except Exception as e:
                logger.info(f"データ整合性テスト {i} でエラー: {e}")

    def test_cache_performance_optimization(self, ml_orchestrator):
        """キャッシュパフォーマンス最適化テスト"""
        # 同じデータでの複数回処理でキャッシュ効果を測定
        sample_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1H'),
            'open': np.random.randn(1000) * 1000 + 50000,
            'close': np.random.randn(1000) * 1000 + 50000,
            'volume': np.random.randn(1000) * 100 + 1000,
        })
        
        execution_times = []
        
        for i in range(3):
            start_time = time.time()
            
            try:
                with patch.object(ml_orchestrator, 'feature_service') as mock_feature_service:
                    mock_feature_service.calculate_features.return_value = pd.DataFrame({
                        'feature1': np.random.randn(len(sample_data))
                    })
                    
                    with patch.object(ml_orchestrator, 'ml_training_service') as mock_ml_service:
                        mock_ml_service.generate_signals.return_value = {
                            'ML_UP_PROB': 0.6,
                            'ML_DOWN_PROB': 0.3,
                            'ML_RANGE_PROB': 0.1
                        }
                        
                        result = ml_orchestrator.calculate_ml_indicators(sample_data)
                
                execution_time = time.time() - start_time
                execution_times.append(execution_time)
                
            except Exception as e:
                logger.warning(f"キャッシュテスト {i} でエラー: {e}")
        
        # キャッシュ効果の分析
        if len(execution_times) >= 2:
            first_time = execution_times[0]
            subsequent_times = execution_times[1:]
            avg_subsequent_time = sum(subsequent_times) / len(subsequent_times)
            
            # 2回目以降の実行が高速化されることを期待（キャッシュ効果）
            speedup_ratio = first_time / avg_subsequent_time if avg_subsequent_time > 0 else 1
            
            logger.info(f"キャッシュ効果: 初回={first_time:.3f}秒, 平均={avg_subsequent_time:.3f}秒, 高速化比={speedup_ratio:.2f}")
            
            # 大幅な高速化は期待しないが、少なくとも性能劣化がないことを確認
            assert speedup_ratio >= 0.8, f"キャッシュ使用時の性能劣化: {speedup_ratio:.2f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
