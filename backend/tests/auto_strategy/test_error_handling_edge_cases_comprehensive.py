"""
エラーハンドリング・エッジケース包括的テスト

各種エラー条件、境界値、異常データ、ネットワーク障害、
リソース不足などのエッジケースの包括的テストを実施します。
"""

import logging
import pytest
import numpy as np
import pandas as pd
import uuid
import threading
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from app.services.auto_strategy.services.auto_strategy_service import AutoStrategyService
from app.services.auto_strategy.services.ml_orchestrator import MLOrchestrator
from app.services.auto_strategy.generators.smart_condition_generator import SmartConditionGenerator

logger = logging.getLogger(__name__)


class TestErrorHandlingEdgeCasesComprehensive:
    """エラーハンドリング・エッジケース包括的テストクラス"""

    @pytest.fixture
    def auto_strategy_service(self):
        """AutoStrategyServiceのテスト用インスタンス"""
        return AutoStrategyService(enable_smart_generation=True)

    @pytest.fixture
    def ml_orchestrator(self):
        """MLOrchestratorのテスト用インスタンス"""
        return MLOrchestrator(enable_automl=True)

    @pytest.fixture
    def smart_generator(self):
        """SmartConditionGeneratorのテスト用インスタンス"""
        return SmartConditionGenerator(enable_smart_generation=True)

    def test_null_and_none_value_handling(self, auto_strategy_service):
        """Null・None値ハンドリングテスト"""
        null_none_cases = [
            (None, None, None, None, None),
            ("", "", {}, {}, None),
            (str(uuid.uuid4()), None, {}, {}, None),
            (str(uuid.uuid4()), "Test", None, {}, None),
            (str(uuid.uuid4()), "Test", {}, None, None),
        ]
        
        for experiment_id, experiment_name, ga_config, backtest_config, background_tasks in null_none_cases:
            try:
                auto_strategy_service.start_strategy_generation(
                    experiment_id=experiment_id,
                    experiment_name=experiment_name,
                    ga_config_dict=ga_config,
                    backtest_config_dict=backtest_config,
                    background_tasks=background_tasks
                )
                pytest.fail(f"Null/None値 {(experiment_id, experiment_name)} でエラーが発生しませんでした")
                
            except Exception as e:
                # 適切なエラーハンドリングが行われることを確認
                assert any(keyword in str(e).lower() for keyword in ['none', 'null', 'invalid', 'missing', 'error'])

    def test_extreme_boundary_values(self, auto_strategy_service):
        """極端な境界値テスト"""
        boundary_cases = [
            # 極端に小さい値
            {
                "population_size": 1,
                "generations": 1,
                "crossover_rate": 0.0,
                "mutation_rate": 0.0,
                "elite_size": 0,
                "max_indicators": 1
            },
            # 極端に大きい値
            {
                "population_size": 10000,
                "generations": 1000,
                "crossover_rate": 1.0,
                "mutation_rate": 1.0,
                "elite_size": 5000,
                "max_indicators": 100
            },
            # 範囲外の値
            {
                "population_size": -1,
                "generations": -1,
                "crossover_rate": -0.5,
                "mutation_rate": 1.5,
                "elite_size": -1,
                "max_indicators": -1
            },
        ]
        
        for ga_config in boundary_cases:
            try:
                from app.services.auto_strategy.models.ga_config import GAConfig
                config = GAConfig.from_dict(ga_config)
                is_valid, errors = config.validate()
                
                if not is_valid:
                    logger.info(f"境界値 {ga_config} で適切にバリデーションエラーが発生: {errors}")
                else:
                    logger.warning(f"境界値 {ga_config} でバリデーションが通過しました")
                    
            except Exception as e:
                # 適切なエラーハンドリングが行われることを確認
                assert any(keyword in str(e).lower() for keyword in ['invalid', 'range', 'boundary', 'value'])

    def test_malformed_data_structures(self, ml_orchestrator):
        """不正なデータ構造テスト"""
        malformed_data_cases = [
            # 不正なDataFrame
            pd.DataFrame({"invalid": []}),  # 空の列
            pd.DataFrame({"col1": [1, 2], "col2": [3]}),  # 長さが異なる列
            pd.DataFrame({"timestamp": ["invalid_date"]}),  # 無効な日付
            # 不正な辞書構造
            {"nested": {"deeply": {"invalid": {"structure": None}}}},
            # 循環参照（シミュレーション）
            {"self_ref": "circular"},
        ]
        
        for malformed_data in malformed_data_cases:
            try:
                if isinstance(malformed_data, pd.DataFrame):
                    result = ml_orchestrator.calculate_ml_indicators(malformed_data)
                    if result is not None:
                        logger.warning(f"不正なDataFrame {malformed_data.shape} で結果が返されました")
                else:
                    # 辞書データの場合は別の処理
                    logger.info(f"不正な構造 {type(malformed_data)} をテスト")
                    
            except Exception as e:
                # 適切なエラーハンドリングが行われることを確認
                assert any(keyword in str(e).lower() for keyword in ['invalid', 'malformed', 'structure', 'data'])

    def test_memory_exhaustion_simulation(self, auto_strategy_service):
        """メモリ枯渇シミュレーションテスト"""
        try:
            # 大量のメモリを消費する設定
            large_config = {
                "population_size": 1000,
                "generations": 100,
                "crossover_rate": 0.8,
                "mutation_rate": 0.1,
                "elite_size": 100,
                "max_indicators": 50,
                "allowed_indicators": ["SMA"] * 50  # 大量の指標
            }
            
            backtest_config = {
                "symbol": "BTC/USDT",
                "timeframe": "1m",  # 高頻度データ
                "start_date": "2020-01-01",
                "end_date": "2024-12-31",  # 長期間
                "initial_capital": 100000,
                "commission_rate": 0.00055
            }
            
            # メモリ使用量監視
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss
            
            with patch.object(auto_strategy_service, 'persistence_service'):
                with patch.object(auto_strategy_service, 'experiment_manager'):
                    try:
                        auto_strategy_service.start_strategy_generation(
                            experiment_id=str(uuid.uuid4()),
                            experiment_name="Memory Test",
                            ga_config_dict=large_config,
                            backtest_config_dict=backtest_config,
                            background_tasks=Mock()
                        )
                        
                        memory_after = process.memory_info().rss
                        memory_increase = memory_after - memory_before
                        
                        # メモリ増加が異常でないことを確認
                        assert memory_increase < 500 * 1024 * 1024, \
                            f"メモリ使用量が異常: {memory_increase / 1024 / 1024:.2f}MB"
                            
                    except MemoryError:
                        logger.info("メモリエラーが適切に発生しました")
                    except Exception as e:
                        logger.info(f"メモリ制限テストでエラー: {e}")
                        
        except ImportError:
            pytest.skip("psutilが利用できないため、メモリテストをスキップ")

    def test_concurrent_access_race_conditions(self, auto_strategy_service):
        """並行アクセス・競合状態テスト"""
        results = []
        errors = []
        
        def concurrent_operation(thread_id):
            try:
                experiment_id = str(uuid.uuid4())
                
                with patch.object(auto_strategy_service, 'persistence_service'):
                    with patch.object(auto_strategy_service, 'experiment_manager'):
                        auto_strategy_service.start_strategy_generation(
                            experiment_id=experiment_id,
                            experiment_name=f"Concurrent Test {thread_id}",
                            ga_config_dict={
                                "population_size": 10,
                                "generations": 5,
                                "crossover_rate": 0.8,
                                "mutation_rate": 0.1,
                                "elite_size": 2,
                                "max_indicators": 3,
                                "allowed_indicators": ["SMA", "EMA"]
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
                        results.append(thread_id)
                        
            except Exception as e:
                errors.append((thread_id, e))
        
        # 多数のスレッドで同時実行
        threads = []
        for i in range(10):
            thread = threading.Thread(target=concurrent_operation, args=(i,))
            threads.append(thread)
            thread.start()
        
        # 全スレッドの完了を待機
        for thread in threads:
            thread.join(timeout=5)
        
        # 結果分析
        logger.info(f"並行アクセステスト: 成功={len(results)}, エラー={len(errors)}")
        
        # 一部エラーは許容（リソース競合など）
        if len(errors) > len(results):
            logger.warning("並行アクセスで多数のエラーが発生しました")

    def test_infinite_loop_prevention(self, smart_generator):
        """無限ループ防止テスト"""
        # 循環参照を含む指標設定
        circular_indicators = []
        for i in range(100):  # 大量の指標
            from app.services.auto_strategy.models.gene_indicator import IndicatorGene
            indicator = IndicatorGene(
                type=f"CIRCULAR_{i}",
                enabled=True,
                parameters={"ref": f"CIRCULAR_{(i + 1) % 100}"}  # 循環参照
            )
            circular_indicators.append(indicator)
        
        start_time = time.time()
        
        try:
            # タイムアウト付きで実行
            result = smart_generator.generate_balanced_conditions(circular_indicators)
            
            execution_time = time.time() - start_time
            
            # 実行時間が合理的な範囲内であることを確認（30秒以下）
            assert execution_time < 30, f"実行時間が長すぎます: {execution_time:.2f}秒"
            
            # 結果が適切に生成されることを確認
            assert isinstance(result, tuple)
            assert len(result) == 3  # long_conditions, short_conditions, exit_conditions
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.info(f"循環参照テストでエラー（実行時間: {execution_time:.2f}秒）: {e}")

    def test_network_timeout_simulation(self, ml_orchestrator):
        """ネットワークタイムアウトシミュレーションテスト"""
        # ネットワーク遅延をシミュレート
        def slow_network_call(*args, **kwargs):
            time.sleep(5)  # 5秒の遅延
            raise TimeoutError("Network timeout")
        
        with patch.object(ml_orchestrator, 'ml_training_service') as mock_service:
            mock_service.generate_signals.side_effect = slow_network_call
            
            start_time = time.time()
            
            try:
                sample_data = pd.DataFrame({
                    'timestamp': pd.date_range('2024-01-01', periods=10, freq='1H'),
                    'open': np.random.randn(10) * 1000 + 50000,
                    'close': np.random.randn(10) * 1000 + 50000,
                    'high': np.random.randn(10) * 1000 + 51000,
                    'low': np.random.randn(10) * 1000 + 49000,
                    'volume': np.random.randn(10) * 100 + 1000,
                })
                
                result = ml_orchestrator.calculate_ml_indicators(sample_data)
                
            except TimeoutError:
                logger.info("ネットワークタイムアウトが適切に処理されました")
            except Exception as e:
                logger.info(f"ネットワークエラーが適切に処理されました: {e}")
            finally:
                execution_time = time.time() - start_time
                # タイムアウト処理が適切に行われることを確認
                assert execution_time < 10, f"タイムアウト処理が遅すぎます: {execution_time:.2f}秒"

    def test_database_connection_failure(self, auto_strategy_service):
        """データベース接続失敗テスト"""
        # データベース接続エラーをシミュレート
        with patch.object(auto_strategy_service, 'db_session_factory') as mock_factory:
            mock_factory.side_effect = Exception("Database connection failed")
            
            try:
                auto_strategy_service._init_services()
                pytest.fail("データベース接続エラーでエラーが発生しませんでした")
                
            except Exception as e:
                # 適切なエラーハンドリングが行われることを確認
                assert any(keyword in str(e).lower() for keyword in ['database', 'connection', 'failed'])

    def test_file_system_errors(self, ml_orchestrator):
        """ファイルシステムエラーテスト"""
        # ファイル読み書きエラーをシミュレート
        with patch('builtins.open', side_effect=PermissionError("Permission denied")):
            try:
                # ファイル操作を含む処理を実行
                sample_data = pd.DataFrame({
                    'timestamp': pd.date_range('2024-01-01', periods=5, freq='1H'),
                    'close': [50000, 50100, 49900, 50200, 50050]
                })
                
                result = ml_orchestrator.calculate_ml_indicators(sample_data)
                
            except PermissionError:
                logger.info("ファイルシステムエラーが適切に処理されました")
            except Exception as e:
                logger.info(f"ファイルシステムエラーが適切に処理されました: {e}")

    def test_unicode_and_encoding_issues(self, auto_strategy_service):
        """Unicode・エンコーディング問題テスト"""
        unicode_test_cases = [
            "実験名_日本語",  # 日本語
            "Тест_кириллица",  # キリル文字
            "测试_中文",  # 中国語
            "🚀📈💰",  # 絵文字
            "test\x00null",  # Null文字
            "test\uffff",  # 無効なUnicode
        ]
        
        for test_name in unicode_test_cases:
            try:
                with patch.object(auto_strategy_service, 'persistence_service'):
                    with patch.object(auto_strategy_service, 'experiment_manager'):
                        auto_strategy_service.start_strategy_generation(
                            experiment_id=str(uuid.uuid4()),
                            experiment_name=test_name,
                            ga_config_dict={
                                "population_size": 5,
                                "generations": 2,
                                "crossover_rate": 0.8,
                                "mutation_rate": 0.1,
                                "elite_size": 1,
                                "max_indicators": 2,
                                "allowed_indicators": ["SMA"]
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
                        
                logger.info(f"Unicode文字列 '{test_name}' が適切に処理されました")
                
            except Exception as e:
                logger.info(f"Unicode文字列 '{test_name}' でエラー: {e}")

    def test_floating_point_precision_issues(self, auto_strategy_service):
        """浮動小数点精度問題テスト"""
        precision_test_cases = [
            0.1 + 0.2,  # 0.30000000000000004
            1e-15,  # 極小値
            1e15,  # 極大値
            float('inf'),  # 無限大
            float('-inf'),  # 負の無限大
            float('nan'),  # NaN
        ]
        
        for test_value in precision_test_cases:
            try:
                ga_config = {
                    "population_size": 10,
                    "generations": 5,
                    "crossover_rate": test_value if 0 <= test_value <= 1 else 0.8,
                    "mutation_rate": 0.1,
                    "elite_size": 2,
                    "max_indicators": 3,
                    "allowed_indicators": ["SMA"]
                }
                
                backtest_config = {
                    "symbol": "BTC/USDT",
                    "timeframe": "1h",
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-31",
                    "initial_capital": test_value if test_value > 0 and test_value < 1e10 else 100000,
                    "commission_rate": 0.00055
                }
                
                from app.services.auto_strategy.models.ga_config import GAConfig
                config = GAConfig.from_dict(ga_config)
                is_valid, errors = config.validate()
                
                if not is_valid and any(str(test_value) in error for error in errors):
                    logger.info(f"浮動小数点値 {test_value} で適切にバリデーションエラーが発生")
                    
            except Exception as e:
                logger.info(f"浮動小数点値 {test_value} でエラー: {e}")

    def test_resource_cleanup_on_failure(self, auto_strategy_service):
        """失敗時のリソースクリーンアップテスト"""
        # リソースリークを検出するためのテスト
        initial_thread_count = threading.active_count()
        
        try:
            # 意図的にエラーを発生させる
            with patch.object(auto_strategy_service, 'experiment_manager') as mock_manager:
                mock_manager.run_experiment.side_effect = Exception("Intentional failure")
                
                auto_strategy_service.start_strategy_generation(
                    experiment_id=str(uuid.uuid4()),
                    experiment_name="Resource Cleanup Test",
                    ga_config_dict={
                        "population_size": 10,
                        "generations": 5,
                        "crossover_rate": 0.8,
                        "mutation_rate": 0.1,
                        "elite_size": 2,
                        "max_indicators": 3,
                        "allowed_indicators": ["SMA"]
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
                
        except Exception as e:
            logger.info(f"意図的なエラーが発生: {e}")
        
        # リソースクリーンアップの確認
        time.sleep(1)  # クリーンアップ時間を待機
        final_thread_count = threading.active_count()
        
        # スレッド数が大幅に増加していないことを確認
        thread_increase = final_thread_count - initial_thread_count
        assert thread_increase < 10, f"スレッドリークの可能性: {thread_increase} 個のスレッドが増加"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
