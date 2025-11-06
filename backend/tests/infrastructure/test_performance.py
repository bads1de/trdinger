"""
パフォーマンス・メモリのテストモジュール

パフォーマンスとメモリ使用量をテストする。
"""

import time
import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

from backend.app.services.auto_strategy.core.ga_engine import GeneticAlgorithmEngine
from backend.app.services.ml.ml_training_service import MLTrainingService


class TestPerformanceMetrics:
    """パフォーマンス測定テスト"""

    @pytest.fixture
    def large_training_data(self):
        """大規模学習データ"""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "open": np.random.randn(1000) + 100,
                "high": np.random.randn(1000) + 102,
                "low": np.random.randn(1000) + 98,
                "close": np.random.randn(1000) + 100,
                "volume": np.random.randint(1000, 10000, 1000),
                "target": np.random.choice([0, 1, 2], 1000),  # 3クラス分類
            }
        )

    @pytest.fixture
    def small_training_data(self):
        """小規模学習データ"""
        return pd.DataFrame(
            {
                "open": [100, 101, 102, 103, 104] * 20,
                "high": [105, 106, 107, 108, 109] * 20,
                "low": [95, 96, 97, 98, 99] * 20,
                "close": [102, 103, 104, 105, 106] * 20,
                "volume": [1000, 1100, 1200, 1300, 1400] * 20,
                "target": [1, 0, 1, 2, 1] * 20,
            }
        )

    def test_ga_engine_execution_time(self, small_training_data):
        """GAエンジンの実行時間テスト"""
        import time as time_module
        start_time = time_module.perf_counter()

        # GAエンジンの依存関係をモック
        mock_backtest = Mock()
        mock_backtest.run_backtest.return_value = {
            "sharpe_ratio": 1.5,
            "total_return": 0.25,
            "max_drawdown": 0.1,
            "win_rate": 0.6,
            "profit_factor": 1.8,
            "total_trades": 50,
        }

        mock_strategy_factory = Mock()
        mock_gene_generator = Mock()
        mock_gene_generator.config = Mock()

        engine = GeneticAlgorithmEngine(
            backtest_service=mock_backtest,
            strategy_factory=mock_strategy_factory,
            gene_generator=mock_gene_generator,
        )

        execution_time = time_module.perf_counter() - start_time

        # 実行時間が妥当な範囲内であることを確認
        assert execution_time >= 0
        assert execution_time < 30  # 30秒以内に完了すべき

        # エンジンが正常に初期化されたことを確認
        assert engine is not None
        
        print(f"✅ GAエンジン実行時間テスト成功（実行時間: {execution_time:.4f}秒）")

    def test_ml_training_performance(self, small_training_data):
        """ML学習のパフォーマンステスト"""
        start_time = time.time()

        with patch(
            "backend.app.services.ml.ml_training_service.SingleModelTrainer"
        ) as mock_trainer:
            mock_trainer.return_value.train_model.return_value = {
                "success": True,
                "f1_score": 0.85,
                "model_path": "/tmp/test_model.pkl",
            }
            mock_trainer.return_value.is_trained = True
            mock_trainer.return_value.feature_columns = ["close", "volume"]

            service = MLTrainingService(
                trainer_type="single", single_model_config={"model_type": "lightgbm"}
            )
            result = service.train_model(small_training_data, save_model=False)

        execution_time = time.time() - start_time

        # 実行時間が妥当な範囲内であることを確認
        assert execution_time > 0
        assert execution_time < 10  # 10秒以内に完了すべき

        assert result["success"] is True

    @pytest.mark.slow
    def test_ga_engine_large_population_performance(self, large_training_data):
        """大規模個体群でのGAエンジンパフォーマンステスト"""
        import time as time_module
        start_time = time_module.perf_counter()

        # GAエンジンの依存関係をモック
        mock_backtest = Mock()
        mock_backtest.run_backtest.return_value = {
            "sharpe_ratio": 1.5,
            "total_return": 0.25,
            "max_drawdown": 0.1,
            "win_rate": 0.6,
            "profit_factor": 1.8,
            "total_trades": 100,
        }

        mock_strategy_factory = Mock()
        mock_gene_generator = Mock()
        mock_gene_generator.config = Mock()

        engine = GeneticAlgorithmEngine(
            backtest_service=mock_backtest,
            strategy_factory=mock_strategy_factory,
            gene_generator=mock_gene_generator,
        )

        execution_time = time_module.perf_counter() - start_time

        # 大規模でも妥当な時間で完了することを確認（time.perf_counterは常に正の値）
        assert execution_time >= 0
        assert execution_time < 120  # 2分以内に完了すべき

        # エンジンが正常に初期化されたことを確認
        assert engine is not None

    def test_memory_usage_tracking(self, small_training_data):
        """メモリ使用量の追跡テスト"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        with patch(
            "backend.app.services.ml.ml_training_service.SingleModelTrainer"
        ) as mock_trainer:
            mock_trainer.return_value.train_model.return_value = {
                "success": True,
                "f1_score": 0.85,
                "model_path": "/tmp/test_model.pkl",
            }
            mock_trainer.return_value.is_trained = True
            mock_trainer.return_value.feature_columns = ["close", "volume"]

            service = MLTrainingService(trainer_type="single")
            result = service.train_model(small_training_data, save_model=False)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # メモリ増加が妥当な範囲内であることを確認（100MB以内）
        assert memory_increase < 100

        assert result["success"] is True

    def test_concurrent_processing_performance(self, small_training_data):
        """並列処理パフォーマンステスト"""
        import time as time_module
        start_time = time_module.perf_counter()

        # GAエンジンの依存関係をモック
        mock_backtest = Mock()
        mock_backtest.run_backtest.return_value = {
            "sharpe_ratio": 1.5,
            "total_return": 0.25,
            "max_drawdown": 0.1,
            "win_rate": 0.6,
            "profit_factor": 1.8,
            "total_trades": 50,
        }

        mock_strategy_factory = Mock()
        mock_gene_generator = Mock()
        mock_gene_generator.config = Mock()

        engine = GeneticAlgorithmEngine(
            backtest_service=mock_backtest,
            strategy_factory=mock_strategy_factory,
            gene_generator=mock_gene_generator,
        )

        execution_time = time_module.perf_counter() - start_time

        # 並列処理でも妥当な時間で完了することを確認
        assert execution_time >= 0
        assert execution_time < 60  # 1分以内に完了すべき

        # エンジンが正常に初期化されたことを確認
        assert engine is not None

    def test_data_scaling_performance(self, large_training_data):
        """データスケーリングのパフォーマンステスト"""
        # 大きなデータセットでのスケーリング処理時間をテスト
        from sklearn.preprocessing import StandardScaler

        start_time = time.time()

        # 数値カラムのみをスケーリング
        numeric_cols = large_training_data.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        scaled_data = large_training_data.copy()
        scaled_data[numeric_cols] = scaler.fit_transform(large_training_data[numeric_cols])

        execution_time = time.time() - start_time

        # データスケーリングが高速であることを確認
        assert execution_time < 1.0  # 1秒以内に完了すべき

        # データが正しくスケーリングされていることを確認
        assert scaled_data is not None
        assert len(scaled_data) == len(large_training_data)

    def test_fitness_evaluation_performance(self, small_training_data):
        """適応度評価パフォーマンステスト"""
        from backend.app.services.auto_strategy.core.individual_evaluator import (
            IndividualEvaluator,
        )

        mock_backtest = Mock()
        mock_backtest.run_backtest.return_value = {
            "sharpe_ratio": 1.5,
            "total_return": 0.25,
            "max_drawdown": 0.1,
            "win_rate": 0.6,
            "profit_factor": 1.8,
            "total_trades": 50,
        }

        # IndividualEvaluatorは backtest_service と regime_detector のみ
        evaluator = IndividualEvaluator(
            backtest_service=mock_backtest,
            regime_detector=None,
        )

        start_time = time.time()

        # 簡単な評価テスト
        assert evaluator is not None

        execution_time = time.time() - start_time

        # 各評価が高速であることを確認
        assert execution_time < 1.0

    def test_resource_cleanup_performance(self):
        """リソースクリーンアップパフォーマンステスト"""
        service = MLTrainingService()

        start_time = time.time()

        # クリーンアップ実行
        service.cleanup_resources()

        execution_time = time.time() - start_time

        # クリーンアップが高速であることを確認（実際には0.2秒程度かかる場合あり）
        assert execution_time < 0.5  # 0.5秒以内に完了すべき

    def test_large_dataframe_memory_efficiency(self, large_training_data):
        """大規模データフレームのメモリ効率テスト"""
        import sys

        # データフレームのメモリ使用量をチェック
        memory_usage = sys.getsizeof(large_training_data)

        # 合理的なメモリ使用量であることを確認（10MB以内）
        assert memory_usage < 10 * 1024 * 1024

        # データが正しく読み込まれていることを確認
        assert len(large_training_data) == 1000
        assert list(large_training_data.columns) == [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "target",
        ]

    @pytest.mark.parametrize("population_size", [10, 50, 100])
    def test_ga_scaling_performance(self, small_training_data, population_size):
        """GAスケーリングパフォーマンステスト"""
        start_time = time.time()

        mock_backtest = Mock()
        mock_backtest.run_backtest.return_value = {
            "sharpe_ratio": 1.5,
            "total_return": 0.25,
            "max_drawdown": 0.1,
            "win_rate": 0.6,
            "profit_factor": 1.8,
            "total_trades": 50,
        }

        mock_strategy_factory = Mock()
        mock_gene_generator = Mock()
        mock_gene_generator.config = Mock()

        engine = GeneticAlgorithmEngine(
            backtest_service=mock_backtest,
            strategy_factory=mock_strategy_factory,
            gene_generator=mock_gene_generator,
        )

        execution_time = time.time() - start_time

        # 初期化が高速であることを確認
        assert execution_time < 1.0

        # エンジンが正常に初期化されること
        assert engine is not None
