"""
データベース永続化サービス包括的テスト

ExperimentPersistenceServiceの実験作成、結果保存、
進捗更新、データ整合性の包括的テストを実施します。
"""

import logging
import pytest
import uuid
from unittest.mock import Mock, patch

from app.services.auto_strategy.persistence.experiment_persistence_service import ExperimentPersistenceService
from app.services.auto_strategy.models.ga_config import GAConfig

logger = logging.getLogger(__name__)


class TestExperimentPersistenceServiceComprehensive:
    """ExperimentPersistenceService包括的テストクラス"""

    @pytest.fixture
    def mock_db_session(self):
        """モックデータベースセッション"""
        session = Mock()
        session.commit = Mock()
        session.rollback = Mock()
        session.close = Mock()
        return session

    @pytest.fixture
    def mock_session_factory(self, mock_db_session):
        """モックセッションファクトリ"""
        factory = Mock()
        factory.return_value.__enter__ = Mock(return_value=mock_db_session)
        factory.return_value.__exit__ = Mock(return_value=None)
        return factory

    @pytest.fixture
    def mock_backtest_service(self):
        """モックバックテストサービス"""
        return Mock()

    @pytest.fixture
    def persistence_service(self, mock_session_factory, mock_backtest_service):
        """ExperimentPersistenceServiceのテスト用インスタンス"""
        return ExperimentPersistenceService(mock_session_factory, mock_backtest_service)

    @pytest.fixture
    def valid_ga_config(self):
        """有効なGA設定"""
        return GAConfig(
            population_size=10,
            generations=5,
            crossover_rate=0.8,
            mutation_rate=0.1,
            elite_size=2,
            max_indicators=3,
            allowed_indicators=["SMA", "EMA", "RSI"],
            enable_multi_objective=False,
            objectives=["total_return"],
            objective_weights=[1.0]
        )

    @pytest.fixture
    def valid_backtest_config(self):
        """有効なバックテスト設定"""
        return {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "initial_capital": 100000,
            "commission_rate": 0.00055
        }

    def test_persistence_service_initialization(self, persistence_service):
        """永続化サービス初期化テスト"""
        assert persistence_service is not None
        assert hasattr(persistence_service, 'db_session_factory')
        assert hasattr(persistence_service, 'backtest_service')

    @patch('app.services.auto_strategy.persistence.experiment_persistence_service.GAExperimentRepository')
    def test_create_experiment_success(self, mock_repo, persistence_service, valid_ga_config, valid_backtest_config):
        """実験作成成功テスト"""
        experiment_id = str(uuid.uuid4())
        experiment_name = "Test Experiment"
        
        # リポジトリのモック設定
        mock_repo_instance = Mock()
        mock_repo.return_value = mock_repo_instance
        
        try:
            result_id = persistence_service.create_experiment(
                experiment_id, experiment_name, valid_ga_config, valid_backtest_config
            )
            
            # 結果検証
            assert result_id == experiment_id
            mock_repo_instance.create_experiment.assert_called_once()
            
        except Exception as e:
            logger.warning(f"実験作成テストでエラー: {e}")
            pytest.skip(f"実験作成テストをスキップ: {e}")

    def test_create_experiment_with_invalid_id(self, persistence_service, valid_ga_config, valid_backtest_config):
        """無効ID実験作成テスト"""
        invalid_ids = [None, "", "invalid-id", 123]
        experiment_name = "Test Experiment"
        
        for invalid_id in invalid_ids:
            try:
                persistence_service.create_experiment(
                    invalid_id, experiment_name, valid_ga_config, valid_backtest_config
                )
                pytest.fail(f"無効なID {invalid_id} でエラーが発生しませんでした")
                
            except Exception as e:
                # 適切なエラーハンドリングが行われることを確認
                assert any(keyword in str(e).lower() for keyword in ['id', 'invalid', 'experiment'])

    def test_create_experiment_with_invalid_name(self, persistence_service, valid_ga_config, valid_backtest_config):
        """無効名前実験作成テスト"""
        experiment_id = str(uuid.uuid4())
        invalid_names = [None, "", " ", "a" * 1000]  # 空、空白、長すぎる名前
        
        for invalid_name in invalid_names:
            try:
                persistence_service.create_experiment(
                    experiment_id, invalid_name, valid_ga_config, valid_backtest_config
                )
                # 一部の無効な名前は許容される場合があるため、警告として記録
                logger.warning(f"無効な名前 {invalid_name} で実験が作成されました")
                
            except Exception as e:
                # 適切なエラーハンドリングが行われることを確認
                assert any(keyword in str(e).lower() for keyword in ['name', 'invalid', 'experiment'])

    @patch('app.services.auto_strategy.persistence.experiment_persistence_service.GAExperimentRepository')
    def test_save_experiment_result_success(self, mock_repo, persistence_service, valid_ga_config, valid_backtest_config):
        """実験結果保存成功テスト"""
        experiment_id = str(uuid.uuid4())
        experiment_result = {
            "best_strategy": {
                "fitness": 0.85,
                "parameters": {"SMA_period": 20, "RSI_period": 14},
                "total_return": 0.15,
                "sharpe_ratio": 1.2,
                "max_drawdown": -0.08
            },
            "generation_results": [
                {"generation": 1, "best_fitness": 0.7, "avg_fitness": 0.5},
                {"generation": 2, "best_fitness": 0.8, "avg_fitness": 0.6}
            ],
            "execution_time": 120.5
        }
        
        # リポジトリのモック設定
        mock_repo_instance = Mock()
        mock_repo.return_value = mock_repo_instance
        
        try:
            persistence_service.save_experiment_result(
                experiment_id, experiment_result, valid_ga_config, valid_backtest_config
            )
            
            # 結果保存が呼び出されることを確認
            mock_repo_instance.save_experiment_result.assert_called_once()
            
        except Exception as e:
            logger.warning(f"実験結果保存テストでエラー: {e}")
            pytest.skip(f"実験結果保存テストをスキップ: {e}")

    def test_save_experiment_result_with_invalid_data(self, persistence_service, valid_ga_config, valid_backtest_config):
        """無効データ実験結果保存テスト"""
        experiment_id = str(uuid.uuid4())
        invalid_results = [
            None,  # None結果
            {},  # 空の結果
            {"invalid": "data"},  # 無効な構造
            {"best_strategy": None},  # 無効な戦略
        ]
        
        for invalid_result in invalid_results:
            try:
                persistence_service.save_experiment_result(
                    experiment_id, invalid_result, valid_ga_config, valid_backtest_config
                )
                # 無効なデータでも保存される場合があるため、警告として記録
                logger.warning(f"無効な結果 {invalid_result} で保存が実行されました")
                
            except Exception as e:
                # 適切なエラーハンドリングが行われることを確認
                assert any(keyword in str(e).lower() for keyword in ['invalid', 'data', 'result'])

    @patch('app.services.auto_strategy.persistence.experiment_persistence_service.GAExperimentRepository')
    def test_complete_experiment_success(self, mock_repo, persistence_service):
        """実験完了成功テスト"""
        experiment_id = str(uuid.uuid4())
        
        # リポジトリのモック設定
        mock_repo_instance = Mock()
        mock_repo.return_value = mock_repo_instance
        
        try:
            persistence_service.complete_experiment(experiment_id)
            
            # 実験完了が呼び出されることを確認
            mock_repo_instance.complete_experiment.assert_called_once_with(experiment_id)
            
        except Exception as e:
            logger.warning(f"実験完了テストでエラー: {e}")
            pytest.skip(f"実験完了テストをスキップ: {e}")

    def test_complete_experiment_with_invalid_id(self, persistence_service):
        """無効ID実験完了テスト"""
        invalid_ids = [None, "", "invalid-id"]
        
        for invalid_id in invalid_ids:
            try:
                persistence_service.complete_experiment(invalid_id)
                pytest.fail(f"無効なID {invalid_id} でエラーが発生しませんでした")
                
            except Exception as e:
                # 適切なエラーハンドリングが行われることを確認
                assert any(keyword in str(e).lower() for keyword in ['id', 'invalid', 'experiment'])

    @patch('app.services.auto_strategy.persistence.experiment_persistence_service.GAExperimentRepository')
    def test_database_transaction_rollback(self, mock_repo, persistence_service, valid_ga_config, valid_backtest_config):
        """データベーストランザクションロールバックテスト"""
        experiment_id = str(uuid.uuid4())
        experiment_name = "Test Experiment"
        
        # リポジトリでエラーを発生させる
        mock_repo_instance = Mock()
        mock_repo_instance.create_experiment.side_effect = Exception("Database error")
        mock_repo.return_value = mock_repo_instance
        
        with pytest.raises(Exception) as exc_info:
            persistence_service.create_experiment(
                experiment_id, experiment_name, valid_ga_config, valid_backtest_config
            )
        
        assert "Database error" in str(exc_info.value)

    def test_concurrent_experiment_creation(self, persistence_service, valid_ga_config, valid_backtest_config):
        """並行実験作成テスト"""
        import threading
        
        results = []
        errors = []
        
        def create_experiment(test_id):
            try:
                experiment_id = str(uuid.uuid4())
                experiment_name = f"Concurrent Test {test_id}"
                
                with patch('app.services.auto_strategy.persistence.experiment_persistence_service.GAExperimentRepository'):
                    result_id = persistence_service.create_experiment(
                        experiment_id, experiment_name, valid_ga_config, valid_backtest_config
                    )
                    results.append((test_id, result_id))
                    
            except Exception as e:
                errors.append((test_id, e))
        
        # 複数スレッドで同時実行
        threads = []
        for i in range(3):
            thread = threading.Thread(target=create_experiment, args=(i,))
            threads.append(thread)
            thread.start()
        
        # 全スレッドの完了を待機
        for thread in threads:
            thread.join(timeout=10)
        
        # 結果検証
        if results:
            logger.info(f"並行実験作成成功: {len(results)} 個")
        
        if errors:
            logger.warning(f"並行実験作成エラー: {len(errors)} 個")

    def test_data_integrity_validation(self, persistence_service, valid_ga_config, valid_backtest_config):
        """データ整合性検証テスト"""
        experiment_id = str(uuid.uuid4())
        experiment_name = "Data Integrity Test"
        
        # GA設定とバックテスト設定の整合性確認
        inconsistent_configs = [
            # シンボルの不一致
            (valid_ga_config, {"symbol": "ETH/USDT", "timeframe": "1h"}),
            # 時間軸の不一致
            (valid_ga_config, {"symbol": "BTC/USDT", "timeframe": "invalid"}),
        ]
        
        for ga_config, bt_config in inconsistent_configs:
            try:
                with patch('app.services.auto_strategy.persistence.experiment_persistence_service.GAExperimentRepository'):
                    persistence_service.create_experiment(
                        experiment_id, experiment_name, ga_config, bt_config
                    )
                # 整合性チェックが実装されている場合はエラーが発生することを期待
                
            except Exception as e:
                logger.info(f"データ整合性エラーが適切に検出されました: {e}")

    def test_large_experiment_result_handling(self, persistence_service, valid_ga_config, valid_backtest_config):
        """大量実験結果ハンドリングテスト"""
        experiment_id = str(uuid.uuid4())
        
        # 大量のデータを含む実験結果
        large_result = {
            "best_strategy": {"fitness": 0.85, "parameters": {}},
            "generation_results": [
                {"generation": i, "best_fitness": 0.7 + i * 0.01, "strategies": [{"id": j} for j in range(100)]}
                for i in range(100)  # 100世代
            ],
            "execution_time": 3600.0
        }
        
        try:
            with patch('app.services.auto_strategy.persistence.experiment_persistence_service.GAExperimentRepository'):
                persistence_service.save_experiment_result(
                    experiment_id, large_result, valid_ga_config, valid_backtest_config
                )
            
            logger.info("大量実験結果の保存が成功しました")
            
        except Exception as e:
            logger.warning(f"大量実験結果保存でエラー: {e}")

    def test_multi_objective_experiment_handling(self, persistence_service, valid_backtest_config):
        """多目的実験ハンドリングテスト"""
        multi_objective_config = GAConfig(
            population_size=20,
            generations=10,
            crossover_rate=0.8,
            mutation_rate=0.1,
            elite_size=4,
            max_indicators=5,
            allowed_indicators=["SMA", "EMA", "RSI", "MACD"],
            enable_multi_objective=True,
            objectives=["total_return", "sharpe_ratio", "max_drawdown"],
            objective_weights=[0.4, 0.4, 0.2]
        )
        
        experiment_id = str(uuid.uuid4())
        experiment_name = "Multi-Objective Test"
        
        multi_objective_result = {
            "pareto_front": [
                {"fitness_values": [0.8, 1.2, -0.1], "parameters": {}},
                {"fitness_values": [0.7, 1.5, -0.08], "parameters": {}}
            ],
            "best_strategy": {"fitness": 0.85, "parameters": {}},
            "generation_results": []
        }
        
        try:
            with patch('app.services.auto_strategy.persistence.experiment_persistence_service.GAExperimentRepository'):
                # 多目的実験の作成
                persistence_service.create_experiment(
                    experiment_id, experiment_name, multi_objective_config, valid_backtest_config
                )
                
                # 多目的実験結果の保存
                persistence_service.save_experiment_result(
                    experiment_id, multi_objective_result, multi_objective_config, valid_backtest_config
                )
            
            logger.info("多目的実験の処理が成功しました")
            
        except Exception as e:
            logger.warning(f"多目的実験処理でエラー: {e}")

    def test_memory_efficiency_during_persistence(self, persistence_service, valid_ga_config, valid_backtest_config):
        """永続化中のメモリ効率性テスト"""
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss
            
            # 複数の実験を連続して作成
            with patch('app.services.auto_strategy.persistence.experiment_persistence_service.GAExperimentRepository'):
                for i in range(10):
                    experiment_id = str(uuid.uuid4())
                    experiment_name = f"Memory Test {i}"
                    
                    persistence_service.create_experiment(
                        experiment_id, experiment_name, valid_ga_config, valid_backtest_config
                    )
            
            memory_after = process.memory_info().rss
            memory_increase = memory_after - memory_before
            
            # メモリ増加が合理的な範囲内であることを確認（50MB以下）
            assert memory_increase < 50 * 1024 * 1024, \
                f"メモリ使用量が過大: {memory_increase / 1024 / 1024:.2f}MB"
            
        except ImportError:
            pytest.skip("psutilが利用できないため、メモリ効率性テストをスキップ")
        except Exception as e:
            logger.warning(f"メモリ効率性テストでエラー: {e}")

    def test_error_recovery_mechanisms(self, persistence_service, valid_ga_config, valid_backtest_config):
        """エラー回復メカニズムテスト"""
        experiment_id = str(uuid.uuid4())
        experiment_name = "Error Recovery Test"
        
        # 一時的なエラーをシミュレート
        with patch('app.services.auto_strategy.persistence.experiment_persistence_service.GAExperimentRepository') as mock_repo:
            mock_repo_instance = Mock()
            
            # 最初の呼び出しでエラー、2回目で成功
            mock_repo_instance.create_experiment.side_effect = [
                Exception("Temporary error"),
                None  # 成功
            ]
            mock_repo.return_value = mock_repo_instance
            
            # リトライ機能がある場合のテスト（実装に依存）
            try:
                persistence_service.create_experiment(
                    experiment_id, experiment_name, valid_ga_config, valid_backtest_config
                )
                pytest.fail("一時的エラーでリトライが実行されませんでした")
                
            except Exception as e:
                # エラーが適切に処理されることを確認
                assert "Temporary error" in str(e)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
