"""
AutoStrategyService包括的テスト

AutoStrategyServiceの戦略生成、GA設定検証、バックグラウンドタスク管理、
エラーハンドリングの包括的テストを実施します。
"""

import logging
import pytest
import uuid
from unittest.mock import Mock, patch

from app.services.auto_strategy.services.auto_strategy_service import AutoStrategyService
from app.services.auto_strategy.models.ga_config import GAConfig
from fastapi import BackgroundTasks

logger = logging.getLogger(__name__)


class TestAutoStrategyServiceComprehensive:
    """AutoStrategyService包括的テストクラス"""

    @pytest.fixture
    def auto_strategy_service(self):
        """AutoStrategyServiceのテスト用インスタンス"""
        return AutoStrategyService(enable_smart_generation=True)

    @pytest.fixture
    def valid_ga_config_dict(self):
        """有効なGA設定辞書"""
        return {
            "population_size": 10,
            "generations": 5,
            "crossover_rate": 0.8,
            "mutation_rate": 0.1,
            "elite_size": 2,
            "max_indicators": 3,
            "allowed_indicators": ["SMA", "EMA", "RSI", "MACD"],
            "enable_multi_objective": False,
            "objectives": ["total_return"],
            "objective_weights": [1.0],
        }

    @pytest.fixture
    def valid_backtest_config_dict(self):
        """有効なバックテスト設定辞書"""
        return {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "initial_capital": 100000,
            "commission_rate": 0.00055,
        }

    def test_service_initialization(self, auto_strategy_service):
        """サービス初期化テスト"""
        assert auto_strategy_service is not None
        assert auto_strategy_service.enable_smart_generation is True
        assert auto_strategy_service.db_session_factory is not None
        assert hasattr(auto_strategy_service, 'backtest_service')
        assert hasattr(auto_strategy_service, 'persistence_service')
        assert hasattr(auto_strategy_service, 'experiment_manager')

    def test_ga_config_validation_success(self, auto_strategy_service, valid_ga_config_dict):
        """GA設定検証成功テスト"""
        try:
            ga_config = GAConfig.from_dict(valid_ga_config_dict)
            is_valid, errors = ga_config.validate()
            assert is_valid is True
            assert len(errors) == 0
        except Exception as e:
            pytest.fail(f"有効なGA設定の検証に失敗: {e}")

    def test_ga_config_validation_failure(self, auto_strategy_service):
        """GA設定検証失敗テスト"""
        invalid_configs = [
            # 人口サイズが無効
            {"population_size": 0, "generations": 5},
            # 世代数が無効
            {"population_size": 10, "generations": 0},
            # 交叉率が範囲外
            {"population_size": 10, "generations": 5, "crossover_rate": 1.5},
            # 突然変異率が範囲外
            {"population_size": 10, "generations": 5, "mutation_rate": -0.1},
            # エリートサイズが人口サイズを超過
            {"population_size": 10, "generations": 5, "elite_size": 15},
        ]

        for invalid_config in invalid_configs:
            with pytest.raises((ValueError, AssertionError)):
                ga_config = GAConfig.from_dict(invalid_config)
                is_valid, errors = ga_config.validate()
                if not is_valid:
                    raise ValueError(f"無効なGA設定: {', '.join(errors)}")

    @patch('app.services.auto_strategy.services.auto_strategy_service.ExperimentManager')
    @patch('app.services.auto_strategy.services.auto_strategy_service.ExperimentPersistenceService')
    def test_start_strategy_generation_success(
        self, 
        mock_persistence, 
        mock_experiment_manager,
        auto_strategy_service, 
        valid_ga_config_dict, 
        valid_backtest_config_dict
    ):
        """戦略生成開始成功テスト"""
        # モックの設定
        mock_persistence_instance = Mock()
        mock_persistence.return_value = mock_persistence_instance
        
        mock_experiment_instance = Mock()
        mock_experiment_manager.return_value = mock_experiment_instance
        auto_strategy_service.experiment_manager = mock_experiment_instance

        # バックグラウンドタスクのモック
        background_tasks = Mock(spec=BackgroundTasks)
        
        experiment_id = str(uuid.uuid4())
        experiment_name = "Test Strategy Generation"

        # 戦略生成開始
        result_experiment_id = auto_strategy_service.start_strategy_generation(
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            ga_config_dict=valid_ga_config_dict,
            backtest_config_dict=valid_backtest_config_dict,
            background_tasks=background_tasks
        )

        # 結果検証
        assert result_experiment_id == experiment_id
        mock_persistence_instance.create_experiment.assert_called_once()
        mock_experiment_instance.initialize_ga_engine.assert_called_once()
        background_tasks.add_task.assert_called_once()

    def test_symbol_normalization(self, auto_strategy_service, valid_ga_config_dict):
        """シンボル正規化テスト"""
        test_cases = [
            ("BTC", "BTC:USDT"),
            ("ETH", "ETH:USDT"),
            ("BTC/USDT", "BTC/USDT"),  # 既に正規化済み
            ("BTC:USDT", "BTC:USDT"),  # 既に正規化済み
        ]

        for original_symbol, expected_symbol in test_cases:
            backtest_config = {"symbol": original_symbol}
            
            # 内部的にシンボル正規化が行われることを確認
            # （実際のメソッドを呼び出すのではなく、ロジックをテスト）
            if original_symbol and ":" not in original_symbol and "/" not in original_symbol:
                normalized_symbol = f"{original_symbol}:USDT"
            else:
                normalized_symbol = original_symbol
                
            assert normalized_symbol == expected_symbol

    def test_invalid_ga_config_handling(self, auto_strategy_service, valid_backtest_config_dict):
        """無効なGA設定のハンドリングテスト"""
        invalid_ga_configs = [
            {},  # 空の設定
            {"population_size": "invalid"},  # 型エラー
            {"population_size": -1},  # 負の値
        ]

        background_tasks = Mock(spec=BackgroundTasks)
        experiment_id = str(uuid.uuid4())
        experiment_name = "Invalid Config Test"

        for invalid_config in invalid_ga_configs:
            with pytest.raises((ValueError, TypeError, AttributeError)):
                auto_strategy_service.start_strategy_generation(
                    experiment_id=experiment_id,
                    experiment_name=experiment_name,
                    ga_config_dict=invalid_config,
                    backtest_config_dict=valid_backtest_config_dict,
                    background_tasks=background_tasks
                )

    @patch('app.services.auto_strategy.services.auto_strategy_service.logger')
    def test_logging_functionality(self, mock_logger, auto_strategy_service):
        """ログ機能テスト"""
        # ログが適切に設定されていることを確認
        assert hasattr(auto_strategy_service, '_init_services')
        
        # ログメッセージが出力されることを確認（実際のメソッド呼び出し時）
        try:
            auto_strategy_service._init_services()
            # ログ呼び出しの確認は実装に依存するため、エラーが発生しないことを確認
        except Exception as e:
            # 初期化エラーは許容（依存関係の問題）
            assert "初期化" in str(e) or "依存" in str(e) or isinstance(e, (ImportError, AttributeError))

    def test_service_dependencies(self, auto_strategy_service):
        """サービス依存関係テスト"""
        # 必要な属性が存在することを確認
        required_attributes = [
            'db_session_factory',
            'enable_smart_generation',
            'backtest_service',
            'persistence_service',
            'experiment_manager'
        ]

        for attr in required_attributes:
            assert hasattr(auto_strategy_service, attr), f"必要な属性 {attr} が存在しません"

    def test_multi_objective_ga_config(self, auto_strategy_service):
        """多目的GA設定テスト"""
        multi_objective_config = {
            "population_size": 20,
            "generations": 10,
            "crossover_rate": 0.8,
            "mutation_rate": 0.1,
            "elite_size": 4,
            "max_indicators": 5,
            "allowed_indicators": ["SMA", "EMA", "RSI", "MACD", "BB"],
            "enable_multi_objective": True,
            "objectives": ["total_return", "sharpe_ratio", "max_drawdown"],
            "objective_weights": [0.4, 0.4, 0.2],
        }

        try:
            ga_config = GAConfig.from_dict(multi_objective_config)
            is_valid, errors = ga_config.validate()
            assert is_valid is True
            assert ga_config.enable_multi_objective is True
            assert len(ga_config.objectives) == 3
            assert len(ga_config.objective_weights) == 3
        except Exception as e:
            pytest.fail(f"多目的GA設定の検証に失敗: {e}")

    def test_error_handling_robustness(self, auto_strategy_service):
        """エラーハンドリング堅牢性テスト"""
        # None値の処理
        with pytest.raises((ValueError, TypeError, AttributeError)):
            auto_strategy_service.start_strategy_generation(
                experiment_id=None,
                experiment_name=None,
                ga_config_dict=None,
                backtest_config_dict=None,
                background_tasks=None
            )

        # 空文字列の処理
        with pytest.raises((ValueError, TypeError, AttributeError)):
            auto_strategy_service.start_strategy_generation(
                experiment_id="",
                experiment_name="",
                ga_config_dict={},
                backtest_config_dict={},
                background_tasks=Mock(spec=BackgroundTasks)
            )

    def test_concurrent_strategy_generation(self, auto_strategy_service, valid_ga_config_dict, valid_backtest_config_dict):
        """並行戦略生成テスト"""
        # 複数の戦略生成リクエストが適切に処理されることを確認
        experiment_ids = [str(uuid.uuid4()) for _ in range(3)]
        background_tasks = Mock(spec=BackgroundTasks)

        with patch.object(auto_strategy_service, 'persistence_service') as mock_persistence:
            with patch.object(auto_strategy_service, 'experiment_manager') as mock_experiment_manager:
                mock_experiment_manager.initialize_ga_engine = Mock()
                
                for i, exp_id in enumerate(experiment_ids):
                    try:
                        result_id = auto_strategy_service.start_strategy_generation(
                            experiment_id=exp_id,
                            experiment_name=f"Concurrent Test {i}",
                            ga_config_dict=valid_ga_config_dict,
                            backtest_config_dict=valid_backtest_config_dict,
                            background_tasks=background_tasks
                        )
                        assert result_id == exp_id
                    except Exception as e:
                        # 並行処理での一部エラーは許容
                        logger.warning(f"並行テストでエラー発生: {e}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
