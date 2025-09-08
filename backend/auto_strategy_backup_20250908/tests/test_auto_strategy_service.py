import pytest
from unittest.mock import Mock, patch, MagicMock
from backend.app.services.auto_strategy.services.auto_strategy_service import AutoStrategyService


class TestAutoStrategyService:
    """AutoStrategyServiceの単体テスト"""

    @pytest.fixture
    def mock_db_session(self):
        """モックデータベースセッション"""
        session = MagicMock()
        return session

    @pytest.fixture
    def mock_backtest_service(self):
        """モックバックテストサービス"""
        return MagicMock()

    @pytest.fixture
    def mock_experiment_manager(self):
        """モック実験マネージャー"""
        return MagicMock()

    @pytest.fixture
    def mock_persistence_service(self):
        """モック永続化サービス"""
        persistence = MagicMock()
        persistence.create_experiment.return_value = None
        persistence.list_experiments.return_value = []
        return persistence

    @patch('backend.app.services.auto_strategy.services.auto_strategy_service.SessionLocal')
    @patch('backend.app.services.auto_strategy.services.auto_strategy_service.BacktestService')
    @patch('backend.app.services.auto_strategy.services.auto_strategy_service.ExperimentPersistenceService')
    @patch('backend.app.services.auto_strategy.services.auto_strategy_service.ExperimentManager')
    def test_init_services_initialization(self,
                                        mock_experiment_manager_class,
                                        mock_persistence_service_class,
                                        mock_backtest_service_class,
                                        mock_session_factory,
                                        mock_db_session,
                                        mock_backtest_service,
                                        mock_experiment_manager,
                                        mock_persistence_service):
        """サービスの初期化テスト"""
        # モックの設定
        mock_session_factory.return_value = mock_db_session
        mock_backtest_service_class.return_value = mock_backtest_service
        mock_persistence_service_class.return_value = mock_persistence_service
        mock_experiment_manager_class.return_value = mock_experiment_manager

        # サービスインスタンス作成
        service = AutoStrategyService()

        # アサーション：プロパティが設定されていること
        assert service.backtest_service == mock_backtest_service
        assert service.persistence_service == mock_persistence_service
        assert service.experiment_manager == mock_experiment_manager

    @patch('backend.app.services.auto_strategy.services.auto_strategy_service.SessionLocal')
    @patch('backend.app.services.auto_strategy.services.auto_strategy_service.ExperimentPersistenceService')
    @patch('backend.app.services.auto_strategy.services.auto_strategy_service.ExperimentManager')
    def test_start_strategy_generation_success(self,
                                             mock_experiment_manager_class,
                                             mock_persistence_service_class,
                                             mock_session_factory,
                                             mock_experiment_manager,
                                             mock_persistence_service):
        """戦略生成開始テスト - 成功ケース"""
        # モックの設定
        from backend.app.services.auto_strategy.config.auto_strategy_config import GAConfig

        mock_session_factory.return_value.__enter__.return_value = MagicMock()
        mock_session_factory.return_value.__exit__.return_value = None
        mock_persistence_service_class.return_value = mock_persistence_service
        mock_experiment_manager_class.return_value = mock_experiment_manager

        # GAConfigモック
        mock_ga_config = Mock(spec=GAConfig)
        mock_ga_config.validate.return_value = (True, [])
        mock_ga_config.from_dict.return_value = mock_ga_config

        # パラメータ設定
        experiment_id = "test-experiment"
        experiment_name = "テスト実験"
        ga_config_dict = {"population_size": 10}
        backtest_config_dict = {"symbol": "BTC/USDT"}

        with patch('backend.app.services.auto_strategy.services.auto_strategy_service.GAConfig', mock_ga_config):
            # サービス作成
            service = AutoStrategyService()

            # BackgroundTasksモック
            background_tasks = MagicMock()

            # 実行
            result = service.start_strategy_generation(
                experiment_id=experiment_id,
                experiment_name=experiment_name,
                ga_config_dict=ga_config_dict,
                backtest_config_dict=backtest_config_dict,
                background_tasks=background_tasks
            )

            # アサーション
            assert result == experiment_id
            mock_persistence_service.create_experiment.assert_called_once()
            mock_experiment_manager.initialize_ga_engine.assert_called_once()
            background_tasks.add_task.assert_called_once()

    @patch('backend.app.services.auto_strategy.services.auto_strategy_service.SessionLocal')
    @patch('backend.app.services.auto_strategy.services.auto_strategy_service.ExperimentPersistenceService')
    @patch('backend.app.services.auto_strategy.services.auto_strategy_service.ExperimentManager')
    def test_start_strategy_generation_invalid_ga_config(self,
                                                       mock_experiment_manager_class,
                                                       mock_persistence_service_class,
                                                       mock_session_factory,
                                                       mock_experiment_manager,
                                                       mock_persistence_service):
        """戦略生成開始テスト - GA設定無効ケース"""
        # モックの設定
        from backend.app.services.auto_strategy.config.auto_strategy_config import GAConfig

        mock_session_factory.return_value.__enter__.return_value = MagicMock()
        mock_session_factory.return_value.__exit__.return_value = None
        mock_persistence_service_class.return_value = mock_persistence_service
        mock_experiment_manager_class.return_value = mock_experiment_manager

        # GAConfigモック - 検証失敗
        mock_ga_config = Mock(spec=GAConfig)
        mock_ga_config.validate.return_value = (False, ["無効な設定"])
        mock_ga_config.from_dict.return_value = mock_ga_config

        with patch('backend.app.services.auto_strategy.services.auto_strategy_service.GAConfig', mock_ga_config):
            service = AutoStrategyService()

            background_tasks = MagicMock()

            with pytest.raises(ValueError, match="無効なGA設定"):
                service.start_strategy_generation(
                    experiment_id="test",
                    experiment_name="テスト",
                    ga_config_dict={},
                    backtest_config_dict={},
                    background_tasks=background_tasks
                )

    @patch('backend.app.services.auto_strategy.services.auto_strategy_service.SessionLocal')
    @patch('backend.app.services.auto_strategy.services.auto_strategy_service.ExperimentPersistenceService')
    @patch('backend.app.services.auto_strategy.services.auto_strategy_service.ExperimentManager')
    def test_list_experiments(self,
                            mock_experiment_manager_class,
                            mock_persistence_service_class,
                            mock_session_factory,
                            mock_experiment_manager,
                            mock_persistence_service):
        """実験一覧取得テスト"""
        mock_session_factory.return_value.__enter__.return_value = MagicMock()
        mock_session_factory.return_value.__exit__.return_value = None
        mock_persistence_service_class.return_value = mock_persistence_service
        mock_experiment_manager_class.return_value = mock_experiment_manager

        # 永続化サービスが返すデータ
        mock_persistence_service.list_experiments.return_value = [
            {"id": "exp1", "name": "実験1"},
            {"id": "exp2", "name": "実験2"}
        ]

        service = AutoStrategyService()

        result = service.list_experiments()

        assert result == [
            {"id": "exp1", "name": "実験1"},
            {"id": "exp2", "name": "実験2"}
        ]
        mock_persistence_service.list_experiments.assert_called_once()

    @patch('backend.app.services.auto_strategy.services.auto_strategy_service.SessionLocal')
    @patch('backend.app.services.auto_strategy.services.auto_strategy_service.ExperimentPersistenceService')
    @patch('backend.app.services.auto_strategy.services.auto_strategy_service.ExperimentManager')
    def test_stop_experiment_success(self,
                                    mock_experiment_manager_class,
                                    mock_persistence_service_class,
                                    mock_session_factory,
                                    mock_experiment_manager,
                                    mock_persistence_service):
        """実験停止テスト - 成功ケース"""
        mock_session_factory.return_value.__enter__.return_value = MagicMock()
        mock_session_factory.return_value.__exit__.return_value = None
        mock_persistence_service_class.return_value = mock_persistence_service
        mock_experiment_manager_class.return_value = mock_experiment_manager

        # マネージャーがTrueを返す
        mock_experiment_manager.stop_experiment.return_value = True

        service = AutoStrategyService()

        result = service.stop_experiment("experiment-id")

        assert result["success"] == True
        assert "正常に停止されました" in result["message"]
        mock_experiment_manager.stop_experiment.assert_called_once_with("experiment-id")

    @patch('backend.app.services.auto_strategy.services.auto_strategy_service.SessionLocal')
    @patch('backend.app.services.auto_strategy.services.auto_strategy_service.ExperimentPersistenceService')
    @patch('backend.app.services.auto_strategy.services.auto_strategy_service.ExperimentManager')
    def test_stop_experiment_failure(self,
                                    mock_experiment_manager_class,
                                    mock_persistence_service_class,
                                    mock_session_factory,
                                    mock_experiment_manager,
                                    mock_persistence_service):
        """実験停止テスト - 失敗ケース"""
        mock_session_factory.return_value.__enter__.return_value = MagicMock()
        mock_session_factory.return_value.__exit__.return_value = None
        mock_persistence_service_class.return_value = mock_persistence_service
        mock_experiment_manager_class.return_value = mock_experiment_manager

        # マネージャーがFalseを返す
        mock_experiment_manager.stop_experiment.return_value = False

        service = AutoStrategyService()

        result = service.stop_experiment("experiment-id")

        assert result["success"] == False
        assert "失敗しました" in result["message"]