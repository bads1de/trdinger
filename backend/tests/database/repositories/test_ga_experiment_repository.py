"""
GAExperimentRepositoryのテストモジュール

GA実験リポジトリの機能をテストします。
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from sqlalchemy.orm import Session

from database.models import GAExperiment
from database.repositories.ga_experiment_repository import GAExperimentRepository


@pytest.fixture
def mock_session() -> MagicMock:
    """モックDBセッション"""
    session = MagicMock(spec=Session)
    session.add = MagicMock()
    session.commit = MagicMock()
    session.refresh = MagicMock()
    session.query = MagicMock()
    session.scalars = MagicMock()
    return session


@pytest.fixture
def repository(mock_session: MagicMock) -> GAExperimentRepository:
    """GAExperimentRepositoryインスタンス"""
    return GAExperimentRepository(mock_session)


@pytest.fixture
def sample_experiment_model() -> GAExperiment:
    """サンプルGAExperimentモデル"""
    return GAExperiment(
        id=1,
        name="test_experiment",
        config={"population": 50, "generations": 100},
        status="running",
        progress=0.5,
        best_fitness=0.85,
        total_generations=100,
        current_generation=50,
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )


class TestRepositoryInitialization:
    """リポジトリ初期化のテスト"""

    def test_repository_initialization(self, mock_session: MagicMock) -> None:
        """リポジトリが正しく初期化される"""
        repo = GAExperimentRepository(mock_session)
        assert repo.db == mock_session
        assert repo.model_class == GAExperiment


class TestCreateExperiment:
    """create_experimentメソッドのテスト"""

    def test_create_experiment_success(
        self, repository: GAExperimentRepository, sample_experiment_model: GAExperiment
    ) -> None:
        """実験が正常に作成される"""
        repository.db.add = MagicMock()
        repository.db.commit = MagicMock()
        repository.db.refresh = MagicMock(side_effect=lambda x: setattr(x, "id", 1))

        experiment = repository.create_experiment(
            name="test_experiment",
            config={"population": 50},
            total_generations=100,
        )

        repository.db.add.assert_called_once()
        repository.db.commit.assert_called_once()
        assert experiment.name == "test_experiment"

    def test_create_experiment_with_custom_status(
        self, repository: GAExperimentRepository
    ) -> None:
        """カスタムステータスで実験が作成される"""
        repository.db.add = MagicMock()
        repository.db.commit = MagicMock()
        repository.db.refresh = MagicMock()

        experiment = repository.create_experiment(
            name="test_experiment",
            config={},
            total_generations=100,
            status="pending",
        )

        assert experiment.status == "pending"


class TestUpdateExperimentStatus:
    """update_experiment_statusメソッドのテスト"""

    def test_update_experiment_status_success(
        self, repository: GAExperimentRepository, sample_experiment_model: GAExperiment
    ) -> None:
        """実験ステータスが更新される"""
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [sample_experiment_model]
        repository.db.scalars.return_value = mock_scalars

        result = repository.update_experiment_status(1, "completed")

        assert result is True
        repository.db.commit.assert_called_once()

    def test_update_experiment_status_not_found(
        self, repository: GAExperimentRepository
    ) -> None:
        """存在しない実験IDの場合Falseが返される"""
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        repository.db.scalars.return_value = mock_scalars

        result = repository.update_experiment_status(999, "completed")

        assert result is False

    def test_update_experiment_status_with_completed_at(
        self, repository: GAExperimentRepository, sample_experiment_model: GAExperiment
    ) -> None:
        """完了時刻付きでステータスが更新される"""
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [sample_experiment_model]
        repository.db.scalars.return_value = mock_scalars

        completed_at = datetime(2024, 1, 2, tzinfo=timezone.utc)
        result = repository.update_experiment_status(1, "completed", completed_at)

        assert result is True


class TestGetExperimentsByStatus:
    """get_experiments_by_statusメソッドのテスト"""

    def test_get_experiments_by_status_success(
        self, repository: GAExperimentRepository, sample_experiment_model: GAExperiment
    ) -> None:
        """ステータス別で実験が取得できる"""
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [sample_experiment_model]
        repository.db.scalars.return_value = mock_scalars

        results = repository.get_experiments_by_status("running")

        assert len(results) == 1
        assert results[0].status == "running"

    def test_get_experiments_by_status_with_limit(
        self, repository: GAExperimentRepository
    ) -> None:
        """リミット指定で実験が取得できる"""
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        repository.db.scalars.return_value = mock_scalars

        repository.get_experiments_by_status("running", limit=10)

        repository.db.scalars.assert_called_once()


class TestGetRecentExperiments:
    """get_recent_experimentsメソッドのテスト"""

    def test_get_recent_experiments_success(
        self, repository: GAExperimentRepository, sample_experiment_model: GAExperiment
    ) -> None:
        """最近の実験が取得できる"""
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [sample_experiment_model]
        repository.db.scalars.return_value = mock_scalars

        results = repository.get_recent_experiments(limit=5)

        assert len(results) == 1


class TestCountByStatus:
    """count_by_statusメソッドのテスト"""

    def test_count_by_status_success(self, repository: GAExperimentRepository) -> None:
        """ステータス別の実験数が取得できる"""
        repository.db.query.return_value.filter.return_value.count.return_value = 3

        count = repository.count_by_status("running")

        assert count == 3

    def test_count_by_status_zero(self, repository: GAExperimentRepository) -> None:
        """該当ステータスが無い場合は0が返る"""
        repository.db.query.return_value.filter.return_value.count.return_value = 0

        count = repository.count_by_status("failed")

        assert count == 0


class TestReconcileStaleRunningExperiments:
    """reconcile_stale_running_experimentsメソッドのテスト"""

    def test_reconcile_stops_running_experiments(
        self, repository: GAExperimentRepository
    ) -> None:
        """running状態の実験がstoppedへ巻き戻される"""
        running_exp = GAExperiment(
            id=1,
            name="stale",
            config={},
            status="running",
            progress=0.5,
            total_generations=10,
            current_generation=3,
        )
        repository.db.query.return_value.filter.return_value.all.return_value = [
            running_exp
        ]

        count = repository.reconcile_stale_running_experiments()

        assert count == 1
        assert running_exp.status == "stopped"
        assert running_exp.error_message is not None
        assert running_exp.completed_at is not None
        repository.db.commit.assert_called_once()

    def test_reconcile_no_running_experiments(
        self, repository: GAExperimentRepository
    ) -> None:
        """running実験が無い場合は0が返りコミットされない"""
        repository.db.query.return_value.filter.return_value.all.return_value = []

        count = repository.reconcile_stale_running_experiments()

        assert count == 0
        repository.db.commit.assert_not_called()


class TestCompleteExperiment:
    """complete_experimentメソッドのテスト"""

    def test_complete_experiment_success(
        self, repository: GAExperimentRepository
    ) -> None:
        """実験が完了状態になる"""
        mock_experiment = MagicMock(spec=GAExperiment)
        mock_query = MagicMock()
        mock_query.filter.return_value.first.return_value = mock_experiment
        repository.db.query.return_value = mock_query

        result = repository.complete_experiment(1, 0.90, 100)

        assert result is True
        repository.db.commit.assert_called_once()

    def test_complete_experiment_not_found(
        self, repository: GAExperimentRepository
    ) -> None:
        """存在しない実験の場合Falseが返される"""
        mock_query = MagicMock()
        mock_query.filter.return_value.first.return_value = None
        repository.db.query.return_value = mock_query

        result = repository.complete_experiment(999, 0.90, 100)

        assert result is False


class TestDeleteAllExperiments:
    """delete_all_experimentsメソッドのテスト"""

    def test_delete_all_experiments_success(
        self, repository: GAExperimentRepository
    ) -> None:
        """全実験が削除される"""
        mock_query = MagicMock()
        mock_query.delete.return_value = 5
        repository.db.query.return_value = mock_query

        count = repository.delete_all_experiments()

        assert count == 5
        repository.db.commit.assert_called_once()


class TestDeleteExperiment:
    """delete_experimentメソッドのテスト"""

    def test_delete_experiment_not_found(
        self, repository: GAExperimentRepository
    ) -> None:
        """実験が見つからない場合はNone"""
        mock_query = MagicMock()
        mock_query.filter.return_value.first.return_value = None
        repository.db.query.return_value = mock_query

        result = repository.delete_experiment(999)

        assert result is None

    def test_delete_experiment_running_raises(
        self, repository: GAExperimentRepository
    ) -> None:
        """実行中の実験は削除できない"""
        running_exp = GAExperiment(
            id=1,
            name="running_exp",
            config={},
            status="running",
            progress=0.5,
            total_generations=10,
            current_generation=3,
        )
        mock_query = MagicMock()
        mock_query.filter.return_value.first.return_value = running_exp
        repository.db.query.return_value = mock_query

        with pytest.raises(ValueError, match="実行中"):
            repository.delete_experiment(1)

    def test_delete_experiment_success(
        self, repository: GAExperimentRepository
    ) -> None:
        """実験と戦略が削除される"""
        from database.models import GeneratedStrategy

        experiment = GAExperiment(
            id=1,
            name="done_exp",
            config={},
            status="completed",
            progress=1.0,
            total_generations=10,
            current_generation=10,
        )
        strategy = MagicMock(spec=GeneratedStrategy)
        strategy.backtest_result_id = 42

        def query_side_effect(*args, **kwargs):
            mock_query = MagicMock()
            if args and args[0] is GAExperiment:
                mock_query.filter.return_value.first.return_value = experiment
            elif args and args[0] is GeneratedStrategy:
                mock_query.filter.return_value.all.return_value = [strategy]
                mock_query.filter.return_value.first.return_value = None
            else:
                # BacktestResult の delete
                mock_query.filter.return_value.delete.return_value = 1
            return mock_query

        repository.db.query.side_effect = query_side_effect

        result = repository.delete_experiment(1)

        assert result == (1, 1)
        repository.db.commit.assert_called_once()


class TestErrorHandling:
    """エラーハンドリングのテスト"""

    def test_create_experiment_handles_error(
        self, repository: GAExperimentRepository
    ) -> None:
        """作成エラーが適切に処理される"""
        repository.db.add.side_effect = Exception("DB Error")

        with pytest.raises(Exception):
            repository.create_experiment("test", {}, 100)
