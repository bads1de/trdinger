"""
GeneratedStrategyRepositoryのテストモジュール

生成戦略リポジトリの機能をテストします。
"""

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.orm import Session

from database.models import GeneratedStrategy
from database.repositories.generated_strategy_repository import (
    GeneratedStrategyRepository,
)


@pytest.fixture
def mock_session() -> MagicMock:
    """モックDBセッション"""
    session = MagicMock(spec=Session)
    session.add = MagicMock()
    session.add_all = MagicMock()
    session.commit = MagicMock()
    session.refresh = MagicMock()
    session.query = MagicMock()
    session.scalars = MagicMock()
    return session


@pytest.fixture
def repository(mock_session: MagicMock) -> GeneratedStrategyRepository:
    """GeneratedStrategyRepositoryインスタンス"""
    return GeneratedStrategyRepository(mock_session)


@pytest.fixture
def sample_strategy_model() -> GeneratedStrategy:
    """サンプルGeneratedStrategyモデル"""
    return GeneratedStrategy(
        id=1,
        experiment_id=100,
        gene_data={
            "id": "strategy_001",
            "indicators": [{"type": "RSI", "period": 14}],
            "entry_conditions": [],
            "exit_conditions": [],
            "risk_management": {},
            "metadata": {},
        },
        generation=10,
        fitness_score=0.85,
        fitness_values=[0.85, 0.90],
        parent_ids=[5, 6],
        backtest_result_id=50,
    )


@pytest.fixture
def sample_gene_data() -> Dict[str, Any]:
    """サンプル遺伝子データ"""
    return {
        "id": "strategy_001",
        "indicators": [{"type": "MACD"}],
        "entry_conditions": [{"condition": "MACD > 0"}],
        "exit_conditions": [{"condition": "MACD < 0"}],
        "risk_management": {"stop_loss": 0.02},
        "metadata": {"created": "2024-01-01"},
    }


class TestRepositoryInitialization:
    """リポジトリ初期化のテスト"""

    def test_repository_initialization(self, mock_session: MagicMock) -> None:
        """リポジトリが正しく初期化される"""
        repo = GeneratedStrategyRepository(mock_session)
        assert repo.db == mock_session
        assert repo.model_class == GeneratedStrategy


class TestSaveStrategy:
    """save_strategyメソッドのテスト"""

    def test_save_strategy_success(
        self,
        repository: GeneratedStrategyRepository,
        sample_gene_data: Dict[str, Any],
    ) -> None:
        """戦略が正常に保存される"""
        # save_strategyは@safe_operationデコレータ内で実行され、
        # 実際のGeneratedStrategyオブジェクトを返す
        result = repository.save_strategy(
            experiment_id=100,
            gene_data=sample_gene_data,
            generation=10,
            fitness_score=0.85,
        )
        
        # 結果が返されることを確認
        assert result is not None
        repository.db.add.assert_called_once()
        repository.db.commit.assert_called_once()
        repository.db.refresh.assert_called_once()

    def test_save_strategy_validates_gene_data(
        self, repository: GeneratedStrategyRepository
    ) -> None:
        """遺伝子データが検証され、不足フィールドが補完される"""
        incomplete_gene_data = {"indicators": []}
        
        # _validate_gene_dataメソッドが不足フィールドを補完するため、
        # 正常に保存される
        result = repository.save_strategy(
            experiment_id=100,
            gene_data=incomplete_gene_data,
            generation=10,
        )
        
        # 結果が返されることを確認
        assert result is not None
        
        # db.addに渡された引数を取得して検証
        call_args = repository.db.add.call_args
        assert call_args is not None
        saved_strategy = call_args[0][0]
        
        # 補完された必須フィールドを確認
        assert "indicators" in saved_strategy.gene_data
        assert "entry_conditions" in saved_strategy.gene_data
        assert "exit_conditions" in saved_strategy.gene_data
        assert "risk_management" in saved_strategy.gene_data
        assert "metadata" in saved_strategy.gene_data


class TestSaveStrategiesBatch:
    """save_strategies_batchメソッドのテスト"""

    def test_save_strategies_batch_success(
        self,
        repository: GeneratedStrategyRepository,
        sample_gene_data: Dict[str, Any],
    ) -> None:
        """戦略が一括保存される"""
        strategies_data = [
            {
                "experiment_id": 100,
                "gene_data": sample_gene_data,
                "generation": 10,
                "fitness_score": 0.85,
            },
            {
                "experiment_id": 100,
                "gene_data": sample_gene_data,
                "generation": 10,
                "fitness_score": 0.90,
            },
        ]
        
        # save_strategies_batchは@safe_operationデコレータ内で実行される
        result = repository.save_strategies_batch(strategies_data)
        
        # 結果が返されることを確認
        assert result is not None
        assert len(result) == 2
        repository.db.add_all.assert_called_once()
        repository.db.commit.assert_called_once()
        
        # refresh が各戦略に対して呼ばれる
        assert repository.db.refresh.call_count == 2


class TestGetStrategiesByExperiment:
    """get_strategies_by_experimentメソッドのテスト"""

    def test_get_strategies_by_experiment_success(
        self,
        repository: GeneratedStrategyRepository,
        sample_strategy_model: GeneratedStrategy,
    ) -> None:
        """実験別で戦略が取得できる"""
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [sample_strategy_model]
        repository.db.scalars.return_value = mock_scalars
        
        results = repository.get_strategies_by_experiment(100)
        
        assert len(results) == 1
        assert results[0].experiment_id == 100

    def test_get_strategies_by_experiment_with_generation(
        self,
        repository: GeneratedStrategyRepository,
        sample_strategy_model: GeneratedStrategy,
    ) -> None:
        """世代指定で戦略が取得できる"""
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [sample_strategy_model]
        repository.db.scalars.return_value = mock_scalars
        
        results = repository.get_strategies_by_experiment(100, generation=10)
        
        assert len(results) == 1


class TestGetStrategiesByGeneration:
    """get_strategies_by_generationメソッドのテスト"""

    def test_get_strategies_by_generation_success(
        self,
        repository: GeneratedStrategyRepository,
        sample_strategy_model: GeneratedStrategy,
    ) -> None:
        """世代別で戦略が取得できる"""
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [sample_strategy_model]
        repository.db.scalars.return_value = mock_scalars
        
        results = repository.get_strategies_by_generation(100, 10)
        
        assert len(results) == 1


class TestGetStrategiesWithBacktestResults:
    """get_strategies_with_backtest_resultsメソッドのテスト"""

    def test_get_strategies_with_backtest_results_success(
        self,
        repository: GeneratedStrategyRepository,
        sample_strategy_model: GeneratedStrategy,
    ) -> None:
        """バックテスト結果付き戦略が取得できる"""
        mock_query = MagicMock()
        mock_query.join.return_value.filter.return_value.offset.return_value.limit.return_value.all.return_value = [
            sample_strategy_model
        ]
        repository.db.query.return_value = mock_query
        
        results = repository.get_strategies_with_backtest_results(limit=10)
        
        assert len(results) == 1


class TestDeleteAllStrategies:
    """delete_all_strategiesメソッドのテスト"""

    def test_delete_all_strategies_success(
        self, repository: GeneratedStrategyRepository
    ) -> None:
        """全戦略が削除される"""
        mock_query = MagicMock()
        mock_query.delete.return_value = 10
        repository.db.query.return_value = mock_query
        
        count = repository.delete_all_strategies()
        
        assert count == 10
        repository.db.commit.assert_called_once()


class TestValidateGeneData:
    """_validate_gene_dataメソッドのテスト"""

    def test_validate_gene_data_adds_missing_fields(
        self, repository: GeneratedStrategyRepository
    ) -> None:
        """欠損フィールドが補完される"""
        incomplete_data = {"indicators": []}
        
        validated = repository._validate_gene_data(incomplete_data)
        
        assert "id" in validated
        assert "entry_conditions" in validated
        assert "exit_conditions" in validated
        assert "risk_management" in validated
        assert "metadata" in validated

    def test_validate_gene_data_preserves_existing_fields(
        self,
        repository: GeneratedStrategyRepository,
        sample_gene_data: Dict[str, Any],
    ) -> None:
        """既存フィールドが保持される"""
        validated = repository._validate_gene_data(sample_gene_data)
        
        assert validated["indicators"] == sample_gene_data["indicators"]
        assert validated["entry_conditions"] == sample_gene_data["entry_conditions"]


class TestErrorHandling:
    """エラーハンドリングのテスト"""

    def test_save_strategy_handles_error(
        self, repository: GeneratedStrategyRepository
    ) -> None:
        """保存エラーが適切に処理される"""
        repository.db.add.side_effect = Exception("DB Error")
        
        with pytest.raises(Exception):
            repository.save_strategy(
                experiment_id=100,
                gene_data={"id": "test"},
                generation=10,
            )