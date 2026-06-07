"""
ExperimentPersistenceService の拡張テスト

既存テスト (``test_experiment_persistence_service.py``) がカバーしていない
プライベートメソッド、進捗更新、pareto front、エッジケースを検証します。
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest

from app.services.auto_strategy.config import GAConfig
from app.services.auto_strategy.genes import StrategyGene
from app.services.auto_strategy.services.experiment_persistence_service import (
    ExperimentPersistenceService,
)


def _make_service() -> tuple[ExperimentPersistenceService, MagicMock]:
    """サービスと session_factory モックを返すヘルパー"""
    session = MagicMock()
    factory = MagicMock()
    factory.return_value.__enter__.return_value = session
    factory.return_value.__exit__.return_value = None
    service = ExperimentPersistenceService(factory)
    return service, factory


def _make_strategy(id_: str = "s1") -> Mock:
    """StrategyGene として振る舞う Mock"""
    s = Mock()
    s.id = id_
    s.fitness = SimpleNamespace(values=(1.0,), valid=True)
    s.__class__ = StrategyGene
    return s


class TestSessionContextManager:
    """``_get_db_session`` コンテキストマネージャのテスト"""

    def test_reuses_active_session(self) -> None:
        service, factory = _make_service()
        active = MagicMock()
        active.closed = False
        service._active_session = active

        with service._get_db_session() as session:
            assert session is active
        # factory は呼ばれない
        factory.assert_not_called()

    def test_creates_new_session_when_none_active(self) -> None:
        service, factory = _make_service()
        new_session = MagicMock()
        factory.return_value.__enter__.return_value = new_session
        service._active_session = None

        with service._get_db_session() as session:
            assert session is new_session
        factory.assert_called_once()
        # 新規セッションはキャッシュされる
        assert service._active_session is new_session

    def test_resets_active_session_on_exception(self) -> None:
        service, factory = _make_service()
        new_session = MagicMock()
        factory.return_value.__enter__.return_value = new_session

        with pytest.raises(RuntimeError, match="boom"):
            with service._get_db_session():
                raise RuntimeError("boom")
        # 例外時はキャッシュクリア
        assert service._active_session is None

    def test_creates_new_session_when_active_closed(self) -> None:
        service, factory = _make_service()
        active = MagicMock()
        active.closed = True
        service._active_session = active

        new_session = MagicMock()
        factory.return_value.__enter__.return_value = new_session

        with service._get_db_session() as session:
            assert session is new_session
        factory.assert_called_once()


class TestCloseActiveSession:
    """``_close_active_session`` のテスト"""

    def test_closes_unclosed_session(self) -> None:
        service, _ = _make_service()
        active = MagicMock()
        active.closed = False
        service._active_session = active

        service._close_active_session()
        active.close.assert_called_once()
        assert service._active_session is None

    def test_skips_already_closed_session(self) -> None:
        service, _ = _make_service()
        active = MagicMock()
        active.closed = True
        service._active_session = active

        service._close_active_session()
        active.close.assert_not_called()
        assert service._active_session is None

    def test_handles_close_exception(self) -> None:
        service, _ = _make_service()
        active = MagicMock()
        active.closed = False
        active.close.side_effect = RuntimeError("close fail")
        service._active_session = active

        # 例外を飲み込んで None にする
        service._close_active_session()
        assert service._active_session is None

    def test_does_nothing_when_no_active_session(self) -> None:
        service, _ = _make_service()
        service._active_session = None
        service._close_active_session()
        assert service._active_session is None


class TestSaveBestStrategy:
    """``_save_best_strategy`` のテスト"""

    def test_saves_with_scalar_fitness(self) -> None:
        service, _ = _make_service()
        db = MagicMock()
        experiment_info = {"db_id": 42}
        result = {
            "best_strategy": _make_strategy("s1"),
            "best_fitness": 0.85,
            "best_evaluation_summary": {"sharpe": 1.5},
        }
        ga_config = GAConfig(generations=10)

        with (
            patch(
                "app.services.auto_strategy.services.experiment_persistence_service.GeneratedStrategyRepository"
            ) as mock_repo_cls,
            patch.object(
                service.serializer,
                "strategy_gene_to_dict",
                return_value={"foo": "bar"},
            ) as mock_to_dict,
        ):
            mock_repo = mock_repo_cls.return_value
            mock_record = Mock(id=999)
            mock_repo.save_strategy.return_value = mock_record

            service._save_best_strategy(
                db, "exp_id", experiment_info, result, ga_config
            )

        mock_to_dict.assert_called_once()
        call_kwargs = mock_repo.save_strategy.call_args[1]
        assert call_kwargs["experiment_id"] == 42
        assert call_kwargs["fitness_score"] == 0.85
        assert call_kwargs["fitness_values"] is None
        assert call_kwargs["generation"] == 10
        # gene_data に evaluation_summary が attach されている
        assert call_kwargs["gene_data"]["foo"] == "bar"

    def test_saves_with_tuple_fitness(self) -> None:
        service, _ = _make_service()
        db = MagicMock()
        experiment_info = {"db_id": 42}
        result = {
            "best_strategy": _make_strategy("s1"),
            "best_fitness": (0.5, 0.3, 0.2),
            "best_evaluation_summary": None,
        }
        ga_config = GAConfig(generations=5)

        with (
            patch(
                "app.services.auto_strategy.services.experiment_persistence_service.GeneratedStrategyRepository"
            ) as mock_repo_cls,
            patch.object(
                service.serializer,
                "strategy_gene_to_dict",
                return_value={},
            ),
        ):
            mock_repo = mock_repo_cls.return_value
            mock_repo.save_strategy.return_value = Mock(id=1)

            service._save_best_strategy(
                db, "exp_id", experiment_info, result, ga_config
            )

        call_kwargs = mock_repo.save_strategy.call_args[1]
        assert call_kwargs["fitness_score"] == 0.5
        assert call_kwargs["fitness_values"] == [0.5, 0.3, 0.2]

    def test_saves_with_empty_tuple_fitness(self) -> None:
        service, _ = _make_service()
        db = MagicMock()
        experiment_info = {"db_id": 1}
        result = {
            "best_strategy": _make_strategy("s1"),
            "best_fitness": (),
            "best_evaluation_summary": None,
        }
        ga_config = GAConfig()

        with (
            patch(
                "app.services.auto_strategy.services.experiment_persistence_service.GeneratedStrategyRepository"
            ) as mock_repo_cls,
            patch.object(
                service.serializer,
                "strategy_gene_to_dict",
                return_value={},
            ),
        ):
            mock_repo = mock_repo_cls.return_value
            mock_repo.save_strategy.return_value = Mock(id=1)

            service._save_best_strategy(
                db, "exp_id", experiment_info, result, ga_config
            )

        call_kwargs = mock_repo.save_strategy.call_args[1]
        assert call_kwargs["fitness_score"] == 0.0

    def test_saves_with_non_numeric_fitness(self) -> None:
        service, _ = _make_service()
        db = MagicMock()
        experiment_info = {"db_id": 1}
        result = {
            "best_strategy": _make_strategy("s1"),
            "best_fitness": "not-a-number",
            "best_evaluation_summary": None,
        }
        ga_config = GAConfig()

        with (
            patch(
                "app.services.auto_strategy.services.experiment_persistence_service.GeneratedStrategyRepository"
            ) as mock_repo_cls,
            patch.object(
                service.serializer,
                "strategy_gene_to_dict",
                return_value={},
            ),
        ):
            mock_repo = mock_repo_cls.return_value
            mock_repo.save_strategy.return_value = Mock(id=1)

            service._save_best_strategy(
                db, "exp_id", experiment_info, result, ga_config
            )

        call_kwargs = mock_repo.save_strategy.call_args[1]
        assert call_kwargs["fitness_score"] == 0.0


class TestSaveOtherStrategies:
    """``_save_other_strategies`` のテスト"""

    def test_returns_early_when_all_strategies_empty(self) -> None:
        service, _ = _make_service()
        db = MagicMock()
        result = {"best_strategy": _make_strategy(), "all_strategies": []}

        with patch(
            "app.services.auto_strategy.services.experiment_persistence_service.GeneratedStrategyRepository"
        ) as mock_repo_cls:
            service._save_other_strategies(db, {"db_id": 1}, result, GAConfig())
            mock_repo_cls.assert_not_called()

    def test_returns_early_when_only_best_strategy(self) -> None:
        service, _ = _make_service()
        db = MagicMock()
        best = _make_strategy("s1")
        result = {"best_strategy": best, "all_strategies": [best]}

        with patch(
            "app.services.auto_strategy.services.experiment_persistence_service.GeneratedStrategyRepository"
        ) as mock_repo_cls:
            service._save_other_strategies(db, {"db_id": 1}, result, GAConfig())
            mock_repo_cls.assert_not_called()

    def test_saves_other_strategies_in_batch(self) -> None:
        service, _ = _make_service()
        db = MagicMock()
        best = _make_strategy("best")
        other1 = _make_strategy("o1")
        other2 = _make_strategy("o2")
        result = {
            "best_strategy": best,
            "all_strategies": [best, other1, other2],
            "fitness_scores": [0.9, 0.5, 0.3],
            "evaluation_summaries": {
                "o1": {"sharpe": 0.5},
                "o2": {"sharpe": 0.3},
            },
        }

        with (
            patch(
                "app.services.auto_strategy.services.experiment_persistence_service.GeneratedStrategyRepository"
            ) as mock_repo_cls,
            patch.object(
                service.serializer,
                "strategy_gene_to_dict",
                return_value={"serialized": True},
            ),
        ):
            mock_repo = mock_repo_cls.return_value
            mock_repo.save_strategies_batch.return_value = 2

            service._save_other_strategies(
                db, {"db_id": 10}, result, GAConfig(generations=20)
            )

        # best 以外の 2 件が batch 保存される
        mock_repo.save_strategies_batch.assert_called_once()
        saved_data = mock_repo.save_strategies_batch.call_args[0][0]
        assert len(saved_data) == 2
        assert all(d["experiment_id"] == 10 for d in saved_data)
        assert all(d["generation"] == 20 for d in saved_data)
        assert saved_data[0]["fitness_score"] == 0.5
        assert saved_data[1]["fitness_score"] == 0.3

    def test_uses_default_fitness_when_index_out_of_range(self) -> None:
        service, _ = _make_service()
        db = MagicMock()
        best = _make_strategy("best")
        other = _make_strategy("o1")
        result = {
            "best_strategy": best,
            "all_strategies": [best, other],
            "fitness_scores": [],  # 空
        }

        with (
            patch(
                "app.services.auto_strategy.services.experiment_persistence_service.GeneratedStrategyRepository"
            ) as mock_repo_cls,
            patch.object(
                service.serializer,
                "strategy_gene_to_dict",
                return_value={},
            ),
        ):
            mock_repo = mock_repo_cls.return_value
            mock_repo.save_strategies_batch.return_value = 1

            service._save_other_strategies(db, {"db_id": 1}, result, GAConfig())

        saved_data = mock_repo.save_strategies_batch.call_args[0][0]
        assert saved_data[0]["fitness_score"] == 0.0


class TestSaveParetoFront:
    """``_save_pareto_front`` のテスト"""

    def test_returns_early_when_pareto_empty(self) -> None:
        service, _ = _make_service()
        db = MagicMock()
        result = {"pareto_front": []}

        with patch(
            "app.services.auto_strategy.services.experiment_persistence_service.GeneratedStrategyRepository"
        ) as mock_repo_cls:
            service._save_pareto_front(db, {"db_id": 1}, result, GAConfig())
            mock_repo_cls.assert_not_called()

    def test_skips_solutions_without_strategy_or_fitness(self) -> None:
        service, _ = _make_service()
        db = MagicMock()
        result = {
            "pareto_front": [
                {"strategy": None, "fitness_values": (0.1,)},
                {"strategy": _make_strategy("s1"), "fitness_values": None},
            ]
        }

        with patch(
            "app.services.auto_strategy.services.experiment_persistence_service.GeneratedStrategyRepository"
        ) as mock_repo_cls:
            mock_repo = mock_repo_cls.return_value
            service._save_pareto_front(db, {"db_id": 1}, result, GAConfig())
            # どちらも None/空なので batch 呼ばれない
            mock_repo.save_strategies_batch.assert_not_called()

    def test_saves_valid_pareto_solutions(self) -> None:
        service, _ = _make_service()
        db = MagicMock()
        s1 = _make_strategy("s1")
        s2 = _make_strategy("s2")
        result = {
            "pareto_front": [
                {"strategy": s1, "fitness_values": (0.8, 0.5)},
                {"strategy": s2, "fitness_values": (0.6, 0.7)},
            ],
            "evaluation_summaries": {"s1": {"sharpe": 0.8}},
        }

        with (
            patch(
                "app.services.auto_strategy.services.experiment_persistence_service.GeneratedStrategyRepository"
            ) as mock_repo_cls,
            patch.object(
                service.serializer,
                "strategy_gene_to_dict",
                return_value={},
            ),
        ):
            mock_repo = mock_repo_cls.return_value
            mock_repo.save_strategies_batch.return_value = 2

            service._save_pareto_front(
                db, {"db_id": 99}, result, GAConfig(generations=10)
            )

        saved_data = mock_repo.save_strategies_batch.call_args[0][0]
        assert len(saved_data) == 2
        assert saved_data[0]["experiment_id"] == 99
        assert saved_data[0]["fitness_score"] == 0.8
        assert saved_data[0]["fitness_values"] == (0.8, 0.5)
        assert saved_data[1]["fitness_score"] == 0.6

    def test_skips_solutions_with_empty_fitness_tuple(self) -> None:
        """fitness_values=() は falsy なので batch 対象外"""
        service, _ = _make_service()
        db = MagicMock()
        s1 = _make_strategy("s1")
        result = {
            "pareto_front": [{"strategy": s1, "fitness_values": ()}],
        }

        with (
            patch(
                "app.services.auto_strategy.services.experiment_persistence_service.GeneratedStrategyRepository"
            ) as mock_repo_cls,
            patch.object(
                service.serializer,
                "strategy_gene_to_dict",
                return_value={},
            ),
        ):
            mock_repo = mock_repo_cls.return_value
            service._save_pareto_front(db, {"db_id": 1}, result, GAConfig())
            # 条件 ``if strategy and fitness_values:`` で空タプルは falsy
            mock_repo.save_strategies_batch.assert_not_called()

    def test_uses_pareto_strategy_id_for_summary_lookup(self) -> None:
        """summary は strategy の id キーで照合される"""
        service, _ = _make_service()
        db = MagicMock()
        s1 = _make_strategy("pareto-id-1")
        result = {
            "pareto_front": [{"strategy": s1, "fitness_values": (0.5,)}],
            "evaluation_summaries": {"pareto-id-1": {"sharpe": 0.5}},
        }

        with (
            patch(
                "app.services.auto_strategy.services.experiment_persistence_service.GeneratedStrategyRepository"
            ) as mock_repo_cls,
            patch.object(
                service.serializer,
                "strategy_gene_to_dict",
                return_value={},
            ),
        ):
            mock_repo = mock_repo_cls.return_value
            mock_repo.save_strategies_batch.return_value = 1

            service._save_pareto_front(db, {"db_id": 1}, result, GAConfig())

        # 1 件 batch される
        mock_repo.save_strategies_batch.assert_called_once()


class TestSaveExperimentResultEdgeCases:
    """``save_experiment_result`` のエッジケース"""

    def test_returns_early_when_no_experiment_info(self) -> None:
        service, factory = _make_service()
        ga_config = GAConfig()
        result = {"best_strategy": Mock(), "best_fitness": 1.0}

        with patch.object(service, "get_experiment_info", return_value=None):
            service.save_experiment_result(
                "exp_id", result, ga_config, {}, experiment_info=None
            )

        # DB セッションは作られない
        factory.assert_not_called()

    def test_fetches_experiment_info_when_invalid_type(self) -> None:
        service, _ = _make_service()
        ga_config = GAConfig()
        result = {"best_strategy": _make_strategy(), "best_fitness": 1.0}

        with (
            patch.object(
                service,
                "get_experiment_info",
                return_value={"db_id": 1, "name": "x", "config": {}},
            ) as mock_get,
            patch(
                "app.services.auto_strategy.services.experiment_persistence_service.GeneratedStrategyRepository"
            ) as mock_repo_cls,
            patch.object(
                service.serializer,
                "strategy_gene_to_dict",
                return_value={},
            ),
        ):
            mock_repo = mock_repo_cls.return_value
            mock_repo.save_strategy.return_value = Mock(id=1)
            # experiment_info に dict 以外を渡すと get_experiment_info で再取得
            service.save_experiment_result(
                "exp_id", result, ga_config, {}, experiment_info="not a dict"
            )
            mock_get.assert_called_once_with("exp_id")
            mock_repo.save_strategy.assert_called_once()

    def test_saves_pareto_when_in_result(self) -> None:
        service, _ = _make_service()
        ga_config = GAConfig(generations=5)
        best = _make_strategy("best")
        other = _make_strategy("other")
        result = {
            "best_strategy": best,
            "best_fitness": 0.9,
            "all_strategies": [best],
            "fitness_scores": [0.9],
            "pareto_front": [{"strategy": other, "fitness_values": (0.7,)}],
        }
        experiment_info = {"db_id": 1}

        with (
            patch(
                "app.services.auto_strategy.services.experiment_persistence_service.GeneratedStrategyRepository"
            ) as mock_repo_cls,
            patch.object(
                service.serializer,
                "strategy_gene_to_dict",
                return_value={},
            ),
        ):
            mock_repo = mock_repo_cls.return_value
            mock_repo.save_strategy.return_value = Mock(id=1)
            mock_repo.save_strategies_batch.return_value = 1

            service.save_experiment_result(
                "exp_id", result, ga_config, {}, experiment_info=experiment_info
            )

        # pareto 含むので save_strategies_batch も呼ばれる
        assert mock_repo.save_strategies_batch.call_count == 1


class TestSaveBacktestResultEdgeCases:
    """``save_backtest_result`` のエッジケース"""

    def test_returns_early_when_result_data_empty(self) -> None:
        service, factory = _make_service()
        service.save_backtest_result({})
        factory.assert_not_called()

    def test_returns_early_when_result_data_none(self) -> None:
        service, factory = _make_service()
        service.save_backtest_result(None)
        factory.assert_not_called()


class TestUpdateExperimentProgress:
    """``update_experiment_progress`` のテスト"""

    def test_returns_false_when_experiment_not_found(self) -> None:
        service, _ = _make_service()
        with patch.object(service, "get_experiment_info", return_value=None):
            result = service.update_experiment_progress(
                "exp_id", current_generation=5, total_generations=10
            )
        assert result is False

    def test_updates_progress_when_found(self) -> None:
        service, _ = _make_service()
        with (
            patch.object(
                service,
                "get_experiment_info",
                return_value={"db_id": 7},
            ),
            patch(
                "app.services.auto_strategy.services.experiment_persistence_service.GAExperimentRepository"
            ) as mock_repo_cls,
        ):
            mock_repo = mock_repo_cls.return_value
            mock_repo.update_progress.return_value = True
            result = service.update_experiment_progress(
                "exp_id",
                current_generation=5,
                total_generations=10,
                best_fitness=0.95,
            )
        assert result is True
        mock_repo.update_progress.assert_called_once_with(7, 5, 10, 0.95)

    def test_uses_active_session(self) -> None:
        service, _ = _make_service()
        active = MagicMock()
        active.closed = False
        service._active_session = active
        with (
            patch.object(
                service,
                "get_experiment_info",
                return_value={"db_id": 1},
            ),
            patch(
                "app.services.auto_strategy.services.experiment_persistence_service.GAExperimentRepository"
            ) as mock_repo_cls,
        ):
            mock_repo = mock_repo_cls.return_value
            mock_repo.update_progress.return_value = True
            service.update_experiment_progress("exp_id", 1, 10)
        # active_session が再利用される
        assert service._active_session is active


class TestGetExperimentDetail:
    """``get_experiment_detail`` のテスト"""

    def test_returns_none_when_not_found(self) -> None:
        service, _ = _make_service()
        with patch(
            "app.services.auto_strategy.services.experiment_persistence_service.GAExperimentRepository"
        ) as mock_repo_cls:
            mock_repo = mock_repo_cls.return_value
            mock_repo.get_by_experiment_id.return_value = None
            assert service.get_experiment_detail("missing") is None

    def test_returns_detail_dict(self) -> None:
        service, _ = _make_service()
        with patch(
            "app.services.auto_strategy.services.experiment_persistence_service.GAExperimentRepository"
        ) as mock_repo_cls:
            mock_repo = mock_repo_cls.return_value
            exp = Mock()
            exp.id = 1
            exp.experiment_id = "exp_001"
            exp.name = "Test"
            exp.status = "running"
            exp.progress = 0.5
            exp.current_generation = 5
            exp.total_generations = 10
            exp.best_fitness = 0.95
            exp.created_at = datetime(2024, 1, 1)
            exp.completed_at = None
            mock_repo.get_by_experiment_id.return_value = exp
            detail = service.get_experiment_detail("exp_001")

        assert detail is not None
        assert detail["id"] == 1
        assert detail["experiment_id"] == "exp_001"
        assert detail["created_at"] == "2024-01-01T00:00:00"
        assert detail["completed_at"] is None


class TestGetStrategyResultKey:
    """``_get_strategy_result_key`` のテスト"""

    def test_uses_id_attribute_when_set(self) -> None:
        s = Mock()
        s.id = "strategy-123"
        assert (
            ExperimentPersistenceService._get_strategy_result_key(s) == "strategy-123"
        )

    def test_uses_id_attribute_when_empty_string_falls_back_to_id(self) -> None:
        """id が空文字のときは id(strategy) にフォールバック"""
        s = Mock()
        s.id = ""
        key = ExperimentPersistenceService._get_strategy_result_key(s)
        assert key == str(id(s))

    def test_uses_id_strategy_when_id_is_none(self) -> None:
        s = Mock()
        s.id = None
        key = ExperimentPersistenceService._get_strategy_result_key(s)
        assert key == str(id(s))

    def test_uses_id_strategy_when_no_id_attribute(self) -> None:
        s = object()  # id 属性なし
        key = ExperimentPersistenceService._get_strategy_result_key(s)
        assert key == str(id(s))
