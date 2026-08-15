"""
Trdinger CLI のユニットテスト
"""

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from app.cli.main import app
from app.cli._services import SynchronousScheduler

runner = CliRunner()


@pytest.fixture
def mock_services():
    """build_services をモックする"""
    auto_service = MagicMock()
    application_service = MagicMock()
    auto_service.get_experiment_detail.return_value = {
        "status": "completed",
        "best_fitness": 0.5,
    }
    auto_service.list_experiments.return_value = []
    auto_service.stop_experiment.return_value = {
        "success": True,
        "message": "停止しました",
    }
    auto_service.delete_experiment.return_value = {
        "success": True,
        "message": "削除しました",
    }
    return auto_service, application_service


def test_synchronous_scheduler_runs_immediately():
    """SynchronousScheduler は add_task をその場で実行する"""
    scheduler = SynchronousScheduler()
    called = []

    def task(arg: str) -> None:
        called.append(arg)

    scheduler.add_task(task, "x")
    assert called == ["x"]


def test_exp_run_rejects_population_too_small(mock_services):
    """個体数 < 2 は BadParameter"""
    with patch("app.cli.main.build_services", return_value=mock_services):
        result = runner.invoke(
            app,
            ["exp", "run", "--population", "1", "--no-validation"],
        )
        assert result.exit_code == 2  # typer.BadParameter


def test_exp_run_rejects_invalid_rate(mock_services):
    """交叉率が範囲外なら BadParameter"""
    with patch("app.cli.main.build_services", return_value=mock_services):
        result = runner.invoke(
            app,
            ["exp", "run", "--crossover-rate", "1.5", "--no-validation"],
        )
        assert result.exit_code == 2


def test_exp_list_empty(mock_services):
    """実験ゼロ件はメッセージ表示"""
    with patch("app.cli.main.build_services", return_value=mock_services):
        result = runner.invoke(app, ["exp", "list"])
        assert result.exit_code == 0
        assert "実験はまだありません" in result.output


def test_exp_show_not_found(mock_services):
    """存在しない実験は exit 1"""
    auto_service, _ = mock_services
    auto_service.get_experiment_detail.return_value = None
    with patch("app.cli.main.build_services", return_value=mock_services):
        result = runner.invoke(app, ["exp", "show", "missing-uuid"])
        assert result.exit_code == 1
        assert "見つかりません" in result.output


def test_exp_stop_success(mock_services):
    """停止成功は exit 0"""
    with patch("app.cli.main.build_services", return_value=mock_services):
        result = runner.invoke(app, ["exp", "stop", "some-uuid"])
        assert result.exit_code == 0
        assert "停止しました" in result.output


def test_exp_stop_failure(mock_services):
    """停止失敗は exit 1"""
    auto_service, _ = mock_services
    auto_service.stop_experiment.return_value = {
        "success": False,
        "message": "実行中の実験が見つかりません",
    }
    with patch("app.cli.main.build_services", return_value=mock_services):
        result = runner.invoke(app, ["exp", "stop", "missing"])
        assert result.exit_code == 1


def test_exp_delete_with_yes(mock_services):
    """--yes 付き削除は確認なしで実行"""
    with patch("app.cli.main.build_services", return_value=mock_services):
        result = runner.invoke(app, ["exp", "delete", "some-uuid", "--yes"])
        assert result.exit_code == 0
        assert "削除" in result.output


def test_strategy_show_invalid_id():
    """不正な戦略IDは exit 1"""
    result = runner.invoke(app, ["strategy", "show", "not-a-number"])
    assert result.exit_code == 1
    assert "不正な戦略ID" in result.output
