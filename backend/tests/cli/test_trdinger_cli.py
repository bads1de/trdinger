"""
Trdinger CLI のユニットテスト
"""

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from app.cli._services import SynchronousScheduler
from app.cli.main import app

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


class TestDataCommands:
    """data コマンド群のテスト"""

    @pytest.fixture
    def mock_data_services(self):
        """build_data_collection_services をモックする"""
        orchestration = MagicMock()
        management = MagicMock()
        return orchestration, management

    @pytest.fixture
    def mock_db(self):
        """SessionLocal をモックする"""
        session = MagicMock()
        session.__enter__.return_value = session
        return session

    @pytest.fixture
    def sync_runner(self):
        """asyncio_run を恒等関数に置き換える（モックはコルーチンを返さないため）"""
        with patch("app.cli.main.asyncio_run", side_effect=lambda coro: coro):
            yield

    def test_data_fetch_starts(self, mock_data_services, mock_db, sync_runner):
        """fetch は収集サービスを呼び、タスクを実行する"""
        orchestration, _ = mock_data_services
        orchestration.start_historical_data_collection = MagicMock(
            return_value={
                "success": True,
                "status": "started",
                "message": "収集を開始しました",
            }
        )
        with (
            patch(
                "app.cli.main.build_data_collection_services",
                return_value=mock_data_services,
            ),
            patch("app.cli.main.SessionLocal", return_value=mock_db),
            patch(
                "app.cli.main.SynchronousBackgroundTasks",
                return_value=MagicMock(),
            ),
        ):
            result = runner.invoke(
                app,
                ["data", "fetch", "--symbol", "BTC/USDT:USDT", "--timeframe", "1h"],
            )
            assert result.exit_code == 0
            assert "started" in result.output
            orchestration.start_historical_data_collection.assert_called_once()

    def test_data_fetch_error(self, mock_data_services, mock_db, sync_runner):
        """fetch の ValueError は exit 1"""
        orchestration, _ = mock_data_services
        orchestration.start_historical_data_collection = MagicMock(
            side_effect=ValueError("無効な時間軸")
        )
        with (
            patch(
                "app.cli.main.build_data_collection_services",
                return_value=mock_data_services,
            ),
            patch("app.cli.main.SessionLocal", return_value=mock_db),
        ):
            result = runner.invoke(app, ["data", "fetch", "--timeframe", "9x"])
            assert result.exit_code == 1
            assert "無効な時間軸" in result.output

    def test_data_update_calls_bulk(self, mock_data_services, mock_db, sync_runner):
        """update は差分更新サービスを呼ぶ"""
        orchestration, _ = mock_data_services
        orchestration.execute_bulk_incremental_update = MagicMock(
            return_value={"success": True, "message": "更新完了"}
        )
        with (
            patch(
                "app.cli.main.build_data_collection_services",
                return_value=mock_data_services,
            ),
            patch("app.cli.main.SessionLocal", return_value=mock_db),
        ):
            result = runner.invoke(app, ["data", "update"])
            assert result.exit_code == 0
            orchestration.execute_bulk_incremental_update.assert_called_once()

    def test_data_status_shows_count(self, mock_data_services, mock_db, sync_runner):
        """status は件数と範囲を表示する"""
        orchestration, _ = mock_data_services
        orchestration.get_collection_status = MagicMock(
            return_value={
                "success": True,
                "data": {
                    "data_count": 100,
                    "latest_timestamp": "2024-06-30T00:00:00",
                    "oldest_timestamp": "2024-01-01T00:00:00",
                },
            }
        )
        with (
            patch(
                "app.cli.main.build_data_collection_services",
                return_value=mock_data_services,
            ),
            patch("app.cli.main.SessionLocal", return_value=mock_db),
        ):
            result = runner.invoke(app, ["data", "status"])
            assert result.exit_code == 0
            assert "100件" in result.output
            assert "2024-01-01" in result.output

    def test_data_overview_shows_totals(self, mock_data_services, sync_runner):
        """overview は全データ種別の合計を表示する"""
        _, management = mock_data_services
        management.get_data_status = MagicMock(
            return_value={
                "success": True,
                "data": {
                    "data_counts": {
                        "ohlcv": 1000,
                        "funding_rates": 50,
                        "open_interest": 30,
                    },
                    "total_records": 1080,
                },
            }
        )
        with patch(
            "app.cli.main.build_data_collection_services",
            return_value=mock_data_services,
        ):
            result = runner.invoke(app, ["data", "overview"])
            assert result.exit_code == 0
            assert "1080件" in result.output

    def test_data_reset_all_with_yes(self, mock_data_services, sync_runner):
        """reset all --yes は確認なしで実行"""
        _, management = mock_data_services
        management.reset_all_data = MagicMock(
            return_value={
                "success": True,
                "message": "全データのリセットが完了しました",
                "data": {"total_deleted": 10},
            }
        )
        with patch(
            "app.cli.main.build_data_collection_services",
            return_value=mock_data_services,
        ):
            result = runner.invoke(app, ["data", "reset", "all", "--yes"])
            assert result.exit_code == 0
            assert "10件" in result.output
            management.reset_all_data.assert_called_once()

    def test_data_reset_symbol(self, mock_data_services, sync_runner):
        """--symbol 指定はシンボル別リセットを呼ぶ"""
        _, management = mock_data_services
        management.reset_data_by_symbol = MagicMock(
            return_value={
                "success": True,
                "message": "完了",
                "data": {"total_deleted": 5},
            }
        )
        with patch(
            "app.cli.main.build_data_collection_services",
            return_value=mock_data_services,
        ):
            result = runner.invoke(
                app, ["data", "reset", "all", "--symbol", "BTC/USDT:USDT", "--yes"]
            )
            assert result.exit_code == 0
            management.reset_data_by_symbol.assert_called_once_with("BTC/USDT:USDT")

    def test_data_reset_invalid_target(self, mock_data_services, sync_runner):
        """不正な対象は exit 2"""
        with patch(
            "app.cli.main.build_data_collection_services",
            return_value=mock_data_services,
        ):
            result = runner.invoke(app, ["data", "reset", "bogus", "--yes"])
            assert result.exit_code == 2
            assert "いずれか" in result.output

    def test_data_reset_cancel(self, mock_data_services):
        """確認で n を選ぶとキャンセル"""
        with patch(
            "app.cli.main.build_data_collection_services",
            return_value=mock_data_services,
        ):
            result = runner.invoke(app, ["data", "reset", "all"], input="n\n")
            assert result.exit_code == 0
            assert "キャンセル" in result.output
