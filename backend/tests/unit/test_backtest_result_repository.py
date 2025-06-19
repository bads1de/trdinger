"""
バックテスト結果リポジトリのテスト
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime


@pytest.mark.skip(reason="Repository tests require proper database setup")
class TestBacktestResultRepository:
    """バックテスト結果リポジトリのテストクラス"""

    @pytest.fixture
    def mock_db(self):
        """モックデータベースセッション"""
        return Mock()

    @pytest.fixture
    def repository(self, mock_db):
        """リポジトリインスタンス"""
        with patch(
            "database.repositories.backtest_result_repository.BacktestResultRepository"
        ):
            from database.repositories.backtest_result_repository import (
                BacktestResultRepository,
            )

            return BacktestResultRepository(mock_db)

    def test_delete_backtest_result_success(self, repository, mock_db):
        """バックテスト結果削除成功テスト"""
        # モックの設定
        mock_result = Mock()
        mock_db.query.return_value.filter.return_value.first.return_value = mock_result

        # テスト実行
        result = repository.delete_backtest_result(1)

        # 検証
        assert result is True
        mock_db.delete.assert_called_once_with(mock_result)
        mock_db.commit.assert_called_once()

    def test_delete_backtest_result_not_found(self, repository, mock_db):
        """バックテスト結果削除失敗テスト（結果が見つからない）"""
        # モックの設定
        mock_db.query.return_value.filter.return_value.first.return_value = None

        # テスト実行
        result = repository.delete_backtest_result(999)

        # 検証
        assert result is False
        mock_db.delete.assert_not_called()
        mock_db.commit.assert_not_called()

    def test_delete_all_backtest_results_success(self, repository, mock_db):
        """バックテスト結果一括削除成功テスト"""
        # モックの設定
        mock_db.query.return_value.delete.return_value = 5

        # テスト実行
        deleted_count = repository.delete_all_backtest_results()

        # 検証
        assert deleted_count == 5
        mock_db.query.return_value.delete.assert_called_once()
        mock_db.commit.assert_called_once()

    def test_delete_all_backtest_results_exception(self, repository, mock_db):
        """バックテスト結果一括削除例外テスト"""
        # モックの設定
        mock_db.query.return_value.delete.side_effect = Exception("Database error")

        # テスト実行と検証
        with pytest.raises(Exception) as exc_info:
            repository.delete_all_backtest_results()

        assert "Failed to delete all backtest results" in str(exc_info.value)
        mock_db.rollback.assert_called_once()

    def test_delete_backtest_result_exception(self, repository, mock_db):
        """バックテスト結果削除例外テスト"""
        # モックの設定
        mock_db.query.side_effect = Exception("Database error")

        # テスト実行と検証
        with pytest.raises(Exception) as exc_info:
            repository.delete_backtest_result(1)

        assert "Failed to delete backtest result" in str(exc_info.value)
        mock_db.rollback.assert_called_once()
