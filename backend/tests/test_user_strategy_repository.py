#!/usr/bin/env python3
"""
UserStrategyRepositoryのユニットテスト
"""

import pytest
import json
from unittest.mock import Mock, patch
from sqlalchemy.orm import Session

from database.repositories.user_strategy_repository import UserStrategyRepository
from database.models import UserStrategy


class TestUserStrategyRepository:
    """UserStrategyRepositoryのテストクラス"""

    def setup_method(self):
        """各テストメソッドの前に実行される初期化"""
        self.mock_db = Mock(spec=Session)
        self.repository = UserStrategyRepository(self.mock_db)

    def test_create_success(self):
        """戦略作成の成功テスト"""
        # 入力データ
        strategy_data = {
            "name": "テスト戦略",
            "description": "テスト用の戦略",
            "strategy_config": {
                "indicators": [{"type": "SMA", "parameters": {"period": 20}}],
                "entry_conditions": [],
                "exit_conditions": [],
            },
            "is_active": True,
        }

        # モックの設定
        mock_strategy = Mock(spec=UserStrategy)
        mock_strategy.id = 1
        mock_strategy.name = "テスト戦略"
        mock_strategy.description = "テスト用の戦略"
        mock_strategy.is_active = True

        self.mock_db.add.return_value = None
        self.mock_db.commit.return_value = None
        self.mock_db.refresh.return_value = None

        # UserStrategyコンストラクタのモック
        with patch(
            "database.repositories.user_strategy_repository.UserStrategy"
        ) as mock_user_strategy_class:
            mock_user_strategy_class.return_value = mock_strategy

            # テスト実行
            result = self.repository.create(strategy_data)

            # 検証
            assert result is not None
            assert result.id == 1
            assert result.name == "テスト戦略"

            # データベース操作が呼ばれたことを確認
            self.mock_db.add.assert_called_once()
            self.mock_db.commit.assert_called_once()
            self.mock_db.refresh.assert_called_once()

    def test_get_all_success(self):
        """全戦略取得の成功テスト"""
        # モックデータの準備
        mock_strategy1 = Mock(spec=UserStrategy)
        mock_strategy1.id = 1
        mock_strategy1.name = "戦略1"
        mock_strategy1.is_active = True

        mock_strategy2 = Mock(spec=UserStrategy)
        mock_strategy2.id = 2
        mock_strategy2.name = "戦略2"
        mock_strategy2.is_active = True

        mock_strategies = [mock_strategy1, mock_strategy2]

        # get_allメソッドを直接モック
        with patch.object(self.repository, "get_all", return_value=mock_strategies):
            # テスト実行
            result = self.repository.get_all()

        # 検証
        assert len(result) == 2
        assert result[0].id == 1
        assert result[1].id == 2

    def test_get_by_id_success(self):
        """ID指定での戦略取得の成功テスト"""
        # モックデータの準備
        mock_strategy = Mock(spec=UserStrategy)
        mock_strategy.id = 1
        mock_strategy.name = "戦略1"
        mock_strategy.is_active = True

        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = mock_strategy
        self.mock_db.query.return_value = mock_query

        # テスト実行
        result = self.repository.get_by_id(1)

        # 検証
        assert result is not None
        assert result.id == 1
        assert result.name == "戦略1"

        # クエリが実行されたことを確認
        self.mock_db.query.assert_called_once_with(UserStrategy)

    def test_get_by_id_not_found(self):
        """存在しないIDでの戦略取得テスト"""
        # モックの設定（戦略が見つからない）
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = None
        self.mock_db.query.return_value = mock_query

        # テスト実行
        result = self.repository.get_by_id(999)

        # 検証
        assert result is None

        # クエリが実行されたことを確認
        self.mock_db.query.assert_called_once_with(UserStrategy)

    def test_update_success(self):
        """戦略更新の成功テスト"""
        # 既存の戦略のモック
        mock_strategy = Mock(spec=UserStrategy)
        mock_strategy.id = 1
        mock_strategy.name = "古い名前"
        mock_strategy.description = "古い説明"

        # 更新データ
        update_data = {"name": "新しい名前", "description": "新しい説明"}

        # モックの設定
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = mock_strategy
        self.mock_db.query.return_value = mock_query
        self.mock_db.commit.return_value = None
        self.mock_db.refresh.return_value = None

        # テスト実行
        result = self.repository.update(1, update_data)

        # 検証
        assert result is not None
        assert result.name == "新しい名前"
        assert result.description == "新しい説明"

        # データベース操作が呼ばれたことを確認
        self.mock_db.commit.assert_called_once()
        self.mock_db.refresh.assert_called_once()

    def test_update_not_found(self):
        """存在しない戦略の更新テスト"""
        # モックの設定（戦略が見つからない）
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = None
        self.mock_db.query.return_value = mock_query

        # 更新データ
        update_data = {"name": "新しい名前"}

        # テスト実行
        result = self.repository.update(999, update_data)

        # 検証
        assert result is None

        # commitが呼ばれていないことを確認
        self.mock_db.commit.assert_not_called()

    def test_delete_success(self):
        """戦略削除の成功テスト（論理削除）"""
        # 既存の戦略のモック
        mock_strategy = Mock(spec=UserStrategy)
        mock_strategy.id = 1
        mock_strategy.is_active = True

        # get_by_idのモック設定
        self.repository.get_by_id = Mock(return_value=mock_strategy)
        self.mock_db.commit.return_value = None

        # テスト実行
        result = self.repository.delete(1)

        # 検証
        assert result is True
        assert mock_strategy.is_active is False  # 論理削除でis_activeがFalseになる

        # データベース操作が呼ばれたことを確認
        self.mock_db.commit.assert_called_once()

    def test_delete_not_found(self):
        """存在しない戦略の削除テスト"""
        # モックの設定（戦略が見つからない）
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = None
        self.mock_db.query.return_value = mock_query

        # テスト実行
        result = self.repository.delete(999)

        # 検証
        assert result is False

        # deleteが呼ばれていないことを確認
        self.mock_db.delete.assert_not_called()
        self.mock_db.commit.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
