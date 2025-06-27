#!/usr/bin/env python3
"""
StrategyBuilderServiceのユニットテスト
"""

import pytest
import json
from unittest.mock import Mock, patch
from sqlalchemy.orm import Session

from app.core.services.strategy_builder_service import StrategyBuilderService
from database.models import UserStrategy


class TestStrategyBuilderService:
    """StrategyBuilderServiceのテストクラス"""

    def setup_method(self):
        """各テストメソッドの前に実行される初期化"""
        self.mock_db = Mock(spec=Session)
        self.mock_user_strategy_repo = Mock()
        self.mock_technical_indicator_service = Mock()
        self.mock_strategy_factory = Mock()

        # StrategyBuilderServiceのインスタンスを作成
        self.service = StrategyBuilderService(self.mock_db)
        self.service.user_strategy_repo = self.mock_user_strategy_repo
        self.service.technical_indicator_service = self.mock_technical_indicator_service
        self.service.strategy_factory = self.mock_strategy_factory

    def test_get_available_indicators_success(self):
        """指標一覧取得の成功テスト"""
        # モックデータの準備
        mock_indicators = {
            "SMA": {
                "name": "SMA",
                "description": "単純移動平均",
                "parameters": {"period": {"type": "integer", "default": 20}},
            },
            "RSI": {
                "name": "RSI",
                "description": "相対力指数",
                "parameters": {"period": {"type": "integer", "default": 14}},
            },
        }
        self.mock_technical_indicator_service.supported_indicators = mock_indicators

        # テスト実行
        result = self.service.get_available_indicators()

        # 検証
        assert isinstance(result, dict)
        assert "trend" in result
        assert "momentum" in result
        assert len(result["trend"]) > 0  # SMAがtrendカテゴリに含まれる
        assert len(result["momentum"]) > 0  # RSIがmomentumカテゴリに含まれる

    def test_validate_strategy_config_valid(self):
        """有効な戦略設定の検証テスト"""
        # 有効な戦略設定
        strategy_config = {
            "indicators": [
                {"type": "SMA", "parameters": {"period": 20}, "enabled": True}
            ],
            "entry_conditions": [
                {"type": "threshold", "indicator": "SMA", "operator": ">", "value": 100}
            ],
            "exit_conditions": [
                {"type": "threshold", "indicator": "SMA", "operator": "<", "value": 95}
            ],
        }

        # モックの設定
        self.mock_strategy_factory.validate_gene.return_value = (True, [])

        # テスト実行
        is_valid, errors = self.service.validate_strategy_config(strategy_config)

        # 検証
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_strategy_config_invalid_missing_fields(self):
        """必須フィールドが不足している戦略設定の検証テスト"""
        # 不正な戦略設定（indicatorsが不足）
        strategy_config = {"entry_conditions": [], "exit_conditions": []}

        # テスト実行
        is_valid, errors = self.service.validate_strategy_config(strategy_config)

        # 検証
        assert is_valid is False
        assert len(errors) > 0
        assert any("indicators" in error for error in errors)

    def test_validate_strategy_config_invalid_empty_indicators(self):
        """指標が空の戦略設定の検証テスト"""
        # 不正な戦略設定（indicatorsが空）
        strategy_config = {
            "indicators": [],
            "entry_conditions": [],
            "exit_conditions": [],
        }

        # テスト実行
        is_valid, errors = self.service.validate_strategy_config(strategy_config)

        # 検証
        assert is_valid is False
        assert len(errors) > 0
        assert any("少なくとも1つの指標" in error for error in errors)

    def test_convert_to_strategy_gene_format(self):
        """戦略設定のStrategyGene形式変換テスト"""
        # 入力データ
        strategy_config = {
            "indicators": [
                {"type": "SMA", "parameters": {"period": 20}, "enabled": True}
            ],
            "entry_conditions": [
                {"type": "threshold", "indicator": "SMA", "operator": ">", "value": 100}
            ],
            "exit_conditions": [
                {"type": "threshold", "indicator": "SMA", "operator": "<", "value": 95}
            ],
        }

        # テスト実行
        result = self.service._convert_to_strategy_gene_format(strategy_config)

        # 検証
        assert "indicators" in result
        assert "entry_conditions" in result
        assert "exit_conditions" in result
        assert "risk_management" in result
        assert "metadata" in result

        # 指標の変換確認
        assert len(result["indicators"]) == 1
        assert result["indicators"][0]["type"] == "SMA"
        assert result["indicators"][0]["parameters"]["period"] == 20

        # 条件の変換確認
        assert len(result["entry_conditions"]) == 1
        assert result["entry_conditions"][0]["left_operand"] == "SMA"
        assert result["entry_conditions"][0]["operator"] == ">"
        assert result["entry_conditions"][0]["right_operand"] == 100

    def test_save_strategy_success(self):
        """戦略保存の成功テスト"""
        # 入力データ
        name = "テスト戦略"
        description = "テスト用の戦略"
        strategy_config = {
            "indicators": [
                {"type": "SMA", "parameters": {"period": 20}, "enabled": True}
            ],
            "entry_conditions": [
                {"type": "threshold", "indicator": "SMA", "operator": ">", "value": 100}
            ],
            "exit_conditions": [
                {"type": "threshold", "indicator": "SMA", "operator": "<", "value": 95}
            ],
        }

        # モックの設定
        mock_user_strategy = Mock(spec=UserStrategy)
        mock_user_strategy.id = 1
        mock_user_strategy.name = name
        mock_user_strategy.description = description

        self.mock_strategy_factory.validate_gene.return_value = (True, [])
        self.mock_user_strategy_repo.create.return_value = mock_user_strategy

        # テスト実行
        result = self.service.save_strategy(name, description, strategy_config)

        # 検証
        assert result is not None
        assert result.id == 1
        assert result.name == name
        assert result.description == description

        # リポジトリのcreateメソッドが呼ばれたことを確認
        self.mock_user_strategy_repo.create.assert_called_once()

    def test_save_strategy_invalid_config(self):
        """無効な戦略設定での保存テスト"""
        # 無効な戦略設定
        strategy_config = {
            "indicators": [],  # 空の指標リスト
            "entry_conditions": [],
            "exit_conditions": [],
        }

        # テスト実行と検証
        with pytest.raises(ValueError) as exc_info:
            self.service.save_strategy("テスト", "説明", strategy_config)

        assert "戦略設定が無効です" in str(exc_info.value)

    def test_get_strategies_success(self):
        """戦略一覧取得の成功テスト"""
        # モックデータの準備
        mock_strategies = [
            Mock(id=1, name="戦略1", description="説明1"),
            Mock(id=2, name="戦略2", description="説明2"),
        ]
        self.mock_user_strategy_repo.get_all.return_value = mock_strategies

        # テスト実行
        result = self.service.get_strategies()

        # 検証
        assert len(result) == 2
        assert result[0].id == 1
        assert result[1].id == 2

        # リポジトリのget_allメソッドが呼ばれたことを確認
        self.mock_user_strategy_repo.get_all.assert_called_once()

    def test_get_strategy_by_id_success(self):
        """ID指定での戦略取得の成功テスト"""
        # モックデータの準備
        mock_strategy = Mock(spec=UserStrategy)
        mock_strategy.id = 1
        mock_strategy.name = "戦略1"
        mock_strategy.description = "説明1"
        self.mock_user_strategy_repo.get_by_id.return_value = mock_strategy

        # テスト実行
        result = self.service.get_strategy_by_id(1)

        # 検証
        assert result is not None
        assert result.id == 1
        assert result.name == "戦略1"

        # リポジトリのget_by_idメソッドが呼ばれたことを確認
        self.mock_user_strategy_repo.get_by_id.assert_called_once_with(1)

    def test_get_strategy_by_id_not_found(self):
        """存在しないIDでの戦略取得テスト"""
        # モックの設定（戦略が見つからない）
        self.mock_user_strategy_repo.get_by_id.return_value = None

        # テスト実行
        result = self.service.get_strategy_by_id(999)

        # 検証
        assert result is None

        # リポジトリのget_by_idメソッドが呼ばれたことを確認
        self.mock_user_strategy_repo.get_by_id.assert_called_once_with(999)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
