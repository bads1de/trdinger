"""
ツール機能のテスト

WeekendFilter などのエントリーフィルターツールをテストします。
"""

import pytest
from unittest.mock import Mock
import pandas as pd

from app.services.auto_strategy.tools import (
    BaseTool,
    ToolContext,
    ToolRegistry,
    tool_registry,
    WeekendFilter,
    weekend_filter,
)
from app.services.auto_strategy.models.tool_gene import ToolGene


class TestWeekendFilter:
    """WeekendFilter ツールのテスト"""

    def test_name(self):
        """ツール名のテスト"""
        assert weekend_filter.name == "weekend_filter"

    def test_description(self):
        """説明のテスト"""
        assert "土曜日" in weekend_filter.description
        assert "日曜日" in weekend_filter.description

    def test_default_params(self):
        """デフォルトパラメータのテスト"""
        params = weekend_filter.get_default_params()
        assert params["enabled"] is True

    def test_should_skip_on_saturday(self):
        """土曜日にスキップするテスト"""
        saturday = pd.Timestamp("2025-12-13 12:00:00")  # 土曜日
        context = ToolContext(timestamp=saturday)
        params = {"enabled": True}

        assert weekend_filter.should_skip_entry(context, params) is True

    def test_should_skip_on_sunday(self):
        """日曜日にスキップするテスト"""
        sunday = pd.Timestamp("2025-12-14 12:00:00")  # 日曜日
        context = ToolContext(timestamp=sunday)
        params = {"enabled": True}

        assert weekend_filter.should_skip_entry(context, params) is True

    def test_should_not_skip_on_weekday(self):
        """平日にスキップしないテスト"""
        wednesday = pd.Timestamp("2025-12-10 12:00:00")  # 水曜日
        context = ToolContext(timestamp=wednesday)
        params = {"enabled": True}

        assert weekend_filter.should_skip_entry(context, params) is False

    def test_should_not_skip_when_disabled(self):
        """無効時にスキップしないテスト"""
        saturday = pd.Timestamp("2025-12-13 12:00:00")  # 土曜日
        context = ToolContext(timestamp=saturday)
        params = {"enabled": False}

        assert weekend_filter.should_skip_entry(context, params) is False

    def test_should_not_skip_without_timestamp(self):
        """タイムスタンプなしでもスキップしないテスト（フェイルセーフ）"""
        context = ToolContext(timestamp=None)
        params = {"enabled": True}

        assert weekend_filter.should_skip_entry(context, params) is False

    def test_mutate_params(self):
        """パラメータ変異のテスト"""
        original_params = {"enabled": True}

        # 何度か変異を実行してエラーが起きないことを確認
        for _ in range(10):
            mutated = weekend_filter.mutate_params(original_params)
            assert "enabled" in mutated
            assert isinstance(mutated["enabled"], bool)


class TestToolRegistry:
    """ToolRegistry のテスト"""

    def test_weekend_filter_registered(self):
        """WeekendFilter がレジストリに登録されているテスト"""
        tool = tool_registry.get("weekend_filter")
        assert tool is not None
        assert isinstance(tool, WeekendFilter)

    def test_get_all(self):
        """すべてのツール取得テスト"""
        tools = tool_registry.get_all()
        assert len(tools) >= 1
        assert any(t.name == "weekend_filter" for t in tools)

    def test_get_names(self):
        """ツール名リスト取得テスト"""
        names = tool_registry.get_names()
        assert "weekend_filter" in names


class TestToolGene:
    """ToolGene モデルのテスト"""

    def test_creation(self):
        """作成テスト"""
        gene = ToolGene(
            tool_name="weekend_filter",
            enabled=True,
            params={"enabled": True},
        )
        assert gene.tool_name == "weekend_filter"
        assert gene.enabled is True
        assert gene.params == {"enabled": True}

    def test_to_dict(self):
        """辞書変換テスト"""
        gene = ToolGene(
            tool_name="weekend_filter",
            enabled=True,
            params={"enabled": True},
        )
        d = gene.to_dict()
        assert d["tool_name"] == "weekend_filter"
        assert d["enabled"] is True
        assert d["params"] == {"enabled": True}

    def test_from_dict(self):
        """辞書からの作成テスト"""
        data = {
            "tool_name": "weekend_filter",
            "enabled": False,
            "params": {"enabled": False},
        }
        gene = ToolGene.from_dict(data)
        assert gene.tool_name == "weekend_filter"
        assert gene.enabled is False
        assert gene.params == {"enabled": False}


class TestUniversalStrategyToolsIntegration:
    """UniversalStrategy のツール統合テスト"""

    def test_tools_block_entry_on_weekend(self):
        """週末にエントリーがブロックされることのテスト"""
        from app.services.auto_strategy.strategies.universal_strategy import (
            UniversalStrategy,
        )
        from app.services.auto_strategy.models.strategy_gene import StrategyGene
        from app.services.auto_strategy.models.tool_gene import ToolGene

        # 土曜日のタイムスタンプを持つモックデータ
        saturday_timestamp = pd.Timestamp("2025-12-13 12:00:00")  # 土曜日

        # モックの作成
        mock_broker = Mock()
        mock_data = Mock()
        mock_data.index = [saturday_timestamp]
        mock_data.Close = [50000.0]
        mock_data.High = [51000.0]
        mock_data.Low = [49000.0]
        mock_data.Volume = [1000.0]

        # 週末フィルター有効なツール遺伝子
        tool_gene = ToolGene(
            tool_name="weekend_filter",
            enabled=True,
            params={"enabled": True},
        )
        strategy_gene = StrategyGene(
            indicators=[],
            tool_genes=[tool_gene],
        )

        params = {"strategy_gene": strategy_gene}

        # UniversalStrategy のインスタンス化
        strategy = UniversalStrategy(mock_broker, mock_data, params)

        # ツールがエントリーをブロックすることを確認
        assert strategy._tools_block_entry() is True

    def test_tools_allow_entry_on_weekday(self):
        """平日にエントリーが許可されることのテスト"""
        from app.services.auto_strategy.strategies.universal_strategy import (
            UniversalStrategy,
        )
        from app.services.auto_strategy.models.strategy_gene import StrategyGene
        from app.services.auto_strategy.models.tool_gene import ToolGene

        # 水曜日のタイムスタンプを持つモックデータ
        wednesday_timestamp = pd.Timestamp("2025-12-10 12:00:00")  # 水曜日

        mock_broker = Mock()
        mock_data = Mock()
        mock_data.index = [wednesday_timestamp]
        mock_data.Close = [50000.0]
        mock_data.High = [51000.0]
        mock_data.Low = [49000.0]
        mock_data.Volume = [1000.0]

        tool_gene = ToolGene(
            tool_name="weekend_filter",
            enabled=True,
            params={"enabled": True},
        )
        strategy_gene = StrategyGene(
            indicators=[],
            tool_genes=[tool_gene],
        )

        params = {"strategy_gene": strategy_gene}
        strategy = UniversalStrategy(mock_broker, mock_data, params)

        # 平日なのでエントリー許可
        assert strategy._tools_block_entry() is False

    def test_tools_allow_entry_when_tool_disabled(self):
        """ツール無効時にエントリーが許可されることのテスト"""
        from app.services.auto_strategy.strategies.universal_strategy import (
            UniversalStrategy,
        )
        from app.services.auto_strategy.models.strategy_gene import StrategyGene
        from app.services.auto_strategy.models.tool_gene import ToolGene

        saturday_timestamp = pd.Timestamp("2025-12-13 12:00:00")

        mock_broker = Mock()
        mock_data = Mock()
        mock_data.index = [saturday_timestamp]
        mock_data.Close = [50000.0]
        mock_data.High = [51000.0]
        mock_data.Low = [49000.0]
        mock_data.Volume = [1000.0]

        # ツール遺伝子は無効
        tool_gene = ToolGene(
            tool_name="weekend_filter",
            enabled=False,
            params={"enabled": True},
        )
        strategy_gene = StrategyGene(
            indicators=[],
            tool_genes=[tool_gene],
        )

        params = {"strategy_gene": strategy_gene}
        strategy = UniversalStrategy(mock_broker, mock_data, params)

        # ツール遺伝子が無効なのでエントリー許可
        assert strategy._tools_block_entry() is False

    def test_tools_allow_entry_when_no_tool_genes(self):
        """ツール遺伝子がない場合にエントリーが許可されることのテスト"""
        from app.services.auto_strategy.strategies.universal_strategy import (
            UniversalStrategy,
        )
        from app.services.auto_strategy.models.strategy_gene import StrategyGene

        saturday_timestamp = pd.Timestamp("2025-12-13 12:00:00")

        mock_broker = Mock()
        mock_data = Mock()
        mock_data.index = [saturday_timestamp]
        mock_data.Close = [50000.0]
        mock_data.High = [51000.0]
        mock_data.Low = [49000.0]
        mock_data.Volume = [1000.0]

        # tool_genes は空
        strategy_gene = StrategyGene(
            indicators=[],
            tool_genes=[],
        )

        params = {"strategy_gene": strategy_gene}
        strategy = UniversalStrategy(mock_broker, mock_data, params)

        # ツール遺伝子がないのでエントリー許可
        assert strategy._tools_block_entry() is False


