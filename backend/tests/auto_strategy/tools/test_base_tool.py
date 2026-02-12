"""
BaseTool のユニットテスト

mutate_params のデフォルト実装（enabled 反転）をテストします。
"""

import pytest
from unittest.mock import patch, MagicMock
from typing import Any, Dict

from app.services.auto_strategy.tools.base import BaseTool, ToolContext


class ConcreteTool(BaseTool):
    """テスト用の具象クラス（抽象メソッドを実装）"""

    @property
    def name(self) -> str:
        return "test_tool"

    def should_skip_entry(self, context: ToolContext, params: Dict[str, Any]) -> bool:
        return False

    def get_default_params(self) -> Dict[str, Any]:
        return {"enabled": True}


class ConcreteToolWithCustomMutation(BaseTool):
    """テスト用: super().mutate_params() + 固有パラメータ変異"""

    @property
    def name(self) -> str:
        return "custom_mutation_tool"

    def should_skip_entry(self, context: ToolContext, params: Dict[str, Any]) -> bool:
        return False

    def get_default_params(self) -> Dict[str, Any]:
        return {"enabled": True, "window_minutes": 15}

    def mutate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        new_params = super().mutate_params(params)
        # 固有パラメータの変異
        if __import__("random").random() < 0.2:
            current = new_params.get("window_minutes", 15)
            delta = __import__("random").randint(-5, 5)
            new_params["window_minutes"] = max(5, min(30, current + delta))
        return new_params


class TestBaseToolMutateParams:
    """BaseTool.mutate_params のデフォルト実装テスト"""

    @pytest.fixture
    def tool(self):
        return ConcreteTool()

    def test_mutate_params_flips_enabled_when_probability_met(self, tool):
        """20%の確率で enabled を反転する"""
        with patch("random.random", return_value=0.1):  # < 0.2 → 変異する
            params = {"enabled": True}
            result = tool.mutate_params(params)
            assert result["enabled"] is False

    def test_mutate_params_keeps_enabled_when_probability_not_met(self, tool):
        """80%の確率で enabled を維持する"""
        with patch("random.random", return_value=0.5):  # >= 0.2 → 変異しない
            params = {"enabled": True}
            result = tool.mutate_params(params)
            assert result["enabled"] is True

    def test_mutate_params_flips_false_to_true(self, tool):
        """enabled=False → True に反転"""
        with patch("random.random", return_value=0.1):
            params = {"enabled": False}
            result = tool.mutate_params(params)
            assert result["enabled"] is True

    def test_mutate_params_does_not_mutate_original(self, tool):
        """元の辞書を変更しない（コピーを返す）"""
        with patch("random.random", return_value=0.1):
            params = {"enabled": True}
            result = tool.mutate_params(params)
            assert params["enabled"] is True  # 元は変わらない
            assert result["enabled"] is False  # 新しい方が変わる

    def test_mutate_params_preserves_extra_keys(self, tool):
        """enabled 以外のキーも保持される"""
        with patch("random.random", return_value=0.5):
            params = {"enabled": True, "some_param": 42}
            result = tool.mutate_params(params)
            assert result["some_param"] == 42

    def test_mutate_params_handles_missing_enabled(self, tool):
        """enabled キーが存在しない場合のデフォルト動作"""
        with patch("random.random", return_value=0.1):
            params = {}
            result = tool.mutate_params(params)
            # get("enabled", True) → not True → False
            assert result["enabled"] is False

    def test_mutate_params_boundary_probability(self, tool):
        """確率境界値: 0.2ちょうどでは変異しない"""
        with patch("random.random", return_value=0.2):
            params = {"enabled": True}
            result = tool.mutate_params(params)
            assert result["enabled"] is True  # 0.2 < 0.2 は False


class TestSubclassMutateParams:
    """サブクラスが super().mutate_params() を呼ぶパターンのテスト"""

    @pytest.fixture
    def tool(self):
        return ConcreteToolWithCustomMutation()

    def test_super_call_handles_enabled_mutation(self, tool):
        """super() により enabled の変異が処理される"""
        with patch("random.random", side_effect=[0.1, 0.5]):
            # 1回目(0.1): enabled 反転 → True→False
            # 2回目(0.5): window_minutes 変化なし
            params = {"enabled": True, "window_minutes": 15}
            result = tool.mutate_params(params)
            assert result["enabled"] is False
            assert result["window_minutes"] == 15

    def test_both_mutations_applied(self, tool):
        """enabled と固有パラメータの両方が変異"""
        with patch("random.random", side_effect=[0.1, 0.1]), \
             patch("random.randint", return_value=3):
            params = {"enabled": True, "window_minutes": 15}
            result = tool.mutate_params(params)
            assert result["enabled"] is False
            assert result["window_minutes"] == 18
