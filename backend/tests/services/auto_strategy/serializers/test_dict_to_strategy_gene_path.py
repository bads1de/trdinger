"""
シード戦略復元パス（dict_to_strategy_gene）のテスト

反復改善ループの PreviousStrategySeedProvider が DB から読み出した
gene_data を StrategyGene へ復元する際に使われる
GeneSerializer.dict_to_strategy_gene の経路を重点的にテストします。
"""

from enum import Enum
from unittest.mock import patch

import pytest

from app.services.auto_strategy.genes import (
    Condition,
    IndicatorGene,
    PositionSizingGene,
    StatefulCondition,
    StrategyGene,
    TPSLGene,
)
from app.services.auto_strategy.serializers.serialization import GeneSerializer


@pytest.fixture
def serializer() -> GeneSerializer:
    return GeneSerializer()


def _seed_gene_data() -> dict:
    """DB に保存されるシード戦略の gene_data（seed_strategy_provider が読む形式）。"""
    return {
        "id": "prev-strategy-001",
        "indicators": [
            {
                "type": "SMA",
                "parameters": {"period": 20},
                "enabled": True,
                "id": "ind-1",
            }
        ],
        "long_entry_conditions": [
            {
                "left_operand": "close",
                "operator": ">",
                "right_operand": "sma",
                "direction": "long",
            }
        ],
        "short_entry_conditions": [],
        "long_exit_conditions": [],
        "short_exit_conditions": [],
        "stateful_conditions": [
            {
                "trigger_condition": {
                    "left_operand": "rsi",
                    "operator": "<",
                    "right_operand": 30,
                    "direction": "long",
                },
                "follow_condition": {
                    "left_operand": "close",
                    "operator": ">",
                    "right_operand": "ema",
                    "direction": "long",
                },
                "lookback_bars": 7,
                "cooldown_bars": 2,
                "direction": "long",
                "enabled": True,
            }
        ],
        "risk_management": {"position_size": 0.1234567, "max_risk": 0.05},
        "tpsl_gene": {
            "enabled": True,
            "method": "fixed_percentage",
            "stop_loss_pct": 0.01,
            "take_profit_pct": 0.03,
        },
        "position_sizing_gene": {
            "enabled": True,
            "method": "fixed_quantity",
            "fixed_quantity": 1000,
        },
        "metadata": {
            "validation": {"passed": True, "pass_rate": 0.8},
            "seed_strategy": "previous",
        },
    }


class TestSeedStrategyRestoration:
    """シード戦略復元のメインパス"""

    def test_restores_full_seed_strategy(self, serializer):
        """DB の gene_data から全フィールドが復元される"""
        data = _seed_gene_data()
        restored = serializer.dict_to_strategy_gene(data, StrategyGene)

        assert isinstance(restored, StrategyGene)
        assert restored.id == "prev-strategy-001"
        assert restored.indicators[0].type == "SMA"
        assert restored.long_entry_conditions[0].left_operand == "close"
        assert restored.long_entry_conditions[0].direction == "long"
        # 自動検証メタデータが保持される（シードの再利用条件）
        assert restored.metadata["validation"]["passed"] is True
        assert restored.metadata["validation"]["pass_rate"] == 0.8
        assert restored.metadata["seed_strategy"] == "previous"
        # サブ遺伝子の復元
        assert isinstance(restored.tpsl_gene, TPSLGene)
        assert isinstance(restored.position_sizing_gene, PositionSizingGene)
        assert isinstance(restored.stateful_conditions[0], StatefulCondition)

    def test_uses_default_strategy_gene_class_when_none(self, serializer):
        """クラス未指定（None）時は StrategyGene が使われる"""
        data = _seed_gene_data()
        restored = serializer.dict_to_strategy_gene(data)  # strategy_gene_class=None
        assert isinstance(restored, StrategyGene)
        assert restored.id == "prev-strategy-001"

    def test_restored_strategy_can_be_reseeded(self, serializer):
        """復元した戦略を再度シリアライズして別のインスタンスを復元できる"""
        data = _seed_gene_data()
        first = serializer.dict_to_strategy_gene(data, StrategyGene)
        second = serializer.dict_to_strategy_gene(data, StrategyGene)

        # キャッシュから返されるが、コピーが返るため同一インスタンスではない
        assert first is not second
        assert first.id == second.id
        assert first.metadata == second.metadata

    def test_restored_gene_mutation_does_not_affect_cache(self, serializer):
        """復元した遺伝子を変更してもキャッシュ済みの復元結果に影響しない"""
        data = _seed_gene_data()
        first = serializer.dict_to_strategy_gene(data, StrategyGene)
        first.metadata["validation"]["passed"] = False

        second = serializer.dict_to_strategy_gene(data, StrategyGene)
        assert second.metadata["validation"]["passed"] is True

    def test_clear_caches(self, serializer):
        """clear_caches で両キャッシュが空になる"""
        data = _seed_gene_data()
        serializer.dict_to_strategy_gene(data, StrategyGene)
        serializer.strategy_gene_to_dict(
            StrategyGene(
                id="x",
                indicators=[IndicatorGene(type="SMA", parameters={"period": 20})],
                metadata={},
            )
        )
        stats = serializer.get_cache_statistics()
        assert stats["serialize_cache_size"] == 1
        assert stats["deserialize_cache_size"] == 1

        serializer.clear_caches()
        stats = serializer.get_cache_statistics()
        assert stats["serialize_cache_size"] == 0
        assert stats["deserialize_cache_size"] == 0


class TestStatefulConditionEdgeCases:
    """StatefulCondition の None / エラー分岐"""

    def test_stateful_condition_to_dict_none(self, serializer):
        """None 入力は None を返す"""
        assert serializer.stateful_condition_to_dict(None) is None

    def test_stateful_condition_to_dict_error(self, serializer):
        """不正オブジェクトは ValueError にラップされる"""
        with pytest.raises(ValueError, match="StatefulConditionの辞書変換に失敗"):
            serializer.stateful_condition_to_dict(object())  # type: ignore[arg-type]

    def test_dict_to_stateful_condition_none(self, serializer):
        """None 入力は None を返す"""
        assert serializer.dict_to_stateful_condition(None) is None

    def test_dict_to_stateful_condition_error(self, serializer):
        """条件が壊れたデータは ValueError にラップされる"""
        with pytest.raises(ValueError, match="StatefulConditionの復元に失敗"):
            serializer.dict_to_stateful_condition(
                {
                    "trigger_condition": {},  # left_operand が無い
                    "follow_condition": {},
                }
            )


class _TestEnum(Enum):
    """Enum 分岐テスト用のプレーンな Enum（str 継承しない）"""

    ALPHA = "alpha"


class _ListLikeIndividual(list):
    """DEAP 個体風の list サブクラス（indicators 属性を持つ）"""

    def __init__(self) -> None:
        super().__init__([1])  # 先頭要素は StrategyGene ではない
        self.indicators = []
        self.long_entry_conditions = []


class TestCacheHelperBranches:
    """キャッシュキー正規化・コピー処理の未カバー分岐"""

    def test_freeze_float_nan(self, serializer):
        assert serializer._freeze_for_cache_key(float("nan")) == ("float", "nan")

    def test_freeze_float_inf(self, serializer):
        assert serializer._freeze_for_cache_key(float("inf")) == ("float", "inf")
        assert serializer._freeze_for_cache_key(float("-inf")) == ("float", "-inf")

    def test_freeze_enum(self, serializer):
        frozen = serializer._freeze_for_cache_key(_TestEnum.ALPHA)
        assert frozen[0] == "enum"
        assert frozen[1] == "_TestEnum"
        assert frozen[2] == "alpha"

    def test_freeze_set(self, serializer):
        frozen = serializer._freeze_for_cache_key({2, 1})
        assert frozen[0] == "set"
        assert set(frozen[1]) == {1, 2}

    def test_freeze_object_with_dict(self, serializer):
        class _Plain:
            def __init__(self):
                self.x = 1

        frozen = serializer._freeze_for_cache_key(_Plain())
        assert frozen[0] == "object"
        # ローカルクラスの __qualname__ は <locals> を含むため末尾で検証
        assert frozen[1].endswith("_Plain")

    def test_freeze_repr_fallback_and_key_fallback(self, serializer):
        class _BadRepr:
            __slots__ = ()

            def __repr__(self):
                raise RuntimeError("bad repr")

        bad = _BadRepr()
        # repr フォールバック自体は例外を伝播させ、上位で object_id にフォールバック
        with pytest.raises(RuntimeError):
            serializer._freeze_for_cache_key(bad)
        key = serializer._generate_dict_cache_key(bad)
        assert key[0] == "object_id"

    def test_copy_cached_value_bytes(self, serializer):
        assert serializer._copy_cached_value(b"abc") == "616263"

    def test_copy_cached_value_enum(self, serializer):
        assert serializer._copy_cached_value(_TestEnum.ALPHA) == "alpha"

    def test_copy_cached_value_tuple_and_set(self, serializer):
        assert serializer._copy_cached_value((1, 2)) == (1, 2)
        assert serializer._copy_cached_value({1, 2}) == [1, 2]

    def test_copy_cached_value_deepcopy_fallback(self, serializer):
        class _BadDeepCopy:
            def __deepcopy__(self, memo):
                raise RuntimeError("no deepcopy")

        value = _BadDeepCopy()
        result = serializer._copy_cached_value(value)
        assert result == repr(value)

    def test_strategy_gene_to_dict_error_wrap(self, serializer):
        """codec のエラーは ValueError にラップされる"""
        with patch.object(
            serializer._strategy_gene_codec,
            "strategy_gene_to_dict",
            side_effect=RuntimeError("boom"),
        ):
            with pytest.raises(ValueError, match="戦略遺伝子の辞書変換に失敗"):
                serializer.strategy_gene_to_dict(
                    StrategyGene(id="x", indicators=[], metadata={})
                )


class TestFromListEdgeCases:
    """GeneSerializer.from_list のエッジケース"""

    def test_from_list_empty_returns_none(self, serializer):
        assert serializer.from_list([], StrategyGene) is None

    def test_from_list_direct_strategy_gene(self, serializer):
        gene = StrategyGene(id="direct", indicators=[], metadata={})
        assert serializer.from_list(gene, StrategyGene) is gene

    def test_from_list_first_element_is_strategy_gene(self, serializer):
        gene = StrategyGene(id="first", indicators=[], metadata={})
        restored = serializer.from_list([gene], StrategyGene)
        assert restored is gene

    def test_from_list_object_with_indicators(self, serializer):
        restored = serializer.from_list(_ListLikeIndividual(), StrategyGene)
        assert restored is not None
        assert restored.indicators == []

    def test_from_list_unrecognized_returns_none(self, serializer):
        assert serializer.from_list([1, 2, 3], StrategyGene) is None


class TestConditionConverterBranches:
    """条件コンバーターの分岐補完"""

    def test_condition_with_direction_round_trip(self, serializer):
        cond = Condition(
            left_operand="close",
            operator=">",
            right_operand="open",
            direction="long",
        )
        data = serializer.condition_to_dict(cond)
        assert data["direction"] == "long"
        restored = serializer.dict_to_condition(data)
        assert restored.direction == "long"

    def test_condition_without_direction_omits_key(self, serializer):
        cond = Condition(left_operand="close", operator=">", right_operand="open")
        data = serializer.condition_to_dict(cond)
        assert "direction" not in data

    def test_dict_to_condition_error(self, serializer):
        with pytest.raises(ValueError, match="条件の復元に失敗"):
            serializer.dict_to_condition({})
