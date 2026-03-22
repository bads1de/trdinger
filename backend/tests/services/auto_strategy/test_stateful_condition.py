"""
StatefulCondition のテスト

シーケンシャルな条件（例：「条件Aが発生後、Nバー以内に条件Bが発生したらエントリー」）
のモデルと評価ロジックをテストします。
"""

import pytest
from unittest.mock import MagicMock

from app.services.auto_strategy.genes.conditions import (
    Condition,
    StatefulCondition,
    StateTracker,
)


class TestStateTracker:
    """StateTracker のテスト"""

    def test_record_and_check_event(self):
        """イベント記録と確認"""
        tracker = StateTracker()

        # イベントを記録
        tracker.record_event("trigger_a", bar_index=10)

        # 同じバーでは True
        assert tracker.was_triggered_within(
            "trigger_a", lookback_bars=5, current_bar=10
        )

        # 5バー以内では True
        assert tracker.was_triggered_within(
            "trigger_a", lookback_bars=5, current_bar=15
        )

        # 5バーを超えると False
        assert not tracker.was_triggered_within(
            "trigger_a", lookback_bars=5, current_bar=16
        )

    def test_unknown_event_returns_false(self):
        """未記録のイベントは False"""
        tracker = StateTracker()

        assert not tracker.was_triggered_within(
            "unknown", lookback_bars=10, current_bar=5
        )

    def test_multiple_events_tracked_separately(self):
        """複数イベントは個別に追跡される"""
        tracker = StateTracker()

        tracker.record_event("event_a", bar_index=5)
        tracker.record_event("event_b", bar_index=10)

        assert tracker.was_triggered_within("event_a", lookback_bars=3, current_bar=8)
        assert not tracker.was_triggered_within(
            "event_a", lookback_bars=3, current_bar=20
        )
        assert tracker.was_triggered_within("event_b", lookback_bars=5, current_bar=14)

    def test_event_can_be_updated(self):
        """同じイベントを再記録すると更新される"""
        tracker = StateTracker()

        tracker.record_event("trigger_a", bar_index=5)
        tracker.record_event("trigger_a", bar_index=15)  # 更新

        # 古いトリガー（bar 5）はもう範囲外
        assert not tracker.was_triggered_within(
            "trigger_a", lookback_bars=5, current_bar=11
        )
        # 新しいトリガー（bar 15）は範囲内
        assert tracker.was_triggered_within(
            "trigger_a", lookback_bars=5, current_bar=18
        )

    def test_reset_clears_all_events(self):
        """リセットで全イベントがクリアされる"""
        tracker = StateTracker()

        tracker.record_event("event_a", bar_index=5)
        tracker.record_event("event_b", bar_index=10)
        tracker.reset()

        assert not tracker.was_triggered_within(
            "event_a", lookback_bars=100, current_bar=5
        )
        assert not tracker.was_triggered_within(
            "event_b", lookback_bars=100, current_bar=10
        )


class TestStatefulCondition:
    """StatefulCondition のテスト"""

    def test_stateful_condition_creation(self):
        """StatefulCondition の作成"""
        trigger = Condition(left_operand="RSI", operator="<", right_operand=30.0)
        follow = Condition(left_operand="close", operator=">", right_operand="SMA_20")

        stateful = StatefulCondition(
            trigger_condition=trigger,
            follow_condition=follow,
            lookback_bars=5,
        )

        assert stateful.trigger_condition == trigger
        assert stateful.follow_condition == follow
        assert stateful.lookback_bars == 5
        assert stateful.cooldown_bars == 0  # デフォルト

    def test_stateful_condition_with_cooldown(self):
        """クールダウン付き StatefulCondition"""
        trigger = Condition(left_operand="RSI", operator="<", right_operand=30.0)
        follow = Condition(left_operand="close", operator=">", right_operand="SMA_20")

        stateful = StatefulCondition(
            trigger_condition=trigger,
            follow_condition=follow,
            lookback_bars=5,
            cooldown_bars=2,
        )

        assert stateful.cooldown_bars == 2

    def test_stateful_condition_validate_valid(self):
        """有効な StatefulCondition のバリデーション"""
        trigger = Condition(left_operand="RSI", operator="<", right_operand=30.0)
        follow = Condition(left_operand="close", operator=">", right_operand="SMA_20")

        stateful = StatefulCondition(
            trigger_condition=trigger,
            follow_condition=follow,
            lookback_bars=5,
        )

        is_valid, errors = stateful.validate()
        assert is_valid
        assert len(errors) == 0

    def test_stateful_condition_validate_invalid_lookback(self):
        """無効な lookback_bars のバリデーション"""
        trigger = Condition(left_operand="RSI", operator="<", right_operand=30.0)
        follow = Condition(left_operand="close", operator=">", right_operand="SMA_20")

        stateful = StatefulCondition(
            trigger_condition=trigger,
            follow_condition=follow,
            lookback_bars=0,  # 無効
        )

        is_valid, errors = stateful.validate()
        assert not is_valid
        assert any("lookback_bars" in e for e in errors)

    def test_stateful_condition_validate_negative_cooldown(self):
        """負のクールダウンは無効"""
        trigger = Condition(left_operand="RSI", operator="<", right_operand=30.0)
        follow = Condition(left_operand="close", operator=">", right_operand="SMA_20")

        stateful = StatefulCondition(
            trigger_condition=trigger,
            follow_condition=follow,
            lookback_bars=5,
            cooldown_bars=-1,  # 無効
        )

        is_valid, errors = stateful.validate()
        assert not is_valid
        assert any("cooldown_bars" in e for e in errors)

    def test_generate_event_name(self):
        """一意のイベント名を生成"""
        trigger = Condition(left_operand="RSI", operator="<", right_operand=30.0)
        follow = Condition(left_operand="close", operator=">", right_operand="SMA_20")

        stateful = StatefulCondition(
            trigger_condition=trigger,
            follow_condition=follow,
            lookback_bars=5,
        )

        event_name = stateful.get_trigger_event_name()
        assert isinstance(event_name, str)
        assert len(event_name) > 0


class TestStatefulConditionEvaluation:
    """ConditionEvaluator での StatefulCondition 評価テスト"""

    @pytest.fixture
    def mock_strategy(self):
        """モック戦略インスタンス"""
        strategy = MagicMock()
        strategy.data = MagicMock()
        strategy.data.Close = [100, 101, 102, 103, 104]
        strategy.data.__len__ = MagicMock(return_value=5)
        # RSI属性
        strategy.RSI = MagicMock()
        strategy.RSI.__getitem__ = MagicMock(return_value=25.0)  # RSI < 30
        # SMA属性
        strategy.SMA_20 = MagicMock()
        strategy.SMA_20.__getitem__ = MagicMock(return_value=100.0)
        return strategy

    def test_evaluate_stateful_condition_trigger_and_follow(self, mock_strategy):
        """トリガー発生後、フォロー条件が成立する場合"""
        from app.services.auto_strategy.core.condition_evaluator import (
            ConditionEvaluator,
        )

        evaluator = ConditionEvaluator()
        tracker = StateTracker()

        trigger = Condition(left_operand=25.0, operator="<", right_operand=30.0)  # True
        follow = Condition(
            left_operand=104.0, operator=">", right_operand=100.0
        )  # True

        stateful = StatefulCondition(
            trigger_condition=trigger,
            follow_condition=follow,
            lookback_bars=5,
        )

        # 現在のバーインデックス
        current_bar = 10

        # トリガー条件を先に記録（bar 8 で発生したと仮定）
        tracker.record_event(stateful.get_trigger_event_name(), bar_index=8)

        # StatefulCondition を評価
        result = evaluator.evaluate_stateful_condition(
            stateful, mock_strategy, tracker, current_bar
        )

        # トリガーが5バー以内（bar 8）にあり、フォロー条件も成立するのでTrue
        assert result is True

    def test_evaluate_stateful_condition_trigger_too_old(self, mock_strategy):
        """トリガーがlookback_barsより古い場合はFalse"""
        from app.services.auto_strategy.core.condition_evaluator import (
            ConditionEvaluator,
        )

        evaluator = ConditionEvaluator()
        tracker = StateTracker()

        trigger = Condition(left_operand=25.0, operator="<", right_operand=30.0)
        follow = Condition(left_operand=104.0, operator=">", right_operand=100.0)

        stateful = StatefulCondition(
            trigger_condition=trigger,
            follow_condition=follow,
            lookback_bars=5,
        )

        current_bar = 20

        # トリガーは bar 10 で発生（10バー前 > lookback_bars 5）
        tracker.record_event(stateful.get_trigger_event_name(), bar_index=10)

        result = evaluator.evaluate_stateful_condition(
            stateful, mock_strategy, tracker, current_bar
        )

        assert result is False

    def test_evaluate_stateful_condition_no_trigger(self, mock_strategy):
        """トリガーが一度も発生していない場合はFalse"""
        from app.services.auto_strategy.core.condition_evaluator import (
            ConditionEvaluator,
        )

        evaluator = ConditionEvaluator()
        tracker = StateTracker()  # 何も記録していない

        trigger = Condition(left_operand=25.0, operator="<", right_operand=30.0)
        follow = Condition(left_operand=104.0, operator=">", right_operand=100.0)

        stateful = StatefulCondition(
            trigger_condition=trigger,
            follow_condition=follow,
            lookback_bars=5,
        )

        result = evaluator.evaluate_stateful_condition(
            stateful, mock_strategy, tracker, current_bar=10
        )

        assert result is False

    def test_evaluate_stateful_condition_follow_fails(self, mock_strategy):
        """トリガーは範囲内だがフォロー条件が不成立の場合はFalse"""
        from app.services.auto_strategy.core.condition_evaluator import (
            ConditionEvaluator,
        )

        evaluator = ConditionEvaluator()
        tracker = StateTracker()

        trigger = Condition(left_operand=25.0, operator="<", right_operand=30.0)
        # フォロー条件が不成立（104 < 200 は True だが、この条件は False になるように設定）
        follow = Condition(
            left_operand=104.0, operator=">", right_operand=200.0
        )  # False

        stateful = StatefulCondition(
            trigger_condition=trigger,
            follow_condition=follow,
            lookback_bars=5,
        )

        tracker.record_event(stateful.get_trigger_event_name(), bar_index=8)

        result = evaluator.evaluate_stateful_condition(
            stateful, mock_strategy, tracker, current_bar=10
        )

        assert result is False

    def test_check_and_record_trigger(self, mock_strategy):
        """トリガー条件の評価と記録"""
        from app.services.auto_strategy.core.condition_evaluator import (
            ConditionEvaluator,
        )

        evaluator = ConditionEvaluator()
        tracker = StateTracker()

        # 成立するトリガー条件
        trigger = Condition(left_operand=25.0, operator="<", right_operand=30.0)
        follow = Condition(left_operand=104.0, operator=">", right_operand=100.0)

        stateful = StatefulCondition(
            trigger_condition=trigger,
            follow_condition=follow,
            lookback_bars=5,
        )

        current_bar = 10

        # トリガーを評価して記録
        triggered = evaluator.check_and_record_trigger(
            stateful, mock_strategy, tracker, current_bar
        )

        assert triggered is True
        # StateTracker にイベントが記録されている
        assert tracker.was_triggered_within(
            stateful.get_trigger_event_name(), lookback_bars=5, current_bar=10
        )


class TestStatefulConditionSerialization:
    """StatefulCondition のシリアライズ/デシリアライズテスト"""

    def test_stateful_condition_to_dict(self):
        """StatefulCondition を辞書に変換"""
        from app.services.auto_strategy.serializers.serialization import (
            DictConverter,
        )

        converter = DictConverter()

        trigger = Condition(left_operand="RSI", operator="<", right_operand=30.0)
        follow = Condition(left_operand="close", operator=">", right_operand=100.0)

        stateful = StatefulCondition(
            trigger_condition=trigger,
            follow_condition=follow,
            lookback_bars=5,
            cooldown_bars=2,
        )

        result = converter.stateful_condition_to_dict(stateful)

        assert result["trigger_condition"]["left_operand"] == "RSI"
        assert result["trigger_condition"]["operator"] == "<"
        assert result["trigger_condition"]["right_operand"] == 30.0
        assert result["follow_condition"]["left_operand"] == "close"
        assert result["follow_condition"]["operator"] == ">"
        assert result["follow_condition"]["right_operand"] == 100.0
        assert result["lookback_bars"] == 5
        assert result["cooldown_bars"] == 2
        assert result["enabled"] is True

    def test_dict_to_stateful_condition(self):
        """辞書から StatefulCondition を復元"""
        from app.services.auto_strategy.serializers.serialization import (
            DictConverter,
        )

        converter = DictConverter()

        data = {
            "trigger_condition": {
                "left_operand": "RSI",
                "operator": "<",
                "right_operand": 30.0,
            },
            "follow_condition": {
                "left_operand": "close",
                "operator": ">",
                "right_operand": 100.0,
            },
            "lookback_bars": 5,
            "cooldown_bars": 2,
            "enabled": True,
        }

        result = converter.dict_to_stateful_condition(data)

        assert result.trigger_condition.left_operand == "RSI"
        assert result.trigger_condition.operator == "<"
        assert result.trigger_condition.right_operand == 30.0
        assert result.follow_condition.left_operand == "close"
        assert result.follow_condition.operator == ">"
        assert result.follow_condition.right_operand == 100.0
        assert result.lookback_bars == 5
        assert result.cooldown_bars == 2
        assert result.enabled is True

    def test_round_trip_serialization(self):
        """シリアライズ→デシリアライズのラウンドトリップ"""
        from app.services.auto_strategy.serializers.serialization import (
            DictConverter,
        )

        converter = DictConverter()

        trigger = Condition(left_operand="MACD", operator=">", right_operand=0.0)
        follow = Condition(left_operand="close", operator=">", right_operand="SMA_50")

        original = StatefulCondition(
            trigger_condition=trigger,
            follow_condition=follow,
            lookback_bars=10,
            cooldown_bars=3,
        )

        # シリアライズ
        as_dict = converter.stateful_condition_to_dict(original)
        # デシリアライズ
        restored = converter.dict_to_stateful_condition(as_dict)

        # 検証
        assert restored.trigger_condition.left_operand == "MACD"
        assert restored.trigger_condition.operator == ">"
        assert restored.follow_condition.right_operand == "SMA_50"
        assert restored.lookback_bars == 10
        assert restored.cooldown_bars == 3


class TestStatefulConditionIntegration:
    """StatefulCondition の統合テスト - 実際のワークフローをシミュレート"""

    def test_full_stateful_condition_workflow(self):
        """
        完全なステートフル条件ワークフロー:
        1. トリガー条件が成立
        2. 数バー後にフォロー条件も成立
        3. エントリーシグナルが発生
        """
        from app.services.auto_strategy.core.condition_evaluator import (
            ConditionEvaluator,
        )

        evaluator = ConditionEvaluator()
        tracker = StateTracker()

        # 「RSI < 30」がトリガー、「close > SMA」がフォロー
        trigger = Condition(left_operand=25.0, operator="<", right_operand=30.0)
        follow = Condition(left_operand=105.0, operator=">", right_operand=100.0)

        stateful = StatefulCondition(
            trigger_condition=trigger,
            follow_condition=follow,
            lookback_bars=5,
        )

        mock_strategy = MagicMock()

        # シミュレーション: Bar 1 - トリガー条件成立
        bar = 1
        triggered = evaluator.check_and_record_trigger(
            stateful, mock_strategy, tracker, bar
        )
        assert triggered is True

        # Bar 1 では、トリガーが「たった今」発生したので、
        # フォロー条件評価は True になるはず
        result_bar1 = evaluator.evaluate_stateful_condition(
            stateful, mock_strategy, tracker, bar
        )
        assert result_bar1 is True, "Bar 1でフォロー条件成立すべき"

        # Bar 3 - lookback_bars (5) 以内なのでまだ有効
        bar = 3
        result_bar3 = evaluator.evaluate_stateful_condition(
            stateful, mock_strategy, tracker, bar
        )
        assert result_bar3 is True, "Bar 3 (2バー後) でもまだ有効"

        # Bar 6 - lookback_bars (5) 以内（5バー後）
        bar = 6
        result_bar6 = evaluator.evaluate_stateful_condition(
            stateful, mock_strategy, tracker, bar
        )
        assert result_bar6 is True, "Bar 6 (5バー後) はギリギリ有効"

        # Bar 7 - lookback_bars (5) を超えた（6バー後）
        bar = 7
        result_bar7 = evaluator.evaluate_stateful_condition(
            stateful, mock_strategy, tracker, bar
        )
        assert result_bar7 is False, "Bar 7 (6バー後) は期限切れ"

    def test_trigger_renewed_within_lookback(self):
        """
        トリガーがlookback期間内に再発生した場合、
        期限が更新されることを確認
        """
        from app.services.auto_strategy.core.condition_evaluator import (
            ConditionEvaluator,
        )

        evaluator = ConditionEvaluator()
        tracker = StateTracker()

        trigger = Condition(left_operand=25.0, operator="<", right_operand=30.0)
        follow = Condition(left_operand=105.0, operator=">", right_operand=100.0)

        stateful = StatefulCondition(
            trigger_condition=trigger,
            follow_condition=follow,
            lookback_bars=3,
        )

        mock_strategy = MagicMock()

        # Bar 1: トリガー発生
        evaluator.check_and_record_trigger(stateful, mock_strategy, tracker, 1)

        # Bar 3: まだ有効（2バー後）
        result_bar3 = evaluator.evaluate_stateful_condition(
            stateful, mock_strategy, tracker, 3
        )
        assert result_bar3 is True

        # Bar 4: トリガーが再発生（期限更新）
        evaluator.check_and_record_trigger(stateful, mock_strategy, tracker, 4)

        # Bar 7: 元のトリガー(Bar 1)からは6バー後で期限切れだが、
        # 更新されたトリガー(Bar 4)からは3バー後なのでまだ有効
        result_bar7 = evaluator.evaluate_stateful_condition(
            stateful, mock_strategy, tracker, 7
        )
        assert result_bar7 is True, "更新されたトリガーからは3バー後なので有効"

        # Bar 8: 更新されたトリガー(Bar 4)からも4バー後で期限切れ
        result_bar8 = evaluator.evaluate_stateful_condition(
            stateful, mock_strategy, tracker, 8
        )
        assert result_bar8 is False, "更新されたトリガーからも4バー後で期限切れ"

    def test_strategy_gene_with_stateful_conditions(self):
        """
        StrategyGene に stateful_conditions を設定して
        シリアライズ/デシリアライズできることを確認
        """
        from app.services.auto_strategy.genes.strategy import StrategyGene
        from app.services.auto_strategy.serializers.serialization import DictConverter

        # StrategyGene を作成
        trigger = Condition(left_operand="RSI", operator="<", right_operand=30.0)
        follow = Condition(left_operand="close", operator=">", right_operand="SMA_20")

        stateful = StatefulCondition(
            trigger_condition=trigger,
            follow_condition=follow,
            lookback_bars=5,
        )

        gene = StrategyGene(
            id="test_gene_1",
            indicators=[],
            stateful_conditions=[stateful],
        )

        # シリアライズ
        converter = DictConverter()
        gene_dict = converter.strategy_gene_to_dict(gene)

        # デシリアライズ
        restored_gene = converter.dict_to_strategy_gene(gene_dict, StrategyGene)

        # 検証
        assert len(restored_gene.stateful_conditions) == 1
        sc = restored_gene.stateful_conditions[0]
        assert sc.trigger_condition.left_operand == "RSI"
        assert sc.follow_condition.right_operand == "SMA_20"
        assert sc.lookback_bars == 5




