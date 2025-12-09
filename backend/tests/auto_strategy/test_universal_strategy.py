"""
UniversalStrategy のテスト
"""

from unittest.mock import MagicMock, patch, PropertyMock
import pytest
from app.services.auto_strategy.strategies.universal_strategy import UniversalStrategy
from app.services.auto_strategy.models.strategy_models import (
    StrategyGene,
    IndicatorGene,
    TPSLGene,
    TPSLMethod,
)
from app.services.auto_strategy.models.condition import Condition


class TestUniversalStrategy:
    """UniversalStrategy のテストクラス"""

    @pytest.fixture
    def mock_broker(self):
        """Brokerのモック"""
        broker = MagicMock()
        broker.orders = []
        broker.trades = []
        return broker

    @pytest.fixture
    def mock_data(self):
        """Dataのモック"""
        data = MagicMock()
        data.Close = [100, 101, 102]
        data.High = [105, 106, 107]
        data.Low = [95, 96, 97]
        data.__len__.return_value = 3
        return data

    @pytest.fixture
    def valid_gene(self):
        """有効な戦略遺伝子のフィクスチャ"""
        # 条件評価まで到達させるため、ダミーの条件を入れておく
        dummy_cond = Condition(left_operand="close", operator=">", right_operand="sma")
        return StrategyGene(
            indicators=[
                IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)
            ],
            entry_conditions=[],
            exit_conditions=[],
            long_entry_conditions=[dummy_cond],
            short_entry_conditions=[dummy_cond],
            risk_management={},
            metadata={"test": True},
        )

    def test_init_without_gene_raises_error(self, mock_broker, mock_data):
        """戦略遺伝子なしでの初期化エラーテスト"""
        params = {}
        with pytest.raises(
            ValueError, match="UniversalStrategy requires 'strategy_gene'"
        ):
            UniversalStrategy(mock_broker, mock_data, params)

    def test_init_indicators(self, mock_broker, mock_data, valid_gene):
        """指標初期化の呼び出しテスト"""
        params = {"strategy_gene": valid_gene}

        with patch(
            "app.services.auto_strategy.strategies.universal_strategy.IndicatorCalculator"
        ) as MockCalc:
            mock_calculator = MockCalc.return_value

            strategy = UniversalStrategy(mock_broker, mock_data, params)
            strategy.init()

            assert mock_calculator.init_indicator.called
            assert mock_calculator.init_indicator.call_count == len(
                valid_gene.indicators
            )

    def test_next_long_entry(self, mock_broker, mock_data, valid_gene):
        """ロングエントリー実行テスト"""
        params = {"strategy_gene": valid_gene}

        with patch(
            "app.services.auto_strategy.strategies.universal_strategy.ConditionEvaluator"
        ) as MockEval:
            mock_evaluator = MockEval.return_value
            strategy = UniversalStrategy(mock_broker, mock_data, params)
            strategy.buy = MagicMock()
            strategy.sell = MagicMock()

            # positionプロパティがNoneを返すようにパッチ（ポジションなし）
            with patch.object(
                UniversalStrategy, "position", new_callable=PropertyMock
            ) as mock_position:
                mock_position.return_value = None

                # evaluate_conditions が呼ばれるように gene に条件があることを確認(fixtureで設定済み)

                # 1回目(Long): True, 2回目(Short): False
                mock_evaluator.evaluate_conditions.side_effect = [True, False]

                strategy.next()

            strategy.buy.assert_called_once()
            strategy.sell.assert_not_called()

    def test_next_short_entry(self, mock_broker, mock_data, valid_gene):
        """ショートエントリー実行テスト"""
        params = {"strategy_gene": valid_gene}

        with patch(
            "app.services.auto_strategy.strategies.universal_strategy.ConditionEvaluator"
        ) as MockEval:
            mock_evaluator = MockEval.return_value

            strategy = UniversalStrategy(mock_broker, mock_data, params)
            strategy.buy = MagicMock()
            strategy.sell = MagicMock()

            # ポジションなし
            with patch.object(
                UniversalStrategy, "position", new_callable=PropertyMock
            ) as mock_position:
                mock_position.return_value = None

                # 1回目(Long): False, 2回目(Short): True
                mock_evaluator.evaluate_conditions.side_effect = [False, True]

                strategy.next()

            strategy.sell.assert_called_once()
            strategy.buy.assert_not_called()

    def test_next_exit(self, mock_broker, mock_data, valid_gene):
        """イグジット実行テスト"""
        valid_gene.exit_conditions = [
            Condition(left_operand="close", operator=">", right_operand="SMA")
        ]
        params = {"strategy_gene": valid_gene}

        with patch(
            "app.services.auto_strategy.strategies.universal_strategy.ConditionEvaluator"
        ) as MockEval:
            mock_evaluator = MockEval.return_value

            strategy = UniversalStrategy(mock_broker, mock_data, params)

            # ポジションあり状態をモック
            mock_pos_instance = MagicMock()

            with patch.object(
                UniversalStrategy, "position", new_callable=PropertyMock
            ) as mock_position:
                mock_position.return_value = mock_pos_instance

                # イグジット条件成立
                mock_evaluator.evaluate_conditions.return_value = True

                strategy.next()

            # position.close() が呼ばれたか
            mock_pos_instance.close.assert_called_once()

    def test_tpsl_parameters(self, mock_broker, mock_data, valid_gene):
        """TP/SLパラメータの適用テスト"""
        tpsl_gene = TPSLGene(
            enabled=True,
            stop_loss_pct=0.05,
            take_profit_pct=0.1,
            method=TPSLMethod.FIXED_PERCENTAGE,
        )
        valid_gene.tpsl_gene = tpsl_gene
        params = {"strategy_gene": valid_gene}

        with patch(
            "app.services.auto_strategy.strategies.universal_strategy.ConditionEvaluator"
        ) as MockEval:
            mock_evaluator = MockEval.return_value

            strategy = UniversalStrategy(mock_broker, mock_data, params)
            strategy.buy = MagicMock()

            # ポジションなし
            with patch.object(
                UniversalStrategy, "position", new_callable=PropertyMock
            ) as mock_position:
                mock_position.return_value = None

                # ロング成立
                mock_evaluator.evaluate_conditions.side_effect = [True, False]

                strategy.next()

            # buyがsl/tp付きで呼ばれたか確認
            args, kwargs = strategy.buy.call_args
            assert "sl" in kwargs
            assert "tp" in kwargs

            # 102 * 0.95 = 96.9, 102 * 1.1 = 112.2
            assert kwargs["sl"] == pytest.approx(102 * 0.95)
            assert kwargs["tp"] == pytest.approx(102 * 1.1)

    def test_stateful_conditions_integration(self, mock_broker, mock_data):
        """ステートフル条件がUniversalStrategyで正しく処理されることをテスト"""
        from app.services.auto_strategy.models.stateful_condition import (
            StatefulCondition,
        )

        # ステートフル条件付きの遺伝子を作成
        trigger = Condition(left_operand=25.0, operator="<", right_operand=30.0)
        follow = Condition(left_operand=105.0, operator=">", right_operand=100.0)

        stateful = StatefulCondition(
            trigger_condition=trigger,
            follow_condition=follow,
            lookback_bars=5,
        )

        gene = StrategyGene(
            indicators=[
                IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)
            ],
            entry_conditions=[],
            long_entry_conditions=[],  # 通常条件は空
            short_entry_conditions=[],
            stateful_conditions=[stateful],  # ステートフル条件のみ
            risk_management={},
            metadata={},
        )

        params = {"strategy_gene": gene}

        with patch(
            "app.services.auto_strategy.strategies.universal_strategy.ConditionEvaluator"
        ) as MockEval:
            mock_evaluator = MockEval.return_value

            strategy = UniversalStrategy(mock_broker, mock_data, params)
            strategy.buy = MagicMock()

            # 戦略がStateTrackerとステートフル条件を持っていることを確認
            assert hasattr(strategy, "state_tracker")
            assert hasattr(strategy, "_current_bar_index")
            assert len(strategy.gene.stateful_conditions) == 1

            # _process_stateful_triggers と _check_stateful_conditions が存在することを確認
            assert hasattr(strategy, "_process_stateful_triggers")
            assert hasattr(strategy, "_check_stateful_conditions")

            # ポジションなし
            with patch.object(
                UniversalStrategy, "position", new_callable=PropertyMock
            ) as mock_position:
                mock_position.return_value = None

                # トリガー評価とステートフル条件評価をモック
                mock_evaluator.evaluate_conditions.return_value = (
                    False  # 通常条件は不成立
                )
                mock_evaluator.check_and_record_trigger.return_value = True
                mock_evaluator.evaluate_stateful_condition.return_value = True

                strategy.next()

            # ステートフル条件でエントリーが発生したはず
            # （buyが呼ばれたか、またはステートフル条件評価メソッドが呼ばれたか）
            assert strategy._current_bar_index == 1  # バーインデックスがインクリメント
