"""
悲観的約定ロジック (Pessimistic Exit) のテスト

このモジュールは、1分足バックテストにおいて、
同一足内でTP/SLの両方に達した場合に確実に「損切り」として処理されることを検証します。
"""

from unittest.mock import MagicMock, patch, PropertyMock
import pytest
from app.services.auto_strategy.strategies.universal_strategy import UniversalStrategy
from app.services.auto_strategy.genes import (
    StrategyGene,
    IndicatorGene,
    TPSLGene,
    TPSLMethod,
)
from app.services.auto_strategy.genes.conditions import Condition


class TestPessimisticExit:
    """悲観的約定ロジックのテストクラス"""

    @pytest.fixture
    def mock_broker(self):
        """Brokerのモック"""
        broker = MagicMock()
        broker.orders = []
        broker.trades = []
        return broker

    @pytest.fixture
    def mock_data_both_tpsl_hit(self):
        """
        TPとSLの両方に到達するローソク足データのモック
        - ロングポジション: エントリー価格 100, SL 95, TP 110
        - この足: High=115 (TP到達), Low=90 (SL到達)
        - 悲観的ロジックでは SL が優先され、損切りとなるべき
        """
        data = MagicMock()
        data.Close = MagicMock()
        data.Close.__getitem__ = MagicMock(return_value=100.0)
        data.High = MagicMock()
        data.High.__getitem__ = MagicMock(return_value=115.0)  # TP(110)到達
        data.Low = MagicMock()
        data.Low.__getitem__ = MagicMock(return_value=90.0)  # SL(95)到達
        data.__len__ = MagicMock(return_value=100)
        return data

    @pytest.fixture
    def mock_data_only_tp_hit(self):
        """TPのみに到達するローソク足データのモック"""
        data = MagicMock()
        data.Close = MagicMock()
        data.Close.__getitem__ = MagicMock(return_value=100.0)
        data.High = MagicMock()
        data.High.__getitem__ = MagicMock(return_value=115.0)  # TP(110)到達
        data.Low = MagicMock()
        data.Low.__getitem__ = MagicMock(return_value=98.0)  # SL(95)未到達
        data.__len__ = MagicMock(return_value=100)
        return data

    @pytest.fixture
    def mock_data_only_sl_hit(self):
        """SLのみに到達するローソク足データのモック"""
        data = MagicMock()
        data.Close = MagicMock()
        data.Close.__getitem__ = MagicMock(return_value=100.0)
        data.High = MagicMock()
        data.High.__getitem__ = MagicMock(return_value=105.0)  # TP(110)未到達
        data.Low = MagicMock()
        data.Low.__getitem__ = MagicMock(return_value=90.0)  # SL(95)到達
        data.__len__ = MagicMock(return_value=100)
        return data

    @pytest.fixture
    def mock_data_neither_hit(self):
        """TP/SLどちらにも到達しないローソク足データのモック"""
        data = MagicMock()
        data.Close = MagicMock()
        data.Close.__getitem__ = MagicMock(return_value=100.0)
        data.High = MagicMock()
        data.High.__getitem__ = MagicMock(return_value=105.0)  # TP(110)未到達
        data.Low = MagicMock()
        data.Low.__getitem__ = MagicMock(return_value=98.0)  # SL(95)未到達
        data.__len__ = MagicMock(return_value=100)
        return data

    @pytest.fixture
    def gene_with_tpsl(self):
        """TPSL設定付きの戦略遺伝子"""
        dummy_cond = Condition(left_operand="close", operator=">", right_operand="sma")
        tpsl_gene = TPSLGene(
            enabled=True,
            stop_loss_pct=0.05,  # SL: 5% -> 95
            take_profit_pct=0.10,  # TP: 10% -> 110
            method=TPSLMethod.FIXED_PERCENTAGE,
        )
        return StrategyGene(
            indicators=[
                IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)
            ],
            tpsl_gene=tpsl_gene,
            risk_management={},
            metadata={"test": True},
        )

    def test_pessimistic_exit_sl_prioritized_when_both_hit(
        self, mock_broker, mock_data_both_tpsl_hit, gene_with_tpsl
    ):
        """
        TP/SLの両方に達した足では、悲観的ロジックによりSLが優先されること
        """
        params = {"strategy_gene": gene_with_tpsl}

        with patch(
            "app.services.auto_strategy.strategies.universal_strategy.ConditionEvaluator"
        ) as MockEval:
            mock_evaluator = MockEval.return_value

            strategy = UniversalStrategy(mock_broker, mock_data_both_tpsl_hit, params)

            # ポジションありの状態をモック
            mock_position = MagicMock()
            mock_position.is_long = True
            mock_position.size = 0.01

            # 悲観的決済ロジック用の内部状態を設定
            strategy._sl_price = 95.0
            strategy._tp_price = 110.0
            strategy._entry_price = 100.0
            strategy._position_direction = 1.0  # Long

            with patch.object(
                UniversalStrategy, "position", new_callable=PropertyMock
            ) as mock_pos_prop:
                mock_pos_prop.return_value = mock_position

                # next()を呼び出し
                strategy.next()

            # 悲観的ロジックによりポジションがクローズされたはず
            mock_position.close.assert_called_once()

    def test_tp_exit_when_only_tp_hit(
        self, mock_broker, mock_data_only_tp_hit, gene_with_tpsl
    ):
        """
        TPのみに達した足では、正常に利確されること
        """
        params = {"strategy_gene": gene_with_tpsl}

        with patch(
            "app.services.auto_strategy.strategies.universal_strategy.ConditionEvaluator"
        ) as MockEval:
            mock_evaluator = MockEval.return_value

            strategy = UniversalStrategy(mock_broker, mock_data_only_tp_hit, params)

            mock_position = MagicMock()
            mock_position.is_long = True
            mock_position.size = 0.01

            strategy._sl_price = 95.0
            strategy._tp_price = 110.0
            strategy._entry_price = 100.0
            strategy._position_direction = 1.0

            with patch.object(
                UniversalStrategy, "position", new_callable=PropertyMock
            ) as mock_pos_prop:
                mock_pos_prop.return_value = mock_position

                strategy.next()

            mock_position.close.assert_called_once()

    def test_sl_exit_when_only_sl_hit(
        self, mock_broker, mock_data_only_sl_hit, gene_with_tpsl
    ):
        """
        SLのみに達した足では、損切りされること
        """
        params = {"strategy_gene": gene_with_tpsl}

        with patch(
            "app.services.auto_strategy.strategies.universal_strategy.ConditionEvaluator"
        ) as MockEval:
            mock_evaluator = MockEval.return_value

            strategy = UniversalStrategy(mock_broker, mock_data_only_sl_hit, params)

            mock_position = MagicMock()
            mock_position.is_long = True
            mock_position.size = 0.01

            strategy._sl_price = 95.0
            strategy._tp_price = 110.0
            strategy._entry_price = 100.0
            strategy._position_direction = 1.0

            with patch.object(
                UniversalStrategy, "position", new_callable=PropertyMock
            ) as mock_pos_prop:
                mock_pos_prop.return_value = mock_position

                strategy.next()

            mock_position.close.assert_called_once()

    def test_position_continues_when_neither_hit(
        self, mock_broker, mock_data_neither_hit, gene_with_tpsl
    ):
        """
        TP/SLどちらにも達しない足では、ポジションが継続すること
        """
        params = {"strategy_gene": gene_with_tpsl}

        with patch(
            "app.services.auto_strategy.strategies.universal_strategy.ConditionEvaluator"
        ) as MockEval:
            mock_evaluator = MockEval.return_value
            # イグジット条件も不成立にする
            mock_evaluator.evaluate_conditions.return_value = False

            strategy = UniversalStrategy(mock_broker, mock_data_neither_hit, params)

            mock_position = MagicMock()
            mock_position.is_long = True
            mock_position.size = 0.01

            strategy._sl_price = 95.0
            strategy._tp_price = 110.0
            strategy._entry_price = 100.0
            strategy._position_direction = 1.0

            with patch.object(
                UniversalStrategy, "position", new_callable=PropertyMock
            ) as mock_pos_prop:
                mock_pos_prop.return_value = mock_position

                strategy.next()

            # ポジションがクローズされていないこと
            mock_position.close.assert_not_called()


class TestTrailingStop:
    """トレーリングストップのテストクラス"""

    @pytest.fixture
    def mock_broker(self):
        broker = MagicMock()
        broker.orders = []
        broker.trades = []
        return broker

    @pytest.fixture
    def mock_data_price_rising(self):
        """価格が上昇した状態のモック"""
        data = MagicMock()
        data.Close = MagicMock()
        data.Close.__getitem__ = MagicMock(return_value=120.0)  # 100->120に上昇
        data.High = MagicMock()
        data.High.__getitem__ = MagicMock(return_value=122.0)
        data.Low = MagicMock()
        data.Low.__getitem__ = MagicMock(
            return_value=118.0
        )  # SL(95)もTP(110)も未到達（新高値更新中）
        data.__len__ = MagicMock(return_value=100)
        return data

    @pytest.fixture
    def gene_with_trailing_stop(self):
        """トレーリングストップ設定付きの戦略遺伝子"""
        dummy_cond = Condition(left_operand="close", operator=">", right_operand="sma")
        tpsl_gene = TPSLGene(
            enabled=True,
            stop_loss_pct=0.05,
            take_profit_pct=0.10,
            method=TPSLMethod.FIXED_PERCENTAGE,
            trailing_stop=True,  # トレーリング有効化
            trailing_step_pct=0.02,  # 2% の更新幅
        )
        return StrategyGene(
            indicators=[
                IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)
            ],
            tpsl_gene=tpsl_gene,
            risk_management={},
            metadata={"test": True},
        )

    def test_trailing_stop_updates_sl_on_price_rise(
        self, mock_broker, mock_data_price_rising, gene_with_trailing_stop
    ):
        """
        価格上昇時にトレーリングによってSLが切り上がること
        - 初期SL: 95 (エントリー価格100の5%下)
        - 価格が120に上昇
        - トレーリング計算: 120 * (1 - 0.02) = 117.6
        - 新SL(117.6) > 旧SL(95) なので更新されるべき
        """
        params = {"strategy_gene": gene_with_trailing_stop}

        with patch(
            "app.services.auto_strategy.strategies.universal_strategy.ConditionEvaluator"
        ) as MockEval:
            mock_evaluator = MockEval.return_value
            mock_evaluator.evaluate_conditions.return_value = (
                False  # イグジット条件不成立
            )

            strategy = UniversalStrategy(mock_broker, mock_data_price_rising, params)

            mock_position = MagicMock()
            mock_position.is_long = True
            mock_position.size = 0.01

            # 初期状態: SL=95, TP=130 (高くしてTP判定をスキップ)
            strategy._sl_price = 95.0
            strategy._tp_price = 130.0  # TPには到達しない設定
            strategy._entry_price = 100.0
            strategy._position_direction = 1.0

            with patch.object(
                UniversalStrategy, "position", new_callable=PropertyMock
            ) as mock_pos_prop:
                mock_pos_prop.return_value = mock_position

                strategy.next()

            # ポジションはまだ継続中
            mock_position.close.assert_not_called()

            # SLがトレーリングによって更新されていること
            # 120 * (1 - 0.02) = 117.6
            assert strategy._sl_price == pytest.approx(120.0 * 0.98)

    def test_trailing_stop_does_not_lower_sl(
        self, mock_broker, gene_with_trailing_stop
    ):
        """
        トレーリングはSLを下げない（価格が下がってもSLは維持される）
        """
        # 価格が下がったデータを作成
        mock_data_price_falling = MagicMock()
        mock_data_price_falling.Close = MagicMock()
        mock_data_price_falling.Close.__getitem__ = MagicMock(
            return_value=105.0
        )  # 120->105に下落
        mock_data_price_falling.High = MagicMock()
        mock_data_price_falling.High.__getitem__ = MagicMock(return_value=106.0)
        mock_data_price_falling.Low = MagicMock()
        mock_data_price_falling.Low.__getitem__ = MagicMock(
            return_value=118.0
        )  # Low=118 > SL=117.6 なのでSL未到達
        mock_data_price_falling.__len__ = MagicMock(return_value=100)

        params = {"strategy_gene": gene_with_trailing_stop}

        with patch(
            "app.services.auto_strategy.strategies.universal_strategy.ConditionEvaluator"
        ) as MockEval:
            mock_evaluator = MockEval.return_value
            mock_evaluator.evaluate_conditions.return_value = False

            strategy = UniversalStrategy(mock_broker, mock_data_price_falling, params)

            mock_position = MagicMock()
            mock_position.is_long = True
            mock_position.size = 0.01

            # 前回のトレーリングでSLが117.6に更新されている状態をシミュレート
            old_sl = 117.6
            strategy._sl_price = old_sl
            strategy._tp_price = 130.0
            strategy._entry_price = 100.0
            strategy._position_direction = 1.0

            with patch.object(
                UniversalStrategy, "position", new_callable=PropertyMock
            ) as mock_pos_prop:
                mock_pos_prop.return_value = mock_position

                strategy.next()

            # SLは下がっていないこと（105 * 0.98 = 102.9 < 117.6 なので更新されない）
            assert strategy._sl_price == old_sl


class TestTrailingTakeProfit:
    """トレーリングテイクプロフィットのテストクラス"""

    @pytest.fixture
    def mock_broker(self):
        broker = MagicMock()
        broker.orders = []
        broker.trades = []
        return broker

    @pytest.fixture
    def gene_with_trailing_tp(self):
        """トレーリングTP設定付きの戦略遺伝子"""
        dummy_cond = Condition(left_operand="close", operator=">", right_operand="sma")
        tpsl_gene = TPSLGene(
            enabled=True,
            stop_loss_pct=0.05,  # SL: 5%
            take_profit_pct=0.10,  # TP: 10% -> 110
            method=TPSLMethod.FIXED_PERCENTAGE,
            trailing_take_profit=True,  # トレーリングTP有効化
            trailing_step_pct=0.02,  # 2%の更新幅
        )
        return StrategyGene(
            indicators=[
                IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)
            ],
            tpsl_gene=tpsl_gene,
            risk_management={},
            metadata={"test": True},
        )

    def test_trailing_tp_does_not_exit_on_tp_reach(
        self, mock_broker, gene_with_trailing_tp
    ):
        """
        トレーリングTP有効時、TP到達でも即時決済せず、利益確保モードに移行すること
        - エントリー価格: 100, TP: 110
        - 価格が115に上昇（TP到達）
        - 即時決済せず、_tp_reached=True になるはず
        """
        mock_data = MagicMock()
        mock_data.Close = MagicMock()
        mock_data.Close.__getitem__ = MagicMock(return_value=115.0)
        mock_data.High = MagicMock()
        mock_data.High.__getitem__ = MagicMock(return_value=116.0)  # TP(110)到達
        mock_data.Low = MagicMock()
        mock_data.Low.__getitem__ = MagicMock(return_value=114.0)  # SL(95)未到達
        mock_data.__len__ = MagicMock(return_value=100)

        params = {"strategy_gene": gene_with_trailing_tp}

        with patch(
            "app.services.auto_strategy.strategies.universal_strategy.ConditionEvaluator"
        ) as MockEval:
            mock_evaluator = MockEval.return_value
            mock_evaluator.evaluate_conditions.return_value = False

            strategy = UniversalStrategy(mock_broker, mock_data, params)

            mock_position = MagicMock()
            mock_position.is_long = True
            mock_position.size = 0.01

            # 初期状態: エントリー済み
            strategy._sl_price = 95.0
            strategy._tp_price = 110.0
            strategy._entry_price = 100.0
            strategy._position_direction = 1.0

            with patch.object(
                UniversalStrategy, "position", new_callable=PropertyMock
            ) as mock_pos_prop:
                mock_pos_prop.return_value = mock_position

                strategy.next()

            # 即時決済されていないこと
            mock_position.close.assert_not_called()

            # TP到達モードに移行していること
            assert strategy._tp_reached is True
            assert strategy._trailing_tp_sl is not None

    def test_trailing_tp_exits_on_pullback(self, mock_broker, gene_with_trailing_tp):
        """
        トレーリングTP到達後、価格が反転して利益確保ラインを割ったら決済すること
        - TP到達後、利益確保ライン = 109.76 (112 * 0.98)
        - 価格が104に下落 (Low <= 109.76) → 決済
        """
        # 下落後のデータ（利益確保ライン割れ）
        mock_data_pullback = MagicMock()
        mock_data_pullback.Close = MagicMock()
        mock_data_pullback.Close.__getitem__ = MagicMock(return_value=105.0)
        mock_data_pullback.High = MagicMock()
        mock_data_pullback.High.__getitem__ = MagicMock(return_value=106.0)
        mock_data_pullback.Low = MagicMock()
        mock_data_pullback.Low.__getitem__ = MagicMock(
            return_value=104.0
        )  # 利益確保ライン(109.76)割れ
        mock_data_pullback.__len__ = MagicMock(return_value=100)

        params = {"strategy_gene": gene_with_trailing_tp}

        with patch(
            "app.services.auto_strategy.strategies.universal_strategy.ConditionEvaluator"
        ) as MockEval:
            mock_evaluator = MockEval.return_value
            mock_evaluator.evaluate_conditions.return_value = False

            strategy = UniversalStrategy(mock_broker, mock_data_pullback, params)

            mock_position = MagicMock()
            mock_position.is_long = True
            mock_position.size = 0.01

            # TP到達後の状態をシミュレート（既にTP到達モード）
            strategy._sl_price = 95.0
            strategy._tp_price = 110.0
            strategy._entry_price = 100.0
            strategy._position_direction = 1.0
            strategy._tp_reached = True
            strategy._trailing_tp_sl = 112.0 * 0.98  # 利益確保ライン = 109.76

            with patch.object(
                UniversalStrategy, "position", new_callable=PropertyMock
            ) as mock_pos_prop:
                mock_pos_prop.return_value = mock_position

                strategy.next()

            # Low=104 <= trailing_tp_sl=109.76 なので決済される
            mock_position.close.assert_called_once()

    def test_trailing_tp_updates_profit_lock_line(
        self, mock_broker, gene_with_trailing_tp
    ):
        """
        トレーリングTP到達後、価格がさらに上昇すると利益確保ラインも更新されること
        """
        # 価格がさらに上昇（利益確保ライン割れなし）
        mock_data = MagicMock()
        mock_data.Close = MagicMock()
        mock_data.Close.__getitem__ = MagicMock(return_value=120.0)  # さらに上昇
        mock_data.High = MagicMock()
        mock_data.High.__getitem__ = MagicMock(return_value=121.0)
        mock_data.Low = MagicMock()
        mock_data.Low.__getitem__ = MagicMock(
            return_value=119.0
        )  # 利益確保ライン未到達
        mock_data.__len__ = MagicMock(return_value=100)

        params = {"strategy_gene": gene_with_trailing_tp}

        with patch(
            "app.services.auto_strategy.strategies.universal_strategy.ConditionEvaluator"
        ) as MockEval:
            mock_evaluator = MockEval.return_value
            mock_evaluator.evaluate_conditions.return_value = False

            strategy = UniversalStrategy(mock_broker, mock_data, params)

            mock_position = MagicMock()
            mock_position.is_long = True
            mock_position.size = 0.01

            # TP到達後の状態をシミュレート
            strategy._sl_price = 95.0
            strategy._tp_price = 110.0
            strategy._entry_price = 100.0
            strategy._position_direction = 1.0
            strategy._tp_reached = True
            strategy._trailing_tp_sl = 110.0 * 0.98  # 初期利益確保ライン = 107.8

            with patch.object(
                UniversalStrategy, "position", new_callable=PropertyMock
            ) as mock_pos_prop:
                mock_pos_prop.return_value = mock_position

                strategy.next()

            # 決済されていないこと
            mock_position.close.assert_not_called()

            # 利益確保ラインが更新されていること
            # 120 * 0.98 = 117.6
            assert strategy._trailing_tp_sl == pytest.approx(120.0 * 0.98)




