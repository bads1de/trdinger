"""
UniversalStrategy の ML フィルター統合テスト

課題 3.1: MLフィルターが「真のフィルター」として機能することを検証します。
MLがエントリー条件成立時にリアルタイムで判断し、危険な相場でのエントリーを拒否できることを確認します。
"""

from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pandas as pd
import pytest

from app.services.auto_strategy.models.condition import Condition
from app.services.auto_strategy.models.strategy_models import (
    IndicatorGene,
    StrategyGene,
    TPSLGene,
    TPSLMethod,
)
from app.services.auto_strategy.strategies.universal_strategy import UniversalStrategy


class TestMLFilterIntegration:
    """ML フィルター統合テストクラス"""

    @pytest.fixture
    def mock_broker(self):
        """Brokerのモック"""
        broker = MagicMock()
        broker.equity = 100000.0
        return broker

    @pytest.fixture
    def mock_data(self):
        """Dataのモック"""
        data = MagicMock()
        data.Close = MagicMock()
        data.Close.__getitem__ = MagicMock(return_value=50000)
        data.Close.__len__ = MagicMock(return_value=100)
        data.High = MagicMock()
        data.High.__getitem__ = MagicMock(return_value=51000)
        data.High.__len__ = MagicMock(return_value=100)
        data.Low = MagicMock()
        data.Low.__getitem__ = MagicMock(return_value=49000)
        data.Low.__len__ = MagicMock(return_value=100)
        data.index = MagicMock()
        data.index.__getitem__ = MagicMock(return_value=pd.Timestamp("2024-01-01"))
        data.__len__ = MagicMock(return_value=100)
        return data

    @pytest.fixture
    def valid_gene(self):
        """有効な戦略遺伝子のフィクスチャ"""
        return StrategyGene(
            indicators=[
                IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True)
            ],
            entry_conditions=[
                Condition(
                    left_operand={"indicator": "RSI_14"},
                    operator="<",
                    right_operand=30.0,
                )
            ],
            exit_conditions=[],
            tpsl_gene=TPSLGene(
                enabled=True,
                method=TPSLMethod.FIXED_PERCENTAGE,
                take_profit_pct=0.02,
                stop_loss_pct=0.01,
            ),
        )

    @pytest.fixture
    def mock_ml_predictor(self):
        """ML予測器のモック"""
        predictor = MagicMock()
        predictor.is_trained.return_value = True
        predictor.predict.return_value = {"up": 0.5, "down": 0.3, "range": 0.2}
        return predictor

    def test_ml_predictor_parameter_accepted(
        self, mock_broker, mock_data, valid_gene, mock_ml_predictor
    ):
        """戦略がml_predictorパラメータを受け入れることをテスト"""
        with patch.object(UniversalStrategy, "init"):
            params = {"strategy_gene": valid_gene, "ml_predictor": mock_ml_predictor}
            strategy = UniversalStrategy(mock_broker, mock_data, params)

            assert hasattr(strategy, "ml_predictor")
            assert strategy.ml_predictor is mock_ml_predictor

    def test_ml_predictor_is_optional(self, mock_broker, mock_data, valid_gene):
        """ml_predictorがオプションであることをテスト"""
        with patch.object(UniversalStrategy, "init"):
            params = {"strategy_gene": valid_gene}
            strategy = UniversalStrategy(mock_broker, mock_data, params)

            assert hasattr(strategy, "ml_predictor")
            assert strategy.ml_predictor is None

    def test_ml_allows_entry_long_signal_positive(
        self, mock_broker, mock_data, valid_gene, mock_ml_predictor
    ):
        """MLがロングエントリーを許可する場合のテスト"""
        # up > down + 0.1 の場合、Longエントリーを許可
        mock_ml_predictor.predict.return_value = {"up": 0.6, "down": 0.2, "range": 0.2}

        with patch.object(UniversalStrategy, "init"):
            params = {"strategy_gene": valid_gene, "ml_predictor": mock_ml_predictor}
            strategy = UniversalStrategy(mock_broker, mock_data, params)
            # _dataを直接設定（backtesting.pyの内部実装に合わせる）
            strategy._data = mock_data

            result = strategy._ml_allows_entry(direction=1.0)
            assert result is True

    def test_ml_allows_entry_long_signal_rejected(
        self, mock_broker, mock_data, valid_gene, mock_ml_predictor
    ):
        """MLがロングエントリーを拒否する場合のテスト"""
        # up < down + 0.1 の場合、Longエントリーを拒否
        mock_ml_predictor.predict.return_value = {"up": 0.3, "down": 0.4, "range": 0.3}

        with patch.object(UniversalStrategy, "init"):
            params = {"strategy_gene": valid_gene, "ml_predictor": mock_ml_predictor}
            strategy = UniversalStrategy(mock_broker, mock_data, params)
            strategy._data = mock_data

            result = strategy._ml_allows_entry(direction=1.0)
            assert result is False

    def test_ml_allows_entry_short_signal_positive(
        self, mock_broker, mock_data, valid_gene, mock_ml_predictor
    ):
        """MLがショートエントリーを許可する場合のテスト"""
        # down > up + 0.1 の場合、Shortエントリーを許可
        mock_ml_predictor.predict.return_value = {"up": 0.2, "down": 0.6, "range": 0.2}

        with patch.object(UniversalStrategy, "init"):
            params = {"strategy_gene": valid_gene, "ml_predictor": mock_ml_predictor}
            strategy = UniversalStrategy(mock_broker, mock_data, params)
            strategy._data = mock_data

            result = strategy._ml_allows_entry(direction=-1.0)
            assert result is True

    def test_ml_allows_entry_short_signal_rejected(
        self, mock_broker, mock_data, valid_gene, mock_ml_predictor
    ):
        """MLがショートエントリーを拒否する場合のテスト"""
        # down < up + 0.1 の場合、Shortエントリーを拒否
        mock_ml_predictor.predict.return_value = {"up": 0.4, "down": 0.3, "range": 0.3}

        with patch.object(UniversalStrategy, "init"):
            params = {"strategy_gene": valid_gene, "ml_predictor": mock_ml_predictor}
            strategy = UniversalStrategy(mock_broker, mock_data, params)
            strategy._data = mock_data

            result = strategy._ml_allows_entry(direction=-1.0)
            assert result is False

    def test_ml_filter_blocks_entry_in_next(
        self, mock_broker, mock_data, valid_gene, mock_ml_predictor
    ):
        """next()でMLフィルターがエントリーをブロックすることをテスト

        このテストでは、_ml_allows_entryがFalseを返す場合に
        buy()が呼ばれないことを検証します。
        """
        with patch.object(UniversalStrategy, "init"):
            params = {
                "strategy_gene": valid_gene,
                "ml_predictor": mock_ml_predictor,
                "ml_filter_threshold": 0.1,
            }
            strategy = UniversalStrategy(mock_broker, mock_data, params)
            strategy._data = mock_data
            strategy.buy = MagicMock()
            strategy.sell = MagicMock()

            # 必須属性の初期化
            strategy._current_bar_index = 0
            strategy._pending_orders = []
            strategy._minute_data = None
            strategy._sl_price = None
            strategy._tp_price = None
            strategy._entry_price = None
            strategy._position_direction = 0.0
            strategy._tp_reached = False
            strategy._trailing_tp_sl = None

            # _ml_allows_entryがFalseを返すようにモック
            with patch.object(
                strategy, "_ml_allows_entry", return_value=False
            ) as mock_ml_check:
                with patch.object(
                    strategy, "_check_long_entry_conditions", return_value=True
                ):
                    with patch.object(
                        strategy, "_check_short_entry_conditions", return_value=False
                    ):
                        with patch.object(
                            strategy, "_check_pending_order_fills", return_value=None
                        ):
                            with patch.object(
                                strategy, "_expire_pending_orders", return_value=None
                            ):
                                with patch.object(
                                    strategy,
                                    "_process_stateful_triggers",
                                    return_value=None,
                                ):
                                    with patch.object(
                                        strategy,
                                        "_get_stateful_entry_direction",
                                        return_value=None,
                                    ):
                                        with patch.object(
                                            type(strategy),
                                            "position",
                                            new_callable=PropertyMock,
                                            return_value=None,
                                        ):
                                            strategy.next()

            # MLフィルターがチェックされたことを確認
            mock_ml_check.assert_called_once_with(1.0)
            # MLフィルターがブロックしたため、buyは呼ばれない
            strategy.buy.assert_not_called()

    def test_ml_filter_allows_entry_in_next(
        self, mock_broker, mock_data, valid_gene, mock_ml_predictor
    ):
        """next()でMLフィルターがエントリーを許可することをテスト

        このテストでは、_ml_allows_entryがTrueを返す場合に
        buy()が呼ばれることを検証します。
        """
        with patch.object(UniversalStrategy, "init"):
            params = {
                "strategy_gene": valid_gene,
                "ml_predictor": mock_ml_predictor,
                "ml_filter_threshold": 0.1,
            }
            strategy = UniversalStrategy(mock_broker, mock_data, params)
            strategy._data = mock_data
            strategy.gene = valid_gene
            strategy.buy = MagicMock()
            strategy.sell = MagicMock()

            # 必須属性の初期化
            strategy._current_bar_index = 0
            strategy._pending_orders = []
            strategy._minute_data = None
            strategy._sl_price = None
            strategy._tp_price = None
            strategy._entry_price = None
            strategy._position_direction = 0.0
            strategy._tp_reached = False
            strategy._trailing_tp_sl = None

            # _ml_allows_entryがTrueを返すようにモック
            with patch.object(
                strategy, "_ml_allows_entry", return_value=True
            ) as mock_ml_check:
                with patch.object(
                    strategy, "_check_long_entry_conditions", return_value=True
                ):
                    with patch.object(
                        strategy, "_check_short_entry_conditions", return_value=False
                    ):
                        with patch.object(
                            strategy, "_check_pending_order_fills", return_value=None
                        ):
                            with patch.object(
                                strategy, "_expire_pending_orders", return_value=None
                            ):
                                with patch.object(
                                    strategy,
                                    "_process_stateful_triggers",
                                    return_value=None,
                                ):
                                    with patch.object(
                                        strategy,
                                        "_get_stateful_entry_direction",
                                        return_value=None,
                                    ):
                                        with patch.object(
                                            strategy,
                                            "_calculate_position_size",
                                            return_value=0.01,
                                        ):
                                            with patch.object(
                                                strategy,
                                                "_get_effective_tpsl_gene",
                                                return_value=None,
                                            ):
                                                with patch.object(
                                                    strategy,
                                                    "_get_effective_entry_gene",
                                                    return_value=None,
                                                ):
                                                    with patch.object(
                                                        type(strategy),
                                                        "position",
                                                        new_callable=PropertyMock,
                                                        return_value=None,
                                                    ):
                                                        strategy.next()

            # MLフィルターがチェックされたことを確認
            mock_ml_check.assert_called_once_with(1.0)
            # MLフィルターが許可したため、buyが呼ばれる
            strategy.buy.assert_called_once()

    def test_ml_predictor_not_trained_allows_entry(
        self, mock_broker, mock_data, valid_gene, mock_ml_predictor
    ):
        """ML予測器が未学習の場合はエントリーを許可するテスト"""
        mock_ml_predictor.is_trained.return_value = False

        with patch.object(UniversalStrategy, "init"):
            params = {"strategy_gene": valid_gene, "ml_predictor": mock_ml_predictor}
            strategy = UniversalStrategy(mock_broker, mock_data, params)
            strategy._data = mock_data

            # 未学習の場合、常にTrue（エントリー許可）を返す
            result = strategy._ml_allows_entry(direction=1.0)
            assert result is True

    def test_ml_filter_threshold_customizable(
        self, mock_broker, mock_data, valid_gene, mock_ml_predictor
    ):
        """MLフィルター閾値がカスタマイズ可能であることをテスト"""
        # 閾値を0.05に設定（up: 0.45, down: 0.35 の場合、0.45 > 0.35 + 0.05 = 0.40 → True）
        mock_ml_predictor.predict.return_value = {
            "up": 0.45,
            "down": 0.35,
            "range": 0.2,
        }

        with patch.object(UniversalStrategy, "init"):
            params = {
                "strategy_gene": valid_gene,
                "ml_predictor": mock_ml_predictor,
                "ml_filter_threshold": 0.05,
            }
            strategy = UniversalStrategy(mock_broker, mock_data, params)
            strategy._data = mock_data

            result = strategy._ml_allows_entry(direction=1.0)
            assert result is True

    def test_prepare_current_features_returns_dataframe(
        self, mock_broker, mock_data, valid_gene
    ):
        """_prepare_current_features()がDataFrameを返すことをテスト"""
        with patch.object(UniversalStrategy, "init"):
            params = {"strategy_gene": valid_gene}
            strategy = UniversalStrategy(mock_broker, mock_data, params)
            strategy._data = mock_data
            strategy.gene = valid_gene

            # 基本的なデータアクセスのモック設定
            mock_data.Close.__getitem__ = MagicMock(
                side_effect=lambda x: (
                    np.array([50000] * 20) if isinstance(x, slice) else 50000
                )
            )
            mock_data.High.__getitem__ = MagicMock(
                side_effect=lambda x: (
                    np.array([51000] * 20) if isinstance(x, slice) else 51000
                )
            )
            mock_data.Low.__getitem__ = MagicMock(
                side_effect=lambda x: (
                    np.array([49000] * 20) if isinstance(x, slice) else 49000
                )
            )
            mock_data.Open = MagicMock()
            mock_data.Open.__getitem__ = MagicMock(
                side_effect=lambda x: (
                    np.array([49500] * 20) if isinstance(x, slice) else 49500
                )
            )
            mock_data.Volume = MagicMock()
            mock_data.Volume.__getitem__ = MagicMock(
                side_effect=lambda x: (
                    np.array([1000] * 20) if isinstance(x, slice) else 1000
                )
            )

            features = strategy._prepare_current_features()

            assert isinstance(features, pd.DataFrame)
            assert not features.empty
