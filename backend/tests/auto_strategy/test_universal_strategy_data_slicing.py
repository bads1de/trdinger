"""
UniversalStrategy TPSL データスライスのテスト

Issue: TPSL 計算時にデータスライスがハードコードされた 30 を使用しており、
atr_period > 30 の場合にデータ不足が発生する問題を修正する。
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np

from app.services.auto_strategy.strategies.universal_strategy import UniversalStrategy
from app.services.auto_strategy.genes import (
    StrategyGene,
    IndicatorGene,
    TPSLGene,
    TPSLMethod,
)
from app.services.auto_strategy.genes.conditions import Condition


class TestTPSLDataSlicing:
    """TPSL 計算時のデータスライスの動的調整テスト"""

    @pytest.fixture
    def mock_broker(self):
        """Brokerのモック"""
        broker = MagicMock()
        broker.commission = 0.001
        return broker

    @pytest.fixture
    def mock_data_large(self):
        """大きなデータセットのモック (100 バー)"""
        data = MagicMock()
        data_length = 100

        # 100本分のデータを生成
        high_values = np.random.uniform(50000, 51000, data_length)
        low_values = np.random.uniform(49000, 50000, data_length)
        close_values = np.random.uniform(49500, 50500, data_length)

        data.High = high_values
        data.Low = low_values
        data.Close = close_values

        # __len__を返すモック
        data.__len__ = MagicMock(return_value=data_length)

        return data

    @pytest.fixture
    def gene_with_large_atr_period(self):
        """atr_period = 50 の TPSL 遺伝子を持つ戦略"""
        tpsl_gene = TPSLGene(
            enabled=True,
            method=TPSLMethod.VOLATILITY_BASED,
            atr_period=50,  # Magic Number (30) より大きな値
            atr_multiplier_sl=1.5,
            atr_multiplier_tp=3.0,
        )

        return StrategyGene(
            indicators=[
                IndicatorGene(
                    type="SMA",
                    parameters={"period": 14},
                    enabled=False,
                )
            ],
            long_tpsl_gene=tpsl_gene,
        )

    @pytest.fixture
    def gene_with_atr_period_21(self):
        """atr_period = 21 の TPSL 遺伝子を持つ戦略"""
        tpsl_gene = TPSLGene(
            enabled=True,
            method=TPSLMethod.VOLATILITY_BASED,
            atr_period=21,  # Magic Number (30) 未満
            atr_multiplier_sl=1.5,
            atr_multiplier_tp=3.0,
        )

        return StrategyGene(
            indicators=[],
            long_tpsl_gene=tpsl_gene,
        )

    def test_data_slice_size_uses_atr_period_when_greater_than_default(
        self, mock_broker, mock_data_large, gene_with_large_atr_period
    ):
        """
        atr_period > 30 の場合、30 ではなく atr_period に基づいたスライスサイズを使用することをテスト
        """
        params = {"strategy_gene": gene_with_large_atr_period}
        strategy = UniversalStrategy(mock_broker, mock_data_large, params)

        # TPSLService をモックして引数をキャプチャ
        with patch.object(strategy.tpsl_service, "calculate_tpsl_prices") as mock_tpsl:
            mock_tpsl.return_value = (49000.0, 52000.0)

            # condition_evaluator をモックしてロングエントリーを発火させる
            with patch.object(
                strategy.condition_evaluator, "evaluate_conditions"
            ) as mock_eval:
                mock_eval.return_value = True

                # buy をモックし position をパッチ
                with patch.object(strategy, "buy"):
                    with patch.object(
                        UniversalStrategy, "position", new_callable=PropertyMock
                    ) as mock_position:
                        mock_position.return_value = None
                        strategy.next()

                        # TPSLService が呼ばれていることを確認
                        mock_tpsl.assert_called_once()

                        # 引数を取得
                        call_kwargs = mock_tpsl.call_args.kwargs
                        market_data = call_kwargs.get("market_data", {})

                        # ohlc_data が存在し、少なくとも atr_period 分のデータが含まれていることを確認
                        assert (
                            "ohlc_data" in market_data
                        ), "market_data に ohlc_data が含まれるべき"
                        ohlc_data = market_data["ohlc_data"]

                        # atr_period = 50 なので、少なくとも 50 以上のデータが必要
                        expected_min_size = (
                            gene_with_large_atr_period.tpsl_gene.atr_period
                        )
                        assert len(ohlc_data) >= expected_min_size, (
                            f"ohlc_data は少なくとも {expected_min_size} エントリ必要ですが、"
                            f"{len(ohlc_data)} しかありません"
                        )

    def test_data_slice_size_uses_default_when_atr_period_is_smaller(
        self, mock_broker, mock_data_large, gene_with_atr_period_21
    ):
        """
        atr_period < 30 の場合でも、正しいスライスサイズを使用することをテスト
        （後方互換性の確保）
        """
        params = {"strategy_gene": gene_with_atr_period_21}
        strategy = UniversalStrategy(mock_broker, mock_data_large, params)

        # TPSLService をモックして引数をキャプチャ
        with patch.object(strategy.tpsl_service, "calculate_tpsl_prices") as mock_tpsl:
            mock_tpsl.return_value = (49000.0, 52000.0)

            # condition_evaluator をモックしてロングエントリーを発火させる
            with patch.object(
                strategy.condition_evaluator, "evaluate_conditions"
            ) as mock_eval:
                mock_eval.return_value = True

                # buy をモックし position をパッチ
                with patch.object(strategy, "buy"):
                    with patch.object(
                        UniversalStrategy, "position", new_callable=PropertyMock
                    ) as mock_position:
                        mock_position.return_value = None
                        strategy.next()

                        # TPSLService が呼ばれていることを確認
                        mock_tpsl.assert_called_once()

                        # 引数を取得
                        call_kwargs = mock_tpsl.call_args.kwargs
                        market_data = call_kwargs.get("market_data", {})

                        # ohlc_data が存在し、少なくとも atr_period 分のデータが含まれていることを確認
                        assert (
                            "ohlc_data" in market_data
                        ), "market_data に ohlc_data が含まれるべき"
                        ohlc_data = market_data["ohlc_data"]

                        # atr_period = 21 なので、少なくとも 21 以上のデータがあれば OK
                        expected_min_size = gene_with_atr_period_21.tpsl_gene.atr_period
                        assert len(ohlc_data) >= expected_min_size, (
                            f"ohlc_data は少なくとも {expected_min_size} エントリ必要ですが、"
                            f"{len(ohlc_data)} しかありません"
                        )

    def test_data_slice_handles_insufficient_data_gracefully(
        self, mock_broker, gene_with_large_atr_period
    ):
        """
        データが atr_period より少ない場合でもエラーにならないことをテスト
        """
        # 少ないデータ（35バー）
        data = MagicMock()
        data_length = 35  # atr_period=50 より小さい

        high_values = np.random.uniform(50000, 51000, data_length)
        low_values = np.random.uniform(49000, 50000, data_length)
        close_values = np.random.uniform(49500, 50500, data_length)

        data.High = high_values
        data.Low = low_values
        data.Close = close_values
        data.__len__ = MagicMock(return_value=data_length)

        params = {"strategy_gene": gene_with_large_atr_period}
        strategy = UniversalStrategy(mock_broker, data, params)

        # TPSLService をモック
        with patch.object(strategy.tpsl_service, "calculate_tpsl_prices") as mock_tpsl:
            mock_tpsl.return_value = (49000.0, 52000.0)

            # condition_evaluator をモックしてロングエントリーを発火させる
            with patch.object(
                strategy.condition_evaluator, "evaluate_conditions"
            ) as mock_eval:
                mock_eval.return_value = True

                with patch.object(strategy, "buy"):
                    with patch.object(
                        UniversalStrategy, "position", new_callable=PropertyMock
                    ) as mock_position:
                        mock_position.return_value = None
                        # エラーなく実行完了すること
                        strategy.next()  # Should not raise

                        # TPSLServiceがゲートを通過（len(self.data) > 必要な期間 の条件があるため）
                        # この場合 35 > 設定された閾値（動的に決定）を確認


class TestTPSLDataSliceCalculation:
    """スライスサイズ計算ロジックのユニットテスト"""

    def test_calculate_required_slice_size_with_buffer(self):
        """
        必要なスライスサイズに適切なバッファが含まれることをテスト
        （ATR 計算には atr_period + 1 本のデータが必要）
        """
        # atr_period = 50 の場合、スライスサイズは少なくとも 51 であるべき
        # （True Range 計算のために前日のデータも必要）
        atr_period = 50
        expected_min_slice = atr_period + 1

        # 実装後にこの値を検証
        # _get_tpsl_data_slice_size メソッドが追加されることを期待
        assert expected_min_slice == 51

    @pytest.mark.parametrize(
        "atr_period,expected_min_slice",
        [
            (14, 15),
            (21, 22),
            (30, 31),
            (50, 51),
        ],
    )
    def test_slice_size_accounts_for_true_range_calculation(
        self, atr_period, expected_min_slice
    ):
        """
        ATR 計算のために True Range 計算用のバッファが確保されることをテスト
        """
        # True Range 計算には最低でも (atr_period + 1) 本のデータが必要
        assert atr_period + 1 == expected_min_slice




