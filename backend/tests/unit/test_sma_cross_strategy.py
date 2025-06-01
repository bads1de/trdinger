"""
SMAクロス戦略のテスト
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from app.core.strategies.sma_cross_strategy import SMACrossStrategy
from app.core.strategies.indicators import SMA


class TestSMACrossStrategy:
    """SMACrossStrategyのテスト"""

    @pytest.fixture
    def sample_data(self):
        """テスト用のOHLCVデータ"""
        # 100日分のサンプルデータを生成
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")

        # トレンドのあるデータを生成（SMAクロスが発生するように）
        np.random.seed(42)
        base_price = 50000
        trend = np.linspace(0, 0.2, 100)  # 20%の上昇トレンド
        noise = np.random.normal(0, 0.02, 100)  # 2%のノイズ

        prices = base_price * (1 + trend + noise)

        # OHLCV データを生成
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            high = price * (1 + abs(np.random.normal(0, 0.01)))
            low = price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = price + np.random.normal(0, price * 0.005)
            close_price = price
            volume = np.random.randint(100, 1000)

            data.append(
                {
                    "Open": open_price,
                    "High": max(open_price, high, close_price),
                    "Low": min(open_price, low, close_price),
                    "Close": close_price,
                    "Volume": volume,
                }
            )

        df = pd.DataFrame(data, index=dates)
        return df

    def test_sma_function(self):
        """SMA関数のテスト"""
        # テストデータ
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        # SMA(5)を計算
        result = SMA(values, 5)

        # 期待値: [NaN, NaN, NaN, NaN, 3, 4, 5, 6, 7, 8]
        expected = pd.Series([np.nan, np.nan, np.nan, np.nan, 3, 4, 5, 6, 7, 8])

        # NaN以外の値を比較
        pd.testing.assert_series_equal(result.dropna(), expected.dropna())

    def test_strategy_class_attributes(self):
        """戦略クラスの属性テスト"""
        # デフォルトパラメータの確認
        assert hasattr(SMACrossStrategy, "n1")
        assert hasattr(SMACrossStrategy, "n2")
        assert SMACrossStrategy.n1 == 20
        assert SMACrossStrategy.n2 == 50

    def test_strategy_init_method(self, sample_data):
        """戦略のinit()メソッドテスト"""
        # Strategyクラスのモック（backtesting.pyの要求に合わせて）
        with patch("backtesting.Strategy.__init__", return_value=None):
            strategy = SMACrossStrategy()
            strategy.data = Mock()
            strategy.data.Close = sample_data["Close"].values
            strategy.I = Mock()

            # SMAインジケーターのモック
            mock_sma1 = Mock()
            mock_sma2 = Mock()
            strategy.I.side_effect = [mock_sma1, mock_sma2]

            # init()メソッドを実行
            strategy.init()

            # SMAインジケーターが正しく設定されたことを確認
            assert strategy.I.call_count == 2
            assert strategy.sma1 == mock_sma1
            assert strategy.sma2 == mock_sma2

    def test_strategy_next_method_golden_cross(self):
        """ゴールデンクロス時のnext()メソッドテスト"""
        with patch("backtesting.Strategy.__init__", return_value=None):
            strategy = SMACrossStrategy()

            # モックの設定
            strategy.sma1 = Mock()
            strategy.sma2 = Mock()
            strategy.buy = Mock()
            strategy.sell = Mock()

            # ゴールデンクロスの状況をシミュレート
            with patch("backtesting.lib.crossover") as mock_crossover:
                # 最初の呼び出し（sma1がsma2を上抜け）でTrueを返す
                mock_crossover.side_effect = [True, False]

                strategy.next()

                # buy()が呼ばれたことを確認
                strategy.buy.assert_called_once()
                strategy.sell.assert_not_called()

    def test_strategy_next_method_death_cross(self):
        """デッドクロス時のnext()メソッドテスト"""
        with patch("backtesting.Strategy.__init__", return_value=None):
            strategy = SMACrossStrategy()

            # モックの設定
            strategy.sma1 = Mock()
            strategy.sma2 = Mock()
            strategy.buy = Mock()
            strategy.sell = Mock()

            # デッドクロスの状況をシミュレート
            with patch("backtesting.lib.crossover") as mock_crossover:
                # 2番目の呼び出し（sma2がsma1を上抜け）でTrueを返す
                mock_crossover.side_effect = [False, True]

                strategy.next()

                # sell()が呼ばれたことを確認
                strategy.sell.assert_called_once()
                strategy.buy.assert_not_called()

    def test_strategy_next_method_no_cross(self):
        """クロスが発生しない場合のnext()メソッドテスト"""
        with patch("backtesting.Strategy.__init__", return_value=None):
            strategy = SMACrossStrategy()

            # モックの設定
            strategy.sma1 = Mock()
            strategy.sma2 = Mock()
            strategy.buy = Mock()
            strategy.sell = Mock()

            # クロスが発生しない状況をシミュレート
            with patch("backtesting.lib.crossover") as mock_crossover:
                # 両方の呼び出しでFalseを返す
                mock_crossover.side_effect = [False, False]

                strategy.next()

                # buy()もsell()も呼ばれないことを確認
                strategy.buy.assert_not_called()
                strategy.sell.assert_not_called()

    def test_strategy_parameter_customization(self):
        """戦略パラメータのカスタマイズテスト"""

        # カスタムパラメータで戦略クラスを作成
        class CustomSMACrossStrategy(SMACrossStrategy):
            n1 = 10
            n2 = 30

        # パラメータが正しく設定されていることを確認
        assert CustomSMACrossStrategy.n1 == 10
        assert CustomSMACrossStrategy.n2 == 30

    def test_sma_calculation_with_real_data(self, sample_data):
        """実際のデータでのSMA計算テスト"""
        close_prices = sample_data["Close"].values

        # SMA(20)を計算
        sma_20 = SMA(close_prices, 20)

        # 基本的な検証
        assert len(sma_20) == len(close_prices)
        assert pd.isna(sma_20.iloc[:19]).all()  # 最初の19個はNaN
        assert not pd.isna(sma_20.iloc[19:]).any()  # 20個目以降はNaNでない

        # 手動計算との比較（20番目の値）
        expected_20th = close_prices[:20].mean()
        assert abs(sma_20.iloc[19] - expected_20th) < 1e-10

    def test_strategy_with_insufficient_data(self):
        """データ不足時の戦略テスト"""
        # 50期間未満のデータ（n2=50なので不足）
        short_data = pd.DataFrame(
            {
                "Open": [100] * 30,
                "High": [105] * 30,
                "Low": [95] * 30,
                "Close": [100] * 30,
                "Volume": [1000] * 30,
            },
            index=pd.date_range("2024-01-01", periods=30),
        )

        with patch("backtesting.Strategy.__init__", return_value=None):
            strategy = SMACrossStrategy()
            strategy.data = Mock()
            strategy.data.Close = short_data["Close"].values
            strategy.I = Mock()

            # SMAの計算で最初の50個がNaNになることを想定
            mock_sma1 = pd.Series([np.nan] * 20 + [100] * 10)
            mock_sma2 = pd.Series([np.nan] * 30)  # 全てNaN
            strategy.I.side_effect = [mock_sma1, mock_sma2]

            # init()は正常に実行されるべき
            strategy.init()

            # SMAが設定されていることを確認
            assert strategy.I.call_count == 2
