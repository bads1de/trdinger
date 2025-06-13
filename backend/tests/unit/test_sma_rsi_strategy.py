"""
SMA + RSI複合戦略のテスト
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from backtesting import Backtest

from app.core.strategies.sma_rsi_strategy import SMARSIStrategy, SMARSIStrategyOptimized
from app.core.strategies.indicators import SMA, RSI


class TestSMARSIStrategy:
    """SMA+RSI戦略のテストクラス"""

    @pytest.fixture
    def sample_data(self):
        """テスト用のサンプルデータ"""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=100, freq="D")

        # トレンドのあるデータを生成
        base_price = 100
        trend = np.linspace(0, 20, 100)  # 上昇トレンド
        noise = np.random.normal(0, 2, 100)  # ノイズ
        close_prices = base_price + trend + noise

        # OHLCV データを生成
        data = pd.DataFrame(
            {
                "Open": (close_prices * (1 + np.random.normal(0, 0.01, 100))).astype(
                    np.float64
                ),
                "High": (
                    close_prices * (1 + np.abs(np.random.normal(0, 0.02, 100)))
                ).astype(np.float64),
                "Low": (
                    close_prices * (1 - np.abs(np.random.normal(0, 0.02, 100)))
                ).astype(np.float64),
                "Close": close_prices.astype(np.float64),
                "Volume": np.random.randint(1000, 10000, 100).astype(np.float64),
            },
            index=dates,
        )

        # 価格の整合性を保つ
        data["High"] = np.maximum(data["High"], data[["Open", "Close"]].max(axis=1))
        data["Low"] = np.minimum(data["Low"], data[["Open", "Close"]].min(axis=1))

        return data

    def test_strategy_class_attributes(self):
        """戦略クラスの属性テスト"""
        # デフォルトパラメータの確認
        assert hasattr(SMARSIStrategy, "sma_short")
        assert hasattr(SMARSIStrategy, "sma_long")
        assert hasattr(SMARSIStrategy, "rsi_period")
        assert hasattr(SMARSIStrategy, "oversold_threshold")
        assert hasattr(SMARSIStrategy, "overbought_threshold")

        assert SMARSIStrategy.sma_short == 20
        assert SMARSIStrategy.sma_long == 50
        assert SMARSIStrategy.rsi_period == 14
        assert SMARSIStrategy.oversold_threshold == 30
        assert SMARSIStrategy.overbought_threshold == 70

    def test_sma_calculation(self):
        """SMA計算のテスト"""
        # テストデータ（float64に変換）
        values = pd.Series(
            [100, 102, 101, 103, 105, 104, 106, 108, 107, 109], dtype=np.float64
        )

        # SMA(5)を計算
        result = SMA(values, 5)

        # 基本的な検証
        assert len(result) == len(values)
        assert pd.isna(result.iloc[:4]).all()  # 最初の4個はNaN
        assert not pd.isna(result.iloc[4:]).any()  # 5個目以降はNaNでない

        # 手動計算との比較（5番目の値）
        expected_5th = np.mean(values[:5])
        assert abs(result.iloc[4] - expected_5th) < 1e-10

    def test_rsi_calculation(self):
        """RSI計算のテスト"""
        # 単調増加データでRSIをテスト（float64に変換）
        values = pd.Series(list(range(100, 120)), dtype=np.float64)  # 100から119まで

        result = RSI(values, 14)

        # 基本的な検証
        assert len(result) == len(values)
        assert not pd.isna(result.iloc[-1])  # 最後の値はNaNでない

        # RSIは0-100の範囲内
        valid_values = result.dropna()
        assert (valid_values >= 0).all()
        assert (valid_values <= 100).all()

        # 単調増加データなのでRSIは高い値になるはず
        assert result.iloc[-1] > 50

    def test_strategy_parameter_customization(self):
        """戦略パラメータのカスタマイズテスト"""

        # カスタムパラメータで戦略クラスを作成
        class CustomSMARSIStrategy(SMARSIStrategy):
            sma_short = 10
            sma_long = 30
            rsi_period = 21
            oversold_threshold = 25
            overbought_threshold = 75

        # パラメータが正しく設定されていることを確認
        assert CustomSMARSIStrategy.sma_short == 10
        assert CustomSMARSIStrategy.sma_long == 30
        assert CustomSMARSIStrategy.rsi_period == 21
        assert CustomSMARSIStrategy.oversold_threshold == 25
        assert CustomSMARSIStrategy.overbought_threshold == 75

    def test_strategy_with_real_data(self, sample_data):
        """実際のデータでの戦略テスト"""

        # バックテストを実行
        bt = Backtest(sample_data, SMARSIStrategy, cash=10000, commission=0.001)

        # カスタムパラメータでテスト
        stats = bt.run(
            sma_short=10,
            sma_long=30,
            rsi_period=14,
            oversold_threshold=30,
            overbought_threshold=70,
            use_risk_management=True,
            sl_pct=0.02,
            tp_pct=0.05,
        )

        # 基本的な結果の検証
        assert stats is not None
        assert "Equity Final [$]" in stats
        assert "# Trades" in stats

        # 取引が発生していることを確認（データによっては0の場合もある）
        print(f"Total trades: {stats['# Trades']}")
        print(f"Final equity: ${stats['Equity Final [$]']:.2f}")
        print(f"Return: {stats['Return [%]']:.2f}%")

    def test_optimized_strategy(self, sample_data):
        """最適化戦略のテスト"""

        # 最適化戦略のバックテスト
        bt = Backtest(
            sample_data, SMARSIStrategyOptimized, cash=10000, commission=0.001
        )

        stats = bt.run(
            sma_short=10,
            sma_long=30,
            rsi_period=14,
            oversold_threshold=30,
            overbought_threshold=70,
            use_risk_management=True,
            volume_filter=True,
            volume_threshold=1.2,
            rsi_confirmation_bars=2,
        )

        # 基本的な結果の検証
        assert stats is not None
        assert "Equity Final [$]" in stats
        assert "# Trades" in stats

        print(f"Optimized strategy trades: {stats['# Trades']}")
        print(f"Optimized final equity: ${stats['Equity Final [$]']:.2f}")
        print(f"Optimized return: {stats['Return [%]']:.2f}%")

    def test_strategy_comparison(self, sample_data):
        """基本戦略と最適化戦略の比較テスト"""

        # 基本戦略
        bt_basic = Backtest(sample_data, SMARSIStrategy, cash=10000, commission=0.001)
        stats_basic = bt_basic.run(
            sma_short=10, sma_long=30, rsi_period=14, use_risk_management=True
        )

        # 最適化戦略
        bt_optimized = Backtest(
            sample_data, SMARSIStrategyOptimized, cash=10000, commission=0.001
        )
        stats_optimized = bt_optimized.run(
            sma_short=10,
            sma_long=30,
            rsi_period=14,
            use_risk_management=True,
            volume_filter=True,
        )

        # 結果の比較
        print("\n=== 戦略比較結果 ===")
        print(f"基本戦略:")
        print(f"  取引数: {stats_basic['# Trades']}")
        print(f"  最終資産: ${stats_basic['Equity Final [$]']:.2f}")
        print(f"  リターン: {stats_basic['Return [%]']:.2f}%")

        print(f"最適化戦略:")
        print(f"  取引数: {stats_optimized['# Trades']}")
        print(f"  最終資産: ${stats_optimized['Equity Final [$]']:.2f}")
        print(f"  リターン: {stats_optimized['Return [%]']:.2f}%")

        # 両方とも有効な結果が得られていることを確認
        assert stats_basic["Equity Final [$]"] > 0
        assert stats_optimized["Equity Final [$]"] > 0

    def test_risk_management_integration(self, sample_data):
        """リスク管理機能の統合テスト"""

        # リスク管理ありの戦略
        bt_with_risk = Backtest(
            sample_data, SMARSIStrategy, cash=10000, commission=0.001
        )
        stats_with_risk = bt_with_risk.run(
            sma_short=10,
            sma_long=30,
            use_risk_management=True,
            sl_pct=0.02,
            tp_pct=0.05,
        )

        # リスク管理なしの戦略
        bt_without_risk = Backtest(
            sample_data, SMARSIStrategy, cash=10000, commission=0.001
        )
        stats_without_risk = bt_without_risk.run(
            sma_short=10, sma_long=30, use_risk_management=False
        )

        # 結果の比較
        print("\n=== リスク管理比較結果 ===")
        print(f"リスク管理あり:")
        print(f"  取引数: {stats_with_risk['# Trades']}")
        print(f"  最大ドローダウン: {stats_with_risk['Max. Drawdown [%]']:.2f}%")

        print(f"リスク管理なし:")
        print(f"  取引数: {stats_without_risk['# Trades']}")
        print(f"  最大ドローダウン: {stats_without_risk['Max. Drawdown [%]']:.2f}%")

        # 基本的な検証
        assert stats_with_risk is not None
        assert stats_without_risk is not None


def create_sample_data():
    """テスト用のサンプルデータ生成関数"""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100, freq="D")

    # トレンドのあるデータを生成
    base_price = 100
    trend = np.linspace(0, 20, 100)  # 上昇トレンド
    noise = np.random.normal(0, 2, 100)  # ノイズ
    close_prices = base_price + trend + noise

    # OHLCV データを生成（float64に変換）
    data = pd.DataFrame(
        {
            "Open": (close_prices * (1 + np.random.normal(0, 0.01, 100))).astype(
                np.float64
            ),
            "High": (
                close_prices * (1 + np.abs(np.random.normal(0, 0.02, 100)))
            ).astype(np.float64),
            "Low": (close_prices * (1 - np.abs(np.random.normal(0, 0.02, 100)))).astype(
                np.float64
            ),
            "Close": close_prices.astype(np.float64),
            "Volume": np.random.randint(1000, 10000, 100).astype(np.float64),
        },
        index=dates,
    )

    # 価格の整合性を保つ
    data["High"] = np.maximum(data["High"], data[["Open", "Close"]].max(axis=1))
    data["Low"] = np.minimum(data["Low"], data[["Open", "Close"]].min(axis=1))

    return data


if __name__ == "__main__":
    # 簡単なテスト実行
    import sys

    sys.path.append("../../..")

    test_instance = TestSMARSIStrategy()

    # サンプルデータ生成
    sample_data = create_sample_data()

    print("=== SMA+RSI戦略テスト ===")

    # 基本テスト
    test_instance.test_strategy_class_attributes()
    print("✅ 戦略クラス属性テスト成功")

    test_instance.test_sma_calculation()
    print("✅ SMA計算テスト成功")

    test_instance.test_rsi_calculation()
    print("✅ RSI計算テスト成功")

    test_instance.test_strategy_parameter_customization()
    print("✅ パラメータカスタマイズテスト成功")

    # 実データテスト
    test_instance.test_strategy_with_real_data(sample_data)
    print("✅ 実データテスト成功")

    test_instance.test_optimized_strategy(sample_data)
    print("✅ 最適化戦略テスト成功")

    test_instance.test_strategy_comparison(sample_data)
    print("✅ 戦略比較テスト成功")

    test_instance.test_risk_management_integration(sample_data)
    print("✅ リスク管理統合テスト成功")

    print("\n🎉 全てのテストが成功しました！")
