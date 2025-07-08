#!/usr/bin/env python3
"""
MACD初期化問題のテストスクリプト

オートストラテジー機能でのMACDインジケーター初期化をテストし、
問題の原因を特定します。
"""

import sys
import os
import logging
from datetime import datetime, timedelta

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.services.auto_strategy.models.strategy_gene import (
    StrategyGene,
    IndicatorGene,
    Condition,
)
from app.core.services.auto_strategy.factories.strategy_factory import StrategyFactory
from app.core.services.backtest_data_service import BacktestDataService
from app.core.services.auto_strategy.calculators.indicator_calculator import (
    IndicatorCalculator,
)
from app.core.services.indicators import TechnicalIndicatorService
import pandas as pd
import numpy as np

# ログ設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_test_data():
    """テスト用のOHLCVデータを作成"""
    logger.info("テスト用データの作成開始")

    # 100日分のテストデータを作成
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
    np.random.seed(42)  # 再現性のため

    # 価格データを生成（ランダムウォーク）
    base_price = 50000
    price_changes = np.random.normal(0, 0.02, 100)  # 2%の標準偏差
    prices = [base_price]

    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1))  # 価格が負にならないように

    # OHLCV データを作成
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        high = close * (1 + abs(np.random.normal(0, 0.01)))
        low = close * (1 - abs(np.random.normal(0, 0.01)))
        open_price = close + np.random.normal(0, close * 0.005)
        volume = np.random.uniform(1000, 10000)

        data.append(
            {
                "Open": open_price,
                "High": max(open_price, high, close),
                "Low": min(open_price, low, close),
                "Close": close,
                "Volume": volume,
            }
        )

    df = pd.DataFrame(data, index=dates)
    logger.info(f"テストデータ作成完了: {df.shape}, カラム: {df.columns.tolist()}")
    return df


def test_macd_direct():
    """MACDを直接計算してテスト"""
    logger.info("=== MACD直接計算テスト ===")

    try:
        df = create_test_data()

        # TechnicalIndicatorServiceを使用してMACDを計算
        service = TechnicalIndicatorService()
        params = {"fast_period": 12, "slow_period": 26, "signal_period": 9}

        logger.info(f"MACD計算開始: パラメータ={params}")
        result = service.calculate_indicator(df, "MACD", params)

        logger.info(f"MACD計算結果: タイプ={type(result)}")
        if isinstance(result, tuple):
            logger.info(f"MACD出力数: {len(result)}")
            for i, output in enumerate(result):
                logger.info(
                    f"出力{i}: 形状={output.shape}, 非NaN数={np.sum(~np.isnan(output))}"
                )

        return True

    except Exception as e:
        logger.error(f"MACD直接計算エラー: {e}", exc_info=True)
        return False


def test_indicator_calculator():
    """IndicatorCalculatorを使用してMACDをテスト"""
    logger.info("=== IndicatorCalculator MACDテスト ===")

    try:
        df = create_test_data()

        # backtesting.pyのデータオブジェクトをシミュレート
        class MockData:
            def __init__(self, df):
                self.df = df

        mock_data = MockData(df)

        # IndicatorCalculatorでMACDを計算
        calculator = IndicatorCalculator()
        params = {"fast_period": 12, "slow_period": 26, "signal_period": 9}

        logger.info(f"IndicatorCalculator MACD計算開始: パラメータ={params}")
        result = calculator.calculate_indicator("MACD", params, mock_data)

        logger.info(f"IndicatorCalculator MACD計算結果: タイプ={type(result)}")
        if isinstance(result, tuple):
            logger.info(f"MACD出力数: {len(result)}")
            for i, output in enumerate(result):
                logger.info(
                    f"出力{i}: 形状={output.shape}, 非NaN数={np.sum(~np.isnan(output))}"
                )

        return True

    except Exception as e:
        logger.error(f"IndicatorCalculator MACDエラー: {e}", exc_info=True)
        return False


def test_strategy_initialization():
    """戦略初期化でのMACDテスト"""
    logger.info("=== 戦略初期化 MACDテスト ===")

    try:
        df = create_test_data()

        # MACDインジケーター遺伝子を作成
        macd_gene = IndicatorGene(
            type="MACD",
            parameters={"fast_period": 12, "slow_period": 26, "signal_period": 9},
            enabled=True,
        )

        # ダミーの条件を作成（MACDテストのため）
        dummy_entry_condition = Condition(
            left_operand="close", operator=">", right_operand="open"
        )
        dummy_exit_condition = Condition(
            left_operand="close", operator="<", right_operand="open"
        )

        # 戦略遺伝子を作成
        strategy_gene = StrategyGene(
            indicators=[macd_gene],
            entry_conditions=[dummy_entry_condition],
            long_entry_conditions=[],
            short_entry_conditions=[],
            exit_conditions=[dummy_exit_condition],
            risk_management={},
            tpsl_gene=None,
        )

        # StrategyFactoryで戦略を作成
        factory = StrategyFactory()
        strategy_class = factory.create_strategy_class(strategy_gene)

        logger.info("戦略クラス作成完了")

        # backtesting.pyのデータオブジェクトをシミュレート
        class MockData:
            def __init__(self, df):
                self.df = df
                # backtesting.pyが期待する属性を追加
                self.Close = df["Close"]
                self.Open = df["Open"]
                self.High = df["High"]
                self.Low = df["Low"]
                self.Volume = df["Volume"]
                self.index = df.index

        mock_data = MockData(df)

        # 戦略インスタンスを作成して初期化
        strategy_instance = strategy_class(data=mock_data)
        logger.info("戦略インスタンス作成完了")

        # 初期化を実行
        strategy_instance.init()
        logger.info("戦略初期化完了")

        # MACD指標が正しく設定されているか確認
        macd_attrs = [attr for attr in dir(strategy_instance) if "MACD" in attr]
        logger.info(f"MACD関連属性: {macd_attrs}")

        return True

    except Exception as e:
        logger.error(f"戦略初期化 MACDエラー: {e}", exc_info=True)
        return False


def main():
    """メイン関数"""
    logger.info("MACD初期化問題テスト開始")

    results = {}

    # テスト1: MACD直接計算
    results["direct"] = test_macd_direct()

    # テスト2: IndicatorCalculator
    results["calculator"] = test_indicator_calculator()

    # テスト3: 戦略初期化
    results["strategy"] = test_strategy_initialization()

    # 結果サマリー
    logger.info("=== テスト結果サマリー ===")
    for test_name, success in results.items():
        status = "成功" if success else "失敗"
        logger.info(f"{test_name}: {status}")

    if all(results.values()):
        logger.info("全てのテストが成功しました")
    else:
        logger.error("一部のテストが失敗しました")

    return all(results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
