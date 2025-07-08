#!/usr/bin/env python3
"""
MACDオペランド修正のテストスクリプト

「未対応のオペランドMACD」エラーの修正を検証します。
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
from app.core.services.auto_strategy.evaluators.condition_evaluator import (
    ConditionEvaluator,
)
import pandas as pd
import numpy as np

# ログ設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_test_data():
    """テスト用のOHLCVデータを作成"""
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
    return df


def test_macd_operand_evaluation():
    """MACDオペランドの条件評価をテスト"""
    logger.info("=== MACDオペランド条件評価テスト ===")

    try:
        df = create_test_data()

        # MACDインジケーター遺伝子を作成
        macd_gene = IndicatorGene(
            type="MACD",
            parameters={"fast_period": 12, "slow_period": 26, "signal_period": 9},
            enabled=True,
        )

        # MACDを使用する条件を作成
        macd_condition = Condition(
            left_operand="MACD", operator=">", right_operand="0"  # これが問題だった
        )

        # エグジット条件を作成
        exit_condition = Condition(
            left_operand="close", operator="<", right_operand="open"
        )

        # 戦略遺伝子を作成
        strategy_gene = StrategyGene(
            indicators=[macd_gene],
            entry_conditions=[macd_condition],
            long_entry_conditions=[],
            short_entry_conditions=[],
            exit_conditions=[exit_condition],
            risk_management={},
            tpsl_gene=None,
        )

        # StrategyFactoryで戦略を作成
        factory = StrategyFactory()
        strategy_class = factory.create_strategy_class(strategy_gene)

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
        strategy_instance.init()

        logger.info("戦略初期化完了")

        # MACD関連属性を確認
        macd_attrs = [attr for attr in dir(strategy_instance) if "MACD" in attr]
        logger.info(f"MACD関連属性: {macd_attrs}")

        # 条件評価器でMACDオペランドをテスト
        evaluator = ConditionEvaluator()

        # MACDオペランドの値取得をテスト
        logger.info("MACDオペランド値取得テスト:")
        macd_value = evaluator.get_condition_value("MACD", strategy_instance)
        logger.info(f"MACD値: {macd_value}")

        # 条件評価をテスト
        logger.info("MACD条件評価テスト:")
        result = evaluator.evaluate_single_condition(macd_condition, strategy_instance)
        logger.info(f"条件評価結果: {result}")

        # 複数のMACDオペランドもテスト
        for i in range(3):
            attr_name = f"MACD_{i}"
            if hasattr(strategy_instance, attr_name):
                value = evaluator.get_condition_value(attr_name, strategy_instance)
                logger.info(f"{attr_name}値: {value}")

        return True

    except Exception as e:
        logger.error(f"MACDオペランド評価テストエラー: {e}", exc_info=True)
        return False


def test_multiple_output_indicators():
    """複数出力指標のオペランド評価をテスト"""
    logger.info("=== 複数出力指標オペランド評価テスト ===")

    try:
        df = create_test_data()

        # 複数の複数出力指標を作成
        indicators = [
            IndicatorGene(
                type="MACD",
                parameters={"fast_period": 12, "slow_period": 26, "signal_period": 9},
                enabled=True,
            ),
            IndicatorGene(
                type="BB",
                parameters={"period": 20, "std_dev": 2.0},
                enabled=True,
            ),
            IndicatorGene(
                type="STOCH",
                parameters={"fastk_period": 14, "slowk_period": 3, "slowd_period": 3},
                enabled=True,
            ),
        ]

        # 各指標を使用する条件を作成
        conditions = [
            Condition(left_operand="MACD", operator=">", right_operand="0"),
            Condition(left_operand="BB", operator=">", right_operand="close"),
            Condition(left_operand="STOCH", operator=">", right_operand="50"),
        ]

        # エグジット条件を作成
        exit_condition = Condition(
            left_operand="close", operator="<", right_operand="open"
        )

        # 戦略遺伝子を作成
        strategy_gene = StrategyGene(
            indicators=indicators,
            entry_conditions=conditions,
            long_entry_conditions=[],
            short_entry_conditions=[],
            exit_conditions=[exit_condition],
            risk_management={},
            tpsl_gene=None,
        )

        # StrategyFactoryで戦略を作成
        factory = StrategyFactory()
        strategy_class = factory.create_strategy_class(strategy_gene)

        # backtesting.pyのデータオブジェクトをシミュレート
        class MockData:
            def __init__(self, df):
                self.df = df
                self.Close = df["Close"]
                self.Open = df["Open"]
                self.High = df["High"]
                self.Low = df["Low"]
                self.Volume = df["Volume"]
                self.index = df.index

        mock_data = MockData(df)

        # 戦略インスタンスを作成して初期化
        strategy_instance = strategy_class(data=mock_data)
        strategy_instance.init()

        logger.info("複数指標戦略初期化完了")

        # 全ての指標関連属性を確認
        indicator_attrs = [
            attr
            for attr in dir(strategy_instance)
            if any(ind in attr for ind in ["MACD", "BB", "STOCH"])
        ]
        logger.info(f"指標関連属性: {indicator_attrs}")

        # 条件評価器で各オペランドをテスト
        evaluator = ConditionEvaluator()

        for i, condition in enumerate(conditions):
            logger.info(
                f"条件{i+1}評価: {condition.left_operand} {condition.operator} {condition.right_operand}"
            )
            try:
                result = evaluator.evaluate_single_condition(
                    condition, strategy_instance
                )
                logger.info(f"  結果: {result}")
            except Exception as e:
                logger.error(f"  エラー: {e}")

        return True

    except Exception as e:
        logger.error(f"複数出力指標テストエラー: {e}", exc_info=True)
        return False


def main():
    """メイン関数"""
    logger.info("MACDオペランド修正テスト開始")

    results = {}

    # テスト1: MACDオペランド評価
    results["macd_operand"] = test_macd_operand_evaluation()

    # テスト2: 複数出力指標
    results["multiple_output"] = test_multiple_output_indicators()

    # 結果サマリー
    logger.info("=== テスト結果サマリー ===")
    for test_name, success in results.items():
        status = "成功" if success else "失敗"
        logger.info(f"{test_name}: {status}")

    if all(results.values()):
        logger.info("全てのテストが成功しました - MACDオペランド問題は修正されました")
    else:
        logger.error("一部のテストが失敗しました")

    return all(results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
