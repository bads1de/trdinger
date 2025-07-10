"""
条件評価デバッグスクリプト

StrategyFactoryでの条件評価ロジックを詳細に調査
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
from app.core.services.auto_strategy.factories.strategy_factory import StrategyFactory
from app.core.services.auto_strategy.models.ga_config import GAConfig
from app.core.services.auto_strategy.models.gene_strategy import StrategyGene, IndicatorGene, Condition
from app.core.services.auto_strategy.evaluators.condition_evaluator import ConditionEvaluator


def create_test_data():
    """テスト用の市場データを作成"""
    dates = pd.date_range('2020-01-01', periods=1000, freq='H')

    # 基本的なOHLCVデータ
    np.random.seed(42)
    close_prices = 100 + np.cumsum(np.random.normal(0, 1, 1000))

    data = pd.DataFrame({
        'Open': np.roll(close_prices, 1),
        'High': close_prices + np.random.uniform(0, 2, 1000),
        'Low': close_prices - np.random.uniform(0, 2, 1000),
        'Close': close_prices,
        'Volume': np.random.uniform(1000, 10000, 1000)
    }, index=dates)

    # 負の価格を防ぐ
    data = data.clip(lower=1.0)

    return data


def debug_condition_evaluation():
    """条件評価の詳細デバッグ"""
    print("🔍 条件評価デバッグ開始")
    print("="*50)

    # 1. テスト戦略を生成（複数回試行して指標ベースの条件を取得）
    print("\n1. テスト戦略生成...")
    ga_config = GAConfig.create_fast()
    generator = RandomGeneGenerator(ga_config, enable_smart_generation=True)

    # 指標ベースの条件が生成されるまで試行
    for attempt in range(10):
        strategy_gene = generator.generate_random_gene()

        # 指標ベースの条件があるかチェック
        has_indicator_conditions = False
        for cond in strategy_gene.long_entry_conditions + strategy_gene.short_entry_conditions:
            if isinstance(cond.left_operand, str) and "_" in cond.left_operand:
                has_indicator_conditions = True
                break
            if isinstance(cond.right_operand, str) and "_" in cond.right_operand:
                has_indicator_conditions = True
                break

        if has_indicator_conditions:
            print(f"   指標ベースの条件を発見（試行 {attempt + 1}）")
            break
    else:
        print("   フォールバック条件のみ生成されました")

    print(f"✅ 戦略生成完了:")
    print(f"   ロング条件数: {len(strategy_gene.long_entry_conditions)}")
    print(f"   ショート条件数: {len(strategy_gene.short_entry_conditions)}")
    print(f"   指標数: {len(strategy_gene.indicators)}")

    # 条件の詳細を表示
    print("\n📋 ロング条件詳細:")
    for i, cond in enumerate(strategy_gene.long_entry_conditions):
        print(f"   {i+1}. {cond.left_operand} {cond.operator} {cond.right_operand}")

    print("\n📋 ショート条件詳細:")
    for i, cond in enumerate(strategy_gene.short_entry_conditions):
        print(f"   {i+1}. {cond.left_operand} {cond.operator} {cond.right_operand}")

    print("\n📋 指標詳細:")
    for i, ind in enumerate(strategy_gene.indicators):
        if ind.enabled:
            print(f"   {i+1}. {ind.type} (期間: {ind.parameters.get('period', 'N/A')})")

    # 2. 簡易条件評価テスト
    print("\n2. 簡易条件評価テスト...")
    test_simple_evaluation(strategy_gene)


def test_simple_evaluation(strategy_gene):
    """簡易条件評価テスト"""
    try:
        # 条件評価器を作成
        evaluator = ConditionEvaluator()

        # モック戦略インスタンス
        class MockStrategy:
            def __init__(self, strategy_gene):
                # 基本的な市場データ
                self.data = pd.DataFrame({
                    'Open': [100, 101, 102],
                    'High': [102, 103, 104],
                    'Low': [99, 100, 101],
                    'Close': [101, 102, 103],
                    'Volume': [1000, 1100, 1200]
                })

                # 指標データ（動的生成）
                self.I = {}
                self._generate_indicators(strategy_gene)

                # 現在の価格データ
                self.close = 102
                self.open = 101
                self.high = 104
                self.low = 101

            def _generate_indicators(self, strategy_gene):
                """戦略遺伝子に基づいて指標データを生成"""
                # 全ての条件から必要な指標を抽出
                all_conditions = (strategy_gene.long_entry_conditions +
                                strategy_gene.short_entry_conditions)

                for condition in all_conditions:
                    # 左オペランドが指標の場合
                    if isinstance(condition.left_operand, str) and "_" in condition.left_operand:
                        indicator_name = condition.left_operand
                        self.I[indicator_name] = pd.Series([50, 60, 70])  # ダミーデータ

                    # 右オペランドが指標の場合
                    if isinstance(condition.right_operand, str) and "_" in condition.right_operand:
                        indicator_name = condition.right_operand
                        self.I[indicator_name] = pd.Series([50, 60, 70])  # ダミーデータ

                print(f"   生成された指標: {list(self.I.keys())}")

        mock_strategy = MockStrategy(strategy_gene)

        # ロング条件を評価
        print("\n📊 ロング条件評価:")
        try:
            long_result = evaluator.evaluate_conditions(
                strategy_gene.long_entry_conditions, mock_strategy
            )
            print(f"   結果: {long_result}")
        except Exception as e:
            print(f"   エラー: {e}")

        # ショート条件を評価
        print("\n📊 ショート条件評価:")
        try:
            short_result = evaluator.evaluate_conditions(
                strategy_gene.short_entry_conditions, mock_strategy
            )
            print(f"   結果: {short_result}")
        except Exception as e:
            print(f"   エラー: {e}")

        # 個別条件の詳細評価
        print("\n🔍 個別条件詳細評価:")
        for i, cond in enumerate(strategy_gene.long_entry_conditions):
            try:
                result = evaluator.evaluate_single_condition(cond, mock_strategy)
                print(f"   ロング条件{i+1} ({cond.left_operand} {cond.operator} {cond.right_operand}): {result}")
            except Exception as e:
                print(f"   ロング条件{i+1}: エラー - {e}")

        for i, cond in enumerate(strategy_gene.short_entry_conditions):
            try:
                result = evaluator.evaluate_single_condition(cond, mock_strategy)
                print(f"   ショート条件{i+1} ({cond.left_operand} {cond.operator} {cond.right_operand}): {result}")
            except Exception as e:
                print(f"   ショート条件{i+1}: エラー - {e}")

    except Exception as e:
        print(f"❌ 簡易評価テストエラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    debug_condition_evaluation()