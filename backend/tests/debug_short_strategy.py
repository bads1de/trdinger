#!/usr/bin/env python3
"""
ショート戦略デバッグスクリプト

AUTO_STRATEGYで生成される戦略遺伝子のロング・ショート条件を確認し、
実際にショートポジションが作成されるかをテストします。
"""

import sys
import os
import logging
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.core.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
from app.core.services.auto_strategy.factories.strategy_factory import StrategyFactory
from app.core.services.auto_strategy.models.gene_strategy import StrategyGene, IndicatorGene, Condition
from app.core.services.auto_strategy.models.ga_config import GAConfig
import pandas as pd
import numpy as np

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_data():
    """テスト用のOHLCVデータを作成"""
    dates = pd.date_range(start="2024-01-01", end="2024-01-10", freq="1H")
    np.random.seed(42)
    
    # 価格データを生成（トレンドのあるデータ）
    base_price = 50000
    price_changes = np.random.normal(0, 100, len(dates))
    prices = [base_price]
    
    for change in price_changes[1:]:
        new_price = prices[-1] + change
        prices.append(max(new_price, 1000))  # 最低価格を設定
    
    data = pd.DataFrame({
        "Open": prices,
        "High": [p * (1 + abs(np.random.normal(0, 0.001))) for p in prices],
        "Low": [p * (1 - abs(np.random.normal(0, 0.001))) for p in prices],
        "Close": prices,
        "Volume": np.random.randint(1000, 10000, len(dates)),
    }, index=dates)
    
    return data

def test_random_gene_generation():
    """ランダム戦略遺伝子の生成テスト"""
    print("=== ランダム戦略遺伝子生成テスト ===")

    # GAConfigを作成
    config = GAConfig()
    generator = RandomGeneGenerator(config)
    
    # 10個の戦略遺伝子を生成してロング・ショート条件を確認
    for i in range(10):
        gene = generator.generate_random_gene()
        
        print(f"\n戦略 {i+1}:")
        print(f"  指標数: {len(gene.indicators)}")
        print(f"  ロング条件数: {len(gene.long_entry_conditions)}")
        print(f"  ショート条件数: {len(gene.short_entry_conditions)}")
        print(f"  ロング・ショート分離: {gene.has_long_short_separation()}")
        
        # 条件の詳細を表示
        if gene.long_entry_conditions:
            print(f"  ロング条件:")
            for j, cond in enumerate(gene.long_entry_conditions):
                print(f"    {j+1}. {cond.left_operand} {cond.operator} {cond.right_operand}")
        
        if gene.short_entry_conditions:
            print(f"  ショート条件:")
            for j, cond in enumerate(gene.short_entry_conditions):
                print(f"    {j+1}. {cond.left_operand} {cond.operator} {cond.right_operand}")

def test_manual_short_strategy():
    """手動でショート戦略を作成してテスト"""
    print("\n=== 手動ショート戦略テスト ===")
    
    # RSIベースのロング・ショート戦略を作成
    gene = StrategyGene(
        indicators=[
            IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True)
        ],
        long_entry_conditions=[
            Condition(left_operand="RSI_14", operator="<", right_operand=30)  # 売られすぎでロング
        ],
        short_entry_conditions=[
            Condition(left_operand="RSI_14", operator=">", right_operand=70)  # 買われすぎでショート
        ],
        exit_conditions=[
            Condition(left_operand="RSI_14", operator="==", right_operand=50)  # 中立で決済
        ],
        risk_management={"position_size": 0.1},
    )
    
    print(f"手動戦略:")
    print(f"  ロング条件数: {len(gene.long_entry_conditions)}")
    print(f"  ショート条件数: {len(gene.short_entry_conditions)}")
    print(f"  ロング・ショート分離: {gene.has_long_short_separation()}")
    
    # 戦略クラスを作成
    factory = StrategyFactory()
    strategy_class = factory.create_strategy_class(gene)
    
    # テストデータを作成
    data = create_test_data()
    
    # RSI計算（簡易版）
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    data["RSI_14"] = calculate_rsi(data["Close"])
    
    # 戦略インスタンスを作成
    strategy_instance = strategy_class(data=data, params={})
    strategy_instance.indicators = {"RSI_14": data["RSI_14"]}
    
    # 異なるRSI値でテスト
    test_cases = [
        (25, "売られすぎ（ロング期待）"),
        (75, "買われすぎ（ショート期待）"),
        (50, "中立"),
    ]
    
    for rsi_value, description in test_cases:
        # RSI値を設定
        data.loc[data.index[-1], "RSI_14"] = rsi_value
        strategy_instance.indicators = {"RSI_14": data["RSI_14"]}
        
        # 条件評価
        long_result = strategy_instance._check_long_entry_conditions()
        short_result = strategy_instance._check_short_entry_conditions()
        
        print(f"\n  RSI={rsi_value} ({description}):")
        print(f"    ロング条件: {long_result}")
        print(f"    ショート条件: {short_result}")

def test_condition_evaluation():
    """条件評価の詳細テスト"""
    print("\n=== 条件評価詳細テスト ===")
    
    from app.core.services.auto_strategy.evaluators.condition_evaluator import ConditionEvaluator
    
    evaluator = ConditionEvaluator()
    
    # モック戦略インスタンス
    class MockStrategy:
        def __init__(self):
            self.indicators = {"RSI_14": pd.Series([75.0])}
            self.data = type('obj', (object,), {
                'Close': pd.Series([50000.0]),
                'Open': pd.Series([49900.0])
            })()
    
    mock_strategy = MockStrategy()
    
    # テスト条件
    test_conditions = [
        Condition(left_operand="RSI_14", operator=">", right_operand=70),  # True期待
        Condition(left_operand="RSI_14", operator="<", right_operand=30),  # False期待
        Condition(left_operand="close", operator=">", right_operand="open"),  # True期待
    ]
    
    for i, condition in enumerate(test_conditions):
        result = evaluator.evaluate_single_condition(condition, mock_strategy)
        print(f"  条件 {i+1}: {condition.left_operand} {condition.operator} {condition.right_operand} = {result}")

def main():
    """メイン実行"""
    print("🔍 ショート戦略デバッグ開始\n")
    
    try:
        test_random_gene_generation()
        test_manual_short_strategy()
        test_condition_evaluation()
        
        print("\n✅ デバッグテスト完了")
        print("\n📋 確認ポイント:")
        print("1. ランダム生成でショート条件が設定されているか")
        print("2. 手動戦略でショート条件が正しく評価されるか")
        print("3. 条件評価器が正しく動作するか")
        
    except Exception as e:
        print(f"❌ デバッグテストエラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
