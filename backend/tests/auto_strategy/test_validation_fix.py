"""
戦略遺伝子検証の修正テスト

TP/SL有効時のイグジット条件検証エラーの修正をテストします。
"""

import pytest
import sys
import os

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from app.core.services.auto_strategy.models.gene_strategy import StrategyGene, IndicatorGene, Condition
from app.core.services.auto_strategy.models.gene_tpsl import TPSLGene, TPSLMethod
from app.core.services.auto_strategy.models.gene_validation import GeneValidator
from app.core.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
from app.core.services.auto_strategy.models.ga_config import GAConfig


def test_validation_with_tpsl_enabled():
    """TP/SL有効時の検証テスト"""
    print("\n=== TP/SL有効時の検証テスト ===")
    
    # TP/SL遺伝子を作成（有効）
    tpsl_gene = TPSLGene(
        method=TPSLMethod.RISK_REWARD_RATIO,
        stop_loss_pct=0.03,
        take_profit_pct=0.06,
        risk_reward_ratio=2.0,
        enabled=True
    )
    
    # 戦略遺伝子を作成（exit_conditionsは空）
    strategy_gene = StrategyGene(
        indicators=[
            IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)
        ],
        entry_conditions=[
            Condition(left_operand="close", operator=">", right_operand="SMA_20")
        ],
        long_entry_conditions=[
            Condition(left_operand="close", operator=">", right_operand="SMA_20")
        ],
        short_entry_conditions=[
            Condition(left_operand="close", operator="<", right_operand="SMA_20")
        ],
        exit_conditions=[],  # 空のイグジット条件
        tpsl_gene=tpsl_gene,
        risk_management={"position_size": 0.1}
    )
    
    # 検証実行
    is_valid, errors = strategy_gene.validate()
    
    print(f"検証結果: {is_valid}")
    print(f"エラー: {errors}")
    print(f"TP/SL有効: {strategy_gene.tpsl_gene.enabled}")
    print(f"exit_conditions数: {len(strategy_gene.exit_conditions)}")
    
    # アサーション
    assert is_valid, f"TP/SL有効時に検証が失敗しました: {errors}"
    assert len(strategy_gene.exit_conditions) == 0, "exit_conditionsが空でありません"
    assert strategy_gene.tpsl_gene.enabled, "TP/SL遺伝子が有効でありません"
    
    print("✅ TP/SL有効時の検証テスト成功")


def test_validation_with_tpsl_disabled():
    """TP/SL無効時の検証テスト"""
    print("\n=== TP/SL無効時の検証テスト ===")
    
    # TP/SL遺伝子を作成（無効）
    tpsl_gene = TPSLGene(
        method=TPSLMethod.RISK_REWARD_RATIO,
        stop_loss_pct=0.03,
        take_profit_pct=0.06,
        risk_reward_ratio=2.0,
        enabled=False
    )
    
    # 戦略遺伝子を作成（exit_conditionsは空）
    strategy_gene = StrategyGene(
        indicators=[
            IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)
        ],
        entry_conditions=[
            Condition(left_operand="close", operator=">", right_operand="SMA_20")
        ],
        long_entry_conditions=[
            Condition(left_operand="close", operator=">", right_operand="SMA_20")
        ],
        short_entry_conditions=[
            Condition(left_operand="close", operator="<", right_operand="SMA_20")
        ],
        exit_conditions=[],  # 空のイグジット条件
        tpsl_gene=tpsl_gene,
        risk_management={"position_size": 0.1}
    )
    
    # 検証実行
    is_valid, errors = strategy_gene.validate()
    
    print(f"検証結果: {is_valid}")
    print(f"エラー: {errors}")
    print(f"TP/SL有効: {strategy_gene.tpsl_gene.enabled}")
    print(f"exit_conditions数: {len(strategy_gene.exit_conditions)}")
    
    # アサーション
    assert not is_valid, "TP/SL無効時にexit_conditions空で検証が成功してしまいました"
    assert "イグジット条件が設定されていません" in str(errors), "期待されるエラーメッセージが含まれていません"
    
    print("✅ TP/SL無効時の検証テスト成功")


def test_validation_with_exit_conditions():
    """イグジット条件ありの検証テスト"""
    print("\n=== イグジット条件ありの検証テスト ===")
    
    # 戦略遺伝子を作成（exit_conditionsあり、TP/SL無効）
    strategy_gene = StrategyGene(
        indicators=[
            IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)
        ],
        entry_conditions=[
            Condition(left_operand="close", operator=">", right_operand="SMA_20")
        ],
        long_entry_conditions=[
            Condition(left_operand="close", operator=">", right_operand="SMA_20")
        ],
        short_entry_conditions=[
            Condition(left_operand="close", operator="<", right_operand="SMA_20")
        ],
        exit_conditions=[
            Condition(left_operand="close", operator="<", right_operand="SMA_20")
        ],
        tpsl_gene=None,  # TP/SL遺伝子なし
        risk_management={"position_size": 0.1}
    )
    
    # 検証実行
    is_valid, errors = strategy_gene.validate()
    
    print(f"検証結果: {is_valid}")
    print(f"エラー: {errors}")
    print(f"TP/SL遺伝子: {strategy_gene.tpsl_gene}")
    print(f"exit_conditions数: {len(strategy_gene.exit_conditions)}")
    
    # アサーション
    assert is_valid, f"イグジット条件ありで検証が失敗しました: {errors}"
    assert len(strategy_gene.exit_conditions) > 0, "exit_conditionsが設定されていません"
    
    print("✅ イグジット条件ありの検証テスト成功")


def test_random_gene_generator_validation():
    """RandomGeneGeneratorで生成された遺伝子の検証テスト"""
    print("\n=== RandomGeneGenerator 検証テスト ===")
    
    ga_config = GAConfig.create_fast()
    generator = RandomGeneGenerator(ga_config)
    
    # 複数の戦略遺伝子を生成してテスト
    for i in range(5):
        print(f"\n--- 戦略 {i+1} ---")
        
        strategy_gene = generator.generate_random_gene()
        is_valid, errors = strategy_gene.validate()
        
        print(f"検証結果: {is_valid}")
        print(f"TP/SL有効: {strategy_gene.tpsl_gene.enabled if strategy_gene.tpsl_gene else False}")
        print(f"exit_conditions数: {len(strategy_gene.exit_conditions)}")
        
        if errors:
            print(f"エラー: {errors}")
        
        # アサーション
        assert is_valid, f"戦略{i+1}の検証が失敗しました: {errors}"
        
        # TP/SL有効時はexit_conditionsが空であることを確認
        if strategy_gene.tpsl_gene and strategy_gene.tpsl_gene.enabled:
            assert len(strategy_gene.exit_conditions) == 0, f"戦略{i+1}: TP/SL有効時にexit_conditionsが空でない"
            print(f"✅ 戦略{i+1}: TP/SL有効時のexit_conditions確認")
        
        print(f"✅ 戦略{i+1}: 検証成功")
    
    print("✅ RandomGeneGenerator 検証テスト成功")


def test_strategy_factory_validation():
    """StrategyFactory での検証テスト"""
    print("\n=== StrategyFactory 検証テスト ===")
    
    from app.core.services.auto_strategy.factories.strategy_factory import StrategyFactory
    
    # TP/SL有効な戦略遺伝子を作成
    tpsl_gene = TPSLGene(
        method=TPSLMethod.RISK_REWARD_RATIO,
        stop_loss_pct=0.03,
        take_profit_pct=0.06,
        risk_reward_ratio=2.0,
        enabled=True
    )
    
    strategy_gene = StrategyGene(
        indicators=[
            IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)
        ],
        entry_conditions=[
            Condition(left_operand="close", operator=">", right_operand="SMA_20")
        ],
        long_entry_conditions=[
            Condition(left_operand="close", operator=">", right_operand="SMA_20")
        ],
        short_entry_conditions=[
            Condition(left_operand="close", operator="<", right_operand="SMA_20")
        ],
        exit_conditions=[],  # 空のイグジット条件
        tpsl_gene=tpsl_gene,
        risk_management={"position_size": 0.1}
    )
    
    # StrategyFactoryで戦略クラスを作成
    factory = StrategyFactory()
    
    try:
        strategy_class = factory.create_strategy_class(strategy_gene)
        print("✅ StrategyFactory での戦略クラス作成成功")
        print(f"戦略クラス名: {strategy_class.__name__}")
    except ValueError as e:
        print(f"❌ StrategyFactory でエラー: {e}")
        raise
    
    print("✅ StrategyFactory 検証テスト成功")


if __name__ == "__main__":
    test_validation_with_tpsl_enabled()
    test_validation_with_tpsl_disabled()
    test_validation_with_exit_conditions()
    test_random_gene_generator_validation()
    test_strategy_factory_validation()
    print("\n🎉 全ての検証修正テストが成功しました！")
