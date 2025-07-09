"""
バックテストサービスの修正テスト

TP/SL有効時のバックテスト実行エラーの修正をテストします。
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.services.auto_strategy.models.gene_strategy import StrategyGene, IndicatorGene, Condition
from app.core.services.auto_strategy.models.gene_tpsl import TPSLGene, TPSLMethod
from app.core.services.auto_strategy.factories.strategy_factory import StrategyFactory
from app.core.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
from app.core.services.auto_strategy.models.ga_config import GAConfig


def test_strategy_factory_with_tpsl_enabled():
    """TP/SL有効時のStrategyFactory動作テスト"""
    print("\n=== TP/SL有効時のStrategyFactory動作テスト ===")
    
    # TP/SL有効な戦略遺伝子を作成
    tpsl_gene = TPSLGene(
        method=TPSLMethod.RISK_REWARD_RATIO,
        stop_loss_pct=0.03,
        take_profit_pct=0.06,
        risk_reward_ratio=2.0,
        enabled=True
    )
    
    strategy_gene = StrategyGene(
        id="test_strategy",
        indicators=[
            IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True),
            IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True)
        ],
        entry_conditions=[
            Condition(left_operand="close", operator=">", right_operand="SMA_20")
        ],
        long_entry_conditions=[
            Condition(left_operand="close", operator=">", right_operand="SMA_20"),
            Condition(left_operand="RSI_14", operator="<", right_operand=30)
        ],
        short_entry_conditions=[
            Condition(left_operand="close", operator="<", right_operand="SMA_20"),
            Condition(left_operand="RSI_14", operator=">", right_operand=70)
        ],
        exit_conditions=[],  # 空のイグジット条件
        tpsl_gene=tpsl_gene,
        risk_management={"position_size": 0.1}
    )
    
    print(f"戦略ID: {strategy_gene.id}")
    print(f"指標数: {len(strategy_gene.indicators)}")
    print(f"TP/SL有効: {strategy_gene.tpsl_gene.enabled}")
    print(f"exit_conditions数: {len(strategy_gene.exit_conditions)}")
    print(f"long_entry_conditions数: {len(strategy_gene.long_entry_conditions)}")
    print(f"short_entry_conditions数: {len(strategy_gene.short_entry_conditions)}")
    
    # 検証
    is_valid, errors = strategy_gene.validate()
    print(f"検証結果: {is_valid}")
    if errors:
        print(f"エラー: {errors}")
    
    assert is_valid, f"戦略遺伝子の検証が失敗しました: {errors}"
    
    # StrategyFactoryで戦略クラスを作成
    factory = StrategyFactory()
    
    try:
        strategy_class = factory.create_strategy_class(strategy_gene)
        print(f"✅ 戦略クラス作成成功: {strategy_class.__name__}")
        
        # 戦略クラスの基本的な属性確認
        assert hasattr(strategy_class, '__init__'), "戦略クラスに__init__メソッドがありません"
        assert hasattr(strategy_class, 'init'), "戦略クラスにinitメソッドがありません"
        assert hasattr(strategy_class, 'next'), "戦略クラスにnextメソッドがありません"
        
        print("✅ 戦略クラスの基本属性確認完了")
        
    except ValueError as e:
        print(f"❌ StrategyFactory でエラー: {e}")
        raise
    
    print("✅ TP/SL有効時のStrategyFactory動作テスト成功")


def test_backtest_config_creation():
    """バックテスト設定作成テスト"""
    print("\n=== バックテスト設定作成テスト ===")
    
    # ランダム戦略遺伝子を生成
    ga_config = GAConfig.create_fast()
    generator = RandomGeneGenerator(ga_config)
    strategy_gene = generator.generate_random_gene()
    
    # 戦略遺伝子を辞書に変換
    gene_dict = strategy_gene.to_dict()
    
    # バックテスト設定を作成
    backtest_config = {
        "strategy_name": "TEST_AUTO_STRATEGY",
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "start_date": "2020-01-01",
        "end_date": "2020-12-31",
        "initial_capital": 100000,
        "commission_rate": 0.00055,
        "strategy_config": {
            "strategy_type": "GENERATED_AUTO",
            "parameters": {
                "strategy_gene": gene_dict
            }
        }
    }
    
    print(f"戦略名: {backtest_config['strategy_name']}")
    print(f"シンボル: {backtest_config['symbol']}")
    print(f"TP/SL有効: {gene_dict.get('tpsl_gene', {}).get('enabled', False)}")
    print(f"exit_conditions数: {len(gene_dict.get('exit_conditions', []))}")
    
    # 設定の基本的な妥当性確認
    assert backtest_config["strategy_config"]["strategy_type"] == "GENERATED_AUTO", "戦略タイプが正しくありません"
    assert "strategy_gene" in backtest_config["strategy_config"]["parameters"], "strategy_geneが含まれていません"
    
    # 戦略遺伝子の妥当性確認
    gene_data = backtest_config["strategy_config"]["parameters"]["strategy_gene"]
    assert len(gene_data.get("indicators", [])) > 0, "指標が含まれていません"
    assert len(gene_data.get("long_entry_conditions", [])) > 0, "long_entry_conditionsが含まれていません"
    assert len(gene_data.get("short_entry_conditions", [])) > 0, "short_entry_conditionsが含まれていません"
    
    # TP/SL有効時のexit_conditions確認
    if gene_data.get("tpsl_gene", {}).get("enabled", False):
        assert len(gene_data.get("exit_conditions", [])) == 0, "TP/SL有効時にexit_conditionsが空でない"
        print("✅ TP/SL有効時のexit_conditions確認")
    
    print("✅ バックテスト設定作成テスト成功")


def test_strategy_gene_serialization_roundtrip():
    """戦略遺伝子のシリアライゼーション往復テスト"""
    print("\n=== 戦略遺伝子シリアライゼーション往復テスト ===")
    
    # ランダム戦略遺伝子を生成
    ga_config = GAConfig.create_fast()
    generator = RandomGeneGenerator(ga_config)
    original_gene = generator.generate_random_gene()
    
    print(f"元の遺伝子 - TP/SL有効: {original_gene.tpsl_gene.enabled if original_gene.tpsl_gene else False}")
    print(f"元の遺伝子 - exit_conditions数: {len(original_gene.exit_conditions)}")
    
    # 辞書に変換
    gene_dict = original_gene.to_dict()
    
    # 辞書から復元
    restored_gene = StrategyGene.from_dict(gene_dict)
    
    print(f"復元後 - TP/SL有効: {restored_gene.tpsl_gene.enabled if restored_gene.tpsl_gene else False}")
    print(f"復元後 - exit_conditions数: {len(restored_gene.exit_conditions)}")
    
    # 検証
    is_valid, errors = restored_gene.validate()
    print(f"復元後の検証結果: {is_valid}")
    if errors:
        print(f"エラー: {errors}")
    
    # アサーション
    assert is_valid, f"復元後の戦略遺伝子の検証が失敗しました: {errors}"
    
    # 基本的な属性の一致確認
    assert len(restored_gene.indicators) == len(original_gene.indicators), "指標数が一致しません"
    assert len(restored_gene.long_entry_conditions) == len(original_gene.long_entry_conditions), "long_entry_conditions数が一致しません"
    assert len(restored_gene.short_entry_conditions) == len(original_gene.short_entry_conditions), "short_entry_conditions数が一致しません"
    assert len(restored_gene.exit_conditions) == len(original_gene.exit_conditions), "exit_conditions数が一致しません"
    
    # TP/SL遺伝子の一致確認
    if original_gene.tpsl_gene and restored_gene.tpsl_gene:
        assert original_gene.tpsl_gene.enabled == restored_gene.tpsl_gene.enabled, "TP/SL有効状態が一致しません"
    
    print("✅ 戦略遺伝子シリアライゼーション往復テスト成功")


def test_multiple_strategy_validation():
    """複数戦略の検証テスト"""
    print("\n=== 複数戦略の検証テスト ===")
    
    ga_config = GAConfig.create_fast()
    generator = RandomGeneGenerator(ga_config)
    factory = StrategyFactory()
    
    success_count = 0
    total_count = 10
    
    for i in range(total_count):
        print(f"\n--- 戦略 {i+1} ---")
        
        try:
            # 戦略遺伝子を生成
            strategy_gene = generator.generate_random_gene()
            
            # 検証
            is_valid, errors = strategy_gene.validate()
            print(f"検証結果: {is_valid}")
            
            if not is_valid:
                print(f"検証エラー: {errors}")
                continue
            
            # StrategyFactoryで戦略クラスを作成
            strategy_class = factory.create_strategy_class(strategy_gene)
            print(f"戦略クラス作成成功: {strategy_class.__name__}")
            
            # TP/SL状態確認
            tpsl_enabled = strategy_gene.tpsl_gene.enabled if strategy_gene.tpsl_gene else False
            exit_conditions_count = len(strategy_gene.exit_conditions)
            
            print(f"TP/SL有効: {tpsl_enabled}")
            print(f"exit_conditions数: {exit_conditions_count}")
            
            # TP/SL有効時のexit_conditions確認
            if tpsl_enabled:
                assert exit_conditions_count == 0, f"戦略{i+1}: TP/SL有効時にexit_conditionsが空でない"
                print(f"✅ 戦略{i+1}: TP/SL有効時のexit_conditions確認")
            
            success_count += 1
            print(f"✅ 戦略{i+1}: 成功")
            
        except Exception as e:
            print(f"❌ 戦略{i+1}: エラー - {e}")
    
    print(f"\n成功率: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
    
    # 少なくとも80%は成功すべき
    assert success_count >= total_count * 0.8, f"成功率が低すぎます: {success_count}/{total_count}"
    
    print("✅ 複数戦略の検証テスト成功")


if __name__ == "__main__":
    test_strategy_factory_with_tpsl_enabled()
    test_backtest_config_creation()
    test_strategy_gene_serialization_roundtrip()
    test_multiple_strategy_validation()
    print("\n🎉 全てのバックテストサービス修正テストが成功しました！")
    print("🎯 「イグジット条件が設定されていません」エラーは修正されました！")
