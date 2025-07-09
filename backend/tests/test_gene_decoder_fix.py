"""
GeneDecoder修正のテスト

exit_conditions の無効化と long/short 条件の分離をテストします。
"""

import pytest
import sys
import os

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.services.auto_strategy.models.gene_decoder import GeneDecoder
from app.core.services.auto_strategy.models.gene_strategy import StrategyGene


def test_gene_decoder_exit_conditions_disabled_when_tpsl_enabled():
    """TP/SL遺伝子が有効な場合にexit_conditionsが空になることをテスト"""
    print("\n=== TP/SL有効時のexit_conditions無効化テスト ===")
    
    decoder = GeneDecoder()
    
    # TP/SL遺伝子を含む32要素のエンコードデータ（TP/SL有効）
    encoded = [
        # 指標部分（10要素）
        0.5, 0.6,  # BB
        0.3, 0.4,  # ADX
        0.2, 0.3,  # ADX
        0.4, 0.5,  # CCI
        0.0, 0.0,  # 未使用
        
        # 条件部分（6要素）
        1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
        
        # TP/SL遺伝子部分（8要素）
        0.4,  # method (RISK_REWARD_RATIO)
        0.336,  # stop_loss_pct
        0.376,  # take_profit_pct
        2.631,  # risk_reward_ratio
        0.25,   # base_stop_loss
        0.603,  # atr_multiplier_sl
        0.35,   # atr_multiplier_tp
        1.371,  # priority
        
        # ポジションサイジング遺伝子部分（8要素）
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
    ]
    
    # デコード実行
    strategy_gene = decoder.decode_list_to_strategy_gene(encoded, StrategyGene)
    
    # 検証
    print(f"指標数: {len(strategy_gene.indicators)}")
    print(f"TP/SL遺伝子有効: {strategy_gene.tpsl_gene and strategy_gene.tpsl_gene.enabled}")
    print(f"exit_conditions数: {len(strategy_gene.exit_conditions)}")
    print(f"long_entry_conditions数: {len(strategy_gene.long_entry_conditions)}")
    print(f"short_entry_conditions数: {len(strategy_gene.short_entry_conditions)}")
    
    # アサーション
    assert strategy_gene.tpsl_gene is not None, "TP/SL遺伝子が生成されていません"
    assert strategy_gene.tpsl_gene.enabled, "TP/SL遺伝子が有効になっていません"
    assert len(strategy_gene.exit_conditions) == 0, f"TP/SL有効時にexit_conditionsが空でない: {strategy_gene.exit_conditions}"
    assert len(strategy_gene.long_entry_conditions) > 0, "long_entry_conditionsが空です"
    assert len(strategy_gene.short_entry_conditions) > 0, "short_entry_conditionsが空です"
    
    print("✅ TP/SL有効時のexit_conditions無効化テスト成功")


def test_gene_decoder_long_short_conditions_generation():
    """ロング・ショート条件が適切に生成されることをテスト"""
    print("\n=== ロング・ショート条件生成テスト ===")
    
    decoder = GeneDecoder()
    
    # BB指標を含むエンコードデータ
    encoded = [
        # 指標部分（BB指標）
        0.5, 0.6,  # BB
        0.0, 0.0,  # 未使用
        0.0, 0.0,  # 未使用
        0.0, 0.0,  # 未使用
        0.0, 0.0,  # 未使用
        
        # 条件部分（6要素）
        1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
        
        # TP/SL遺伝子部分（8要素）
        0.4, 0.336, 0.376, 2.631, 0.25, 0.603, 0.35, 1.371,
        
        # ポジションサイジング遺伝子部分（8要素）
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
    ]
    
    # デコード実行
    strategy_gene = decoder.decode_list_to_strategy_gene(encoded, StrategyGene)
    
    # 検証
    print(f"指標: {[ind.type for ind in strategy_gene.indicators]}")
    print(f"long_entry_conditions: {[f'{c.left_operand} {c.operator} {c.right_operand}' for c in strategy_gene.long_entry_conditions]}")
    print(f"short_entry_conditions: {[f'{c.left_operand} {c.operator} {c.right_operand}' for c in strategy_gene.short_entry_conditions]}")
    
    # アサーション
    assert len(strategy_gene.indicators) > 0, "指標が生成されていません"
    assert len(strategy_gene.long_entry_conditions) > 0, "long_entry_conditionsが空です"
    assert len(strategy_gene.short_entry_conditions) > 0, "short_entry_conditionsが空です"
    
    # 条件の内容をチェック
    long_condition = strategy_gene.long_entry_conditions[0]
    short_condition = strategy_gene.short_entry_conditions[0]
    
    print(f"ロング条件: {long_condition.left_operand} {long_condition.operator} {long_condition.right_operand}")
    print(f"ショート条件: {short_condition.left_operand} {short_condition.operator} {short_condition.right_operand}")
    
    print("✅ ロング・ショート条件生成テスト成功")


def test_gene_decoder_json_output():
    """JSONシリアライゼーションのテスト"""
    print("\n=== JSONシリアライゼーションテスト ===")
    
    decoder = GeneDecoder()
    
    # テストデータ
    encoded = [
        0.5, 0.6,  # BB
        0.3, 0.4,  # ADX
        0.2, 0.3,  # ADX
        0.4, 0.5,  # CCI
        0.0, 0.0,  # 未使用
        1.0, 0.0, 1.0, 0.0, 1.0, 0.0,  # 条件
        0.4, 0.336, 0.376, 2.631, 0.25, 0.603, 0.35, 1.371,  # TP/SL
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5  # ポジションサイジング
    ]
    
    # デコード実行
    strategy_gene = decoder.decode_list_to_strategy_gene(encoded, StrategyGene)
    
    # JSON変換
    gene_dict = strategy_gene.to_dict()
    
    print("生成されたJSON構造:")
    print(f"- indicators: {len(gene_dict.get('indicators', []))}")
    print(f"- entry_conditions: {len(gene_dict.get('entry_conditions', []))}")
    print(f"- long_entry_conditions: {len(gene_dict.get('long_entry_conditions', []))}")
    print(f"- short_entry_conditions: {len(gene_dict.get('short_entry_conditions', []))}")
    print(f"- exit_conditions: {len(gene_dict.get('exit_conditions', []))}")
    print(f"- tpsl_gene: {'あり' if gene_dict.get('tpsl_gene') else 'なし'}")
    print(f"- position_sizing_gene: {'あり' if gene_dict.get('position_sizing_gene') else 'なし'}")
    
    # アサーション
    assert len(gene_dict.get('long_entry_conditions', [])) > 0, "long_entry_conditionsがJSONに含まれていません"
    assert len(gene_dict.get('short_entry_conditions', [])) > 0, "short_entry_conditionsがJSONに含まれていません"
    assert len(gene_dict.get('exit_conditions', [])) == 0, "exit_conditionsが空でありません"
    assert gene_dict.get('tpsl_gene') is not None, "tpsl_geneがJSONに含まれていません"
    
    print("✅ JSONシリアライゼーションテスト成功")


if __name__ == "__main__":
    test_gene_decoder_exit_conditions_disabled_when_tpsl_enabled()
    test_gene_decoder_long_short_conditions_generation()
    test_gene_decoder_json_output()
    print("\n🎉 全てのテストが成功しました！")
