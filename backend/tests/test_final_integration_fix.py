"""
最終統合テスト - 修正の確認

ユーザーが報告した問題が修正されていることを確認します。
"""

import pytest
import sys
import os
import json

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.services.auto_strategy.models.gene_decoder import GeneDecoder
from app.core.services.auto_strategy.models.gene_strategy import StrategyGene
from app.core.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
from app.core.services.auto_strategy.models.ga_config import GAConfig


def test_user_reported_issue_fix():
    """ユーザーが報告した問題の修正を確認"""
    print("\n=== ユーザー報告問題の修正確認テスト ===")
    
    # ユーザーが提供したJSONと同じ構造を生成
    ga_config = GAConfig.create_fast()
    generator = RandomGeneGenerator(ga_config)
    
    # 複数の戦略を生成してテスト
    for i in range(5):
        print(f"\n--- テスト戦略 {i+1} ---")
        
        strategy_gene = generator.generate_random_gene()
        gene_dict = strategy_gene.to_dict()
        
        # ユーザーが提供したJSONと同じ構造を作成
        strategy_config = {
            "strategy_config": {
                "strategy_type": "GENERATED_AUTO",
                "parameters": {
                    "strategy_gene": gene_dict
                }
            },
            "experiment_id": f"{i+1}",
            "db_experiment_id": i+1,
            "fitness_score": 0
        }
        
        # 問題1: exit_conditions がTP/SL有効時に表示される問題
        tpsl_enabled = gene_dict.get('tpsl_gene', {}).get('enabled', False)
        exit_conditions_count = len(gene_dict.get('exit_conditions', []))
        
        print(f"TP/SL有効: {tpsl_enabled}")
        print(f"exit_conditions数: {exit_conditions_count}")
        
        if tpsl_enabled:
            assert exit_conditions_count == 0, f"戦略{i+1}: TP/SL有効時にexit_conditionsが表示されています"
            print(f"✅ 戦略{i+1}: exit_conditions無効化確認")
        
        # 問題2: long_entry_conditions と short_entry_conditions が空の問題
        long_conditions_count = len(gene_dict.get('long_entry_conditions', []))
        short_conditions_count = len(gene_dict.get('short_entry_conditions', []))
        
        print(f"long_entry_conditions数: {long_conditions_count}")
        print(f"short_entry_conditions数: {short_conditions_count}")
        
        assert long_conditions_count > 0, f"戦略{i+1}: long_entry_conditionsが空です"
        assert short_conditions_count > 0, f"戦略{i+1}: short_entry_conditionsが空です"
        print(f"✅ 戦略{i+1}: ロング・ショート条件生成確認")
        
        # 条件の内容確認
        long_conditions = gene_dict.get('long_entry_conditions', [])
        short_conditions = gene_dict.get('short_entry_conditions', [])
        
        if long_conditions:
            long_condition = long_conditions[0]
            print(f"ロング条件例: {long_condition['left_operand']} {long_condition['operator']} {long_condition['right_operand']}")
        
        if short_conditions:
            short_condition = short_conditions[0]
            print(f"ショート条件例: {short_condition['left_operand']} {short_condition['operator']} {short_condition['right_operand']}")
        
        # JSONシリアライゼーション確認
        json_str = json.dumps(strategy_config, ensure_ascii=False, indent=2)
        parsed = json.loads(json_str)
        
        # パースされたJSONの構造確認
        parsed_gene = parsed["strategy_config"]["parameters"]["strategy_gene"]
        assert "long_entry_conditions" in parsed_gene, f"戦略{i+1}: JSONにlong_entry_conditionsが含まれていません"
        assert "short_entry_conditions" in parsed_gene, f"戦略{i+1}: JSONにshort_entry_conditionsが含まれていません"
        assert len(parsed_gene["long_entry_conditions"]) > 0, f"戦略{i+1}: JSONのlong_entry_conditionsが空です"
        assert len(parsed_gene["short_entry_conditions"]) > 0, f"戦略{i+1}: JSONのshort_entry_conditionsが空です"
        
        print(f"✅ 戦略{i+1}: JSON構造確認完了")
    
    print("✅ ユーザー報告問題の修正確認テスト成功")


def test_gene_decoder_specific_case():
    """GeneDecoderの特定ケースをテスト"""
    print("\n=== GeneDecoder 特定ケーステスト ===")
    
    # ユーザーが提供したJSONに近いエンコードデータを作成
    encoded = [
        # BB指標 (period=90, std_dev=4.26)
        0.5, 0.6,
        # ADX指標 (period=13)
        0.3, 0.4,
        # ADX指標 (period=11)
        0.2, 0.3,
        # CCI指標 (period=12)
        0.4, 0.5,
        # 未使用
        0.0, 0.0,
        
        # 条件部分（6要素）
        1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
        
        # TP/SL遺伝子部分（8要素）
        0.4,    # method (RISK_REWARD_RATIO)
        0.0504, # stop_loss_pct
        0.1129, # take_profit_pct
        3.131,  # risk_reward_ratio
        0.0375, # base_stop_loss
        2.603,  # atr_multiplier_sl
        5.35,   # atr_multiplier_tp
        1.0,    # priority
        
        # ポジションサイジング遺伝子部分（8要素）
        0.5, 0.5399, 0.5, 0.975, 0.01724, 0.2012, 4.368, 1.371
    ]
    
    decoder = GeneDecoder()
    strategy_gene = decoder.decode_list_to_strategy_gene(encoded, StrategyGene)
    gene_dict = strategy_gene.to_dict()
    
    print(f"指標数: {len(gene_dict.get('indicators', []))}")
    print(f"指標タイプ: {[ind['type'] for ind in gene_dict.get('indicators', [])]}")
    print(f"TP/SL有効: {gene_dict.get('tpsl_gene', {}).get('enabled', False)}")
    print(f"exit_conditions数: {len(gene_dict.get('exit_conditions', []))}")
    print(f"long_entry_conditions数: {len(gene_dict.get('long_entry_conditions', []))}")
    print(f"short_entry_conditions数: {len(gene_dict.get('short_entry_conditions', []))}")
    
    # ユーザーが提供したJSONと同じ構造を作成
    result_json = {
        "strategy_config": {
            "strategy_type": "GENERATED_AUTO",
            "parameters": {
                "strategy_gene": gene_dict
            }
        },
        "experiment_id": "6",
        "db_experiment_id": 6,
        "fitness_score": 0
    }
    
    # 修正確認
    assert len(gene_dict.get('exit_conditions', [])) == 0, "exit_conditionsが空でありません"
    assert len(gene_dict.get('long_entry_conditions', [])) > 0, "long_entry_conditionsが空です"
    assert len(gene_dict.get('short_entry_conditions', [])) > 0, "short_entry_conditions が空です"
    
    print("✅ GeneDecoder 特定ケーステスト成功")
    
    # 結果のJSONを表示
    print("\n--- 修正後のJSON出力 ---")
    json_str = json.dumps(result_json, ensure_ascii=False, indent=2)
    print(json_str[:1000] + "..." if len(json_str) > 1000 else json_str)


def test_comparison_with_user_json():
    """ユーザー提供JSONとの比較テスト"""
    print("\n=== ユーザー提供JSONとの比較テスト ===")
    
    # ユーザーが提供したJSONの問題点を再現
    user_json_issues = {
        "exit_conditions_present": True,  # 修正前: exit_conditionsが表示されていた
        "long_entry_conditions_empty": True,  # 修正前: long_entry_conditionsが空だった
        "short_entry_conditions_empty": True,  # 修正前: short_entry_conditionsが空だった
    }
    
    # 修正後の動作を確認
    ga_config = GAConfig.create_fast()
    generator = RandomGeneGenerator(ga_config)
    strategy_gene = generator.generate_random_gene()
    gene_dict = strategy_gene.to_dict()
    
    # 修正後の状態
    fixed_json_state = {
        "exit_conditions_present": len(gene_dict.get('exit_conditions', [])) > 0,
        "long_entry_conditions_empty": len(gene_dict.get('long_entry_conditions', [])) == 0,
        "short_entry_conditions_empty": len(gene_dict.get('short_entry_conditions', [])) == 0,
    }
    
    print("修正前の問題:")
    print(f"- exit_conditions表示: {user_json_issues['exit_conditions_present']}")
    print(f"- long_entry_conditions空: {user_json_issues['long_entry_conditions_empty']}")
    print(f"- short_entry_conditions空: {user_json_issues['short_entry_conditions_empty']}")
    
    print("\n修正後の状態:")
    print(f"- exit_conditions表示: {fixed_json_state['exit_conditions_present']}")
    print(f"- long_entry_conditions空: {fixed_json_state['long_entry_conditions_empty']}")
    print(f"- short_entry_conditions空: {fixed_json_state['short_entry_conditions_empty']}")
    
    # 修正確認
    if gene_dict.get('tpsl_gene', {}).get('enabled', False):
        assert not fixed_json_state['exit_conditions_present'], "TP/SL有効時にexit_conditionsが表示されています"
    assert not fixed_json_state['long_entry_conditions_empty'], "long_entry_conditionsが空です"
    assert not fixed_json_state['short_entry_conditions_empty'], "short_entry_conditionsが空です"
    
    print("\n✅ 全ての問題が修正されました！")
    print("✅ ユーザー提供JSONとの比較テスト成功")


if __name__ == "__main__":
    test_user_reported_issue_fix()
    test_gene_decoder_specific_case()
    test_comparison_with_user_json()
    print("\n🎉 全ての最終統合テストが成功しました！")
    print("🎯 ユーザーが報告した問題は完全に修正されています！")
