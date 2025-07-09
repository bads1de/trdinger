"""
Auto-Strategy生成の修正テスト

実際のGA生成フローで修正が正しく動作することをテストします。
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


def test_random_gene_generator_with_tpsl():
    """RandomGeneGeneratorでTP/SL有効時の動作をテスト"""
    print("\n=== RandomGeneGenerator TP/SL有効時テスト ===")
    
    # GA設定を作成
    ga_config = GAConfig(
        population_size=10,
        generations=5,
        mutation_rate=0.1,
        crossover_rate=0.8,
        elite_size=2,
        max_indicators=5,
        min_indicators=2,
        max_conditions=3,
        min_conditions=1,
    )
    
    # ランダム遺伝子生成器を初期化
    generator = RandomGeneGenerator(ga_config)
    
    # ランダム戦略遺伝子を生成
    strategy_gene = generator.generate_random_gene()
    
    print(f"指標数: {len(strategy_gene.indicators)}")
    print(f"TP/SL遺伝子: {'あり' if strategy_gene.tpsl_gene else 'なし'}")
    print(f"TP/SL有効: {strategy_gene.tpsl_gene.enabled if strategy_gene.tpsl_gene else 'N/A'}")
    print(f"exit_conditions数: {len(strategy_gene.exit_conditions)}")
    print(f"long_entry_conditions数: {len(strategy_gene.long_entry_conditions)}")
    print(f"short_entry_conditions数: {len(strategy_gene.short_entry_conditions)}")
    
    # アサーション
    assert strategy_gene.tpsl_gene is not None, "TP/SL遺伝子が生成されていません"
    
    if strategy_gene.tpsl_gene.enabled:
        assert len(strategy_gene.exit_conditions) == 0, f"TP/SL有効時にexit_conditionsが空でない: {len(strategy_gene.exit_conditions)}"
        print("✅ TP/SL有効時のexit_conditions無効化確認")
    
    assert len(strategy_gene.long_entry_conditions) > 0, "long_entry_conditionsが空です"
    assert len(strategy_gene.short_entry_conditions) > 0, "short_entry_conditionsが空です"
    
    print("✅ RandomGeneGenerator TP/SL有効時テスト成功")


def test_gene_decoder_with_real_encoded_data():
    """実際のエンコードデータでGeneDecoderをテスト"""
    print("\n=== 実際のエンコードデータでのGeneDecoderテスト ===")
    
    # 実際のGA生成フローをシミュレート
    ga_config = GAConfig(
        population_size=10,
        generations=5,
        mutation_rate=0.1,
        crossover_rate=0.8,
        elite_size=2,
        max_indicators=5,
        min_indicators=2,
        max_conditions=3,
        min_conditions=1,
    )
    
    # ランダム遺伝子を生成してエンコード
    generator = RandomGeneGenerator(ga_config)
    original_gene = generator.generate_random_gene()
    
    # エンコード
    from app.core.services.auto_strategy.models.gene_strategy import encode_gene_to_list
    encoded = encode_gene_to_list(original_gene)
    
    print(f"エンコードデータ長: {len(encoded)}")
    print(f"元の遺伝子 - TP/SL有効: {original_gene.tpsl_gene.enabled if original_gene.tpsl_gene else False}")
    print(f"元の遺伝子 - exit_conditions数: {len(original_gene.exit_conditions)}")
    
    # デコード
    decoder = GeneDecoder()
    decoded_gene = decoder.decode_list_to_strategy_gene(encoded, StrategyGene)
    
    print(f"デコード後 - TP/SL有効: {decoded_gene.tpsl_gene.enabled if decoded_gene.tpsl_gene else False}")
    print(f"デコード後 - exit_conditions数: {len(decoded_gene.exit_conditions)}")
    print(f"デコード後 - long_entry_conditions数: {len(decoded_gene.long_entry_conditions)}")
    print(f"デコード後 - short_entry_conditions数: {len(decoded_gene.short_entry_conditions)}")
    
    # アサーション
    if decoded_gene.tpsl_gene and decoded_gene.tpsl_gene.enabled:
        assert len(decoded_gene.exit_conditions) == 0, "TP/SL有効時にexit_conditionsが空でない"
        print("✅ エンコード・デコード後のexit_conditions無効化確認")
    
    assert len(decoded_gene.long_entry_conditions) > 0, "long_entry_conditionsが空です"
    assert len(decoded_gene.short_entry_conditions) > 0, "short_entry_conditionsが空です"
    
    print("✅ 実際のエンコードデータでのGeneDecoderテスト成功")


def test_strategy_gene_json_serialization():
    """StrategyGeneのJSONシリアライゼーションをテスト"""
    print("\n=== StrategyGene JSONシリアライゼーションテスト ===")
    
    # ランダム戦略遺伝子を生成
    ga_config = GAConfig(
        population_size=10,
        generations=5,
        mutation_rate=0.1,
        crossover_rate=0.8,
        elite_size=2,
        max_indicators=5,
        min_indicators=2,
        max_conditions=3,
        min_conditions=1,
    )
    
    generator = RandomGeneGenerator(ga_config)
    strategy_gene = generator.generate_random_gene()
    
    # JSONに変換
    gene_dict = strategy_gene.to_dict()
    json_str = json.dumps(gene_dict, indent=2, ensure_ascii=False)
    
    print("生成されたJSON構造:")
    print(f"- indicators: {len(gene_dict.get('indicators', []))}")
    print(f"- entry_conditions: {len(gene_dict.get('entry_conditions', []))}")
    print(f"- long_entry_conditions: {len(gene_dict.get('long_entry_conditions', []))}")
    print(f"- short_entry_conditions: {len(gene_dict.get('short_entry_conditions', []))}")
    print(f"- exit_conditions: {len(gene_dict.get('exit_conditions', []))}")
    print(f"- tpsl_gene enabled: {gene_dict.get('tpsl_gene', {}).get('enabled', False)}")
    
    # ユーザーが提供したJSONと同じ構造かチェック
    expected_fields = [
        'indicators', 'entry_conditions', 'long_entry_conditions', 
        'short_entry_conditions', 'exit_conditions', 'tpsl_gene', 
        'position_sizing_gene', 'risk_management', 'metadata'
    ]
    
    for field in expected_fields:
        assert field in gene_dict, f"必要なフィールド '{field}' がJSONに含まれていません"
    
    # TP/SL有効時のexit_conditions確認
    if gene_dict.get('tpsl_gene', {}).get('enabled', False):
        assert len(gene_dict.get('exit_conditions', [])) == 0, "TP/SL有効時にexit_conditionsが空でない"
        print("✅ JSON内でのexit_conditions無効化確認")
    
    # long/short条件の存在確認
    assert len(gene_dict.get('long_entry_conditions', [])) > 0, "long_entry_conditionsがJSONに含まれていません"
    assert len(gene_dict.get('short_entry_conditions', [])) > 0, "short_entry_conditionsがJSONに含まれていません"
    
    print("✅ StrategyGene JSONシリアライゼーションテスト成功")
    
    # 実際のJSONサンプルを出力（デバッグ用）
    print("\n--- 生成されたJSONサンプル ---")
    sample_output = {
        "strategy_config": {
            "strategy_type": "GENERATED_AUTO",
            "parameters": {
                "strategy_gene": gene_dict
            }
        },
        "experiment_id": "test",
        "db_experiment_id": 999,
        "fitness_score": 0
    }
    print(json.dumps(sample_output, indent=2, ensure_ascii=False)[:1000] + "...")


if __name__ == "__main__":
    test_random_gene_generator_with_tpsl()
    test_gene_decoder_with_real_encoded_data()
    test_strategy_gene_json_serialization()
    print("\n🎉 全てのAuto-Strategy生成テストが成功しました！")
