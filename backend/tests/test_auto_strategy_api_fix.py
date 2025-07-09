"""
Auto-Strategy API修正のテスト

実際のAPIエンドポイントで修正が正しく動作することをテストします。
"""

import pytest
import sys
import os
import json
import asyncio
from unittest.mock import patch, MagicMock

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.services.auto_strategy.services.auto_strategy_service import AutoStrategyService
from app.core.services.auto_strategy.models.ga_config import GAConfig
from app.core.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator


def test_ga_config_creation():
    """GAConfig作成をテスト"""
    print("\n=== GAConfig 作成テスト ===")

    # GA設定辞書
    ga_config_dict = {
        "population_size": 10,
        "generations": 5,
        "mutation_rate": 0.1,
        "crossover_rate": 0.8,
        "elite_size": 2,
        "max_indicators": 5,
        "fitness_weights": {
            "total_return": 0.3,
            "sharpe_ratio": 0.4,
            "max_drawdown": 0.2,
            "win_rate": 0.1,
        }
    }

    # GAConfigオブジェクトを作成
    ga_config = GAConfig.from_dict(ga_config_dict)

    print(f"個体数: {ga_config.population_size}")
    print(f"世代数: {ga_config.generations}")
    print(f"最大指標数: {ga_config.max_indicators}")
    print(f"フィットネス重み: {ga_config.fitness_weights}")

    # 検証
    is_valid, errors = ga_config.validate()
    print(f"設定有効性: {is_valid}")
    if errors:
        print(f"エラー: {errors}")

    # アサーション
    assert is_valid, f"GA設定が無効です: {errors}"
    assert ga_config.population_size == 10, "個体数が正しく設定されていません"
    assert ga_config.generations == 5, "世代数が正しく設定されていません"
    assert ga_config.max_indicators == 5, "最大指標数が正しく設定されていません"

    print("✅ GAConfig 作成テスト成功")


def test_random_gene_generator_json_output():
    """RandomGeneGeneratorの出力JSONをテスト"""
    print("\n=== RandomGeneGenerator JSON出力テスト ===")
    
    # GA設定
    ga_config = GAConfig.create_fast()
    
    # ランダム遺伝子生成器
    generator = RandomGeneGenerator(ga_config)
    
    # 複数の戦略遺伝子を生成してテスト
    for i in range(3):
        print(f"\n--- 戦略遺伝子 {i+1} ---")
        
        strategy_gene = generator.generate_random_gene()
        gene_dict = strategy_gene.to_dict()
        
        print(f"指標数: {len(gene_dict.get('indicators', []))}")
        print(f"TP/SL有効: {gene_dict.get('tpsl_gene', {}).get('enabled', False)}")
        print(f"exit_conditions数: {len(gene_dict.get('exit_conditions', []))}")
        print(f"long_entry_conditions数: {len(gene_dict.get('long_entry_conditions', []))}")
        print(f"short_entry_conditions数: {len(gene_dict.get('short_entry_conditions', []))}")
        
        # アサーション
        assert len(gene_dict.get('indicators', [])) > 0, f"戦略{i+1}: 指標が生成されていません"
        assert len(gene_dict.get('long_entry_conditions', [])) > 0, f"戦略{i+1}: long_entry_conditionsが空です"
        assert len(gene_dict.get('short_entry_conditions', [])) > 0, f"戦略{i+1}: short_entry_conditionsが空です"
        
        # TP/SL有効時のexit_conditions確認
        if gene_dict.get('tpsl_gene', {}).get('enabled', False):
            assert len(gene_dict.get('exit_conditions', [])) == 0, f"戦略{i+1}: TP/SL有効時にexit_conditionsが空でない"
            print(f"✅ 戦略{i+1}: TP/SL有効時のexit_conditions無効化確認")
        
        # ユーザーが提供したJSONと同じ構造の確認
        strategy_config = {
            "strategy_config": {
                "strategy_type": "GENERATED_AUTO",
                "parameters": {
                    "strategy_gene": gene_dict
                }
            },
            "experiment_id": f"test_{i+1}",
            "db_experiment_id": i+1,
            "fitness_score": 0
        }
        
        # JSONシリアライゼーション確認
        json_str = json.dumps(strategy_config, ensure_ascii=False, indent=2)
        assert len(json_str) > 0, f"戦略{i+1}: JSONシリアライゼーションに失敗"
        
        # JSONパース確認
        parsed = json.loads(json_str)
        assert parsed["strategy_config"]["parameters"]["strategy_gene"]["indicators"], f"戦略{i+1}: JSON内に指標が含まれていません"
        
        print(f"✅ 戦略{i+1}: JSON構造確認完了")
    
    print("✅ RandomGeneGenerator JSON出力テスト成功")


def test_strategy_gene_serialization_compatibility():
    """StrategyGeneのシリアライゼーション互換性をテスト"""
    print("\n=== StrategyGene シリアライゼーション互換性テスト ===")
    
    # ユーザーが提供したJSONと同じ構造を生成
    ga_config = GAConfig.create_fast()
    generator = RandomGeneGenerator(ga_config)
    strategy_gene = generator.generate_random_gene()
    
    # 辞書に変換
    gene_dict = strategy_gene.to_dict()
    
    # ユーザーが提供したJSONと同じフィールドが存在することを確認
    expected_fields = [
        'id', 'indicators', 'entry_conditions', 'long_entry_conditions',
        'short_entry_conditions', 'exit_conditions', 'risk_management',
        'tpsl_gene', 'position_sizing_gene', 'metadata'
    ]
    
    for field in expected_fields:
        assert field in gene_dict, f"必要なフィールド '{field}' が存在しません"
        print(f"✅ フィールド '{field}' 存在確認")
    
    # 特定フィールドの型確認
    assert isinstance(gene_dict['indicators'], list), "indicators は list である必要があります"
    assert isinstance(gene_dict['entry_conditions'], list), "entry_conditions は list である必要があります"
    assert isinstance(gene_dict['long_entry_conditions'], list), "long_entry_conditions は list である必要があります"
    assert isinstance(gene_dict['short_entry_conditions'], list), "short_entry_conditions は list である必要があります"
    assert isinstance(gene_dict['exit_conditions'], list), "exit_conditions は list である必要があります"
    
    if gene_dict['tpsl_gene']:
        assert isinstance(gene_dict['tpsl_gene'], dict), "tpsl_gene は dict である必要があります"
        assert 'enabled' in gene_dict['tpsl_gene'], "tpsl_gene に enabled フィールドが必要です"
    
    if gene_dict['position_sizing_gene']:
        assert isinstance(gene_dict['position_sizing_gene'], dict), "position_sizing_gene は dict である必要があります"
    
    print("✅ StrategyGene シリアライゼーション互換性テスト成功")
    
    # 実際の出力例を表示
    print("\n--- 生成されたJSON例 ---")
    sample_output = {
        "strategy_config": {
            "strategy_type": "GENERATED_AUTO",
            "parameters": {
                "strategy_gene": gene_dict
            }
        },
        "experiment_id": "sample",
        "db_experiment_id": 999,
        "fitness_score": 0
    }
    
    json_output = json.dumps(sample_output, indent=2, ensure_ascii=False)
    print(json_output[:800] + "..." if len(json_output) > 800 else json_output)


if __name__ == "__main__":
    test_ga_config_creation()
    test_random_gene_generator_json_output()
    test_strategy_gene_serialization_compatibility()
    print("\n🎉 全てのAuto-Strategy APIテストが成功しました！")
