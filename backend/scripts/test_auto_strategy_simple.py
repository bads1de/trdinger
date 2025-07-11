#!/usr/bin/env python3
"""
Auto-strategyの簡単なテスト

エラーが解決されたかを確認します。
"""

import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.core.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
from app.core.services.auto_strategy.models.ga_config import GAConfig


def test_auto_strategy():
    """Auto-strategyの基本テスト"""
    print("=== Auto-strategy基本テスト ===")
    
    try:
        # 設定を作成
        config = GAConfig(
            population_size=5,
            generations=1,
            mutation_rate=0.1,
            crossover_rate=0.8,
            elite_size=1,
            max_indicators=3,
            min_conditions=1,
            max_conditions=2,
        )
        
        # ランダム遺伝子生成器を作成
        generator = RandomGeneGenerator(config)
        
        print("遺伝子生成器を作成しました")
        
        # 複数の戦略遺伝子を生成してテスト
        for i in range(3):
            print(f"\n戦略 {i+1} を生成中...")
            
            try:
                gene = generator.generate_random_gene()
                print(f"✅ 戦略 {i+1}: 生成成功")
                print(f"   指標数: {len(gene.indicators)}")
                print(f"   エントリー条件数: {len(gene.entry_conditions)}")
                print(f"   ロングエントリー条件数: {len(gene.long_entry_conditions)}")
                print(f"   ショートエントリー条件数: {len(gene.short_entry_conditions)}")
                
                # 検証テスト
                is_valid, errors = gene.validate()
                if is_valid:
                    print(f"   ✅ 検証: 成功")
                else:
                    print(f"   ❌ 検証: 失敗 - {errors}")
                
            except Exception as e:
                print(f"❌ 戦略 {i+1}: エラー - {str(e)}")
        
        print(f"\n=== テスト完了 ===")
        
    except Exception as e:
        print(f"❌ 初期化エラー: {str(e)}")


if __name__ == "__main__":
    test_auto_strategy()
