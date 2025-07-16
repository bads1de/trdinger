#!/usr/bin/env python3
"""
戦略作成のテストスクリプト

修正後の検証ロジックで戦略ファクトリーが正常に動作することを確認します。
"""

import sys
import os
import logging
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.core.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
from app.core.services.auto_strategy.models.ga_config import GAConfig
from app.core.services.auto_strategy.factories.strategy_factory import StrategyFactory

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_strategy_creation():
    """戦略作成のテスト"""
    print("=== 戦略作成テスト開始 ===")
    
    # GAコンフィグを作成
    config = GAConfig()
    
    # ジェネレーターとファクトリーを初期化
    generator = RandomGeneGenerator(config)
    factory = StrategyFactory()
    
    success_count = 0
    total_count = 20
    
    for i in range(total_count):
        try:
            # 戦略遺伝子を生成
            gene = generator.generate_random_gene()
            
            # 戦略クラスを作成
            strategy_class = factory.create_strategy_class(gene)
            
            print(f"戦略 #{i+1}: 成功 - {strategy_class.__name__}")
            success_count += 1
            
            # 戦略の詳細情報を表示（最初の3つのみ）
            if i < 3:
                print(f"  指標数: {len(gene.indicators)}")
                print(f"  エントリー条件数: {len(gene.entry_conditions)}")
                print(f"  ロングエントリー条件数: {len(gene.long_entry_conditions)}")
                print(f"  ショートエントリー条件数: {len(gene.short_entry_conditions)}")
                
                # 条件の詳細（最初の条件のみ）
                if gene.long_entry_conditions:
                    cond = gene.long_entry_conditions[0]
                    print(f"  ロング条件例: {cond.left_operand} {cond.operator} {cond.right_operand}")
                
        except Exception as e:
            print(f"戦略 #{i+1}: 失敗 - {e}")
            logger.error(f"戦略作成エラー: {e}", exc_info=True)
    
    print(f"\n=== 結果 ===")
    print(f"総テスト数: {total_count}")
    print(f"成功数: {success_count}")
    print(f"成功率: {success_count/total_count*100:.1f}%")
    
    if success_count == total_count:
        print("✅ すべての戦略作成が成功しました！")
    else:
        print("❌ 一部の戦略作成が失敗しました。")

if __name__ == "__main__":
    test_strategy_creation()
