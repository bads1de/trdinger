#!/usr/bin/env python3
"""
無効な条件エラーのデバッグスクリプト

「エントリー条件0が無効です, ロングエントリー条件0が無効です」エラーを再現・調査します。
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
from app.core.services.auto_strategy.models.gene_validation import GeneValidator
from app.core.services.auto_strategy.factories.strategy_factory import StrategyFactory

# ログ設定
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def debug_condition_validation():
    """条件検証のデバッグ"""
    print("=== 無効な条件エラーのデバッグ開始 ===")
    
    # GAコンフィグを作成
    config = GAConfig()
    
    # ジェネレーターを初期化
    generator = RandomGeneGenerator(config)
    validator = GeneValidator()
    factory = StrategyFactory()
    
    # 複数の戦略遺伝子を生成してテスト
    invalid_count = 0
    total_count = 100
    
    for i in range(total_count):
        try:
            # 戦略遺伝子を生成
            gene = generator.generate_random_gene()
            
            # 検証を実行
            is_valid, errors = validator.validate_strategy_gene(gene)
            
            if not is_valid:
                invalid_count += 1
                print(f"\n--- 無効な戦略遺伝子 #{i+1} ---")
                print(f"エラー: {errors}")
                
                # 詳細な条件情報を出力
                print(f"指標数: {len(gene.indicators)}")
                print(f"エントリー条件数: {len(gene.entry_conditions)}")
                print(f"ロングエントリー条件数: {len(gene.long_entry_conditions)}")
                print(f"ショートエントリー条件数: {len(gene.short_entry_conditions)}")
                print(f"イグジット条件数: {len(gene.exit_conditions)}")
                
                # 各条件の詳細を出力
                if gene.entry_conditions:
                    print("\n[エントリー条件]")
                    for j, condition in enumerate(gene.entry_conditions):
                        print(f"  条件{j}: {condition.__dict__}")
                        is_valid_cond, error_detail = validator.validate_condition(condition)
                        if not is_valid_cond:
                            print(f"    -> 無効: {error_detail}")
                
                if gene.long_entry_conditions:
                    print("\n[ロングエントリー条件]")
                    for j, condition in enumerate(gene.long_entry_conditions):
                        print(f"  条件{j}: {condition.__dict__}")
                        is_valid_cond, error_detail = validator.validate_condition(condition)
                        if not is_valid_cond:
                            print(f"    -> 無効: {error_detail}")
                
                if gene.short_entry_conditions:
                    print("\n[ショートエントリー条件]")
                    for j, condition in enumerate(gene.short_entry_conditions):
                        print(f"  条件{j}: {condition.__dict__}")
                        is_valid_cond, error_detail = validator.validate_condition(condition)
                        if not is_valid_cond:
                            print(f"    -> 無効: {error_detail}")
                
                # 戦略ファクトリーでの検証も試行
                try:
                    strategy_class = factory.create_strategy_class(gene)
                    print("戦略ファクトリーでの検証: 成功")
                except Exception as e:
                    print(f"戦略ファクトリーでの検証: 失敗 - {e}")
                
                # 最初の無効な戦略で詳細分析を停止
                if invalid_count >= 3:
                    break
                    
        except Exception as e:
            print(f"戦略遺伝子生成エラー #{i+1}: {e}")
            logger.error(f"戦略遺伝子生成エラー: {e}", exc_info=True)
    
    print(f"\n=== 結果 ===")
    print(f"総生成数: {total_count}")
    print(f"無効な戦略数: {invalid_count}")
    print(f"無効率: {invalid_count/total_count*100:.1f}%")

def test_specific_condition_patterns():
    """特定の条件パターンをテスト"""
    print("\n=== 特定の条件パターンのテスト ===")
    
    validator = GeneValidator()
    
    # 問題のある条件パターンをテスト
    from app.core.services.auto_strategy.models.gene_strategy import Condition
    
    test_conditions = [
        # 空文字列オペランド
        Condition(left_operand="", operator=">", right_operand="close"),
        Condition(left_operand="close", operator=">", right_operand=""),
        
        # None オペランド
        Condition(left_operand=None, operator=">", right_operand="close"),
        Condition(left_operand="close", operator=">", right_operand=None),
        
        # 無効な演算子
        Condition(left_operand="close", operator="invalid", right_operand="open"),
        
        # 無効な指標名
        Condition(left_operand="INVALID_INDICATOR", operator=">", right_operand="close"),
        
        # 辞書形式のオペランド（正常）
        Condition(
            left_operand={"type": "indicator", "name": "SMA"},
            operator=">",
            right_operand="close"
        ),
        
        # 辞書形式のオペランド（異常）
        Condition(
            left_operand={"type": "indicator", "name": ""},
            operator=">",
            right_operand="close"
        ),
    ]
    
    for i, condition in enumerate(test_conditions):
        print(f"\n--- テスト条件 #{i+1} ---")
        print(f"条件: {condition.__dict__}")
        
        is_valid, error_detail = validator.validate_condition(condition)
        print(f"検証結果: {'有効' if is_valid else '無効'}")
        if not is_valid:
            print(f"エラー詳細: {error_detail}")

if __name__ == "__main__":
    debug_condition_validation()
    test_specific_condition_patterns()
