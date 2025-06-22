#!/usr/bin/env python3
"""
GeneEncodingのテストスクリプト

エンコード・デコード処理の問題を調査します。
"""

import sys
import os
import logging

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from app.core.services.auto_strategy.models.gene_encoding import GeneEncoder
from app.core.services.auto_strategy.models.strategy_gene import StrategyGene, IndicatorGene, Condition

# ログ設定
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_indicator_id_mapping():
    """指標IDマッピングをテスト"""
    print("指標IDマッピングテスト開始")
    print("="*50)
    
    try:
        encoder = GeneEncoder()
        
        print(f"指標ID数: {len(encoder.indicator_ids)}")
        print(f"ID→指標マッピング数: {len(encoder.id_to_indicator)}")
        
        # 重要な指標のIDを確認
        important_indicators = ["SMA", "EMA", "RSI", "MACD", "PSAR"]
        print("\n重要な指標のID:")
        for indicator in important_indicators:
            indicator_id = encoder.indicator_ids.get(indicator, "未登録")
            print(f"  {indicator}: {indicator_id}")
        
        # PSARのIDを特に確認
        psar_id = encoder.indicator_ids.get("PSAR", 0)
        print(f"\nPSAR ID: {psar_id}")
        print(f"最大ID: {max(encoder.indicator_ids.values())}")
        
        # ID→指標の逆引きも確認
        print(f"\nID {psar_id} → {encoder.id_to_indicator.get(psar_id, '未登録')}")
        
        return True
        
    except Exception as e:
        print(f"指標IDマッピングテストエラー: {e}")
        logger.error(f"指標IDマッピングテストエラー: {e}", exc_info=True)
        return False

def test_decode_calculation():
    """デコード計算をテスト"""
    print("\n" + "="*50)
    print("デコード計算テスト開始")
    print("="*50)
    
    try:
        encoder = GeneEncoder()
        indicator_count = len(encoder.indicator_ids) - 1  # 0を除く
        
        print(f"指標数（0除く）: {indicator_count}")
        
        # 様々な値でデコード計算をテスト
        test_values = [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
        
        print("\nデコード計算結果:")
        print("元値 -> 正規化値 -> 指標ID -> 指標タイプ")
        
        for val in test_values:
            # 現在のデコード計算
            normalized_val = min(0.99, max(0.05, val))
            indicator_id = int(normalized_val * indicator_count) + 1
            indicator_id = max(1, min(indicator_count, indicator_id))
            
            indicator_type = encoder.id_to_indicator.get(indicator_id, "未登録")
            
            print(f"{val:4.2f} -> {normalized_val:4.2f} -> {indicator_id:2d} -> {indicator_type}")
        
        return True
        
    except Exception as e:
        print(f"デコード計算テストエラー: {e}")
        logger.error(f"デコード計算テストエラー: {e}", exc_info=True)
        return False

def test_encode_decode_cycle():
    """エンコード・デコードサイクルをテスト"""
    print("\n" + "="*50)
    print("エンコード・デコードサイクルテスト開始")
    print("="*50)
    
    try:
        encoder = GeneEncoder()
        
        # 多様な指標を持つテスト戦略を作成
        test_indicators = [
            IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True),
            IndicatorGene(type="EMA", parameters={"period": 12}, enabled=True),
            IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
            IndicatorGene(type="MACD", parameters={"fast_period": 12, "slow_period": 26}, enabled=True),
            IndicatorGene(type="PSAR", parameters={"period": 10}, enabled=True),
        ]
        
        test_gene = StrategyGene(
            indicators=test_indicators,
            entry_conditions=[Condition(left_operand="close", operator=">", right_operand="SMA_20")],
            exit_conditions=[Condition(left_operand="close", operator="<", right_operand="SMA_20")],
            risk_management={"stop_loss": 0.03, "take_profit": 0.15, "position_size": 0.1}
        )
        
        print("元の戦略:")
        for i, indicator in enumerate(test_gene.indicators):
            print(f"  指標{i+1}: {indicator.type} - {indicator.parameters}")
        
        # エンコード
        encoded = encoder.encode_strategy_gene_to_list(test_gene)
        print(f"\nエンコード結果: {encoded}")
        
        # デコード
        decoded_gene = encoder.decode_list_to_strategy_gene(encoded, StrategyGene)
        
        print(f"\nデコード後の戦略:")
        for i, indicator in enumerate(decoded_gene.indicators):
            print(f"  指標{i+1}: {indicator.type} - {indicator.parameters}")
        
        # 比較
        print(f"\n比較:")
        print(f"元の指標数: {len(test_gene.indicators)}")
        print(f"デコード後指標数: {len(decoded_gene.indicators)}")
        
        # 指標タイプの一致確認
        original_types = [ind.type for ind in test_gene.indicators]
        decoded_types = [ind.type for ind in decoded_gene.indicators]
        
        print(f"元の指標タイプ: {original_types}")
        print(f"デコード後タイプ: {decoded_types}")
        
        return True
        
    except Exception as e:
        print(f"エンコード・デコードサイクルテストエラー: {e}")
        logger.error(f"エンコード・デコードサイクルテストエラー: {e}", exc_info=True)
        return False

def test_fallback_individual_decode():
    """フォールバック個体のデコードをテスト"""
    print("\n" + "="*50)
    print("フォールバック個体デコードテスト開始")
    print("="*50)
    
    try:
        encoder = GeneEncoder()
        
        # フォールバック個体のような数値リストを生成
        import random
        random.seed(42)  # 再現性のため
        
        fallback_individual = []
        
        # 指標部分（5指標 × 2値）
        for i in range(5):
            if i == 0:
                indicator_id = random.uniform(0.1, 0.95)
            elif i < 3:
                indicator_id = random.uniform(0.05, 0.9) if random.random() < 0.8 else 0.0
            else:
                indicator_id = random.uniform(0.1, 0.9) if random.random() < 0.3 else 0.0
            
            param_val = random.uniform(0.1, 0.9)
            fallback_individual.extend([indicator_id, param_val])
        
        # 条件部分（6値）
        for i in range(6):
            if i < 3:
                fallback_individual.append(random.uniform(0.2, 0.8))
            else:
                fallback_individual.append(random.uniform(0.1, 0.9))
        
        print(f"フォールバック個体: {fallback_individual}")
        
        # デコード
        decoded_gene = encoder.decode_list_to_strategy_gene(fallback_individual, StrategyGene)
        
        print(f"\nデコード結果:")
        print(f"指標数: {len(decoded_gene.indicators)}")
        for i, indicator in enumerate(decoded_gene.indicators):
            print(f"  指標{i+1}: {indicator.type} - {indicator.parameters}")
        
        # 指標の多様性を確認
        indicator_types = [ind.type for ind in decoded_gene.indicators]
        unique_types = set(indicator_types)
        
        print(f"\n多様性分析:")
        print(f"総指標数: {len(indicator_types)}")
        print(f"ユニーク指標数: {len(unique_types)}")
        print(f"ユニーク指標: {list(unique_types)}")
        
        return len(unique_types) > 1  # 複数の異なる指標があれば成功
        
    except Exception as e:
        print(f"フォールバック個体デコードテストエラー: {e}")
        logger.error(f"フォールバック個体デコードテストエラー: {e}", exc_info=True)
        return False

def main():
    """メイン実行関数"""
    print("GeneEncoding詳細テスト")
    print("="*50)
    
    results = []
    
    # 1. 指標IDマッピングテスト
    results.append(test_indicator_id_mapping())
    
    # 2. デコード計算テスト
    results.append(test_decode_calculation())
    
    # 3. エンコード・デコードサイクルテスト
    results.append(test_encode_decode_cycle())
    
    # 4. フォールバック個体デコードテスト
    results.append(test_fallback_individual_decode())
    
    # 結果まとめ
    print("\n" + "="*50)
    print("テスト結果まとめ")
    print("="*50)
    
    test_names = [
        "指標IDマッピング",
        "デコード計算",
        "エンコード・デコードサイクル",
        "フォールバック個体デコード"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✓ 成功" if result else "✗ 失敗"
        print(f"{i+1}. {name}: {status}")
    
    all_passed = all(results)
    print(f"\n総合結果: {'✓ 全テスト成功' if all_passed else '✗ 一部テスト失敗'}")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
