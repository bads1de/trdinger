"""
OI/FR対応機能の簡単なテスト

StrategyFactoryのOI/FR機能が正しく動作するかを確認します。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.services.auto_strategy.models.strategy_gene import (
    StrategyGene, IndicatorGene, Condition
)
from app.core.services.auto_strategy.factories.strategy_factory import StrategyFactory

def test_oi_fr_validation():
    """OI/FR条件の妥当性検証テスト"""
    print("=== OI/FR条件妥当性検証テスト ===")
    
    factory = StrategyFactory()
    
    # OI/FR条件を含む戦略遺伝子を作成
    gene = StrategyGene(
        indicators=[
            IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True),
        ],
        entry_conditions=[
            Condition(left_operand="FundingRate", operator=">", right_operand=0.0005),
            Condition(left_operand="OpenInterest", operator=">", right_operand=10000000),
        ],
        exit_conditions=[
            Condition(left_operand="close", operator="<", right_operand=95),
        ]
    )
    
    # 妥当性検証
    is_valid, errors = factory.validate_gene(gene)
    
    print(f"妥当性: {is_valid}")
    if errors:
        print(f"エラー: {errors}")
    else:
        print("✅ エラーなし")
    
    return is_valid

def test_strategy_class_generation():
    """戦略クラス生成テスト"""
    print("\n=== 戦略クラス生成テスト ===")
    
    factory = StrategyFactory()
    
    gene = StrategyGene(
        indicators=[
            IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True),
        ],
        entry_conditions=[
            Condition(left_operand="FundingRate", operator=">", right_operand=0.001),
        ],
        exit_conditions=[
            Condition(left_operand="OpenInterest", operator="<", right_operand=5000000),
        ]
    )
    
    try:
        strategy_class = factory.create_strategy_class(gene)
        print(f"✅ 戦略クラス生成成功: {strategy_class.__name__}")
        return True
    except Exception as e:
        print(f"❌ 戦略クラス生成失敗: {e}")
        return False

def test_oi_fr_data_access():
    """OI/FRデータアクセステスト"""
    print("\n=== OI/FRデータアクセステスト ===")
    
    factory = StrategyFactory()
    
    gene = StrategyGene(
        indicators=[],
        entry_conditions=[
            Condition(left_operand="FundingRate", operator=">", right_operand=0.001),
        ],
        exit_conditions=[
            Condition(left_operand="close", operator="<", right_operand=95),
        ]
    )
    
    try:
        strategy_class = factory.create_strategy_class(gene)
        
        # モックデータを作成
        class MockData:
            def __init__(self):
                self.Close = [100, 101, 102]
                self.FundingRate = [0.0001, 0.0005, 0.0015]
                self.OpenInterest = [8000000, 9000000, 12000000]
        
        # 戦略インスタンス作成
        mock_data = MockData()
        strategy_instance = strategy_class(data=mock_data, params={})
        
        # OI/FRデータアクセステスト
        fr_value = strategy_instance._get_oi_fr_value("FundingRate")
        oi_value = strategy_instance._get_oi_fr_value("OpenInterest")
        
        print(f"FundingRate値: {fr_value}")
        print(f"OpenInterest値: {oi_value}")
        
        # 期待値チェック
        if fr_value == 0.0015 and oi_value == 12000000:
            print("✅ OI/FRデータアクセス成功")
            return True
        else:
            print(f"❌ 期待値と異なります。期待: FR=0.0015, OI=12000000")
            return False
            
    except Exception as e:
        print(f"❌ OI/FRデータアクセステスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メインテスト実行"""
    print("🧪 OI/FR対応機能テスト開始\n")
    
    results = []
    
    # テスト実行
    results.append(test_oi_fr_validation())
    results.append(test_strategy_class_generation())
    results.append(test_oi_fr_data_access())
    
    # 結果サマリー
    print(f"\n📊 テスト結果サマリー:")
    print(f"  成功: {sum(results)}/{len(results)}")
    print(f"  失敗: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("🎉 全てのテストが成功しました！")
        print("✅ StrategyFactoryのOI/FR対応が正常に動作しています。")
    else:
        print("⚠️ 一部のテストが失敗しました。")
    
    return all(results)

if __name__ == "__main__":
    main()
