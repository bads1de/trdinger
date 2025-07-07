#!/usr/bin/env python3
"""
StrategyFactoryのPosition Sizing計算をデバッグするスクリプト
"""

import sys
import os

# パスを追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_strategy_factory_debug():
    """StrategyFactoryのPosition Sizing計算をデバッグ"""
    print("=== StrategyFactory Position Sizing デバッグ ===")
    
    try:
        from app.core.services.auto_strategy.models.position_sizing_gene import PositionSizingGene, PositionSizingMethod
        from app.core.services.auto_strategy.factories.strategy_factory import StrategyFactory
        from app.core.services.auto_strategy.models.strategy_gene import StrategyGene
        
        # テスト用の遺伝子を作成
        test_gene = PositionSizingGene(
            method=PositionSizingMethod.FIXED_RATIO,
            fixed_ratio=0.1,  # 10%
            max_position_size=50.0,
            enabled=True
        )
        
        strategy_gene = StrategyGene(
            id="test_strategy",
            indicators=[],
            entry_conditions=[],
            exit_conditions=[],
            risk_management={"position_size": 0.1},
            position_sizing_gene=test_gene,
        )
        
        factory = StrategyFactory()
        
        # パラメータ
        account_balance = 100000.0
        current_price = 50000.0
        
        print(f"📊 入力パラメータ:")
        print(f"  - 口座残高: {account_balance}")
        print(f"  - 現在価格: {current_price}")
        print(f"  - Position Sizing Gene: {test_gene}")
        print(f"  - 有効フラグ: {test_gene.enabled}")
        print(f"  - 方式: {test_gene.method}")
        print(f"  - Fixed Ratio: {test_gene.fixed_ratio}")
        print(f"  - Max Position Size: {test_gene.max_position_size}")
        
        # StrategyFactoryの_calculate_position_sizeメソッドを直接呼び出し
        try:
            calculated_size = factory._calculate_position_size(
                strategy_gene, 
                account_balance=account_balance, 
                current_price=current_price, 
                data=None
            )
            print(f"✅ StrategyFactoryでの計算結果: {calculated_size}")
            
        except Exception as e:
            print(f"❌ StrategyFactory計算エラー: {e}")
            import traceback
            traceback.print_exc()
        
        # Position Sizing Calculatorを直接テスト
        print(f"\n📊 Position Sizing Calculator直接テスト:")
        
        try:
            from app.core.services.auto_strategy.calculators.position_sizing_calculator import PositionSizingCalculatorService
            
            calculator = PositionSizingCalculatorService()
            
            # 市場データを準備
            market_data = {
                "atr": current_price * 0.02,  # 2% ATR
                "atr_source": "test"
            }
            
            # 取引履歴（空）
            trade_history = []
            
            result = calculator.calculate_position_size(
                gene=test_gene,
                account_balance=account_balance,
                current_price=current_price,
                symbol="BTCUSDT",
                market_data=market_data,
                trade_history=trade_history,
                use_cache=False
            )
            
            print(f"✅ Calculator直接計算結果:")
            print(f"  - Position Size: {result.position_size}")
            print(f"  - Method Used: {result.method_used}")
            print(f"  - Confidence Score: {result.confidence_score}")
            print(f"  - Calculation Details: {result.calculation_details}")
            print(f"  - Warnings: {result.warnings}")
            
        except Exception as e:
            print(f"❌ Calculator直接テストエラー: {e}")
            import traceback
            traceback.print_exc()
        
        # PositionSizingGeneの直接計算もテスト
        print(f"\n📊 PositionSizingGene直接計算テスト:")
        
        try:
            direct_size = test_gene.calculate_position_size(
                account_balance=account_balance,
                current_price=current_price
            )
            print(f"✅ Gene直接計算結果: {direct_size}")
            
        except Exception as e:
            print(f"❌ Gene直接計算エラー: {e}")
            import traceback
            traceback.print_exc()
        
        return True
        
    except Exception as e:
        print(f"\n❌ 全体エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_strategy_factory_debug()
    sys.exit(0 if success else 1)
