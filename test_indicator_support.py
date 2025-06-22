#!/usr/bin/env python3
"""
指標サポート状況のテストスクリプト

IndicatorInitializerでPSAR指標がサポートされているかを確認します。
"""

import sys
import os
import logging

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from app.core.services.auto_strategy.factories.indicator_initializer import IndicatorInitializer

# ログ設定
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_indicator_support():
    """指標サポート状況をテスト"""
    print("指標サポート状況テスト開始")
    print("="*50)
    
    try:
        initializer = IndicatorInitializer()
        
        # サポートされている指標リストを取得
        supported_indicators = initializer.get_supported_indicators()
        
        print(f"サポートされている指標数: {len(supported_indicators)}")
        print("\nサポートされている指標一覧:")
        for i, indicator in enumerate(sorted(supported_indicators), 1):
            print(f"  {i:2d}. {indicator}")
        
        # PSAR指標の確認
        print(f"\nPSAR指標のサポート状況:")
        is_psar_supported = initializer.is_supported_indicator("PSAR")
        print(f"  PSAR: {'✓ サポート済み' if is_psar_supported else '✗ 未サポート'}")
        
        # indicator_adaptersの詳細確認
        print(f"\nindicator_adaptersの詳細:")
        print(f"  総数: {len(initializer.indicator_adapters)}")
        
        # PSARが含まれているかチェック
        if "PSAR" in initializer.indicator_adapters:
            print(f"  PSAR: 登録済み - {initializer.indicator_adapters['PSAR']}")
        else:
            print(f"  PSAR: 未登録")
            
        # VolatilityAdapterの確認
        print(f"\nVolatilityAdapterの確認:")
        try:
            from app.core.services.indicators.adapters.volatility_adapter import VolatilityAdapter
            
            # PSARメソッドが存在するかチェック
            if hasattr(VolatilityAdapter, 'psar'):
                print(f"  VolatilityAdapter.psar: ✓ 存在")
            else:
                print(f"  VolatilityAdapter.psar: ✗ 存在しない")
                
        except Exception as e:
            print(f"  VolatilityAdapter確認エラー: {e}")
        
        return is_psar_supported
        
    except Exception as e:
        print(f"テストエラー: {e}")
        logger.error(f"テストエラー: {e}", exc_info=True)
        return False

def test_strategy_factory_validation():
    """StrategyFactoryの検証をテスト"""
    print("\n" + "="*50)
    print("StrategyFactory検証テスト開始")
    print("="*50)
    
    try:
        from app.core.services.auto_strategy.factories.strategy_factory import StrategyFactory
        from app.core.services.auto_strategy.models.strategy_gene import StrategyGene, IndicatorGene, Condition
        
        # テスト用のPSAR戦略遺伝子を作成
        psar_indicator = IndicatorGene(
            type="PSAR",
            parameters={"period": 12},
            enabled=True
        )
        
        entry_condition = Condition(
            left_operand="close",
            operator=">",
            right_operand="PSAR_12"
        )
        
        exit_condition = Condition(
            left_operand="close",
            operator="<",
            right_operand="PSAR_12"
        )
        
        test_gene = StrategyGene(
            indicators=[psar_indicator],
            entry_conditions=[entry_condition],
            exit_conditions=[exit_condition],
            risk_management={"stop_loss": 0.03, "take_profit": 0.15, "position_size": 0.1}
        )
        
        # StrategyFactoryで検証
        factory = StrategyFactory()
        is_valid, errors = factory.validate_gene(test_gene)
        
        print(f"PSAR戦略遺伝子の検証結果:")
        print(f"  有効性: {'✓ 有効' if is_valid else '✗ 無効'}")
        
        if errors:
            print(f"  エラー:")
            for i, error in enumerate(errors, 1):
                print(f"    {i}. {error}")
        else:
            print(f"  エラー: なし")
            
        return is_valid
        
    except Exception as e:
        print(f"StrategyFactory検証テストエラー: {e}")
        logger.error(f"StrategyFactory検証テストエラー: {e}", exc_info=True)
        return False

def main():
    """メイン実行関数"""
    print("指標サポート状況の詳細テスト")
    print("="*50)
    
    # 1. 指標サポート状況テスト
    indicator_support = test_indicator_support()
    
    # 2. StrategyFactory検証テスト
    strategy_validation = test_strategy_factory_validation()
    
    # 結果まとめ
    print("\n" + "="*50)
    print("テスト結果まとめ")
    print("="*50)
    
    print(f"1. 指標サポート: {'✓ PSAR対応' if indicator_support else '✗ PSAR未対応'}")
    print(f"2. 戦略検証: {'✓ PSAR戦略有効' if strategy_validation else '✗ PSAR戦略無効'}")
    
    overall_success = indicator_support and strategy_validation
    print(f"\n総合結果: {'✓ 全テスト成功' if overall_success else '✗ 一部テスト失敗'}")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
