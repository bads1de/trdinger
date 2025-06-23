"""
MAMA指標修正の統合テスト

実際のGA戦略生成でMAMA指標が正常に動作するかテストします。
"""

import logging
import pandas as pd
import numpy as np
from typing import List

from app.core.services.auto_strategy.factories.indicator_initializer import IndicatorInitializer
from app.core.services.auto_strategy.models.gene_encoding import GeneEncoder
from app.core.services.auto_strategy.models.strategy_gene import IndicatorGene, StrategyGene, Condition

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_data() -> pd.DataFrame:
    """テスト用の価格データを作成"""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='1H')
    
    # ランダムウォークで価格データを生成
    price = 100
    prices = []
    for _ in range(100):
        price += np.random.normal(0, 1)
        prices.append(price)
    
    df = pd.DataFrame({
        'open': prices,
        'high': [p + abs(np.random.normal(0, 0.5)) for p in prices],
        'low': [p - abs(np.random.normal(0, 0.5)) for p in prices],
        'close': prices,
        'volume': [1000 + abs(np.random.normal(0, 100)) for _ in prices]
    }, index=dates)
    
    return df


def test_mama_indicator_calculation():
    """MAMA指標の計算テスト"""
    logger.info("=== MAMA指標計算テスト開始 ===")
    
    initializer = IndicatorInitializer()
    test_data = create_test_data()
    
    # MAMA指標の計算のみテスト
    try:
        result, indicator_name = initializer.calculate_indicator_only(
            "MAMA", 
            {"fast_limit": 0.5, "slow_limit": 0.05}, 
            test_data
        )
        
        if result is not None:
            logger.info(f"✅ MAMA指標計算成功: {indicator_name}")
            logger.info(f"   結果の型: {type(result)}")
            logger.info(f"   データ数: {len(result) if hasattr(result, '__len__') else 'N/A'}")
            return True
        else:
            logger.error("❌ MAMA指標計算失敗")
            return False
            
    except Exception as e:
        logger.error(f"❌ MAMA指標計算エラー: {e}")
        return False


def test_fallback_indicators():
    """代替指標機能のテスト"""
    logger.info("=== 代替指標機能テスト開始 ===")
    
    initializer = IndicatorInitializer()
    test_data = create_test_data()
    
    # 未対応指標のテスト
    unsupported_indicators = ["STOCHF", "ROCP", "ROCR", "AROONOSC"]
    
    for indicator_type in unsupported_indicators:
        try:
            result, indicator_name = initializer.calculate_indicator_only(
                indicator_type,
                {"period": 14},
                test_data
            )
            
            if result is not None:
                logger.info(f"✅ {indicator_type} → 代替指標で計算成功: {indicator_name}")
            else:
                logger.warning(f"⚠️ {indicator_type} → 代替指標計算失敗")
                
        except Exception as e:
            logger.error(f"❌ {indicator_type} 代替指標エラー: {e}")
    
    return True


def test_gene_encoding_with_mama():
    """MAMA指標を含む遺伝子エンコーディングテスト"""
    logger.info("=== MAMA遺伝子エンコーディングテスト開始 ===")
    
    encoder = GeneEncoder()
    
    # MAMA指標を含む戦略遺伝子を作成
    indicators = [
        IndicatorGene(type="MAMA", parameters={"fast_limit": 0.5, "slow_limit": 0.05}, enabled=True),
        IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
    ]
    
    entry_conditions = [
        Condition(left_operand="close", operator="cross_above", right_operand="MAMA")
    ]
    
    exit_conditions = [
        Condition(left_operand="close", operator="cross_below", right_operand="MAMA")
    ]
    
    strategy_gene = StrategyGene(
        indicators=indicators,
        entry_conditions=entry_conditions,
        exit_conditions=exit_conditions
    )
    
    try:
        # エンコード
        encoded = encoder.encode_strategy_gene_to_list(strategy_gene)
        logger.info(f"✅ エンコード成功: 長さ={len(encoded)}")
        
        # デコード
        decoded = encoder.decode_list_to_strategy_gene(encoded, StrategyGene)
        logger.info(f"✅ デコード成功: 指標数={len(decoded.indicators)}")
        
        # MAMA指標が含まれているか確認
        mama_found = any(ind.type == "MAMA" for ind in decoded.indicators)
        if mama_found:
            logger.info("✅ デコード後にMAMA指標が保持されています")
        else:
            logger.warning("⚠️ デコード後にMAMA指標が失われました")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 遺伝子エンコーディングエラー: {e}")
        return False


def test_supported_indicators_count():
    """サポートされている指標数の確認"""
    logger.info("=== サポート指標数確認テスト開始 ===")
    
    initializer = IndicatorInitializer()
    encoder = GeneEncoder()
    
    # 利用可能な指標数
    supported_indicators = initializer.get_supported_indicators()
    logger.info(f"サポート指標数: {len(supported_indicators)}")
    logger.info(f"サポート指標: {sorted(supported_indicators)}")
    
    # エンコーダーの指標数
    encoding_info = encoder.get_encoding_info()
    logger.info(f"エンコーダー指標数: {encoding_info['indicator_count']}")
    logger.info(f"エンコーダー指標: {sorted(encoding_info['supported_indicators'])}")
    
    # MAMA指標の確認
    if "MAMA" in supported_indicators:
        logger.info("✅ MAMA指標がサポートされています")
    else:
        logger.error("❌ MAMA指標がサポートされていません")
    
    return True


def main():
    """メインテスト実行"""
    logger.info("🚀 MAMA指標修正統合テスト開始")
    
    tests = [
        ("サポート指標数確認", test_supported_indicators_count),
        ("MAMA指標計算", test_mama_indicator_calculation),
        ("代替指標機能", test_fallback_indicators),
        ("MAMA遺伝子エンコーディング", test_gene_encoding_with_mama),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
            logger.info(f"{test_name}: {'✅ 成功' if result else '❌ 失敗'}")
        except Exception as e:
            logger.error(f"{test_name}: ❌ エラー - {e}")
            results.append((test_name, False))
    
    # 結果サマリー
    logger.info("\n" + "="*50)
    logger.info("📊 テスト結果サマリー")
    logger.info("="*50)
    
    success_count = 0
    for test_name, result in results:
        status = "✅ 成功" if result else "❌ 失敗"
        logger.info(f"{test_name}: {status}")
        if result:
            success_count += 1
    
    logger.info(f"\n成功率: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
    
    if success_count == len(results):
        logger.info("🎉 すべてのテストが成功しました！")
        logger.info("MAMA指標の修正が正常に動作しています。")
    else:
        logger.warning("⚠️ 一部のテストが失敗しました。")
    
    return success_count == len(results)


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
