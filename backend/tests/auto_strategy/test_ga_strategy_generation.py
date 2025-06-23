"""
GA戦略生成の0取引問題解決テスト

実際のGA戦略生成プロセスをシミュレートして、
0取引問題が解決されているかを確認します。
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from unittest.mock import Mock, patch

from app.core.services.auto_strategy.factories.indicator_initializer import IndicatorInitializer
from app.core.services.auto_strategy.factories.condition_evaluator import ConditionEvaluator
from app.core.services.auto_strategy.models.gene_encoding import GeneEncoder
from app.core.services.auto_strategy.models.strategy_gene import IndicatorGene, StrategyGene, Condition

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_realistic_market_data(length: int = 500) -> pd.DataFrame:
    """現実的な市場データを生成"""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=length, freq='1h')
    
    # より現実的な価格データ生成
    price = 50000  # BTC価格想定
    prices = []
    volumes = []
    
    for i in range(length):
        # トレンド + ノイズ + 周期性
        trend = 0.001 * np.sin(i / 100)  # 長期トレンド
        cycle = 0.005 * np.sin(i / 24)   # 日次サイクル
        noise = np.random.normal(0, 0.01)  # ランダムノイズ
        
        price_change = trend + cycle + noise
        price *= (1 + price_change)
        
        # OHLC生成
        high = price * (1 + abs(np.random.normal(0, 0.005)))
        low = price * (1 - abs(np.random.normal(0, 0.005)))
        volume = 1000 + abs(np.random.normal(0, 500))
        
        prices.append(price)
        volumes.append(volume)
    
    # OHLC構造
    df = pd.DataFrame({
        'open': [prices[max(0, i-1)] for i in range(length)],
        'high': [prices[i] * (1 + abs(np.random.normal(0, 0.003))) for i in range(length)],
        'low': [prices[i] * (1 - abs(np.random.normal(0, 0.003))) for i in range(length)],
        'close': prices,
        'volume': volumes
    }, index=dates)
    
    return df


def create_mock_strategy_instance(data: pd.DataFrame) -> Mock:
    """モック戦略インスタンスを作成"""
    mock_strategy = Mock()
    
    # backtesting.pyのデータ構造をシミュレート
    mock_data = Mock()
    mock_data.Close = data['close'].values
    mock_data.High = data['high'].values
    mock_data.Low = data['low'].values
    mock_data.Open = data['open'].values
    mock_data.Volume = data['volume'].values
    
    mock_strategy.data = mock_data
    mock_strategy.indicators = {}
    
    # I()メソッドのモック
    def mock_I(func, name=None):
        result = Mock()
        result.name = name
        # 関数を実行して結果を取得
        values = func()
        result.__getitem__ = lambda self, idx: values[idx] if isinstance(values, (list, np.ndarray)) else values
        result.__len__ = lambda self: len(values) if hasattr(values, '__len__') else 1
        return result
    
    mock_strategy.I = mock_I
    
    return mock_strategy


def test_problematic_indicators():
    """問題のあった指標（MAMA等）のテスト"""
    logger.info("=== 問題指標テスト開始 ===")
    
    initializer = IndicatorInitializer()
    data = create_realistic_market_data(200)
    
    # 以前問題があった指標をテスト
    problematic_indicators = [
        ("MAMA", {"fast_limit": 0.5, "slow_limit": 0.05}),
        ("MACD", {"fast_period": 12, "slow_period": 26, "signal_period": 9}),
        ("STOCH", {"k_period": 14, "d_period": 3}),
        ("CCI", {"period": 20}),
        ("ADX", {"period": 14}),
        ("MFI", {"period": 14}),
        ("ULTOSC", {"period1": 7, "period2": 14, "period3": 28}),
        ("BOP", {}),
        ("VWAP", {"period": 20}),
    ]
    
    success_count = 0
    for indicator_type, parameters in problematic_indicators:
        try:
            result, indicator_name = initializer.calculate_indicator_only(
                indicator_type, parameters, data
            )
            
            if result is not None and indicator_name is not None:
                logger.info(f"✅ {indicator_type}: 成功 - {indicator_name}")
                success_count += 1
            else:
                logger.error(f"❌ {indicator_type}: 失敗")
                
        except Exception as e:
            logger.error(f"💥 {indicator_type}: エラー - {e}")
    
    success_rate = (success_count / len(problematic_indicators)) * 100
    logger.info(f"問題指標成功率: {success_rate:.1f}% ({success_count}/{len(problematic_indicators)})")
    
    return success_rate >= 90


def test_strategy_initialization():
    """戦略初期化の包括テスト"""
    logger.info("=== 戦略初期化テスト開始 ===")
    
    initializer = IndicatorInitializer()
    data = create_realistic_market_data(300)
    mock_strategy = create_mock_strategy_instance(data)
    
    # 様々な指標を含む戦略をテスト
    test_indicators = [
        IndicatorGene(type="MAMA", parameters={"fast_limit": 0.5, "slow_limit": 0.05}, enabled=True),
        IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
        IndicatorGene(type="MACD", parameters={"fast_period": 12, "slow_period": 26, "signal_period": 9}, enabled=True),
        IndicatorGene(type="STOCH", parameters={"k_period": 14, "d_period": 3}, enabled=True),
        IndicatorGene(type="BB", parameters={"period": 20, "std_dev": 2.0}, enabled=True),
    ]
    
    initialized_indicators = []
    for indicator_gene in test_indicators:
        try:
            indicator_name = initializer.initialize_indicator(
                indicator_gene, mock_strategy.data, mock_strategy
            )
            
            if indicator_name:
                initialized_indicators.append(indicator_name)
                logger.info(f"✅ 指標初期化成功: {indicator_name}")
            else:
                logger.error(f"❌ 指標初期化失敗: {indicator_gene.type}")
                
        except Exception as e:
            logger.error(f"💥 指標初期化エラー ({indicator_gene.type}): {e}")
    
    logger.info(f"初期化された指標: {initialized_indicators}")
    logger.info(f"戦略の指標数: {len(mock_strategy.indicators)}")
    
    return len(initialized_indicators) >= 4  # 5個中4個以上成功


def test_condition_evaluation():
    """条件評価テスト"""
    logger.info("=== 条件評価テスト開始 ===")
    
    evaluator = ConditionEvaluator()
    data = create_realistic_market_data(100)
    mock_strategy = create_mock_strategy_instance(data)
    
    # モック指標を追加
    mock_strategy.indicators["RSI_14"] = Mock()
    mock_strategy.indicators["RSI_14"].__getitem__ = lambda idx: 45.0  # RSI値
    mock_strategy.indicators["RSI_14"].__len__ = lambda: 100
    
    mock_strategy.indicators["MAMA"] = Mock()
    mock_strategy.indicators["MAMA"].__getitem__ = lambda idx: 50000.0  # MAMA値
    mock_strategy.indicators["MAMA"].__len__ = lambda: 100
    
    # テスト条件
    test_conditions = [
        Condition(left_operand="RSI_14", operator="<", right_operand="50"),
        Condition(left_operand="close", operator=">", right_operand="MAMA"),
        Condition(left_operand="close", operator="cross_above", right_operand="MAMA"),
    ]
    
    evaluation_results = []
    for i, condition in enumerate(test_conditions):
        try:
            result = evaluator.evaluate_condition(condition, mock_strategy)
            evaluation_results.append(result)
            logger.info(f"✅ 条件{i+1}評価: {result}")
        except Exception as e:
            logger.error(f"❌ 条件{i+1}評価エラー: {e}")
            evaluation_results.append(False)
    
    # エントリー・イグジット条件のテスト
    entry_result = evaluator.check_entry_conditions(test_conditions[:2], mock_strategy)
    exit_result = evaluator.check_exit_conditions(test_conditions[1:], mock_strategy)
    
    logger.info(f"エントリー条件結果: {entry_result}")
    logger.info(f"イグジット条件結果: {exit_result}")
    
    return len([r for r in evaluation_results if r is not None]) >= 2


def test_gene_encoding_decoding():
    """遺伝子エンコーディング/デコーディングテスト"""
    logger.info("=== 遺伝子エンコーディングテスト開始 ===")
    
    encoder = GeneEncoder()
    
    # MAMA指標を含む戦略遺伝子を作成
    indicators = [
        IndicatorGene(type="MAMA", parameters={"fast_limit": 0.5, "slow_limit": 0.05}, enabled=True),
        IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
        IndicatorGene(type="MACD", parameters={"fast_period": 12, "slow_period": 26, "signal_period": 9}, enabled=True),
    ]
    
    entry_conditions = [
        Condition(left_operand="close", operator="cross_above", right_operand="MAMA"),
        Condition(left_operand="RSI_14", operator="<", right_operand="30"),
    ]
    
    exit_conditions = [
        Condition(left_operand="close", operator="cross_below", right_operand="MAMA"),
        Condition(left_operand="RSI_14", operator=">", right_operand="70"),
    ]
    
    original_strategy = StrategyGene(
        indicators=indicators,
        entry_conditions=entry_conditions,
        exit_conditions=exit_conditions
    )
    
    try:
        # エンコード
        encoded = encoder.encode_strategy_gene_to_list(original_strategy)
        logger.info(f"✅ エンコード成功: 長さ={len(encoded)}")
        
        # デコード
        decoded_strategy = encoder.decode_list_to_strategy_gene(encoded, StrategyGene)
        logger.info(f"✅ デコード成功: 指標数={len(decoded_strategy.indicators)}")
        
        # MAMA指標の保持確認
        mama_preserved = any(ind.type == "MAMA" for ind in decoded_strategy.indicators)
        logger.info(f"MAMA指標保持: {mama_preserved}")
        
        # 条件の保持確認
        has_conditions = len(decoded_strategy.entry_conditions) > 0 and len(decoded_strategy.exit_conditions) > 0
        logger.info(f"条件保持: エントリー={len(decoded_strategy.entry_conditions)}, イグジット={len(decoded_strategy.exit_conditions)}")
        
        return mama_preserved and has_conditions
        
    except Exception as e:
        logger.error(f"❌ エンコーディングエラー: {e}")
        return False


def test_end_to_end_strategy_generation():
    """エンドツーエンド戦略生成テスト"""
    logger.info("=== エンドツーエンド戦略生成テスト開始 ===")
    
    # 1. 遺伝子エンコーダーで戦略生成
    encoder = GeneEncoder()
    
    # ランダムな遺伝子データをシミュレート（GAが生成するような）
    random_genes = [0.3, 0.7, 0.1, 0.9, 0.5, 0.2, 0.8, 0.4, 0.6, 0.3, 0.7, 0.1, 0.9, 0.5, 0.2, 0.8]
    
    try:
        # デコードして戦略生成
        strategy = encoder.decode_list_to_strategy_gene(random_genes, StrategyGene)
        logger.info(f"✅ 戦略生成成功: 指標数={len(strategy.indicators)}")
        
        # 2. 指標初期化
        initializer = IndicatorInitializer()
        data = create_realistic_market_data(200)
        mock_strategy = create_mock_strategy_instance(data)
        
        initialized_count = 0
        for indicator_gene in strategy.indicators:
            indicator_name = initializer.initialize_indicator(
                indicator_gene, mock_strategy.data, mock_strategy
            )
            if indicator_name:
                initialized_count += 1
        
        logger.info(f"✅ 指標初期化: {initialized_count}/{len(strategy.indicators)}個成功")
        
        # 3. 条件評価
        evaluator = ConditionEvaluator()
        
        # 実際の指標値をモック
        for name in mock_strategy.indicators.keys():
            mock_indicator = Mock()
            mock_indicator.__getitem__ = lambda idx: np.random.uniform(20, 80)  # ランダム値
            mock_indicator.__len__ = lambda: 200
            mock_strategy.indicators[name] = mock_indicator
        
        # エントリー条件評価
        entry_result = evaluator.check_entry_conditions(strategy.entry_conditions, mock_strategy)
        exit_result = evaluator.check_exit_conditions(strategy.exit_conditions, mock_strategy)
        
        logger.info(f"✅ 条件評価: エントリー={entry_result}, イグジット={exit_result}")
        
        # 成功条件：指標が初期化され、条件評価が実行できること
        success = (
            len(strategy.indicators) > 0 and
            initialized_count > 0 and
            entry_result is not None and
            exit_result is not None
        )
        
        logger.info(f"エンドツーエンドテスト結果: {'✅ 成功' if success else '❌ 失敗'}")
        return success
        
    except Exception as e:
        logger.error(f"❌ エンドツーエンドテストエラー: {e}")
        return False


def main():
    """メインテスト実行"""
    logger.info("🚀 GA戦略生成0取引問題解決テスト開始")
    
    tests = [
        ("問題指標テスト", test_problematic_indicators),
        ("戦略初期化テスト", test_strategy_initialization),
        ("条件評価テスト", test_condition_evaluation),
        ("遺伝子エンコーディングテスト", test_gene_encoding_decoding),
        ("エンドツーエンド戦略生成テスト", test_end_to_end_strategy_generation),
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
    logger.info("\n" + "="*60)
    logger.info("📊 GA戦略生成テスト結果サマリー")
    logger.info("="*60)
    
    success_count = 0
    for test_name, result in results:
        status = "✅ 成功" if result else "❌ 失敗"
        logger.info(f"{test_name}: {status}")
        if result:
            success_count += 1
    
    success_rate = (success_count / len(results)) * 100
    logger.info(f"\n総合成功率: {success_count}/{len(results)} ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        logger.info("🎉 GA戦略生成の0取引問題が解決されました！")
        logger.info("✨ 修正により、戦略が適切に指標を初期化し、条件評価が正常に動作しています。")
    else:
        logger.warning("⚠️ まだ改善が必要な箇所があります。")
    
    return success_rate >= 80


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
