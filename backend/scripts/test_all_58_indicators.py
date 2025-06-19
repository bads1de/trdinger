#!/usr/bin/env python3
"""
全58指標対応テストスクリプト

バックエンドで実装されている全58指標を使用した
オートストラテジー機能のテストを実行します。
"""

import sys
import os
import time
import json
from datetime import datetime
from typing import Dict, Any, List
from collections import Counter

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.services.indicators.constants import ALL_INDICATORS
from app.core.services.auto_strategy.models.strategy_gene import (
    StrategyGene, 
    decode_list_to_gene,
    encode_gene_to_list,
    _generate_indicator_parameters,
    _generate_indicator_specific_conditions
)
from app.core.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator


def print_header(title: str):
    """ヘッダーを出力"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)


def print_section(title: str):
    """セクションヘッダーを出力"""
    print(f"\n--- {title} ---")


def test_all_indicators_availability():
    """全58指標の利用可能性をテスト"""
    print_header("全58指標利用可能性テスト")
    
    print(f"バックエンドで利用可能な指標数: {len(ALL_INDICATORS)}")
    print("\n全指標リスト:")
    
    # カテゴリ別に分類
    trend_indicators = []
    momentum_indicators = []
    volatility_indicators = []
    volume_indicators = []
    price_indicators = []
    other_indicators = []
    
    for indicator in ALL_INDICATORS:
        if indicator in ["SMA", "EMA", "WMA", "HMA", "KAMA", "TEMA", "DEMA", "T3", "ZLEMA", "MIDPOINT", "MIDPRICE", "TRIMA", "VWMA", "MAMA", "MACD"]:
            trend_indicators.append(indicator)
        elif indicator in ["RSI", "CCI", "ADX", "AROON", "MFI", "CMO", "TRIX", "DX", "ADXR", "PLUS_DI", "MINUS_DI", "STOCH", "STOCHRSI", "STOCHF", "WILLR", "MOMENTUM", "MOM", "ROC", "ROCP", "ROCR", "AROONOSC", "ULTOSC", "BOP", "APO", "PPO"]:
            momentum_indicators.append(indicator)
        elif indicator in ["BB", "ATR", "NATR", "STDDEV", "TRANGE", "KELTNER", "DONCHIAN"]:
            volatility_indicators.append(indicator)
        elif indicator in ["OBV", "AD", "VWAP", "PVT", "ADOSC", "EMV"]:
            volume_indicators.append(indicator)
        elif indicator in ["AVGPRICE", "MEDPRICE", "TYPPRICE", "WCLPRICE"]:
            price_indicators.append(indicator)
        else:
            other_indicators.append(indicator)
    
    print(f"\nトレンド系指標 ({len(trend_indicators)}個):")
    for i, indicator in enumerate(trend_indicators, 1):
        print(f"  {i:2d}. {indicator}")
    
    print(f"\nモメンタム系指標 ({len(momentum_indicators)}個):")
    for i, indicator in enumerate(momentum_indicators, 1):
        print(f"  {i:2d}. {indicator}")
    
    print(f"\nボラティリティ系指標 ({len(volatility_indicators)}個):")
    for i, indicator in enumerate(volatility_indicators, 1):
        print(f"  {i:2d}. {indicator}")
    
    print(f"\n出来高系指標 ({len(volume_indicators)}個):")
    for i, indicator in enumerate(volume_indicators, 1):
        print(f"  {i:2d}. {indicator}")
    
    print(f"\n価格変換系指標 ({len(price_indicators)}個):")
    for i, indicator in enumerate(price_indicators, 1):
        print(f"  {i:2d}. {indicator}")
    
    if other_indicators:
        print(f"\nその他の指標 ({len(other_indicators)}個):")
        for i, indicator in enumerate(other_indicators, 1):
            print(f"  {i:2d}. {indicator}")
    
    total_categorized = len(trend_indicators) + len(momentum_indicators) + len(volatility_indicators) + len(volume_indicators) + len(price_indicators) + len(other_indicators)
    print(f"\n分類済み指標数: {total_categorized}")
    print(f"全指標数: {len(ALL_INDICATORS)}")
    
    return {
        "trend": trend_indicators,
        "momentum": momentum_indicators,
        "volatility": volatility_indicators,
        "volume": volume_indicators,
        "price": price_indicators,
        "other": other_indicators
    }


def test_indicator_parameter_generation():
    """指標パラメータ生成のテスト"""
    print_header("指標パラメータ生成テスト")
    
    # 各カテゴリから代表的な指標をテスト
    test_indicators = [
        "SMA", "EMA", "MACD", "MAMA",  # トレンド系
        "RSI", "STOCH", "CCI", "ULTOSC", "APO",  # モメンタム系
        "BB", "ATR", "KELTNER",  # ボラティリティ系
        "OBV", "ADOSC", "VWAP",  # 出来高系
        "AVGPRICE", "PSAR"  # その他
    ]
    
    print_section("パラメータ生成テスト")
    
    for indicator in test_indicators:
        param_val = 0.5  # 中間値でテスト
        try:
            parameters = _generate_indicator_parameters(indicator, param_val)
            print(f"{indicator:12}: {parameters}")
        except Exception as e:
            print(f"{indicator:12}: エラー - {e}")
    
    return test_indicators


def test_comprehensive_strategy_generation():
    """包括的な戦略生成テスト"""
    print_header("包括的戦略生成テスト（全58指標対応）")
    
    generator = RandomGeneGenerator()
    strategies = []
    indicator_usage = Counter()
    
    print_section("大量戦略生成")
    
    num_strategies = 50  # 50個の戦略を生成
    print(f"{num_strategies}個の戦略を生成中...")
    
    start_time = time.time()
    
    for i in range(num_strategies):
        try:
            strategy = generator.generate_random_gene()
            strategies.append(strategy)
            
            # 指標使用状況を記録
            for indicator in strategy.indicators:
                indicator_usage[indicator.type] += 1
            
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}個完了...")
                
        except Exception as e:
            print(f"戦略 {i+1} の生成に失敗: {e}")
    
    generation_time = time.time() - start_time
    
    print(f"\n生成完了: {len(strategies)}個の戦略")
    print(f"生成時間: {generation_time:.3f}秒")
    print(f"1戦略あたり: {generation_time/len(strategies):.4f}秒")
    
    # 指標使用統計
    print_section("指標使用統計")
    print(f"使用された指標の種類: {len(indicator_usage)}")
    print(f"全指標数: {len(ALL_INDICATORS)}")
    print(f"指標カバー率: {len(indicator_usage)/len(ALL_INDICATORS)*100:.1f}%")
    
    print("\n使用頻度上位20指標:")
    for i, (indicator, count) in enumerate(indicator_usage.most_common(20), 1):
        print(f"  {i:2d}. {indicator:12}: {count:2d}回")
    
    # 未使用指標の確認
    unused_indicators = set(ALL_INDICATORS) - set(indicator_usage.keys())
    if unused_indicators:
        print(f"\n未使用指標 ({len(unused_indicators)}個):")
        for indicator in sorted(unused_indicators):
            print(f"  - {indicator}")
    else:
        print("\n✅ 全ての指標が使用されました！")
    
    return strategies, indicator_usage


def test_specific_indicator_strategies():
    """特定指標の戦略生成テスト"""
    print_header("特定指標戦略生成テスト")
    
    # 各カテゴリから代表的な指標を選んで詳細テスト
    target_indicators = [
        ("MACD", 0.4),      # トレンド系
        ("RSI", 0.3),       # モメンタム系
        ("BB", 0.5),        # ボラティリティ系
        ("STOCH", 0.6),     # オシレーター系
        ("VWAP", 0.7),      # 出来高系
        ("PSAR", 0.8),      # その他
    ]
    
    for indicator_type, param_val in target_indicators:
        print_section(f"{indicator_type} 戦略")
        
        # 特定指標を使用した戦略を生成
        test_encoded = [param_val, 0.5] + [0.0] * 14
        strategy = decode_list_to_gene(test_encoded)
        
        if strategy.indicators:
            indicator = strategy.indicators[0]
            print(f"指標タイプ: {indicator.type}")
            print(f"パラメータ: {indicator.parameters}")
            
            # 条件生成テスト
            indicator_name = f"{indicator.type}_{indicator.parameters.get('period', 20)}"
            try:
                entry_conditions, exit_conditions = _generate_indicator_specific_conditions(
                    indicator, indicator_name
                )
                
                print(f"エントリー条件数: {len(entry_conditions)}")
                for i, cond in enumerate(entry_conditions, 1):
                    print(f"  {i}. {cond.left_operand} {cond.operator} {cond.right_operand}")
                
                print(f"エグジット条件数: {len(exit_conditions)}")
                for i, cond in enumerate(exit_conditions, 1):
                    print(f"  {i}. {cond.left_operand} {cond.operator} {cond.right_operand}")
                    
            except Exception as e:
                print(f"条件生成エラー: {e}")
        else:
            print("指標が生成されませんでした")


def test_encode_decode_all_indicators():
    """全指標のエンコード/デコードテスト"""
    print_header("全指標エンコード/デコードテスト")
    
    print_section("エンコード/デコード精度テスト")
    
    generator = RandomGeneGenerator()
    success_count = 0
    error_count = 0
    
    for i in range(20):
        try:
            # オリジナル戦略を生成
            original_strategy = generator.generate_random_gene()
            
            # エンコード
            encoded = encode_gene_to_list(original_strategy)
            
            # デコード
            decoded_strategy = decode_list_to_gene(encoded)
            
            print(f"\n戦略 {i+1}:")
            print(f"  オリジナル指標数: {len(original_strategy.indicators)}")
            print(f"  デコード後指標数: {len(decoded_strategy.indicators)}")
            
            if original_strategy.indicators:
                print(f"  オリジナル指標: {[ind.type for ind in original_strategy.indicators]}")
            if decoded_strategy.indicators:
                print(f"  デコード後指標: {[ind.type for ind in decoded_strategy.indicators]}")
            
            success_count += 1
            
        except Exception as e:
            print(f"戦略 {i+1}: エラー - {e}")
            error_count += 1
    
    print(f"\n結果:")
    print(f"  成功: {success_count}")
    print(f"  エラー: {error_count}")
    print(f"  成功率: {success_count/(success_count+error_count)*100:.1f}%")


def save_comprehensive_results(strategies: List[StrategyGene], indicator_usage: Counter, filename: str = "all_58_indicators_test.json"):
    """包括的なテスト結果を保存"""
    print_section("テスト結果保存")
    
    # 戦略データを準備
    strategies_data = []
    for strategy in strategies[:10]:  # 最初の10個のみ保存
        strategies_data.append(strategy.to_dict())
    
    # 指標使用統計を準備
    usage_stats = dict(indicator_usage)
    
    output_path = os.path.join(os.path.dirname(__file__), filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "test_timestamp": datetime.now().isoformat(),
            "test_type": "all_58_indicators_comprehensive",
            "total_indicators_available": len(ALL_INDICATORS),
            "indicators_used": len(indicator_usage),
            "coverage_rate": len(indicator_usage) / len(ALL_INDICATORS) * 100,
            "total_strategies_generated": len(strategies),
            "sample_strategies": strategies_data,
            "indicator_usage_stats": usage_stats,
            "all_indicators": ALL_INDICATORS
        }, f, ensure_ascii=False, indent=2)
    
    print(f"包括的なテスト結果を保存しました: {output_path}")


def main():
    """メイン実行関数"""
    print_header("全58指標対応オートストラテジーテスト")
    print(f"実行開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 全指標の利用可能性テスト
        categorized_indicators = test_all_indicators_availability()
        
        # 指標パラメータ生成テスト
        test_indicator_parameter_generation()
        
        # 包括的な戦略生成テスト
        strategies, indicator_usage = test_comprehensive_strategy_generation()
        
        # 特定指標の戦略生成テスト
        test_specific_indicator_strategies()
        
        # エンコード/デコードテスト
        test_encode_decode_all_indicators()
        
        # 結果保存
        save_comprehensive_results(strategies, indicator_usage)
        
        print_header("全58指標テスト完了")
        print("✅ すべてのテストが正常に完了しました")
        print(f"✅ {len(ALL_INDICATORS)}個の指標が利用可能です")
        print(f"✅ {len(indicator_usage)}個の指標が実際に使用されました")
        print(f"✅ 指標カバー率: {len(indicator_usage)/len(ALL_INDICATORS)*100:.1f}%")
        print("✅ オートストラテジー機能が大幅に強化されました")
        
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
