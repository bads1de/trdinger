#!/usr/bin/env python3
"""
Phase 4 新規指標のテスト
PLUS_DI, MINUS_DI, ROCP, ROCR, STOCHF指標のテスト
"""

import sys
import os
import pandas as pd
import numpy as np

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def create_test_data():
    """テスト用のOHLCVデータを生成"""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    
    base_price = 50000
    returns = np.random.normal(0, 0.02, 100)
    close_prices = base_price * np.exp(np.cumsum(returns))
    
    return pd.DataFrame({
        'open': close_prices * (1 + np.random.normal(0, 0.001, 100)),
        'high': close_prices * (1 + np.abs(np.random.normal(0, 0.01, 100))),
        'low': close_prices * (1 - np.abs(np.random.normal(0, 0.01, 100))),
        'close': close_prices,
        'volume': np.random.randint(1000, 10000, 100),
    }, index=dates)

def test_phase4_indicators():
    """Phase 4 新規指標のテスト"""
    print("🧪 Phase 4 新規指標テスト")
    print("=" * 70)
    
    try:
        from app.core.services.indicators import (
            get_momentum_indicator,
            PLUSDIIndicator,
            MINUSDIIndicator,
            ROCPIndicator,
            ROCRIndicator,
            STOCHFIndicator,
        )
        
        # テストデータ準備
        df = create_test_data()
        print(f"📊 テストデータ生成完了: {len(df)} 日分")
        
        # Phase 4 指標リスト
        phase4_indicators = [
            ("PLUS_DI", PLUSDIIndicator, 14),
            ("MINUS_DI", MINUSDIIndicator, 14),
            ("ROCP", ROCPIndicator, 10),
            ("ROCR", ROCRIndicator, 10),
            ("STOCHF", STOCHFIndicator, 5),
        ]
        
        print(f"\n📊 テスト対象指標: {', '.join([name for name, _, _ in phase4_indicators])}")
        
        success_count = 0
        
        for indicator_name, indicator_class, period in phase4_indicators:
            print(f"\n🔍 {indicator_name} テスト")
            print("-" * 30)
            
            try:
                # 直接クラスでテスト
                indicator = indicator_class()
                
                if indicator_name == "STOCHF":
                    # STOCHFは辞書を返す
                    result = indicator.calculate(df, period, fastd_period=3, fastd_matype=0)
                    assert isinstance(result, dict)
                    assert "fastk" in result
                    assert "fastd" in result
                    assert isinstance(result["fastk"], pd.Series)
                    assert isinstance(result["fastd"], pd.Series)
                    valid_k = result["fastk"].dropna()
                    valid_d = result["fastd"].dropna()
                    print(f"  ✅ FastK: {len(valid_k)} 個の有効値")
                    print(f"  ✅ FastD: {len(valid_d)} 個の有効値")
                else:
                    # その他は単一Seriesを返す
                    result = indicator.calculate(df, period)
                    assert isinstance(result, pd.Series)
                    valid_values = result.dropna()
                    print(f"  ✅ {len(valid_values)} 個の有効値")
                
                # ファクトリー関数でテスト
                factory_indicator = get_momentum_indicator(indicator_name)
                assert factory_indicator.indicator_type == indicator_name
                print(f"  ✅ ファクトリー関数テスト成功")
                
                # 説明テスト
                description = indicator.get_description()
                assert isinstance(description, str)
                assert len(description) > 0
                print(f"  ✅ 説明: {description[:50]}...")
                
                success_count += 1
                print(f"  🎉 {indicator_name} テスト成功")
                
            except Exception as e:
                print(f"  ❌ {indicator_name} テスト失敗: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n📊 テスト結果: {success_count}/{len(phase4_indicators)} 成功")
        
        if success_count == len(phase4_indicators):
            print("🎉 全てのPhase 4指標が正常に動作しました！")
            return True
        else:
            print("⚠️  一部の指標でエラーが発生しました")
            return False
        
    except Exception as e:
        print(f"❌ Phase 4指標テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_auto_strategy_integration():
    """オートストラテジー統合テスト"""
    print("\n🧪 オートストラテジー統合テスト")
    print("=" * 70)
    
    try:
        from app.core.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
        
        generator = RandomGeneGenerator()
        total_indicators = len(generator.available_indicators)
        
        print(f"📊 現在の利用可能指標数: {total_indicators}")
        
        # Phase 4指標が含まれているかチェック
        phase4_indicators = ["PLUS_DI", "MINUS_DI", "ROCP", "ROCR", "STOCHF"]
        
        found_count = 0
        for indicator in phase4_indicators:
            if indicator in generator.available_indicators:
                print(f"✅ {indicator}: 利用可能リストに含まれています")
                found_count += 1
            else:
                print(f"❌ {indicator}: 利用可能リストに含まれていません")
        
        print(f"\n📊 Phase 4統合結果: {found_count}/{len(phase4_indicators)} 統合済み")
        
        # パラメータ生成テスト
        param_success = 0
        for indicator in phase4_indicators:
            try:
                params = generator._generate_indicator_parameters(indicator)
                assert isinstance(params, dict)
                assert "period" in params
                print(f"✅ {indicator}: パラメータ生成成功 - {params}")
                param_success += 1
            except Exception as e:
                print(f"❌ {indicator}: パラメータ生成失敗 - {e}")
        
        print(f"\n📊 パラメータ生成結果: {param_success}/{len(phase4_indicators)} 成功")
        
        if found_count == len(phase4_indicators) and param_success == len(phase4_indicators):
            print("🎉 Phase 4指標のオートストラテジー統合が完了しました！")
            return True
        else:
            print("⚠️  統合に問題があります")
            return False
        
    except Exception as e:
        print(f"❌ 統合テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_indicator_count():
    """指標数確認テスト"""
    print("\n🧪 指標数確認テスト")
    print("=" * 70)
    
    try:
        from app.core.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
        
        generator = RandomGeneGenerator()
        total_indicators = len(generator.available_indicators)
        
        print(f"📊 現在の利用可能指標数: {total_indicators}")
        
        # 期待される指標数（前回47 + 今回5 = 52）
        expected_count = 52
        
        if total_indicators >= expected_count:
            print(f"✅ 指標数確認成功: {total_indicators}種類（期待値: {expected_count}以上）")
            print(f"🎯 目標の50種類を超えました！")
            
            # 全指標リストを表示
            print("\n📋 利用可能指標一覧:")
            for i, indicator in enumerate(sorted(generator.available_indicators), 1):
                print(f"  {i:2d}. {indicator}")
            
            return True
        else:
            print(f"❌ 指標数不足: {total_indicators}種類（期待値: {expected_count}以上）")
            return False
        
    except Exception as e:
        print(f"❌ 指標数確認テスト失敗: {e}")
        return False

def main():
    """メイン実行関数"""
    print("🚀 Phase 4 新規指標テスト開始")
    print("=" * 70)
    
    # テスト実行
    test1_result = test_phase4_indicators()
    test2_result = test_auto_strategy_integration()
    test3_result = test_indicator_count()
    
    print("\n" + "=" * 70)
    print("📊 最終結果")
    print("=" * 70)
    
    if test1_result and test2_result and test3_result:
        print("🎉 全てのテストが成功しました！")
        print("✅ Phase 4 新規指標（PLUS_DI, MINUS_DI, ROCP, ROCR, STOCHF）が正常に動作します")
        print("🎯 指標数が50種類を超えました！")
        return True
    else:
        print("❌ 一部のテストが失敗しました")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
