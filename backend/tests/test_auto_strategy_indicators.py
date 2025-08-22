#!/usr/bin/env python3
"""
オートストラテジー指標使用テストスクリプト

VALID_INDICATOR_TYPESの全指標をテストして問題を特定します。
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService
from app.services.auto_strategy.config.constants import VALID_INDICATOR_TYPES

def create_test_data():
    """テスト用のOHLCVデータを生成"""
    np.random.seed(42)
    
    n = 100
    dates = pd.date_range('2024-01-01', periods=n, freq='1h')
    
    price = 50000
    prices = [price]
    
    for _ in range(n-1):
        change = np.random.normal(0, 0.02)
        price = price * (1 + change)
        prices.append(price)
    
    data = []
    for i, close in enumerate(prices):
        high = close * (1 + abs(np.random.normal(0, 0.01)))
        low = close * (1 - abs(np.random.normal(0, 0.01)))
        open_price = prices[i-1] if i > 0 else close
        volume = np.random.randint(1000000, 10000000)
        
        data.append({
            'Open': open_price,
            'High': max(open_price, high, close),
            'Low': min(open_price, low, close),
            'Close': close,
            'Volume': volume
        })
    
    return pd.DataFrame(data, index=dates)

def get_test_parameters(indicator_type):
    """指標タイプに応じたテストパラメータを生成"""
    # 基本パラメータマッピング
    param_map = {
        # 基本パラメータ
        "period": 14,
        "length": 14,
        
        # ストキャスティクス系
        "fastk_period": 5,
        "slowk_period": 3,
        "slowd_period": 3,
        "fastd_period": 3,
        "k_period": 5,
        "d_period": 3,
        
        # MACD系
        "fast": 12,
        "slow": 26,
        "signal": 9,
        "fast_period": 12,
        "slow_period": 26,
        "signal_period": 9,
        
        # KST系
        "r1": 10,
        "r2": 15,
        "r3": 20,
        "r4": 30,
        "n1": 10,
        "n2": 10,
        "n3": 10,
        "n4": 15,
        
        # その他特殊パラメータ
        "fastperiod": 12,
        "slowperiod": 26,
        "matype": 0,
        "k": 14,
        "d": 3,
        "smooth_k": 1,
        "tclength": 10,
        "mom": 20,
        "acceleration": 0.02,
        "maximum": 0.2,
        "std": 2.0,
        "std_dev": 2.0,
        "scalar": 2.0,
        "nbdev": 1.0,
        "fast": 13,
        "medium": 14,
        "slow": 28,
    }
    
    # 指標別特殊設定
    special_cases = {
        # ストキャスティクス系
        "STOCH": {"fastk_period": 5, "slowk_period": 3, "slowd_period": 3},
        "STOCHF": {"fastk_period": 5, "fastd_period": 3},
        "STOCHRSI": {"period": 14, "k_period": 5, "d_period": 3},
        
        # MACD系
        "MACD": {"fast": 12, "slow": 26, "signal": 9},
        "MACDEXT": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
        "MACDFIX": {"signal_period": 9},
        
        # 特殊パラメータ指標
        "KDJ": {"k": 14, "d": 3},
        "KST": {"r1": 10, "r2": 15, "r3": 20, "r4": 30},
        "STC": {"period": 10, "fast": 23, "slow": 50},
        "SMI": {"fast": 13, "slow": 25, "signal": 2},
        "PVO": {"fast": 12, "slow": 26, "signal": 9},
        
        # SAR
        "SAR": {"acceleration": 0.02, "maximum": 0.2},
        
        # ボリンジャーバンド
        "BBANDS": {"period": 20, "std": 2.0},
        "BB": {"period": 20, "std": 2.0},
        
        # ATR系
        "ATR": {"period": 14},
        "NATR": {"period": 14},
        
        # ULTOSC
        "ULTOSC": {"fast": 7, "medium": 14, "slow": 28},
        
        # RMI
        "RMI": {"length": 20, "mom": 20},
        
        # パラメータ不要な指標（空辞書）
        "OBV": {},
        "VWAP": {},
        "AD": {},
        "ADOSC": {},
        "AO": {},
        "BOP": {},
        "PPO": {},
        "APO": {},
        "TYPPRICE": {},
        "AVGPRICE": {},
        "MEDPRICE": {},
        "WCLPRICE": {},
        "NVI": {},
        "PVI": {},
        "PVT": {},
        "CMF": {},
        
        # パターン認識系（パラメータ不要）
        "CDL_DOJI": {},
        "CDL_HAMMER": {},
        "CDL_HANGING_MAN": {},
        "CDL_HARAMI": {},
        "CDL_PIERCING": {},
        "CDL_DARK_CLOUD_COVER": {},
        "CDL_THREE_BLACK_CROWS": {},
        "CDL_THREE_WHITE_SOLDIERS": {},
        "CDL_MARUBOZU": {},
        "CDL_SPINNING_TOP": {},
        "CDL_SHOOTING_STAR": {},
        "CDL_ENGULFING": {},
        "CDL_MORNING_STAR": {},
        "CDL_EVENING_STAR": {},
        "HAMMER": {},
        "ENGULFING_PATTERN": {},
        "MORNING_STAR": {},
        "EVENING_STAR": {},
    }
    
    if indicator_type in special_cases:
        return special_cases[indicator_type]
    else:
        # デフォルトパラメータ
        return {"period": 14}

def test_all_valid_indicators():
    """VALID_INDICATOR_TYPESの全指標をテスト"""
    print("🧪 VALID_INDICATOR_TYPES全指標テスト")
    print("=" * 60)
    
    df = create_test_data()
    service = TechnicalIndicatorService()
    
    success_count = 0
    error_count = 0
    total_count = len(VALID_INDICATOR_TYPES)
    
    errors = []
    
    for indicator_type in sorted(VALID_INDICATOR_TYPES):
        try:
            params = get_test_parameters(indicator_type)
            print(f"📊 {indicator_type} テスト: {params}")
            
            result = service.calculate_indicator(df, indicator_type, params)
            
            if isinstance(result, (np.ndarray, tuple)):
                print(f"  ✅ {indicator_type} 正常動作")
                success_count += 1
            else:
                print(f"  ❌ {indicator_type} 結果形式エラー: {type(result)}")
                error_count += 1
                errors.append(f"{indicator_type}: 結果形式エラー {type(result)}")
                
        except Exception as e:
            print(f"  ❌ {indicator_type} エラー: {e}")
            error_count += 1
            errors.append(f"{indicator_type}: {e}")
    
    print(f"\n全指標テスト結果: {success_count}/{total_count} 成功, {error_count} エラー")
    
    if errors:
        print("\n⚠️  発見されたエラー:")
        for error in errors:
            print(f"  - {error}")
    
    return success_count == total_count, errors

def test_auto_strategy_integration():
    """オートストラテジー統合テスト"""
    print("\n🧪 オートストラテジー統合テスト")
    print("=" * 60)
    
    try:
        from app.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
        from app.services.auto_strategy.models.ga_config import GAConfig
        
        # 基本設定
        config = GAConfig()
        generator = RandomGeneGenerator(config)
        
        # 指標生成テスト
        print("📊 指標生成テスト")
        for i in range(5):
            try:
                indicators = generator._generate_random_indicators()
                print(f"  ✅ 指標{i+1}: {len(indicators)}個生成 - {[ind.type for ind in indicators[:3]]}...")
            except Exception as e:
                print(f"  ❌ 指標{i+1} 生成エラー: {e}")
                return False
        
        # 戦略生成テスト
        print("\n📊 戦略生成テスト")
        try:
            strategy = generator.generate_random_gene()
            print(f"  ✅ 戦略生成成功: {len(strategy.indicators)}個の指標, {len(strategy.long_entry_conditions)}個のロング条件")
        except Exception as e:
            print(f"  ❌ 戦略生成エラー: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ オートストラテジー統合テストエラー: {e}")
        return False

def check_no_length_indicators_completeness():
    """NO_LENGTH_INDICATORSの完全性チェック"""
    print("\n🧪 NO_LENGTH_INDICATORS完全性チェック")
    print("=" * 60)
    
    from app.services.indicators.parameter_manager import NO_LENGTH_INDICATORS
    
    # パラメータ不要と思われる指標をチェック
    suspected_no_length = {
        # ボリューム系
        "AD", "ADOSC", "OBV", "VWAP", "NVI", "PVI", "PVT", "CMF", "EOM", "KVO",
        
        # 価格変換系
        "TYPPRICE", "AVGPRICE", "MEDPRICE", "WCLPRICE", "HA_CLOSE", "HA_OHLC",
        
        # 特殊パラメータ指標
        "SAR", "AO", "BOP", "PPO", "APO", "ULTOSC", "STC", "KDJ", "KST", "SMI", "PVO",
        
        # パターン認識系
        "CDL_DOJI", "CDL_HAMMER", "CDL_HANGING_MAN", "CDL_HARAMI", "CDL_PIERCING",
        "CDL_DARK_CLOUD_COVER", "CDL_THREE_BLACK_CROWS", "CDL_THREE_WHITE_SOLDIERS",
        "CDL_MARUBOZU", "CDL_SPINNING_TOP", "CDL_SHOOTING_STAR", "CDL_ENGULFING",
        "CDL_MORNING_STAR", "CDL_EVENING_STAR", "HAMMER", "ENGULFING_PATTERN",
        "MORNING_STAR", "EVENING_STAR",
        
        # ストキャスティクス系
        "STOCH", "STOCHF", "STOCHRSI",
        
        # その他
        "RSI_EMA_CROSS", "ICHIMOKU",
    }
    
    missing_indicators = suspected_no_length - NO_LENGTH_INDICATORS
    extra_indicators = NO_LENGTH_INDICATORS - suspected_no_length
    
    print(f"現在のNO_LENGTH_INDICATORS: {len(NO_LENGTH_INDICATORS)}個")
    print(f"疑似対象指標: {len(suspected_no_length)}個")
    
    if missing_indicators:
        print(f"\n⚠️  NO_LENGTH_INDICATORSに追加が必要な可能性: {len(missing_indicators)}個")
        for indicator in sorted(missing_indicators):
            print(f"  - {indicator}")
    
    if extra_indicators:
        print(f"\n💡 NO_LENGTH_INDICATORSにある追加指標: {len(extra_indicators)}個")
        for indicator in sorted(extra_indicators):
            print(f"  - {indicator}")
    
    return len(missing_indicators) == 0

if __name__ == "__main__":
    print("オートストラテジー指標使用テストスクリプト")
    print("=" * 70)
    
    all_passed = True
    
    # 全指標テスト
    success, errors = test_all_valid_indicators()
    all_passed &= success
    
    # オートストラテジー統合テスト
    auto_success = test_auto_strategy_integration()
    all_passed &= auto_success
    
    # NO_LENGTH_INDICATORS完全性チェック
    completeness = check_no_length_indicators_completeness()
    all_passed &= completeness
    
    print("\n" + "=" * 70)
    if all_passed:
        print("🎊 すべてのテストが成功しました！")
        print("✅ オートストラテジーでの指標使用に問題はありません")
    else:
        print("⚠️  問題が発見されました")
        if errors:
            print("修正が必要なエラー:")
            for error in errors[:10]:  # 最初の10個のエラーを表示
                print(f"  - {error}")
            if len(errors) > 10:
                print(f"  ... その他{len(errors) - 10}個のエラー")
    print("=" * 70)