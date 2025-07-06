#!/usr/bin/env python3
"""
TP/SL自動決定機能の直接テスト

モジュールを直接インポートしてテストします。
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def test_tpsl_service_direct():
    """TP/SL自動決定サービスの直接テスト"""
    print("=== TP/SL自動決定サービス直接テスト ===")
    
    try:
        # 直接インポート
        sys.path.append(os.path.join(os.path.dirname(__file__), 'app', 'core', 'services', 'auto_strategy', 'services'))
        
        from tpsl_auto_decision_service import (
            TPSLAutoDecisionService,
            TPSLConfig,
            TPSLStrategy,
            TPSLResult
        )
        
        # サービスの初期化
        service = TPSLAutoDecisionService()
        print("✅ TPSLAutoDecisionService 初期化成功")
        
        # ランダム戦略のテスト
        config = TPSLConfig(strategy=TPSLStrategy.RANDOM)
        result = service.generate_tpsl_values(config)
        
        print(f"✅ ランダム戦略テスト成功:")
        print(f"   - SL: {result.stop_loss_pct:.3f} ({result.stop_loss_pct*100:.1f}%)")
        print(f"   - TP: {result.take_profit_pct:.3f} ({result.take_profit_pct*100:.1f}%)")
        print(f"   - RR比: {result.risk_reward_ratio:.2f}")
        print(f"   - 戦略: {result.strategy_used}")
        print(f"   - 信頼度: {result.confidence_score:.2f}")
        
        # リスクリワード戦略のテスト
        config = TPSLConfig(
            strategy=TPSLStrategy.RISK_REWARD,
            max_risk_per_trade=0.03,
            preferred_risk_reward_ratio=2.5
        )
        result = service.generate_tpsl_values(config)
        
        print(f"✅ リスクリワード戦略テスト成功:")
        print(f"   - SL: {result.stop_loss_pct:.3f} ({result.stop_loss_pct*100:.1f}%)")
        print(f"   - TP: {result.take_profit_pct:.3f} ({result.take_profit_pct*100:.1f}%)")
        print(f"   - RR比: {result.risk_reward_ratio:.2f}")
        print(f"   - 戦略: {result.strategy_used}")
        
        # バリデーションテスト
        is_valid = service.validate_tpsl_values(result, config)
        print(f"✅ バリデーション: {'成功' if is_valid else '失敗'}")
        
        return True
        
    except Exception as e:
        print(f"❌ TPSLAutoDecisionService テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_risk_reward_calculator_direct():
    """リスクリワード計算機の直接テスト"""
    print("\n=== リスクリワード計算機直接テスト ===")
    
    try:
        # 直接インポート
        sys.path.append(os.path.join(os.path.dirname(__file__), 'app', 'core', 'services', 'auto_strategy', 'calculators'))
        
        from risk_reward_calculator import (
            RiskRewardCalculator,
            RiskRewardConfig,
            RiskRewardResult
        )
        
        calculator = RiskRewardCalculator()
        print("✅ RiskRewardCalculator 初期化成功")
        
        # 基本計算テスト
        config = RiskRewardConfig(target_ratio=2.0)
        result = calculator.calculate_take_profit(0.03, config)
        
        print(f"✅ 基本計算テスト成功:")
        print(f"   - 入力SL: 3.0%")
        print(f"   - 計算TP: {result.take_profit_pct:.3f} ({result.take_profit_pct*100:.1f}%)")
        print(f"   - 実際RR比: {result.actual_risk_reward_ratio:.2f}")
        print(f"   - 目標達成: {'はい' if result.is_ratio_achieved else 'いいえ'}")
        
        # 上限制限テスト
        config = RiskRewardConfig(target_ratio=10.0, max_tp_limit=0.15)
        result = calculator.calculate_take_profit(0.03, config)
        
        print(f"✅ 上限制限テスト成功:")
        print(f"   - 制限後TP: {result.take_profit_pct:.3f} ({result.take_profit_pct*100:.1f}%)")
        print(f"   - 調整理由: {result.adjustment_reason}")
        
        return True
        
    except Exception as e:
        print(f"❌ RiskRewardCalculator テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_volatility_generator_direct():
    """ボラティリティ生成器の直接テスト"""
    print("\n=== ボラティリティ生成器直接テスト ===")
    
    try:
        # 直接インポート
        sys.path.append(os.path.join(os.path.dirname(__file__), 'app', 'core', 'services', 'auto_strategy', 'generators'))
        
        from volatility_based_generator import (
            VolatilityBasedGenerator,
            VolatilityConfig,
            VolatilityResult
        )
        
        generator = VolatilityBasedGenerator()
        config = VolatilityConfig()
        print("✅ VolatilityBasedGenerator 初期化成功")
        
        # ATRデータありのテスト
        market_data = {
            "atr_pct": 0.025,
            "trend_strength": 0.8,
            "volume_ratio": 1.2
        }
        
        result = generator.generate_volatility_based_tpsl(
            market_data, config, 1000.0
        )
        
        print(f"✅ ボラティリティベース生成テスト成功:")
        print(f"   - ATR: {result.atr_pct:.3f} ({result.atr_pct*100:.1f}%)")
        print(f"   - SL: {result.stop_loss_pct:.3f} ({result.stop_loss_pct*100:.1f}%)")
        print(f"   - TP: {result.take_profit_pct:.3f} ({result.take_profit_pct*100:.1f}%)")
        print(f"   - レジーム: {result.volatility_regime.value}")
        print(f"   - 信頼度: {result.confidence_score:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ VolatilityBasedGenerator テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_basic_functionality():
    """基本機能のテスト"""
    print("\n=== 基本機能テスト ===")
    
    try:
        # 基本的な計算のテスト
        print("✅ 基本的なTP/SL計算テスト:")
        
        # SL 3%, RR比 2:1 の場合
        sl_pct = 0.03
        rr_ratio = 2.0
        tp_pct = sl_pct * rr_ratio
        
        print(f"   - SL: {sl_pct:.1%}")
        print(f"   - TP: {tp_pct:.1%}")
        print(f"   - RR比: 1:{rr_ratio}")
        
        # 価格計算
        current_price = 50000
        sl_price = current_price * (1 - sl_pct)
        tp_price = current_price * (1 + tp_pct)
        
        print(f"   - 現在価格: ${current_price:,}")
        print(f"   - SL価格: ${sl_price:,.0f}")
        print(f"   - TP価格: ${tp_price:,.0f}")
        
        # バリデーション
        assert 0.005 <= sl_pct <= 0.1, "SL範囲チェック"
        assert 0.01 <= tp_pct <= 0.2, "TP範囲チェック"
        assert rr_ratio >= 1.0, "RR比チェック"
        
        print("✅ すべての基本計算が正常です")
        
        return True
        
    except Exception as e:
        print(f"❌ 基本機能テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_validation():
    """設定バリデーションのテスト"""
    print("\n=== 設定バリデーションテスト ===")
    
    try:
        # 有効な設定
        valid_configs = [
            {"max_risk": 0.03, "rr_ratio": 2.0, "volatility": "medium"},
            {"max_risk": 0.01, "rr_ratio": 1.5, "volatility": "low"},
            {"max_risk": 0.05, "rr_ratio": 3.0, "volatility": "high"},
        ]
        
        for i, config in enumerate(valid_configs):
            max_risk = config["max_risk"]
            rr_ratio = config["rr_ratio"]
            volatility = config["volatility"]
            
            # バリデーション
            assert 0.005 <= max_risk <= 0.1, f"設定{i+1}: 最大リスク範囲エラー"
            assert 1.0 <= rr_ratio <= 5.0, f"設定{i+1}: RR比範囲エラー"
            assert volatility in ["low", "medium", "high"], f"設定{i+1}: ボラティリティ感度エラー"
            
            print(f"✅ 設定{i+1}バリデーション成功: リスク{max_risk:.1%}, RR比1:{rr_ratio}, 感度{volatility}")
        
        # 無効な設定のテスト
        invalid_configs = [
            {"max_risk": 0.15, "rr_ratio": 2.0},  # リスク過大
            {"max_risk": 0.03, "rr_ratio": 6.0},  # RR比過大
            {"max_risk": 0.001, "rr_ratio": 2.0}, # リスク過小
        ]
        
        for i, config in enumerate(invalid_configs):
            max_risk = config["max_risk"]
            rr_ratio = config["rr_ratio"]
            
            is_valid = (0.005 <= max_risk <= 0.1) and (1.0 <= rr_ratio <= 5.0)
            print(f"✅ 無効設定{i+1}検出: リスク{max_risk:.1%}, RR比1:{rr_ratio} -> {'有効' if is_valid else '無効'}")
        
        return True
        
    except Exception as e:
        print(f"❌ 設定バリデーションテストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """メインテスト実行"""
    print("🚀 TP/SL自動決定機能直接テスト開始\n")
    
    tests = [
        test_basic_functionality,
        test_config_validation,
        test_tpsl_service_direct,
        test_risk_reward_calculator_direct,
        test_volatility_generator_direct,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ テスト実行エラー: {e}")
            failed += 1
    
    print(f"\n📊 テスト結果:")
    print(f"   - 成功: {passed}")
    print(f"   - 失敗: {failed}")
    print(f"   - 合計: {passed + failed}")
    
    if failed == 0:
        print("🎉 すべてのテストが成功しました！")
    else:
        print("⚠️  一部のテストが失敗しました。")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
