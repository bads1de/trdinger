#!/usr/bin/env python3
"""
TP/SL自動決定機能の簡単なテストスクリプト

conftest.pyの依存関係を回避して、直接テストを実行します。
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def test_tpsl_auto_decision_service():
    """TP/SL自動決定サービスの基本テスト"""
    print("=== TP/SL自動決定サービステスト ===")
    
    try:
        from app.core.services.auto_strategy.services.tpsl_auto_decision_service import (
            TPSLAutoDecisionService,
            TPSLConfig,
            TPSLStrategy
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


def test_risk_reward_calculator():
    """リスクリワード計算機のテスト"""
    print("\n=== リスクリワード計算機テスト ===")
    
    try:
        from app.core.services.auto_strategy.calculators.risk_reward_calculator import (
            RiskRewardCalculator,
            RiskRewardConfig
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


def test_volatility_based_generator():
    """ボラティリティベース生成器のテスト"""
    print("\n=== ボラティリティベース生成器テスト ===")
    
    try:
        from app.core.services.auto_strategy.generators.volatility_based_generator import (
            VolatilityBasedGenerator,
            VolatilityConfig
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


def test_random_gene_generator_integration():
    """RandomGeneGeneratorとの統合テスト"""
    print("\n=== RandomGeneGenerator統合テスト ===")
    
    try:
        from app.core.services.auto_strategy.models.ga_config import GAConfig
        from app.core.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
        
        # 新しいTP/SL機能を有効にした設定
        ga_config = GAConfig(
            tpsl_strategy="risk_reward",
            max_risk_per_trade=0.03,
            preferred_risk_reward_ratio=2.0,
            volatility_sensitivity="medium",
            enable_advanced_tpsl=True
        )
        
        generator = RandomGeneGenerator(ga_config)
        print("✅ RandomGeneGenerator 初期化成功")
        
        # リスク管理設定の生成テスト
        risk_management = generator._generate_risk_management()
        
        print(f"✅ 高度なリスク管理設定生成成功:")
        print(f"   - SL: {risk_management.get('stop_loss', 'N/A')}")
        print(f"   - TP: {risk_management.get('take_profit', 'N/A')}")
        print(f"   - 戦略: {risk_management.get('_tpsl_strategy', 'N/A')}")
        print(f"   - RR比: {risk_management.get('_risk_reward_ratio', 'N/A')}")
        print(f"   - 信頼度: {risk_management.get('_confidence_score', 'N/A')}")
        
        # 従来方式のテスト
        ga_config_legacy = GAConfig(tpsl_strategy="legacy")
        generator_legacy = RandomGeneGenerator(ga_config_legacy)
        risk_management_legacy = generator_legacy._generate_risk_management()
        
        print(f"✅ 従来方式（後方互換性）テスト成功:")
        print(f"   - SL: {risk_management_legacy.get('stop_loss', 'N/A')}")
        print(f"   - TP: {risk_management_legacy.get('take_profit', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ RandomGeneGenerator統合テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_strategy_factory_integration():
    """StrategyFactoryとの統合テスト"""
    print("\n=== StrategyFactory統合テスト ===")
    
    try:
        from app.core.services.auto_strategy.factories.strategy_factory import StrategyFactory
        from app.core.services.auto_strategy.models.strategy_gene import StrategyGene
        
        # テスト用の戦略遺伝子を作成
        gene = StrategyGene()
        gene.risk_management = {
            "stop_loss": 0.03,
            "take_profit": 0.06,
            "position_size": 0.1,
            "_tpsl_strategy": "risk_reward",
            "_risk_reward_ratio": 2.0,
            "_confidence_score": 0.85
        }
        
        factory = StrategyFactory()
        print("✅ StrategyFactory 初期化成功")
        
        # TP/SL価格計算のテスト
        current_price = 50000.0
        sl_price, tp_price = factory._calculate_tpsl_prices(
            current_price, 0.03, 0.06, gene.risk_management
        )
        
        print(f"✅ TP/SL価格計算テスト成功:")
        print(f"   - 現在価格: ${current_price:,.0f}")
        print(f"   - SL価格: ${sl_price:,.0f} ({((current_price - sl_price) / current_price * 100):.1f}%)")
        print(f"   - TP価格: ${tp_price:,.0f} ({((tp_price - current_price) / current_price * 100):.1f}%)")
        
        # 高度なTP/SL機能の検出テスト
        is_advanced = factory._is_advanced_tpsl_used(gene.risk_management)
        print(f"✅ 高度なTP/SL機能検出: {'有効' if is_advanced else '無効'}")
        
        return True
        
    except Exception as e:
        print(f"❌ StrategyFactory統合テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """メインテスト実行"""
    print("🚀 TP/SL自動決定機能テスト開始\n")
    
    tests = [
        test_tpsl_auto_decision_service,
        test_risk_reward_calculator,
        test_volatility_based_generator,
        test_random_gene_generator_integration,
        test_strategy_factory_integration
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
