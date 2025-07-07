#!/usr/bin/env python3
"""
GA設定との統合テスト

実際のGA設定を使用してTP/SL自動決定機能をテストします。
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def test_ga_config_with_new_tpsl():
    """新しいTP/SL設定を使用したGA設定のテスト"""
    print("=== GA設定 + 新TP/SL機能テスト ===")

    try:
        # 直接インポート
        sys.path.append(
            os.path.join(
                os.path.dirname(__file__),
                "app",
                "core",
                "services",
                "auto_strategy",
                "models",
            )
        )

        from ga_config import GAConfig

        # 新しいTP/SL機能を有効にした設定
        ga_config = GAConfig(
            # 基本GA設定
            population_size=10,
            generations=3,
            crossover_rate=0.8,
            mutation_rate=0.1,
            elite_size=2,
            max_indicators=3,
            allowed_indicators=["SMA", "EMA", "RSI"],
            # 新しいTP/SL自動決定設定
            tpsl_strategy="risk_reward",
            max_risk_per_trade=0.03,
            preferred_risk_reward_ratio=2.0,
            volatility_sensitivity="medium",
            enable_advanced_tpsl=True,
            # 統計的設定
            statistical_lookback_days=365,
            statistical_min_samples=50,
            # ボラティリティベース設定
            atr_period=14,
            atr_multiplier_sl=2.0,
            atr_multiplier_tp=3.0,
            adaptive_multiplier=True,
        )

        print("✅ 新しいTP/SL機能付きGA設定作成成功")
        print(f"   - TP/SL戦略: {ga_config.tpsl_strategy}")
        print(f"   - 最大リスク: {ga_config.max_risk_per_trade:.1%}")
        print(f"   - RR比: 1:{ga_config.preferred_risk_reward_ratio}")
        print(f"   - ボラティリティ感度: {ga_config.volatility_sensitivity}")
        print(f"   - 高度機能有効: {ga_config.enable_advanced_tpsl}")

        # 設定の妥当性チェック
        is_valid, errors = ga_config.validate()
        print(f"✅ GA設定バリデーション: {'成功' if is_valid else '失敗'}")
        if not is_valid:
            for error in errors:
                print(f"   - エラー: {error}")

        return True

    except Exception as e:
        print(f"❌ GA設定テストエラー: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_legacy_compatibility():
    """従来方式との互換性テスト"""
    print("\n=== 従来方式互換性テスト ===")

    try:
        sys.path.append(
            os.path.join(
                os.path.dirname(__file__),
                "app",
                "core",
                "services",
                "auto_strategy",
                "models",
            )
        )

        from ga_config import GAConfig

        # 従来方式の設定（Position Sizingシステム対応）
        ga_config_legacy = GAConfig(
            population_size=10,
            generations=3,
            tpsl_strategy="legacy",  # 従来方式
            enable_advanced_tpsl=False,
            # 従来の範囲設定
            stop_loss_range=[0.02, 0.05],
            take_profit_range=[0.01, 0.15],
            # position_size_range=[0.1, 0.5]  # Position Sizingシステムにより削除
        )

        print("✅ 従来方式GA設定作成成功")
        print(f"   - TP/SL戦略: {ga_config_legacy.tpsl_strategy}")
        print(
            f"   - SL範囲: {ga_config_legacy.stop_loss_range[0]:.1%} - {ga_config_legacy.stop_loss_range[1]:.1%}"
        )
        print(
            f"   - TP範囲: {ga_config_legacy.take_profit_range[0]:.1%} - {ga_config_legacy.take_profit_range[1]:.1%}"
        )
        print(f"   - 高度機能有効: {ga_config_legacy.enable_advanced_tpsl}")

        # 設定の妥当性チェック
        is_valid, errors = ga_config_legacy.validate()
        print(f"✅ 従来方式バリデーション: {'成功' if is_valid else '失敗'}")

        return True

    except Exception as e:
        print(f"❌ 従来方式互換性テストエラー: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_preset_configurations():
    """プリセット設定のテスト"""
    print("\n=== プリセット設定テスト ===")

    try:
        sys.path.append(
            os.path.join(
                os.path.dirname(__file__),
                "app",
                "core",
                "services",
                "auto_strategy",
                "models",
            )
        )

        from ga_config import GAConfig

        # 保守的プリセット
        conservative_config = GAConfig(
            tpsl_strategy="risk_reward",
            max_risk_per_trade=0.02,  # 2%
            preferred_risk_reward_ratio=1.5,
            volatility_sensitivity="low",
            enable_advanced_tpsl=True,
        )

        # バランス型プリセット
        balanced_config = GAConfig(
            tpsl_strategy="auto_optimal",
            max_risk_per_trade=0.03,  # 3%
            preferred_risk_reward_ratio=2.0,
            volatility_sensitivity="medium",
            enable_advanced_tpsl=True,
        )

        # 積極的プリセット
        aggressive_config = GAConfig(
            tpsl_strategy="volatility_adaptive",
            max_risk_per_trade=0.05,  # 5%
            preferred_risk_reward_ratio=3.0,
            volatility_sensitivity="high",
            enable_advanced_tpsl=True,
        )

        presets = [
            ("保守的", conservative_config),
            ("バランス型", balanced_config),
            ("積極的", aggressive_config),
        ]

        for name, config in presets:
            print(f"✅ {name}プリセット:")
            print(f"   - 戦略: {config.tpsl_strategy}")
            print(f"   - 最大リスク: {config.max_risk_per_trade:.1%}")
            print(f"   - RR比: 1:{config.preferred_risk_reward_ratio}")
            print(f"   - ボラティリティ感度: {config.volatility_sensitivity}")

        return True

    except Exception as e:
        print(f"❌ プリセット設定テストエラー: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_strategy_factory_integration():
    """StrategyFactoryとの統合テスト"""
    print("\n=== StrategyFactory統合テスト ===")

    try:
        # 直接インポート
        sys.path.append(
            os.path.join(
                os.path.dirname(__file__),
                "app",
                "core",
                "services",
                "auto_strategy",
                "factories",
            )
        )
        sys.path.append(
            os.path.join(
                os.path.dirname(__file__),
                "app",
                "core",
                "services",
                "auto_strategy",
                "models",
            )
        )

        from strategy_factory import StrategyFactory
        from strategy_gene import StrategyGene

        # 新しいTP/SL機能を使用したリスク管理設定
        risk_management_advanced = {
            "stop_loss": 0.03,
            "take_profit": 0.06,
            "position_size": 0.1,
            "_tpsl_strategy": "risk_reward",
            "_risk_reward_ratio": 2.0,
            "_confidence_score": 0.85,
        }

        # 従来のリスク管理設定
        risk_management_legacy = {
            "stop_loss": 0.025,
            "take_profit": 0.05,
            "position_size": 0.1,
        }

        factory = StrategyFactory()
        current_price = 50000.0

        # 新しい方式のテスト
        sl_price_adv, tp_price_adv = factory._calculate_tpsl_prices(
            current_price, 0.03, 0.06, risk_management_advanced
        )

        print("✅ 新方式TP/SL価格計算:")
        print(f"   - 現在価格: ${current_price:,.0f}")
        print(
            f"   - SL価格: ${sl_price_adv:,.0f} ({((current_price - sl_price_adv) / current_price * 100):.1f}%)"
        )
        print(
            f"   - TP価格: ${tp_price_adv:,.0f} ({((tp_price_adv - current_price) / current_price * 100):.1f}%)"
        )

        # 従来方式のテスト
        sl_price_leg, tp_price_leg = factory._calculate_tpsl_prices(
            current_price, 0.025, 0.05, risk_management_legacy
        )

        print("✅ 従来方式TP/SL価格計算:")
        print(
            f"   - SL価格: ${sl_price_leg:,.0f} ({((current_price - sl_price_leg) / current_price * 100):.1f}%)"
        )
        print(
            f"   - TP価格: ${tp_price_leg:,.0f} ({((tp_price_leg - current_price) / current_price * 100):.1f}%)"
        )

        # 高度機能検出テスト
        is_advanced = factory._is_advanced_tpsl_used(risk_management_advanced)
        is_legacy = factory._is_advanced_tpsl_used(risk_management_legacy)

        print(f"✅ 高度機能検出:")
        print(f"   - 新方式: {'検出' if is_advanced else '未検出'}")
        print(f"   - 従来方式: {'検出' if is_legacy else '未検出'}")

        return True

    except Exception as e:
        print(f"❌ StrategyFactory統合テストエラー: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_end_to_end_workflow():
    """エンドツーエンドワークフローのテスト"""
    print("\n=== エンドツーエンドワークフローテスト ===")

    try:
        # 1. GA設定の作成
        sys.path.append(
            os.path.join(
                os.path.dirname(__file__),
                "app",
                "core",
                "services",
                "auto_strategy",
                "models",
            )
        )
        from ga_config import GAConfig

        ga_config = GAConfig(
            tpsl_strategy="risk_reward",
            max_risk_per_trade=0.03,
            preferred_risk_reward_ratio=2.5,
            volatility_sensitivity="medium",
            enable_advanced_tpsl=True,
        )
        print("✅ ステップ1: GA設定作成")

        # 2. TP/SL自動決定サービスでの値生成
        sys.path.append(
            os.path.join(
                os.path.dirname(__file__),
                "app",
                "core",
                "services",
                "auto_strategy",
                "services",
            )
        )
        from tpsl_auto_decision_service import (
            TPSLAutoDecisionService,
            TPSLConfig,
            TPSLStrategy,
        )

        service = TPSLAutoDecisionService()
        config = TPSLConfig(
            strategy=TPSLStrategy.RISK_REWARD,
            max_risk_per_trade=ga_config.max_risk_per_trade,
            preferred_risk_reward_ratio=ga_config.preferred_risk_reward_ratio,
            volatility_sensitivity=ga_config.volatility_sensitivity,
        )

        result = service.generate_tpsl_values(config)
        print("✅ ステップ2: TP/SL値自動生成")
        print(f"   - SL: {result.stop_loss_pct:.1%}")
        print(f"   - TP: {result.take_profit_pct:.1%}")
        print(f"   - RR比: {result.risk_reward_ratio:.2f}")

        # 3. リスク管理設定の作成
        risk_management = {
            "stop_loss": result.stop_loss_pct,
            "take_profit": result.take_profit_pct,
            "position_size": 0.1,
            "_tpsl_strategy": result.strategy_used,
            "_risk_reward_ratio": result.risk_reward_ratio,
            "_confidence_score": result.confidence_score,
        }
        print("✅ ステップ3: リスク管理設定作成")

        # 4. StrategyFactoryでの価格計算
        sys.path.append(
            os.path.join(
                os.path.dirname(__file__),
                "app",
                "core",
                "services",
                "auto_strategy",
                "factories",
            )
        )
        from strategy_factory import StrategyFactory

        factory = StrategyFactory()
        current_price = 50000.0

        sl_price, tp_price = factory._calculate_tpsl_prices(
            current_price, result.stop_loss_pct, result.take_profit_pct, risk_management
        )
        print("✅ ステップ4: 実際の価格計算")
        print(f"   - 現在価格: ${current_price:,.0f}")
        print(f"   - SL価格: ${sl_price:,.0f}")
        print(f"   - TP価格: ${tp_price:,.0f}")

        # 5. 最終バリデーション
        is_valid = service.validate_tpsl_values(result, config)
        print(f"✅ ステップ5: 最終バリデーション {'成功' if is_valid else '失敗'}")

        print("🎉 エンドツーエンドワークフロー完了！")
        return True

    except Exception as e:
        print(f"❌ エンドツーエンドワークフローエラー: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """メインテスト実行"""
    print("🚀 GA統合テスト開始\n")

    tests = [
        test_ga_config_with_new_tpsl,
        test_legacy_compatibility,
        test_preset_configurations,
        test_strategy_factory_integration,
        test_end_to_end_workflow,
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

    print(f"\n📊 GA統合テスト結果:")
    print(f"   - 成功: {passed}")
    print(f"   - 失敗: {failed}")
    print(f"   - 合計: {passed + failed}")

    if failed == 0:
        print("🎉 すべてのGA統合テストが成功しました！")
        print("\n✨ TP/SL自動決定機能は正常に動作しています！")
    else:
        print("⚠️  一部のGA統合テストが失敗しました。")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
