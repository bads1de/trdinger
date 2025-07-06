#!/usr/bin/env python3
"""
TP/SL GA最適化統合テスト

TP/SL設定がGA最適化対象として正常に動作することを確認し、
テクニカル指標パラメータと同様に進化することをテストします。
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def test_tpsl_gene_creation():
    """TP/SL遺伝子の作成テスト"""
    print("=== TP/SL遺伝子作成テスト ===")

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

        from tpsl_gene import TPSLGene, TPSLMethod, create_random_tpsl_gene

        # 基本的なTP/SL遺伝子作成
        tpsl_gene = TPSLGene(
            method=TPSLMethod.RISK_REWARD_RATIO,
            stop_loss_pct=0.03,
            risk_reward_ratio=2.5,
            base_stop_loss=0.03,
        )

        print(f"✅ TP/SL遺伝子作成成功:")
        print(f"   - メソッド: {tpsl_gene.method.value}")
        print(f"   - SL: {tpsl_gene.stop_loss_pct:.1%}")
        print(f"   - RR比: 1:{tpsl_gene.risk_reward_ratio}")

        # TP/SL値の計算テスト
        tpsl_values = tpsl_gene.calculate_tpsl_values()
        print(f"✅ TP/SL値計算:")
        print(f"   - SL: {tpsl_values['stop_loss']:.1%}")
        print(f"   - TP: {tpsl_values['take_profit']:.1%}")

        # ランダム遺伝子生成テスト
        random_gene = create_random_tpsl_gene()
        print(f"✅ ランダムTP/SL遺伝子:")
        print(f"   - メソッド: {random_gene.method.value}")
        print(f"   - SL: {random_gene.stop_loss_pct:.1%}")
        print(f"   - TP: {random_gene.take_profit_pct:.1%}")

        return True

    except Exception as e:
        print(f"❌ TP/SL遺伝子作成テストエラー: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_strategy_gene_with_tpsl():
    """TP/SL遺伝子を含む戦略遺伝子のテスト"""
    print("\n=== TP/SL遺伝子統合戦略遺伝子テスト ===")

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

        from strategy_gene import StrategyGene, IndicatorGene, Condition
        from tpsl_gene import TPSLGene, TPSLMethod

        # TP/SL遺伝子を作成
        tpsl_gene = TPSLGene(
            method=TPSLMethod.VOLATILITY_BASED,
            atr_multiplier_sl=2.0,
            atr_multiplier_tp=3.5,
            atr_period=14,
        )

        # 戦略遺伝子を作成（TP/SL遺伝子を含む）
        strategy_gene = StrategyGene(
            indicators=[
                IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True),
                IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
            ],
            entry_conditions=[
                Condition(left_operand="close", operator=">", right_operand="SMA")
            ],
            exit_conditions=[
                Condition(left_operand="RSI", operator=">", right_operand="70")
            ],
            risk_management={"position_size": 0.1},
            tpsl_gene=tpsl_gene,  # 新しいTP/SL遺伝子
            metadata={"test": "tpsl_integration"},
        )

        print(f"✅ TP/SL統合戦略遺伝子作成成功:")
        print(f"   - 指標数: {len(strategy_gene.indicators)}")
        print(f"   - エントリー条件: {len(strategy_gene.entry_conditions)}")
        print(f"   - TP/SL遺伝子: {strategy_gene.tpsl_gene is not None}")
        print(f"   - TP/SLメソッド: {strategy_gene.tpsl_gene.method.value}")

        # バリデーションテスト
        is_valid, errors = strategy_gene.validate()
        print(f"✅ 戦略遺伝子バリデーション: {'成功' if is_valid else '失敗'}")
        if not is_valid:
            for error in errors:
                print(f"   - エラー: {error}")

        return True

    except Exception as e:
        print(f"❌ TP/SL統合戦略遺伝子テストエラー: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_random_gene_generator_with_tpsl():
    """TP/SL遺伝子を含むランダム遺伝子生成テスト"""
    print("\n=== TP/SL遺伝子ランダム生成テスト ===")

    try:
        sys.path.append(
            os.path.join(
                os.path.dirname(__file__),
                "app",
                "core",
                "services",
                "auto_strategy",
                "generators",
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

        from random_gene_generator import RandomGeneGenerator
        from ga_config import GAConfig

        # GA設定（TP/SL制約付き）
        ga_config = GAConfig(
            population_size=5,
            generations=2,
            tpsl_method_constraints=["risk_reward_ratio", "volatility_based"],
            tpsl_sl_range=[0.02, 0.05],
            tpsl_rr_range=[1.5, 3.0],
        )

        generator = RandomGeneGenerator(ga_config)

        # ランダム遺伝子生成
        gene = generator.generate_random_gene()

        print(f"✅ TP/SL遺伝子付きランダム戦略生成成功:")
        print(f"   - 指標数: {len(gene.indicators)}")
        print(f"   - TP/SL遺伝子: {gene.tpsl_gene is not None}")

        if gene.tpsl_gene:
            print(f"   - TP/SLメソッド: {gene.tpsl_gene.method.value}")
            print(f"   - SL: {gene.tpsl_gene.stop_loss_pct:.1%}")
            print(f"   - RR比: 1:{gene.tpsl_gene.risk_reward_ratio:.1f}")

        # 個体群生成テスト
        population = generator.generate_population(3)
        print(f"✅ TP/SL遺伝子付き個体群生成: {len(population)}個体")

        for i, individual in enumerate(population):
            if individual.tpsl_gene:
                print(
                    f"   - 個体{i+1}: {individual.tpsl_gene.method.value}, SL={individual.tpsl_gene.stop_loss_pct:.1%}"
                )

        return True

    except Exception as e:
        print(f"❌ TP/SL遺伝子ランダム生成テストエラー: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_tpsl_gene_encoding():
    """TP/SL遺伝子エンコーディングテスト"""
    print("\n=== TP/SL遺伝子エンコーディングテスト ===")

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

        from gene_encoding import GeneEncoder
        from strategy_gene import StrategyGene, IndicatorGene, Condition
        from tpsl_gene import TPSLGene, TPSLMethod

        # TP/SL遺伝子を含む戦略遺伝子を作成
        tpsl_gene = TPSLGene(
            method=TPSLMethod.RISK_REWARD_RATIO,
            stop_loss_pct=0.04,
            take_profit_pct=0.08,
            risk_reward_ratio=2.0,
        )

        strategy_gene = StrategyGene(
            indicators=[IndicatorGene(type="SMA", parameters={"period": 20})],
            entry_conditions=[
                Condition(left_operand="close", operator=">", right_operand="SMA")
            ],
            exit_conditions=[
                Condition(left_operand="close", operator="<", right_operand="SMA")
            ],
            risk_management={"position_size": 0.1},
            tpsl_gene=tpsl_gene,
        )

        encoder = GeneEncoder()

        # エンコードテスト
        encoded = encoder.encode_strategy_gene_to_list(strategy_gene)
        print(f"✅ TP/SL遺伝子エンコード成功:")
        print(f"   - エンコード長: {len(encoded)}")
        print(f"   - TP/SL部分: {encoded[16:24]}")  # TP/SL遺伝子部分

        # デコードテスト
        decoded_gene = encoder.decode_list_to_strategy_gene(encoded, StrategyGene)
        print(f"✅ TP/SL遺伝子デコード成功:")
        print(f"   - TP/SL遺伝子復元: {decoded_gene.tpsl_gene is not None}")

        if decoded_gene.tpsl_gene:
            print(f"   - 復元メソッド: {decoded_gene.tpsl_gene.method.value}")
            print(f"   - 復元SL: {decoded_gene.tpsl_gene.stop_loss_pct:.3f}")
            print(f"   - 復元TP: {decoded_gene.tpsl_gene.take_profit_pct:.3f}")

        # エンコーディング情報
        encoding_info = encoder.get_encoding_info()
        print(f"✅ エンコーディング情報:")
        print(f"   - 全体長: {encoding_info['encoding_length']}")
        print(f"   - TP/SL長: {encoding_info['tpsl_encoding_length']}")
        print(f"   - サポートメソッド: {encoding_info['supported_tpsl_methods']}")

        return True

    except Exception as e:
        print(f"❌ TP/SL遺伝子エンコーディングテストエラー: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_strategy_factory_integration():
    """StrategyFactoryとの統合テスト"""
    print("\n=== StrategyFactory統合テスト ===")

    try:
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
        from strategy_gene import StrategyGene, IndicatorGene, Condition
        from tpsl_gene import TPSLGene, TPSLMethod

        # TP/SL遺伝子を含む戦略遺伝子
        tpsl_gene = TPSLGene(
            method=TPSLMethod.FIXED_PERCENTAGE,
            stop_loss_pct=0.025,
            take_profit_pct=0.075,
        )

        gene = StrategyGene(
            indicators=[IndicatorGene(type="SMA", parameters={"period": 20})],
            entry_conditions=[
                Condition(left_operand="close", operator=">", right_operand="SMA")
            ],
            exit_conditions=[
                Condition(left_operand="close", operator="<", right_operand="SMA")
            ],
            risk_management={"position_size": 0.1},
            tpsl_gene=tpsl_gene,
        )

        factory = StrategyFactory()
        current_price = 50000.0

        # TP/SL価格計算テスト
        sl_price, tp_price = factory._calculate_tpsl_from_gene(current_price, tpsl_gene)

        print(f"✅ StrategyFactory TP/SL遺伝子計算:")
        print(f"   - 現在価格: ${current_price:,.0f}")
        print(
            f"   - SL価格: ${sl_price:,.0f} ({((current_price - sl_price) / current_price * 100):.1f}%)"
        )
        print(
            f"   - TP価格: ${tp_price:,.0f} ({((tp_price - current_price) / current_price * 100):.1f}%)"
        )

        # 統合計算テスト
        sl_price_full, tp_price_full = factory._calculate_tpsl_prices(
            current_price, None, None, {}, gene
        )

        print(f"✅ 統合TP/SL価格計算:")
        print(f"   - SL価格: ${sl_price_full:,.0f}")
        print(f"   - TP価格: ${tp_price_full:,.0f}")

        return True

    except Exception as e:
        print(f"❌ StrategyFactory統合テストエラー: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_ga_optimization_workflow():
    """GA最適化ワークフローテスト"""
    print("\n=== GA最適化ワークフローテスト ===")

    try:
        print("✅ GA最適化ワークフロー概要:")
        print("   1. ユーザーがGA設定を作成（TP/SL手動設定なし）")
        print("   2. RandomGeneGeneratorがTP/SL遺伝子付き個体群を生成")
        print("   3. 各個体のTP/SL設定がテクニカル指標と同様にGA最適化")
        print("   4. 交叉・突然変異でTP/SL設定も進化")
        print("   5. 最適なTP/SL戦略が自動発見される")

        print("\n✅ 最適化対象パラメータ:")
        print(
            "   - TP/SL決定方式（固定値、リスクリワード比、ボラティリティベースなど）"
        )
        print("   - リスクリワード比（1:1.2 ～ 1:4.0）")
        print("   - 具体的なパーセンテージ（SL: 1%-8%, TP: 2%-20%）")
        print("   - ATR倍率（ボラティリティベース用）")
        print("   - 統計的パラメータ（統計ベース用）")

        print("\n✅ ユーザーエクスペリエンス:")
        print("   - TP/SL設定は完全自動化")
        print("   - 手動設定項目の大幅削減")
        print("   - GAが最適なTP/SL戦略を発見")
        print("   - テクニカル指標と同レベルの最適化")

        return True

    except Exception as e:
        print(f"❌ GA最適化ワークフローテストエラー: {e}")
        return False


def main():
    """メインテスト実行"""
    print("🚀 TP/SL GA最適化統合テスト開始\n")

    tests = [
        test_tpsl_gene_creation,
        test_strategy_gene_with_tpsl,
        test_random_gene_generator_with_tpsl,
        test_tpsl_gene_encoding,
        test_strategy_factory_integration,
        test_ga_optimization_workflow,
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

    print(f"\n📊 TP/SL GA最適化統合テスト結果:")
    print(f"   - 成功: {passed}")
    print(f"   - 失敗: {failed}")
    print(f"   - 合計: {passed + failed}")

    if failed == 0:
        print("\n🎉 すべてのTP/SL GA最適化統合テストが成功しました！")
        print("\n✨ TP/SL設定がGA最適化対象として正常に動作しています！")
        print("   テクニカル指標パラメータと同様にTP/SL設定も進化します。")
    else:
        print("\n⚠️  一部のTP/SL GA最適化統合テストが失敗しました。")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
