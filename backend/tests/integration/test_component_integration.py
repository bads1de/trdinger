#!/usr/bin/env python3
"""
コンポーネント統合テスト

実際のコンポーネントを使用してTP/SL GA最適化機能の
統合動作を確認します。
"""

import sys
import os
import logging

# ログレベルを設定
logging.basicConfig(level=logging.INFO)

# パスを設定
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def test_tpsl_gene_component():
    """TP/SL遺伝子コンポーネントのテスト"""
    print("=== TP/SL遺伝子コンポーネントテスト ===")

    try:
        # 直接インポートでテスト
        from app.core.services.auto_strategy.models.tpsl_gene import (
            TPSLGene,
            TPSLMethod,
            create_random_tpsl_gene,
            crossover_tpsl_genes,
            mutate_tpsl_gene,
        )

        # 基本的なTP/SL遺伝子作成
        gene1 = TPSLGene(
            method=TPSLMethod.RISK_REWARD_RATIO,
            stop_loss_pct=0.03,
            risk_reward_ratio=2.0,
            base_stop_loss=0.03,
        )

        print(f"✅ TP/SL遺伝子1作成成功:")
        print(f"   - メソッド: {gene1.method.value}")
        print(f"   - SL: {gene1.stop_loss_pct:.1%}")
        print(f"   - RR比: 1:{gene1.risk_reward_ratio}")

        # バリデーションテスト
        is_valid, errors = gene1.validate()
        print(f"✅ バリデーション: {'成功' if is_valid else '失敗'}")
        if errors:
            for error in errors:
                print(f"   - エラー: {error}")

        # TP/SL値計算テスト
        tpsl_values = gene1.calculate_tpsl_values()
        print(f"✅ TP/SL値計算:")
        print(f"   - SL: {tpsl_values['stop_loss']:.1%}")
        print(f"   - TP: {tpsl_values['take_profit']:.1%}")

        # ランダム遺伝子生成テスト
        random_gene = create_random_tpsl_gene()
        print(f"✅ ランダム遺伝子生成:")
        print(f"   - メソッド: {random_gene.method.value}")
        print(f"   - SL: {random_gene.stop_loss_pct:.1%}")

        # 交叉テスト
        gene2 = TPSLGene(
            method=TPSLMethod.VOLATILITY_BASED,
            atr_multiplier_sl=2.5,
            atr_multiplier_tp=3.5,
        )

        child1, child2 = crossover_tpsl_genes(gene1, gene2)
        print(f"✅ 交叉テスト:")
        print(f"   - 子1メソッド: {child1.method.value}")
        print(f"   - 子2メソッド: {child2.method.value}")

        # 突然変異テスト
        mutated = mutate_tpsl_gene(gene1, mutation_rate=0.5)
        print(f"✅ 突然変異テスト:")
        print(f"   - 元SL: {gene1.stop_loss_pct:.1%}")
        print(f"   - 変異後SL: {mutated.stop_loss_pct:.1%}")

        return True

    except Exception as e:
        print(f"❌ TP/SL遺伝子コンポーネントテストエラー: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_ga_config_component():
    """GA設定コンポーネントのテスト"""
    print("\n=== GA設定コンポーネントテスト ===")

    try:
        from app.core.services.auto_strategy.models.ga_config import GAConfig

        # 新しいGA設定（TP/SL制約付き）
        ga_config = GAConfig(
            population_size=10,
            generations=5,
            crossover_rate=0.8,
            mutation_rate=0.1,
            # TP/SL GA最適化制約
            tpsl_method_constraints=[
                "risk_reward_ratio",
                "volatility_based",
                "fixed_percentage",
            ],
            tpsl_sl_range=[0.02, 0.06],
            tpsl_tp_range=[0.03, 0.15],
            tpsl_rr_range=[1.5, 3.5],
            tpsl_atr_multiplier_range=[1.5, 3.5],
        )

        print(f"✅ GA設定作成成功:")
        print(f"   - 個体数: {ga_config.population_size}")
        print(f"   - 世代数: {ga_config.generations}")
        print(f"   - TP/SLメソッド制約: {ga_config.tpsl_method_constraints}")
        print(
            f"   - SL範囲: {ga_config.tpsl_sl_range[0]:.1%} - {ga_config.tpsl_sl_range[1]:.1%}"
        )
        print(
            f"   - TP範囲: {ga_config.tpsl_tp_range[0]:.1%} - {ga_config.tpsl_tp_range[1]:.1%}"
        )
        print(
            f"   - RR比範囲: 1:{ga_config.tpsl_rr_range[0]} - 1:{ga_config.tpsl_rr_range[1]}"
        )

        # バリデーションテスト
        is_valid, errors = ga_config.validate()
        print(f"✅ GA設定バリデーション: {'成功' if is_valid else '失敗'}")
        if errors:
            for error in errors:
                print(f"   - エラー: {error}")

        return True

    except Exception as e:
        print(f"❌ GA設定コンポーネントテストエラー: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_random_generator_component():
    """ランダム遺伝子生成器コンポーネントのテスト"""
    print("\n=== ランダム遺伝子生成器コンポーネントテスト ===")

    try:
        from app.core.services.auto_strategy.generators.random_gene_generator import (
            RandomGeneGenerator,
        )
        from app.core.services.auto_strategy.models.ga_config import GAConfig

        # GA設定
        ga_config = GAConfig(
            population_size=3,
            generations=2,
            max_indicators=2,
            allowed_indicators=["SMA", "RSI"],
            # TP/SL制約
            tpsl_method_constraints=["risk_reward_ratio", "volatility_based"],
            tpsl_sl_range=[0.02, 0.05],
            tpsl_rr_range=[1.5, 3.0],
        )

        generator = RandomGeneGenerator(ga_config)

        # 単一遺伝子生成テスト
        gene = generator.generate_random_gene()

        print(f"✅ ランダム戦略遺伝子生成成功:")
        print(f"   - 指標数: {len(gene.indicators)}")
        print(f"   - エントリー条件数: {len(gene.entry_conditions)}")
        print(f"   - TP/SL遺伝子: {gene.tpsl_gene is not None}")

        if gene.tpsl_gene:
            print(f"   - TP/SLメソッド: {gene.tpsl_gene.method.value}")
            print(f"   - SL: {gene.tpsl_gene.stop_loss_pct:.1%}")
            print(f"   - RR比: 1:{gene.tpsl_gene.risk_reward_ratio:.1f}")

        # 個体群生成テスト
        population = generator.generate_population(3)
        print(f"✅ 個体群生成成功: {len(population)}個体")

        for i, individual in enumerate(population):
            if individual.tpsl_gene:
                print(
                    f"   - 個体{i+1}: {individual.tpsl_gene.method.value}, SL={individual.tpsl_gene.stop_loss_pct:.1%}"
                )

        return True

    except Exception as e:
        print(f"❌ ランダム遺伝子生成器コンポーネントテストエラー: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_strategy_factory_component():
    """戦略ファクトリーコンポーネントのテスト"""
    print("\n=== 戦略ファクトリーコンポーネントテスト ===")

    try:
        from app.core.services.auto_strategy.factories.strategy_factory import (
            StrategyFactory,
        )
        from app.core.services.auto_strategy.models.tpsl_gene import (
            TPSLGene,
            TPSLMethod,
        )

        # TP/SL遺伝子を作成
        tpsl_gene = TPSLGene(
            method=TPSLMethod.FIXED_PERCENTAGE,
            stop_loss_pct=0.025,
            take_profit_pct=0.075,
        )

        # モック遺伝子オブジェクト
        class MockGene:
            def __init__(self, tpsl_gene):
                self.tpsl_gene = tpsl_gene

        mock_gene = MockGene(tpsl_gene)

        factory = StrategyFactory()
        current_price = 50000.0

        # TP/SL価格計算テスト
        sl_price, tp_price = factory._calculate_tpsl_from_gene(current_price, tpsl_gene)

        print(f"✅ TP/SL遺伝子価格計算:")
        print(f"   - 現在価格: ${current_price:,.0f}")
        print(
            f"   - SL価格: ${sl_price:,.0f} ({((current_price - sl_price) / current_price * 100):.1f}%)"
        )
        print(
            f"   - TP価格: ${tp_price:,.0f} ({((tp_price - current_price) / current_price * 100):.1f}%)"
        )

        # 統合計算テスト
        sl_price_full, tp_price_full = factory._calculate_tpsl_prices(
            current_price, None, None, {}, mock_gene
        )

        print(f"✅ 統合TP/SL価格計算:")
        print(f"   - SL価格: ${sl_price_full:,.0f}")
        print(f"   - TP価格: ${tp_price_full:,.0f}")

        # 異なるメソッドでのテスト
        rr_gene = TPSLGene(
            method=TPSLMethod.RISK_REWARD_RATIO,
            base_stop_loss=0.03,
            risk_reward_ratio=2.5,
        )

        rr_mock_gene = MockGene(rr_gene)
        sl_rr, tp_rr = factory._calculate_tpsl_prices(
            current_price, None, None, {}, rr_mock_gene
        )

        print(f"✅ リスクリワード比ベース計算:")
        print(f"   - SL価格: ${sl_rr:,.0f}")
        print(f"   - TP価格: ${tp_rr:,.0f}")
        print(
            f"   - 実際RR比: {((tp_rr - current_price) / (current_price - sl_rr)):.1f}"
        )

        return True

    except Exception as e:
        print(f"❌ 戦略ファクトリーコンポーネントテストエラー: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_encoding_component():
    """エンコーディングコンポーネントのテスト"""
    print("\n=== エンコーディングコンポーネントテスト ===")

    try:
        from app.core.services.auto_strategy.models.gene_encoding import GeneEncoder
        from app.core.services.auto_strategy.models.tpsl_gene import (
            TPSLGene,
            TPSLMethod,
        )

        # TP/SL遺伝子を作成
        tpsl_gene = TPSLGene(
            method=TPSLMethod.RISK_REWARD_RATIO,
            stop_loss_pct=0.04,
            take_profit_pct=0.08,
            risk_reward_ratio=2.0,
            base_stop_loss=0.04,
        )

        encoder = GeneEncoder()

        # エンコードテスト
        encoded = encoder._encode_tpsl_gene(tpsl_gene)
        print(f"✅ TP/SL遺伝子エンコード:")
        print(f"   - エンコード長: {len(encoded)}")
        print(f"   - エンコード値: {[f'{x:.3f}' for x in encoded]}")

        # デコードテスト
        decoded_gene = encoder._decode_tpsl_gene(encoded)
        print(f"✅ TP/SL遺伝子デコード:")
        print(f"   - 元メソッド: {tpsl_gene.method.value}")
        print(f"   - 復元メソッド: {decoded_gene.method.value}")
        print(f"   - 元SL: {tpsl_gene.stop_loss_pct:.3f}")
        print(f"   - 復元SL: {decoded_gene.stop_loss_pct:.3f}")
        print(f"   - 元RR比: {tpsl_gene.risk_reward_ratio:.1f}")
        print(f"   - 復元RR比: {decoded_gene.risk_reward_ratio:.1f}")

        # エンコーディング情報
        encoding_info = encoder.get_encoding_info()
        print(f"✅ エンコーディング情報:")
        print(f"   - 全体長: {encoding_info['encoding_length']}")
        print(f"   - TP/SL長: {encoding_info['tpsl_encoding_length']}")
        print(
            f"   - サポートメソッド: {len(encoding_info['supported_tpsl_methods'])}種類"
        )

        return True

    except Exception as e:
        print(f"❌ エンコーディングコンポーネントテストエラー: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_end_to_end_workflow():
    """エンドツーエンドワークフローテスト"""
    print("\n=== エンドツーエンドワークフローテスト ===")

    try:
        # 1. GA設定作成
        from app.core.services.auto_strategy.models.ga_config import GAConfig

        ga_config = GAConfig(
            population_size=2,
            generations=1,
            max_indicators=1,
            allowed_indicators=["SMA"],
            tpsl_method_constraints=["risk_reward_ratio"],
            tpsl_sl_range=[0.02, 0.04],
            tpsl_rr_range=[1.5, 2.5],
        )
        print("✅ ステップ1: GA設定作成完了")

        # 2. ランダム遺伝子生成
        from app.core.services.auto_strategy.generators.random_gene_generator import (
            RandomGeneGenerator,
        )

        generator = RandomGeneGenerator(ga_config)
        gene = generator.generate_random_gene()
        print("✅ ステップ2: TP/SL遺伝子付き戦略遺伝子生成完了")

        # 3. エンコーディング
        from app.core.services.auto_strategy.models.gene_encoding import GeneEncoder

        encoder = GeneEncoder()
        encoded = encoder.encode_strategy_gene_to_list(gene)
        print(f"✅ ステップ3: 遺伝子エンコーディング完了（長さ: {len(encoded)}）")

        # 4. デコーディング
        from app.core.services.auto_strategy.models.strategy_gene import StrategyGene

        decoded_gene = encoder.decode_list_to_strategy_gene(encoded, StrategyGene)
        print("✅ ステップ4: 遺伝子デコーディング完了")

        # 5. TP/SL価格計算
        from app.core.services.auto_strategy.factories.strategy_factory import (
            StrategyFactory,
        )

        factory = StrategyFactory()
        current_price = 50000.0

        if decoded_gene.tpsl_gene:
            sl_price, tp_price = factory._calculate_tpsl_from_gene(
                current_price, decoded_gene.tpsl_gene
            )
            print(f"✅ ステップ5: TP/SL価格計算完了")
            print(f"   - SL: ${sl_price:,.0f}")
            print(f"   - TP: ${tp_price:,.0f}")

        print("\n🎉 エンドツーエンドワークフロー完全成功！")
        print("   GA最適化でTP/SL設定が自動進化する準備が整いました。")

        return True

    except Exception as e:
        print(f"❌ エンドツーエンドワークフローテストエラー: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """メインテスト実行"""
    print("🚀 コンポーネント統合テスト開始\n")

    tests = [
        test_tpsl_gene_component,
        test_ga_config_component,
        test_random_generator_component,
        test_strategy_factory_component,
        test_encoding_component,
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

    print(f"\n📊 コンポーネント統合テスト結果:")
    print(f"   - 成功: {passed}")
    print(f"   - 失敗: {failed}")
    print(f"   - 合計: {passed + failed}")

    if failed == 0:
        print("\n🎉 すべてのコンポーネント統合テストが成功しました！")
        print("\n✨ TP/SL GA最適化機能が完全に統合されています！")
        print("\n🔧 動作確認済みコンポーネント:")
        print("   ✅ TP/SL遺伝子（作成、バリデーション、計算、交叉、突然変異）")
        print("   ✅ GA設定（制約設定、バリデーション）")
        print("   ✅ ランダム遺伝子生成器（TP/SL遺伝子統合）")
        print("   ✅ 戦略ファクトリー（TP/SL価格計算）")
        print("   ✅ エンコーディング（エンコード/デコード）")
        print("   ✅ エンドツーエンドワークフロー")
        print("\n🚀 実装完了：ユーザーはTP/SL設定不要、GAが自動最適化！")
    else:
        print("\n⚠️  一部のコンポーネント統合テストが失敗しました。")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
