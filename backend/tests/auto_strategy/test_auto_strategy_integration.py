"""
オートストラテジー統合テスト

実際のオートストラテジーサービスでロング・ショート戦略が正しく生成・実行されるかをテストします。
"""

import sys
import os
import json

# パス設定
sys.path.append(os.path.join(os.path.dirname(__file__), "app"))


def test_auto_strategy_service():
    """オートストラテジーサービスのテスト"""
    print("=== オートストラテジーサービス統合テスト ===")

    try:
        from app.core.services.auto_strategy.generators.random_gene_generator import (
            RandomGeneGenerator,
        )
        from app.core.services.auto_strategy.models.ga_config import GAConfig

        # GAConfigを作成
        config = GAConfig()
        generator = RandomGeneGenerator(config)

        # ランダム戦略を生成
        print("ランダム戦略を生成中...")
        gene = generator.generate_random_gene()

        # ロング・ショート条件の確認
        has_long = len(gene.long_entry_conditions) > 0
        has_short = len(gene.short_entry_conditions) > 0
        has_legacy = len(gene.entry_conditions) > 0

        print(f"✅ 生成された戦略:")
        print(f"   - 指標数: {len(gene.indicators)}")
        print(f"   - ロング条件数: {len(gene.long_entry_conditions)}")
        print(f"   - ショート条件数: {len(gene.short_entry_conditions)}")
        print(f"   - 従来エントリー条件数: {len(gene.entry_conditions)}")
        print(f"   - エグジット条件数: {len(gene.exit_conditions)}")

        # 戦略をJSONに変換
        strategy_json = gene.to_json()
        strategy_dict = json.loads(strategy_json)

        print(f"\n✅ JSON変換成功:")
        print(f"   - JSONサイズ: {len(strategy_json)} 文字")
        print(
            f"   - long_entry_conditions存在: {'long_entry_conditions' in strategy_dict}"
        )
        print(
            f"   - short_entry_conditions存在: {'short_entry_conditions' in strategy_dict}"
        )

        # ロング・ショート条件の詳細表示
        if has_long:
            print(f"\n📈 ロング条件:")
            for i, cond in enumerate(gene.long_entry_conditions):
                print(
                    f"   {i+1}. {cond.left_operand} {cond.operator} {cond.right_operand}"
                )

        if has_short:
            print(f"\n📉 ショート条件:")
            for i, cond in enumerate(gene.short_entry_conditions):
                print(
                    f"   {i+1}. {cond.left_operand} {cond.operator} {cond.right_operand}"
                )

        # 戦略の妥当性チェック
        is_valid, errors = gene.validate()
        print(f"\n✅ 戦略妥当性: {'有効' if is_valid else '無効'}")
        if errors:
            print(f"   エラー: {errors}")

        assert is_valid, f"生成された戦略が無効です: {errors}"
        assert has_long or has_short, "ロング・ショート条件のいずれかが必要です"

        print("\n🎉 オートストラテジーサービス統合テスト成功！")
        return True

    except Exception as e:
        print(f"❌ オートストラテジーサービステスト失敗: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_strategy_factory_integration():
    """StrategyFactoryとの統合テスト"""
    print("\n=== StrategyFactory統合テスト ===")

    try:
        from app.core.services.auto_strategy.generators.random_gene_generator import (
            RandomGeneGenerator,
        )
        from app.core.services.auto_strategy.factories.strategy_factory import (
            StrategyFactory,
        )
        from app.core.services.auto_strategy.models.ga_config import GAConfig

        # 戦略を生成
        config = GAConfig()
        generator = RandomGeneGenerator(config)
        gene = generator.generate_random_gene()

        # StrategyFactoryで戦略クラスを生成
        factory = StrategyFactory()
        strategy_class = factory.create_strategy_class(gene)

        print(f"✅ 戦略クラス生成成功: {strategy_class.__name__}")

        # 戦略クラスのメソッド確認
        required_methods = [
            "_check_long_entry_conditions",
            "_check_short_entry_conditions",
            "_check_entry_conditions",
            "_check_exit_conditions",
        ]

        for method_name in required_methods:
            assert hasattr(
                strategy_class, method_name
            ), f"必要なメソッド {method_name} が存在しません"
            print(f"   ✅ {method_name} メソッド存在")

        print("\n🎉 StrategyFactory統合テスト成功！")
        return True

    except Exception as e:
        print(f"❌ StrategyFactory統合テスト失敗: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_json_structure():
    """JSON構造テスト"""
    print("\n=== JSON構造テスト ===")

    try:
        from app.core.services.auto_strategy.generators.random_gene_generator import (
            RandomGeneGenerator,
        )
        from app.core.services.auto_strategy.models.ga_config import GAConfig
        import json

        # 戦略を生成
        config = GAConfig()
        generator = RandomGeneGenerator(config)
        gene = generator.generate_random_gene()

        # JSONに変換
        strategy_json = gene.to_json()
        strategy_dict = json.loads(strategy_json)

        # 必要なフィールドの確認
        required_fields = [
            "indicators",
            "entry_conditions",
            "long_entry_conditions",
            "short_entry_conditions",
            "exit_conditions",
            "risk_management",
        ]

        print("✅ JSON構造確認:")
        for field in required_fields:
            exists = field in strategy_dict
            print(f"   - {field}: {'存在' if exists else '不存在'}")
            if field in ["long_entry_conditions", "short_entry_conditions"]:
                # 新しいフィールドは存在することを確認
                assert exists, f"新しいフィールド {field} が存在しません"

        # JSONサイズの確認
        json_size = len(strategy_json)
        print(f"\n✅ JSONサイズ: {json_size} 文字")

        # 構造の詳細表示（サンプル）
        print(f"\n📋 JSON構造サンプル:")
        print(f"   - indicators: {len(strategy_dict.get('indicators', []))} 個")
        print(
            f"   - long_entry_conditions: {len(strategy_dict.get('long_entry_conditions', []))} 個"
        )
        print(
            f"   - short_entry_conditions: {len(strategy_dict.get('short_entry_conditions', []))} 個"
        )
        print(
            f"   - exit_conditions: {len(strategy_dict.get('exit_conditions', []))} 個"
        )

        print("\n🎉 JSON構造テスト成功！")
        return True, json_size

    except Exception as e:
        print(f"❌ JSON構造テスト失敗: {e}")
        import traceback

        traceback.print_exc()
        return False, 0


def main():
    """メインテスト実行"""
    print("🚀 オートストラテジー統合テスト開始\n")

    tests = [
        test_auto_strategy_service,
        test_strategy_factory_integration,
        test_json_structure,
    ]

    passed = 0
    total = len(tests)
    json_size = 0

    for test in tests:
        try:
            if test == test_json_structure:
                result, size = test()
                if result:
                    passed += 1
                    json_size = size
            else:
                if test():
                    passed += 1
        except Exception as e:
            print(f"❌ テスト実行エラー: {e}")

    print(f"\n📊 統合テスト結果: {passed}/{total} 成功")

    if passed == total:
        print("🎉 全ての統合テストが成功しました！")
        print("\n🎯 確認済み機能:")
        print("✅ オートストラテジーサービスでロング・ショート戦略生成")
        print("✅ StrategyFactoryでロング・ショート戦略クラス生成")
        print("✅ JSON構造にロング・ショート条件が含まれる")
        print(f"✅ JSONサイズ: {json_size} 文字")

        if json_size > 1000:
            print(
                f"\n⚠️ JSONサイズが大きいため({json_size}文字)、フロントエンドで折りたたみ表示の実装を推奨"
            )

        return True
    else:
        print("❌ 一部の統合テストが失敗しました")
        return False


if __name__ == "__main__":
    main()
