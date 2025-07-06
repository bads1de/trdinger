"""
後方互換性修正のテストケース

GENERATED_AUTO戦略タイプでentry_conditionsからlong/short条件への
変換が正しく動作することを確認します。
"""

import pytest
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))


def test_backward_compatibility_with_empty_long_short_conditions():
    """
    空のlong_entry_conditions/short_entry_conditionsが設定されている場合の
    後方互換性をテスト
    """
    try:
        from app.core.services.auto_strategy.models.strategy_gene import (
            StrategyGene,
            IndicatorGene,
            Condition,
        )

        # 問題のあるJSONパラメータを再現
        gene = StrategyGene(
            id="test_gene",
            indicators=[
                IndicatorGene(
                    type="ADX",
                    parameters={"period": 25},
                    enabled=True
                )
            ],
            entry_conditions=[
                Condition(
                    left_operand="ADX",
                    operator="<",
                    right_operand=30
                )
            ],
            exit_conditions=[
                Condition(
                    left_operand="ADX",
                    operator=">",
                    right_operand=70
                )
            ],
            long_entry_conditions=[],  # 明示的に空の配列
            short_entry_conditions=[],  # 明示的に空の配列
            risk_management={"position_size": 0.1},
        )

        # 修正後の動作をテスト
        long_conditions = gene.get_effective_long_conditions()
        short_conditions = gene.get_effective_short_conditions()

        print(f"✅ ロング条件数: {len(long_conditions)}")
        print(f"✅ ショート条件数: {len(short_conditions)}")

        # 期待される結果
        assert len(long_conditions) > 0, "ロング条件が取得できませんでした（後方互換性の問題）"
        assert len(short_conditions) > 0, "ショート条件が取得できませんでした（後方互換性の問題）"
        
        # 条件の内容を確認
        assert long_conditions[0].left_operand == "ADX", "ロング条件の内容が正しくありません"
        assert short_conditions[0].left_operand == "ADX", "ショート条件の内容が正しくありません"

        print("✅ 後方互換性テスト成功: 空のlong/short条件でもentry_conditionsが使用されました")
        return True

    except Exception as e:
        print(f"❌ 後方互換性テストエラー: {e}")
        return False


def test_strategy_factory_condition_check():
    """
    StrategyFactoryの条件チェックロジックをテスト
    """
    try:
        from app.core.services.auto_strategy.models.strategy_gene import (
            StrategyGene,
            IndicatorGene,
            Condition,
        )
        from app.core.services.auto_strategy.factories.strategy_factory import (
            StrategyFactory,
        )

        # テスト用の戦略遺伝子
        gene = StrategyGene(
            id="test_gene",
            indicators=[
                IndicatorGene(
                    type="ADX",
                    parameters={"period": 25},
                    enabled=True
                )
            ],
            entry_conditions=[
                Condition(
                    left_operand="ADX",
                    operator="<",
                    right_operand=30
                )
            ],
            exit_conditions=[
                Condition(
                    left_operand="ADX",
                    operator=">",
                    right_operand=70
                )
            ],
            long_entry_conditions=[],  # 明示的に空の配列
            short_entry_conditions=[],  # 明示的に空の配列
            risk_management={"position_size": 0.1},
        )

        # StrategyFactoryを初期化
        factory = StrategyFactory()
        
        # 戦略クラスを作成
        strategy_class = factory.create_strategy_class(gene)
        
        print("✅ 戦略クラス作成成功")
        print(f"✅ 戦略クラス名: {strategy_class.__name__}")

        # 条件チェックメソッドが存在することを確認
        assert hasattr(strategy_class, '_check_long_entry_conditions'), "ロング条件チェックメソッドが存在しません"
        assert hasattr(strategy_class, '_check_short_entry_conditions'), "ショート条件チェックメソッドが存在しません"

        print("✅ StrategyFactory条件チェックテスト成功")
        return True

    except Exception as e:
        print(f"❌ StrategyFactory条件チェックテストエラー: {e}")
        return False


def test_json_deserialization():
    """
    JSONからの戦略遺伝子デシリアライゼーションをテスト
    """
    try:
        from app.core.services.auto_strategy.models.strategy_gene import StrategyGene

        # 問題のあるJSONデータを再現
        json_data = {
            "id": "",
            "indicators": [
                {
                    "type": "ADX",
                    "parameters": {"period": 25},
                    "enabled": True
                }
            ],
            "entry_conditions": [
                {
                    "left_operand": "ADX",
                    "operator": "<",
                    "right_operand": 30
                }
            ],
            "long_entry_conditions": [],  # 明示的に空の配列
            "short_entry_conditions": [],  # 明示的に空の配列
            "exit_conditions": [
                {
                    "left_operand": "ADX",
                    "operator": ">",
                    "right_operand": 70
                }
            ],
            "risk_management": {"position_size": 0.1},
            "tpsl_gene": {
                "method": "adaptive",
                "stop_loss_pct": 0.0226,
                "take_profit_pct": 0.084,
                "risk_reward_ratio": 2.713,
                "enabled": True
            },
            "metadata": {
                "generated_by": "GeneEncoder_decode",
                "source": "fallback_individual"
            }
        }

        # 辞書からStrategyGeneを復元
        gene = StrategyGene.from_dict(json_data)

        # 修正後の動作をテスト
        long_conditions = gene.get_effective_long_conditions()
        short_conditions = gene.get_effective_short_conditions()

        print(f"✅ デシリアライゼーション後のロング条件数: {len(long_conditions)}")
        print(f"✅ デシリアライゼーション後のショート条件数: {len(short_conditions)}")

        # 期待される結果
        assert len(long_conditions) > 0, "デシリアライゼーション後にロング条件が取得できませんでした"
        assert len(short_conditions) > 0, "デシリアライゼーション後にショート条件が取得できませんでした"

        print("✅ JSONデシリアライゼーションテスト成功")
        return True

    except Exception as e:
        print(f"❌ JSONデシリアライゼーションテストエラー: {e}")
        return False


def run_all_tests():
    """全てのテストを実行"""
    print("🧪 後方互換性修正テストを開始します...\n")

    tests = [
        ("後方互換性テスト", test_backward_compatibility_with_empty_long_short_conditions),
        ("StrategyFactory条件チェックテスト", test_strategy_factory_condition_check),
        ("JSONデシリアライゼーションテスト", test_json_deserialization),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"📋 {test_name}を実行中...")
        try:
            result = test_func()
            results.append((test_name, result))
            print(f"{'✅' if result else '❌'} {test_name}: {'成功' if result else '失敗'}\n")
        except Exception as e:
            print(f"❌ {test_name}: エラー - {e}\n")
            results.append((test_name, False))

    # 結果サマリー
    print("📊 テスト結果サマリー:")
    success_count = sum(1 for _, result in results if result)
    total_count = len(results)
    
    for test_name, result in results:
        print(f"  {'✅' if result else '❌'} {test_name}")
    
    print(f"\n🎯 成功: {success_count}/{total_count}")
    return success_count == total_count


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
