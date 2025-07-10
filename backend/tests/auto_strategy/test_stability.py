"""
長期安定性テスト（簡略版）

SmartConditionGeneratorの基本安定性確認
"""

import time
import os
import sys
import random

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from app.core.services.auto_strategy.generators.smart_condition_generator import SmartConditionGenerator
from app.core.services.auto_strategy.models.gene_strategy import IndicatorGene


def test_basic_stability():
    """基本安定性テスト（簡略版）"""
    print("🚀 SmartConditionGenerator 基本安定性テスト")
    print("="*50)

    generator = SmartConditionGenerator(enable_smart_generation=True)

    test_indicators = [
        IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
        IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)
    ]

    # 連続実行テスト（簡略版）
    print("\n=== 連続実行テスト (100回) ===")
    success_count = 0
    error_count = 0

    for i in range(100):
        try:
            long_conds, short_conds, exit_conds = generator.generate_balanced_conditions(test_indicators)

            if len(long_conds) > 0 and len(short_conds) > 0:
                success_count += 1
            else:
                error_count += 1

        except Exception as e:
            error_count += 1
            print(f"   エラー {i+1}: {e}")

    success_rate = (success_count / 100) * 100

    print(f"\n📊 結果:")
    print(f"   成功: {success_count}/100")
    print(f"   エラー: {error_count}/100")
    print(f"   成功率: {success_rate:.1f}%")

    # エラー復旧テスト
    print("\n=== エラー復旧テスト ===")

    # 空の指標リスト
    try:
        long_conds, short_conds, exit_conds = generator.generate_balanced_conditions([])
        if len(long_conds) > 0 and len(short_conds) > 0:
            print("   ✅ 空リスト復旧: 成功")
            recovery_ok = True
        else:
            print("   ❌ 空リスト復旧: 失敗")
            recovery_ok = False
    except Exception as e:
        print(f"   ❌ 空リスト復旧: エラー {e}")
        recovery_ok = False

    # 総合判定
    overall_ok = success_rate >= 95 and recovery_ok

    print(f"\n🎯 総合判定:")
    if overall_ok:
        print("   ✅ 安定性: 良好 - 本格運用可能")
    else:
        print("   ⚠️  安定性: 要注意")

    return overall_ok


if __name__ == "__main__":
    success = test_basic_stability()

    if success:
        print("\n🎉 安定性テストが成功しました！")
        exit(0)
    else:
        print("\n💥 安定性テストで問題が発見されました。")
        exit(1)