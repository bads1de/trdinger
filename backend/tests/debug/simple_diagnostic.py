#!/usr/bin/env python3
"""
シンプルロング・ショートバランス診断

循環インポートを避けたシンプルな診断スクリプト
"""

import sys
import os
import logging

# パス設定
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# 必要な依存を最小限に
def test_condition_generator_simple():
    """簡易ConditionGeneratorテスト"""
    print("簡易ロング・ショートバランス診断開始")

    # ランタイムにインポートして循環回避
    try:
        # 部分的インポート
        from app.services.auto_strategy.config.constants import IndicatorType
        from app.services.auto_strategy.models.strategy_models import Condition

        print("モデルインポート成功")
    except Exception as e:
        print(f"モデルインポート失敗: {e}")
        return

    # ConditionGeneratorのインポートテスト
    try:
        import importlib.util
        import os

        # ConditionGeneratorを動的インポート
        spec = importlib.util.spec_from_file_location(
            "condition_generator",
            os.path.join(os.path.dirname(__file__), '..', '..', 'app', 'services', 'auto_strategy', 'generators', 'condition_generator.py')
        )

        condition_generator_module = importlib.util.module_from_spec(spec)

        # 必要なモジュールを事前に準備
        sys.modules['app.services.auto_strategy.config.constants'] = importlib.import_module('app.services.auto_strategy.config.constants')
        sys.modules['app.services.auto_strategy.models.strategy_models'] = importlib.import_module('app.services.auto_strategy.models.strategy_models')

        spec.loader.exec_module(condition_generator_module)
        print("ConditionGenerator動的インポート成功")

        # クラス取得
        ConditionGenerator = condition_generator_module.ConditionGenerator

        # IndicatorGene もインポート
        sys.modules['app.services.auto_strategy.utils.common_utils'] = importlib.import_module('app.services.auto_strategy.utils.common_utils')

        from app.services.auto_strategy.models.strategy_models import IndicatorGene

        # 診断テスト実行
        generator = ConditionGenerator(enable_smart_generation=True)

        print("\n=== 診断テスト ===")

        # 統計指標テスト
        statistics_indicators = [
            IndicatorGene(type="RSI", enabled=True),
            IndicatorGene(type="STOCH", enabled=True),
        ]

        print("統計指標テスト")
        long_conds, short_conds, exit_conds = generator.generate_balanced_conditions(statistics_indicators)

        print(f"統計指標 - ロング条件数: {len(long_conds)}, ショート条件数: {len(short_conds)}")

        if len(short_conds) == 0:
            print("⚠️ 問題検出: 統計指標でショート条件が生成されていない!")

        # パターン指標テスト
        pattern_indicators = [
            IndicatorGene(type="CDL_HAMMER", enabled=True),
            IndicatorGene(type="CDL_ENGULFING", enabled=True),
        ]

        print("\nパターン指標テスト")
        long_conds, short_conds, exit_conds = generator.generate_balanced_conditions(pattern_indicators)

        print(f"パターン指標 - ロング条件数: {len(long_conds)}, ショート条件数: {len(short_conds)}")

        if len(short_conds) == 0:
            print("⚠️ 問題検出: パターン指標でショート条件が生成されていない!")

        # トレンド指標テスト (比較)
        trend_indicators = [
            IndicatorGene(type="SMA", enabled=True),
            IndicatorGene(type="EMA", enabled=True),
        ]

        print("\nトレンド指標テスト")
        long_conds, short_conds, exit_conds = generator.generate_balanced_conditions(trend_indicators)

        print(f"トレンド指標 - ロング条件数: {len(long_conds)}, ショート条件数: {len(short_conds)}")

        if len(short_conds) > 0:
            print("✅ トレンド指標はショート条件が生成されている")
        else:
            print("⚠️ トレンド指標でもショート条件が生成されていない")

    except Exception as e:
        print(f"テスト実行失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_condition_generator_simple()