#!/usr/bin/env python3
"""
自動戦略生成機能の直接テスト（APIを使わずに）
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))


def test_direct_ga():
    """GA機能を直接テスト"""
    print("🧬 自動戦略生成機能 直接テスト")
    print("=" * 60)

    try:
        # 1. 必要なモジュールのインポート
        print("1. モジュールインポート中...")
        from app.core.services.auto_strategy.models.strategy_gene import (
            StrategyGene,
            IndicatorGene,
            Condition,
        )
        from app.core.services.auto_strategy.models.ga_config import GAConfig
        from app.core.services.auto_strategy.services.auto_strategy_service import (
            AutoStrategyService,
        )

        print("  ✅ インポート完了")

        # 2. GA設定作成
        print("\n2. GA設定作成中...")
        ga_config = GAConfig(
            population_size=3,  # 非常に小さなテスト
            generations=2,
            crossover_rate=0.8,
            mutation_rate=0.2,
            elite_size=1,
            max_indicators=2,
            allowed_indicators=["SMA", "RSI"],
        )
        print(
            f"  ✅ GA設定: 個体数{ga_config.population_size}, 世代数{ga_config.generations}"
        )

        # 3. バックテスト設定
        print("\n3. バックテスト設定作成中...")
        backtest_config = {
            "symbol": "BTC/USDT",
            "timeframe": "1d",
            "start_date": "2024-01-01",
            "end_date": "2024-04-09",
            "initial_capital": 100000,
            "commission_rate": 0.001,
        }
        print(
            f"  ✅ バックテスト設定: {backtest_config['symbol']} {backtest_config['timeframe']}"
        )

        # 4. AutoStrategyServiceの初期化
        print("\n4. AutoStrategyService初期化中...")
        service = AutoStrategyService()
        print("  ✅ サービス初期化完了")

        # 5. 戦略生成開始
        print("\n5. 戦略生成開始...")
        from fastapi import BackgroundTasks

        background_tasks = BackgroundTasks()

        experiment_id = service.start_strategy_generation(
            experiment_name="Direct_Test_Daily_BTC",
            ga_config_dict=ga_config.to_dict(),
            backtest_config_dict=backtest_config,
            background_tasks=background_tasks,
        )
        print(f"  ✅ 実験開始: {experiment_id}")

        # 6. 進捗監視
        print("\n6. 進捗監視中...")
        import time

        max_wait = 60  # 最大1分待機
        start_time = time.time()

        while time.time() - start_time < max_wait:
            progress = service.get_experiment_progress(experiment_id)
            if progress:
                print(
                    f"  世代: {progress.current_generation}/{progress.total_generations}"
                )
                print(f"  最高フィットネス: {progress.best_fitness:.4f}")
                print(f"  ステータス: {progress.status}")

                if progress.status == "completed":
                    print("  🎉 実験完了!")
                    break
                elif progress.status == "error":
                    print(f"  ❌ エラー: {progress.error_message}")
                    return False

            time.sleep(2)

        # 7. 結果取得
        print("\n7. 結果取得中...")
        result = service.get_experiment_result(experiment_id)
        if result:
            print("  ✅ 結果取得成功")
            print(f"  最高フィットネス: {result['best_fitness']:.4f}")
            print(f"  実行時間: {result['execution_time']:.2f}秒")
            print(f"  完了世代数: {result['generations_completed']}")

            # 最優秀戦略の詳細
            best_strategy = result["best_strategy"]
            print(f"\n🏆 最優秀戦略:")
            print(f"  指標数: {len(best_strategy.indicators)}")
            for i, indicator in enumerate(best_strategy.indicators, 1):
                print(f"    {i}. {indicator.type} - {indicator.parameters}")

            print(f"  エントリー条件数: {len(best_strategy.entry_conditions)}")
            for i, condition in enumerate(best_strategy.entry_conditions, 1):
                print(f"    {i}. {condition}")

            print(f"  エグジット条件数: {len(best_strategy.exit_conditions)}")
            for i, condition in enumerate(best_strategy.exit_conditions, 1):
                print(f"    {i}. {condition}")
        else:
            print("  ⚠️ 結果がまだ利用できません")

        print("\n✅ 直接テスト完了")
        return True

    except Exception as e:
        print(f"\n❌ テストエラー: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_direct_ga()
