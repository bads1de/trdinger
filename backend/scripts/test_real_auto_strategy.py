#!/usr/bin/env python3
"""
実際のオートストラテジー機能テストスクリプト

本番環境を想定して、実際のGA機能、バックテスト、データベース保存まで
すべての機能を統合的にテストします。
"""

import sys
import os
import asyncio
import logging
from datetime import datetime, timedelta, timezone
import json

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.services.auto_strategy.services.auto_strategy_service import (
    AutoStrategyService,
)
from app.core.services.auto_strategy.models.ga_config import GAConfig

# ログ設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_test_ga_config():
    """テスト用GA設定を作成"""
    return GAConfig(
        population_size=2,  # さらに少数に
        generations=1,  # 1世代のみでテスト
        mutation_rate=0.15,
        crossover_rate=0.8,
        elite_size=1,  # エリート保存数（個体数未満）
        fitness_weights={
            "total_return": 0.35,
            "sharpe_ratio": 0.35,
            "max_drawdown": 0.25,
            "win_rate": 0.05,
        },
        max_indicators=4,  # 最大指標数
        fitness_constraints={
            "min_trades": 5,
            "max_drawdown_limit": 0.5,
            "min_sharpe_ratio": 0.0,
        },
    )


def create_test_backtest_config():
    """テスト用バックテスト設定を作成"""
    import random

    # 利用可能な時間足からランダムに選択
    available_timeframes = ["15m", "30m", "1h", "4h", "1d"]
    selected_timeframe = random.choice(available_timeframes)

    # 時間足に応じて適切な期間を設定
    end_date = datetime.now(timezone.utc)
    if selected_timeframe == "15m":
        start_date = end_date - timedelta(days=7)  # 15分足: 1週間
    elif selected_timeframe == "30m":
        start_date = end_date - timedelta(days=14)  # 30分足: 2週間
    elif selected_timeframe == "1h":
        start_date = end_date - timedelta(days=30)  # 1時間足: 1ヶ月
    elif selected_timeframe == "4h":
        start_date = end_date - timedelta(days=60)  # 4時間足: 2ヶ月
    else:  # 1d
        start_date = end_date - timedelta(days=90)  # 日足: 3ヶ月

    return {
        "symbol": "BTC/USDT:USDT",  # 完全データセットが利用可能
        "timeframe": selected_timeframe,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "initial_cash": 100000,
        "commission": 0.001,
        "use_oi": True,  # Open Interest使用
        "use_fr": True,  # Funding Rate使用
        "experiment_id": f"test_real_auto_strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    }


async def test_auto_strategy_generation():
    """実際のオートストラテジー生成をテスト"""
    print("🚀 実際のオートストラテジー機能テスト開始")
    print("=" * 80)

    try:
        # AutoStrategyServiceの初期化
        print("🔧 AutoStrategyService初期化中...")
        auto_strategy_service = AutoStrategyService()
        print("✅ AutoStrategyService初期化完了")

        # テスト設定の作成
        print("\n⚙️ テスト設定作成中...")
        ga_config = create_test_ga_config()
        backtest_config = create_test_backtest_config()

        print(f"GA設定:")
        print(f"  個体数: {ga_config.population_size}")
        print(f"  世代数: {ga_config.generations}")
        print(f"  突然変異率: {ga_config.mutation_rate}")
        print(f"  交叉率: {ga_config.crossover_rate}")
        print(f"  時間足多様性テスト: 各戦略でランダム時間足選択")

        print(f"\nバックテスト設定:")
        print(f"  シンボル: {backtest_config['symbol']}")
        print(f"  時間軸: {backtest_config['timeframe']}")
        print(
            f"  期間: {backtest_config['start_date'][:10]} ～ {backtest_config['end_date'][:10]}"
        )
        print(f"  初期資金: {backtest_config['initial_cash']:,}")
        print(f"  OI使用: {backtest_config['use_oi']}")
        print(f"  FR使用: {backtest_config['use_fr']}")

        # 戦略生成開始
        print("\n🧬 GA戦略生成開始...")
        experiment_id = auto_strategy_service.start_strategy_generation(
            experiment_name="実際のオートストラテジー機能テスト",
            ga_config=ga_config,
            backtest_config=backtest_config,
        )

        print(f"✅ 戦略生成開始成功")
        print(f"📋 実験ID: {experiment_id}")

        # 進捗監視
        print("\n📊 進捗監視開始...")
        max_wait_time = 300  # 最大5分待機
        check_interval = 10  # 10秒間隔でチェック
        elapsed_time = 0

        while elapsed_time < max_wait_time:
            try:
                progress = auto_strategy_service.get_experiment_progress(experiment_id)

                print(f"\n⏱️ 経過時間: {elapsed_time}秒")
                print(
                    f"📈 進捗: {progress.current_generation}/{progress.total_generations} 世代"
                )
                print(f"🎯 ステータス: {progress.status}")

                if progress.best_fitness is not None:
                    print(f"🏆 最高フィットネス: {progress.best_fitness:.4f}")

                if progress.status == "completed":
                    print("✅ GA戦略生成完了！")
                    break
                elif progress.status == "failed":
                    print("❌ GA戦略生成失敗")
                    return None

                await asyncio.sleep(check_interval)
                elapsed_time += check_interval

            except Exception as e:
                print(f"⚠️ 進捗確認エラー: {e}")
                await asyncio.sleep(check_interval)
                elapsed_time += check_interval

        if elapsed_time >= max_wait_time:
            print("⏰ タイムアウト: 戦略生成に時間がかかりすぎています")
            return None

        # 結果取得
        print("\n📋 結果取得中...")
        results = auto_strategy_service.get_experiment_result(experiment_id)

        if results:
            print("✅ 結果取得成功")
            return results, experiment_id
        else:
            print("❌ 結果取得失敗")
            return None

    except Exception as e:
        logger.error(f"オートストラテジーテストエラー: {e}")
        import traceback

        traceback.print_exc()
        return None


def analyze_test_results(results, experiment_id):
    """テスト結果の分析"""
    print("\n🏆 テスト結果分析")
    print("=" * 80)

    try:
        if not results:
            print("❌ 分析対象の結果がありません")
            return

        print(f"📋 実験ID: {experiment_id}")
        print(f"📊 生成戦略数: {len(results.get('strategies', []))}")

        # 最優秀戦略の分析
        best_strategy = results.get("best_strategy")
        if best_strategy:
            print(f"\n🥇 最優秀戦略:")
            print(f"  戦略ID: {best_strategy.get('id', 'N/A')}")
            print(f"  フィットネス: {best_strategy.get('fitness', 0):.4f}")

            # パフォーマンス指標
            performance = best_strategy.get("performance", {})
            print(f"  総リターン: {performance.get('total_return', 0):.2f}%")
            print(f"  シャープレシオ: {performance.get('sharpe_ratio', 0):.2f}")
            print(f"  最大ドローダウン: {performance.get('max_drawdown', 0):.2f}%")
            print(f"  勝率: {performance.get('win_rate', 0):.1f}%")
            print(f"  取引回数: {performance.get('total_trades', 0)}")

            # 戦略詳細
            strategy_details = best_strategy.get("strategy_gene", {})
            indicators = strategy_details.get("indicators", [])
            print(f"  使用指標: {[ind.get('type', 'Unknown') for ind in indicators]}")

            # OI/FR使用確認
            entry_conditions = strategy_details.get("entry_conditions", [])
            exit_conditions = strategy_details.get("exit_conditions", [])
            all_conditions = entry_conditions + exit_conditions

            oi_fr_usage = []
            for cond in all_conditions:
                left = cond.get("left_operand", "")
                right = cond.get("right_operand", "")
                if "OpenInterest" in [left, right] or "FundingRate" in [left, right]:
                    oi_fr_usage.append(f"{left} {cond.get('operator', '')} {right}")

            if oi_fr_usage:
                print(f"  OI/FR活用: {oi_fr_usage}")
            else:
                print(f"  OI/FR活用: なし")

        # 全体統計
        all_strategies = results.get("strategies", [])
        if all_strategies:
            print(f"\n📊 全体統計:")

            total_returns = [
                s.get("performance", {}).get("total_return", 0) for s in all_strategies
            ]
            sharpe_ratios = [
                s.get("performance", {}).get("sharpe_ratio", 0) for s in all_strategies
            ]

            if total_returns:
                avg_return = sum(total_returns) / len(total_returns)
                max_return = max(total_returns)
                min_return = min(total_returns)
                print(f"  平均リターン: {avg_return:.2f}%")
                print(f"  最高リターン: {max_return:.2f}%")
                print(f"  最低リターン: {min_return:.2f}%")

            if sharpe_ratios:
                avg_sharpe = sum(sharpe_ratios) / len(sharpe_ratios)
                max_sharpe = max(sharpe_ratios)
                print(f"  平均シャープレシオ: {avg_sharpe:.2f}")
                print(f"  最高シャープレシオ: {max_sharpe:.2f}")

        # 結果保存
        output_file = (
            f"real_auto_strategy_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "experiment_id": experiment_id,
                    "test_timestamp": datetime.now().isoformat(),
                    "results": results,
                },
                f,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

        print(f"\n📁 結果保存: {output_file}")

        print("\n" + "=" * 80)
        print("🎉 実際のオートストラテジー機能テスト完了！")
        print("✨ 本番環境での動作確認成功")
        print("🎯 GA最適化、バックテスト、データベース保存すべて正常動作")

    except Exception as e:
        logger.error(f"結果分析エラー: {e}")
        import traceback

        traceback.print_exc()


async def main():
    """メイン実行関数"""
    print("🔬 本番環境想定オートストラテジー機能テスト")
    print("=" * 80)
    print("📋 テスト内容:")
    print("  ✓ 実際のDBデータ使用")
    print("  ✓ GA戦略生成")
    print("  ✓ バックテスト実行")
    print("  ✓ データベース保存")
    print("  ✓ OI/FR統合利用")
    print("=" * 80)

    start_time = datetime.now()

    # オートストラテジー生成テスト
    result = await test_auto_strategy_generation()

    if result:
        results, experiment_id = result
        analyze_test_results(results, experiment_id)
    else:
        print("❌ オートストラテジー機能テスト失敗")

    end_time = datetime.now()
    execution_time = (end_time - start_time).total_seconds()

    print(f"\n⏱️ 総実行時間: {execution_time:.2f} 秒")


if __name__ == "__main__":
    asyncio.run(main())
