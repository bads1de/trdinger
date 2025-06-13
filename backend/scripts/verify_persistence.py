"""
永続化処理の最終確認スクリプト

データベースに保存されたGA実験と戦略データを確認します。
"""

import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connection import SessionLocal
from database.repositories.ga_experiment_repository import GAExperimentRepository
from database.repositories.generated_strategy_repository import (
    GeneratedStrategyRepository,
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def verify_ga_experiments():
    """GA実験データの確認"""
    print("\n=== GA実験データ確認 ===")

    db = SessionLocal()
    try:
        repo = GAExperimentRepository(db)

        # 全実験を取得
        experiments = repo.get_recent_experiments(limit=10)
        print(f"総実験数: {len(experiments)}")

        for exp in experiments:
            print(f"\n実験ID: {exp.id}")
            print(f"  名前: {exp.name}")
            print(f"  ステータス: {exp.status}")
            print(f"  進捗: {exp.progress:.2%}")
            print(f"  現在世代: {exp.current_generation}/{exp.total_generations}")
            print(f"  最高フィットネス: {exp.best_fitness}")
            print(f"  作成日時: {exp.created_at}")
            print(f"  完了日時: {exp.completed_at}")

            # 設定の一部を表示
            config = exp.config
            if config:
                print(f"  個体数: {config.get('population_size', 'N/A')}")
                print(f"  世代数: {config.get('generations', 'N/A')}")

        # 統計情報
        stats = repo.get_experiment_statistics()
        print("\n統計情報:")
        print(f"  総実験数: {stats.get('total_experiments', 0)}")
        print(f"  実行中: {stats.get('running_experiments', 0)}")
        print(f"  完了: {stats.get('completed_experiments', 0)}")
        print(f"  エラー: {stats.get('error_experiments', 0)}")
        print(f"  最高フィットネス: {stats.get('best_fitness', 'N/A')}")

    finally:
        db.close()


def verify_generated_strategies():
    """生成戦略データの確認"""
    print("\n=== 生成戦略データ確認 ===")

    db = SessionLocal()
    try:
        repo = GeneratedStrategyRepository(db)

        # 最高戦略を取得
        best_strategies = repo.get_best_strategies(limit=5)
        print(f"最高戦略数: {len(best_strategies)}")

        for i, strategy in enumerate(best_strategies, 1):
            print(f"\n戦略{i}:")
            print(f"  ID: {strategy.id}")
            print(f"  実験ID: {strategy.experiment_id}")
            print(f"  世代: {strategy.generation}")
            print(f"  フィットネス: {strategy.fitness_score}")
            print(f"  作成日時: {strategy.created_at}")

            # 遺伝子データの一部を表示
            gene_data = strategy.gene_data
            if gene_data:
                print(f"  戦略ID: {gene_data.get('id', 'N/A')}")
                indicators = gene_data.get("indicators", [])
                print(f"  指標数: {len(indicators)}")
                if indicators:
                    print(f"  指標例: {indicators[0].get('type', 'N/A')}")

        # 実験別の戦略数を確認
        db_experiments = SessionLocal()
        try:
            exp_repo = GAExperimentRepository(db_experiments)
            experiments = exp_repo.get_recent_experiments(limit=5)

            print(f"\n実験別戦略数:")
            for exp in experiments:
                strategies = repo.get_strategies_by_experiment(exp.id)
                print(f"  実験{exp.id} ({exp.name}): {len(strategies)} 戦略")

                if strategies:
                    # 世代統計
                    for gen in range(1, exp.total_generations + 1):
                        gen_strategies = repo.get_strategies_by_generation(exp.id, gen)
                        if gen_strategies:
                            gen_stats = repo.get_generation_statistics(exp.id, gen)
                            print(
                                f"    世代{gen}: {len(gen_strategies)} 戦略, "
                                f"最高: {gen_stats.get('best_fitness', 'N/A'):.4f}"
                            )
        finally:
            db_experiments.close()

    finally:
        db.close()


def verify_data_integrity():
    """データ整合性の確認"""
    print("\n=== データ整合性確認 ===")

    db = SessionLocal()
    try:
        exp_repo = GAExperimentRepository(db)
        strategy_repo = GeneratedStrategyRepository(db)

        experiments = exp_repo.get_recent_experiments(limit=10)

        for exp in experiments:
            strategies = strategy_repo.get_strategies_by_experiment(exp.id)

            print(f"\n実験{exp.id} ({exp.name}):")
            print(f"  ステータス: {exp.status}")
            print(f"  戦略数: {len(strategies)}")

            # 完了した実験には戦略が存在するはず
            if exp.status == "completed" and len(strategies) == 0:
                print(f"  ⚠️ 警告: 完了した実験に戦略がありません")
            elif exp.status == "completed" and len(strategies) > 0:
                print(f"  ✅ 正常: 完了した実験に戦略が存在します")

            # フィットネススコアの整合性
            if strategies:
                max_fitness = max(
                    s.fitness_score for s in strategies if s.fitness_score is not None
                )
                if exp.best_fitness and abs(exp.best_fitness - max_fitness) > 0.0001:
                    print(
                        f"  ⚠️ 警告: 実験の最高フィットネス({exp.best_fitness})と"
                        f"戦略の最高フィットネス({max_fitness})が一致しません"
                    )
                else:
                    print(f"  ✅ 正常: フィットネススコアが整合しています")

    finally:
        db.close()


def main():
    """メイン処理"""
    try:
        print("🔍 DB永続化処理の最終確認")

        # GA実験データ確認
        verify_ga_experiments()

        # 生成戦略データ確認
        verify_generated_strategies()

        # データ整合性確認
        verify_data_integrity()

        print("\n🎉 永続化処理確認完了！")
        return True

    except Exception as e:
        print(f"\n❌ 確認エラー: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
