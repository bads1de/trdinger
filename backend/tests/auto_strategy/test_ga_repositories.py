"""
GA関連リポジトリのテスト

GAExperimentRepositoryとGeneratedStrategyRepositoryの基本機能をテストします。
"""

import sys
import os
import pytest
from datetime import datetime

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connection import SessionLocal
from database.repositories.ga_experiment_repository import GAExperimentRepository
from database.repositories.generated_strategy_repository import (
    GeneratedStrategyRepository,
)


def test_ga_experiment_repository():
    """GAExperimentRepositoryのテスト"""
    print("\n=== GAExperimentRepository テスト開始 ===")

    db = SessionLocal()
    try:
        repo = GAExperimentRepository(db)

        # 1. 実験作成テスト
        print("1. 実験作成テスト")
        experiment = repo.create_experiment(
            name="テスト実験1",
            config={
                "population_size": 50,
                "generations": 10,
                "crossover_rate": 0.8,
                "mutation_rate": 0.1,
            },
            total_generations=10,
        )

        assert experiment.id is not None
        assert experiment.name == "テスト実験1"
        assert experiment.status == "running"
        print(f"✅ 実験作成成功: ID={experiment.id}")

        # 2. 進捗更新テスト
        print("2. 進捗更新テスト")
        success = repo.update_experiment_progress(
            experiment_id=experiment.id,
            current_generation=5,
            progress=0.5,
            best_fitness=0.75,
        )

        assert success == True

        # 更新確認
        updated_experiment = repo.get_experiment_by_id(experiment.id)
        assert updated_experiment.current_generation == 5
        assert updated_experiment.progress == 0.5
        assert updated_experiment.best_fitness == 0.75
        print("✅ 進捗更新成功")

        # 3. ステータス更新テスト
        print("3. ステータス更新テスト")
        success = repo.update_experiment_status(
            experiment_id=experiment.id, status="completed"
        )

        assert success == True

        # 更新確認
        completed_experiment = repo.get_experiment_by_id(experiment.id)
        assert completed_experiment.status == "completed"
        assert completed_experiment.completed_at is not None
        print("✅ ステータス更新成功")

        # 4. 実験完了テスト
        print("4. 実験完了テスト")
        experiment2 = repo.create_experiment(
            name="テスト実験2", config={"test": "config"}, total_generations=20
        )

        success = repo.complete_experiment(
            experiment_id=experiment2.id, best_fitness=0.95, final_generation=20
        )

        assert success == True

        # 完了確認
        final_experiment = repo.get_experiment_by_id(experiment2.id)
        assert final_experiment.status == "completed"
        assert final_experiment.best_fitness == 0.95
        assert final_experiment.current_generation == 20
        assert final_experiment.progress == 1.0
        print("✅ 実験完了成功")

        # 5. 統計情報テスト
        print("5. 統計情報テスト")
        stats = repo.get_experiment_statistics()

        assert stats["total_experiments"] >= 2
        assert stats["completed_experiments"] >= 2
        assert stats["best_fitness"] == 0.95
        print(f"✅ 統計情報取得成功: {stats}")

        print("=== GAExperimentRepository テスト完了 ===")
        return experiment.id, experiment2.id

    finally:
        db.close()


def test_generated_strategy_repository(experiment_ids):
    """GeneratedStrategyRepositoryのテスト"""
    print("\n=== GeneratedStrategyRepository テスト開始 ===")

    db = SessionLocal()
    try:
        repo = GeneratedStrategyRepository(db)
        experiment_id = experiment_ids[0]

        # 1. 戦略保存テスト
        print("1. 戦略保存テスト")
        strategy = repo.save_strategy(
            experiment_id=experiment_id,
            gene_data={
                "id": "test_strategy_1",
                "indicators": [
                    {"type": "SMA", "parameters": {"period": 20}},
                    {"type": "RSI", "parameters": {"period": 14}},
                ],
                "entry_conditions": [],
                "exit_conditions": [],
            },
            generation=1,
            fitness_score=0.85,
        )

        assert strategy.id is not None
        assert strategy.experiment_id == experiment_id
        assert strategy.fitness_score == 0.85
        print(f"✅ 戦略保存成功: ID={strategy.id}")

        # 2. 一括保存テスト
        print("2. 戦略一括保存テスト")
        strategies_data = []
        for i in range(3):
            strategies_data.append(
                {
                    "experiment_id": experiment_id,
                    "gene_data": {
                        "id": f"test_strategy_{i+2}",
                        "indicators": [
                            {"type": "EMA", "parameters": {"period": 10 + i}}
                        ],
                        "entry_conditions": [],
                        "exit_conditions": [],
                        "risk_management": {},
                        "metadata": {},
                    },
                    "generation": 2,
                    "fitness_score": 0.7 + i * 0.05,
                }
            )

        saved_strategies = repo.save_strategies_batch(strategies_data)
        assert len(saved_strategies) == 3
        print(f"✅ 一括保存成功: {len(saved_strategies)} 件")

        # 3. 実験別戦略取得テスト
        print("3. 実験別戦略取得テスト")
        experiment_strategies = repo.get_strategies_by_experiment(experiment_id)
        assert len(experiment_strategies) >= 4  # 1個 + 3個
        print(f"✅ 実験別戦略取得成功: {len(experiment_strategies)} 件")

        # 4. 最高戦略取得テスト
        print("4. 最高戦略取得テスト")
        best_strategies = repo.get_best_strategies(experiment_id=experiment_id, limit=2)
        assert len(best_strategies) >= 2
        assert best_strategies[0].fitness_score >= best_strategies[1].fitness_score
        print(
            f"✅ 最高戦略取得成功: 最高フィットネス={best_strategies[0].fitness_score}"
        )

        # 5. 世代統計テスト
        print("5. 世代統計テスト")
        gen_stats = repo.get_generation_statistics(experiment_id, 2)
        assert gen_stats["strategy_count"] == 3
        assert "best_fitness" in gen_stats
        assert "average_fitness" in gen_stats
        print(f"✅ 世代統計取得成功: {gen_stats}")

        print("=== GeneratedStrategyRepository テスト完了 ===")

    finally:
        db.close()


def main():
    """メインテスト実行"""
    try:
        print("🧪 GA関連リポジトリテスト開始")

        # GAExperimentRepositoryテスト
        experiment_ids = test_ga_experiment_repository()

        # GeneratedStrategyRepositoryテスト
        test_generated_strategy_repository(experiment_ids)

        print("\n🎉 全テスト完了！")
        return True

    except Exception as e:
        print(f"\n❌ テストエラー: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
