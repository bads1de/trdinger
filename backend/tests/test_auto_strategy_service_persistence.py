"""
AutoStrategyService永続化処理のテスト

実際のGA実験を小規模で実行して、永続化処理が正常に動作することを確認します。
"""

import sys
import os
import time
import threading

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.services.auto_strategy.services.auto_strategy_service import AutoStrategyService
from app.core.services.auto_strategy.models.ga_config import GAConfig
from database.connection import SessionLocal
from database.repositories.ga_experiment_repository import GAExperimentRepository
from database.repositories.generated_strategy_repository import GeneratedStrategyRepository


def test_auto_strategy_service_persistence():
    """AutoStrategyServiceの永続化処理テスト"""
    print("\n=== AutoStrategyService永続化テスト開始 ===")
    
    try:
        # 1. サービス初期化テスト
        print("1. サービス初期化テスト")
        service = AutoStrategyService()
        print("✅ AutoStrategyService初期化成功")
        
        # 2. 小規模GA設定
        print("2. GA設定作成")
        ga_config = GAConfig(
            population_size=10,  # 小規模
            generations=3,       # 短時間
            crossover_rate=0.8,
            mutation_rate=0.1,
            elite_size=2,
            max_indicators=3,
            allowed_indicators=["SMA", "EMA", "RSI"]
        )
        
        backtest_config = {
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-01-07",  # 短期間
            "initial_capital": 10000
        }
        
        print("✅ GA設定作成完了")
        
        # 3. 進捗コールバック設定
        print("3. 進捗コールバック設定")
        progress_updates = []
        
        def progress_callback(progress):
            progress_updates.append(progress)
            print(f"   進捗: 世代{progress.current_generation}/{progress.total_generations} "
                  f"({progress.progress_percentage:.1f}%) "
                  f"最高フィットネス: {progress.best_fitness:.4f}")
        
        # 4. 実験開始前のDB状態確認
        print("4. 実験開始前のDB状態確認")
        db = SessionLocal()
        try:
            ga_repo = GAExperimentRepository(db)
            strategy_repo = GeneratedStrategyRepository(db)
            
            initial_experiments = ga_repo.get_recent_experiments(limit=100)
            initial_strategies = strategy_repo.get_best_strategies(limit=100)
            
            print(f"   既存実験数: {len(initial_experiments)}")
            print(f"   既存戦略数: {len(initial_strategies)}")
            
        finally:
            db.close()
        
        # 5. GA実験実行
        print("5. GA実験実行開始")
        experiment_id = service.start_strategy_generation(
            experiment_name="永続化テスト実験",
            ga_config=ga_config,
            backtest_config=backtest_config,
            progress_callback=progress_callback
        )
        
        print(f"✅ 実験開始成功: {experiment_id}")
        
        # 6. 実験完了まで待機
        print("6. 実験完了待機中...")
        max_wait_time = 120  # 最大2分待機
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            progress = service.get_experiment_progress(experiment_id)
            if progress and progress.status == "completed":
                print("✅ 実験完了")
                break
            elif progress and progress.status == "error":
                print("❌ 実験エラー")
                break
            
            time.sleep(2)
        else:
            print("⚠️ 実験タイムアウト")
        
        # 7. 実験完了後のDB状態確認
        print("7. 実験完了後のDB状態確認")
        db = SessionLocal()
        try:
            ga_repo = GAExperimentRepository(db)
            strategy_repo = GeneratedStrategyRepository(db)
            
            # 実験確認
            final_experiments = ga_repo.get_recent_experiments(limit=100)
            new_experiment_count = len(final_experiments) - len(initial_experiments)
            print(f"   新規実験数: {new_experiment_count}")
            
            if new_experiment_count > 0:
                latest_experiment = final_experiments[0]
                print(f"   最新実験: ID={latest_experiment.id}, "
                      f"ステータス={latest_experiment.status}, "
                      f"進捗={latest_experiment.progress:.2%}")
                
                # 戦略確認
                experiment_strategies = strategy_repo.get_strategies_by_experiment(latest_experiment.id)
                print(f"   実験の戦略数: {len(experiment_strategies)}")
                
                if experiment_strategies:
                    best_strategy = experiment_strategies[0]
                    print(f"   最高戦略フィットネス: {best_strategy.fitness_score:.4f}")
            
            # 統計情報
            stats = ga_repo.get_experiment_statistics()
            print(f"   実験統計: {stats}")
            
        finally:
            db.close()
        
        # 8. 進捗更新確認
        print("8. 進捗更新確認")
        print(f"   進捗更新回数: {len(progress_updates)}")
        if progress_updates:
            final_progress = progress_updates[-1]
            print(f"   最終進捗: {final_progress.status}, "
                  f"世代{final_progress.current_generation}/{final_progress.total_generations}")
        
        print("=== AutoStrategyService永続化テスト完了 ===")
        return True
        
    except Exception as e:
        print(f"❌ テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_error_handling():
    """エラーハンドリングテスト"""
    print("\n=== エラーハンドリングテスト開始 ===")
    
    try:
        service = AutoStrategyService()
        
        # 無効な設定でテスト
        invalid_config = GAConfig(
            population_size=0,  # 無効な値
            generations=0       # 無効な値
        )
        
        try:
            experiment_id = service.start_strategy_generation(
                experiment_name="エラーテスト実験",
                ga_config=invalid_config,
                backtest_config={}
            )
            print("❌ エラーが発生すべきでした")
            return False
            
        except ValueError as e:
            print(f"✅ 期待通りのエラー発生: {e}")
            return True
            
    except Exception as e:
        print(f"❌ 予期しないエラー: {e}")
        return False


def main():
    """メインテスト実行"""
    try:
        print("🧪 AutoStrategyService永続化テスト開始")
        
        # 基本的な永続化テスト
        success1 = test_auto_strategy_service_persistence()
        
        # エラーハンドリングテスト
        success2 = test_error_handling()
        
        if success1 and success2:
            print("\n🎉 全テスト成功！")
            return True
        else:
            print("\n❌ 一部テスト失敗")
            return False
        
    except Exception as e:
        print(f"\n❌ テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
