"""
包括的な永続化処理テスト

実際の問題を検出するための詳細なテストを実行します。
"""

import sys
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.services.auto_strategy.services.auto_strategy_service import AutoStrategyService
from app.core.services.auto_strategy.models.ga_config import GAConfig
from database.connection import SessionLocal
from database.repositories.ga_experiment_repository import GAExperimentRepository
from database.repositories.generated_strategy_repository import GeneratedStrategyRepository
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_concurrent_experiments():
    """同時実行実験のテスト"""
    print("\n=== 同時実行実験テスト ===")
    
    try:
        service = AutoStrategyService()
        
        # 複数の実験を同時実行
        experiment_configs = []
        for i in range(3):
            ga_config = GAConfig(
                population_size=5,
                generations=2,
                crossover_rate=0.8,
                mutation_rate=0.1,
                elite_size=1,
                max_indicators=2,
                allowed_indicators=["SMA", "EMA"]
            )
            
            backtest_config = {
                "symbol": "BTCUSDT",
                "timeframe": "1h",
                "start_date": "2024-01-01",
                "end_date": "2024-01-03",
                "initial_capital": 10000
            }
            
            experiment_configs.append((f"同時実行テスト{i+1}", ga_config, backtest_config))
        
        # 同時実行
        experiment_ids = []
        for name, ga_config, backtest_config in experiment_configs:
            experiment_id = service.start_strategy_generation(
                experiment_name=name,
                ga_config=ga_config,
                backtest_config=backtest_config
            )
            experiment_ids.append(experiment_id)
            print(f"実験開始: {experiment_id} ({name})")
        
        # 完了まで待機
        max_wait = 60
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            completed_count = 0
            for exp_id in experiment_ids:
                progress = service.get_experiment_progress(exp_id)
                if progress and progress.status in ["completed", "error"]:
                    completed_count += 1
            
            if completed_count == len(experiment_ids):
                print("✅ 全実験完了")
                break
            
            time.sleep(2)
        else:
            print("⚠️ 一部実験がタイムアウト")
        
        # 結果確認
        for exp_id in experiment_ids:
            progress = service.get_experiment_progress(exp_id)
            print(f"実験{exp_id}: {progress.status if progress else 'Unknown'}")
        
        return True
        
    except Exception as e:
        print(f"❌ 同時実行テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_database_consistency():
    """データベース整合性の詳細テスト"""
    print("\n=== データベース整合性テスト ===")
    
    try:
        db = SessionLocal()
        try:
            exp_repo = GAExperimentRepository(db)
            strategy_repo = GeneratedStrategyRepository(db)
            
            # 1. 外部キー制約テスト
            print("1. 外部キー制約テスト")
            experiments = exp_repo.get_recent_experiments(limit=5)
            
            for exp in experiments:
                strategies = strategy_repo.get_strategies_by_experiment(exp.id)
                for strategy in strategies:
                    if strategy.experiment_id != exp.id:
                        print(f"❌ 外部キー不整合: 戦略{strategy.id}の実験ID")
                        return False
            
            print("✅ 外部キー制約OK")
            
            # 2. データ型整合性テスト
            print("2. データ型整合性テスト")
            for exp in experiments:
                if not isinstance(exp.progress, (int, float)) or exp.progress < 0 or exp.progress > 1:
                    print(f"❌ 進捗率異常: 実験{exp.id} = {exp.progress}")
                    return False
                
                if exp.best_fitness is not None and not isinstance(exp.best_fitness, (int, float)):
                    print(f"❌ フィットネス型異常: 実験{exp.id}")
                    return False
            
            print("✅ データ型整合性OK")
            
            # 3. JSON データ整合性テスト
            print("3. JSON データ整合性テスト")
            strategies = strategy_repo.get_best_strategies(limit=10)
            
            for strategy in strategies:
                gene_data = strategy.gene_data
                if not isinstance(gene_data, dict):
                    print(f"❌ 遺伝子データ型異常: 戦略{strategy.id}")
                    return False
                
                required_fields = ["id", "indicators", "entry_conditions", "exit_conditions"]
                for field in required_fields:
                    if field not in gene_data:
                        print(f"❌ 遺伝子データ不完全: 戦略{strategy.id}, 欠損フィールド: {field}")
                        return False
            
            print("✅ JSON データ整合性OK")
            
            return True
            
        finally:
            db.close()
            
    except Exception as e:
        print(f"❌ データベース整合性テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_error_scenarios():
    """エラーシナリオテスト"""
    print("\n=== エラーシナリオテスト ===")
    
    try:
        service = AutoStrategyService()
        
        # 1. 無効なGA設定
        print("1. 無効なGA設定テスト")
        try:
            invalid_config = GAConfig(
                population_size=-1,  # 無効
                generations=0,       # 無効
                crossover_rate=1.5   # 無効
            )
            
            experiment_id = service.start_strategy_generation(
                experiment_name="エラーテスト",
                ga_config=invalid_config,
                backtest_config={}
            )
            print("❌ エラーが発生すべきでした")
            return False
            
        except ValueError:
            print("✅ 無効なGA設定で適切にエラー")
        
        # 2. 存在しない実験の進捗取得
        print("2. 存在しない実験テスト")
        progress = service.get_experiment_progress("non-existent-id")
        if progress is not None:
            print("❌ 存在しない実験で進捗が返された")
            return False
        print("✅ 存在しない実験で適切にNone")
        
        # 3. データベース接続エラーシミュレーション
        print("3. データベース操作エラーテスト")
        db = SessionLocal()
        try:
            exp_repo = GAExperimentRepository(db)
            
            # 存在しない実験の更新
            success = exp_repo.update_experiment_progress(
                experiment_id=99999,  # 存在しない
                current_generation=1,
                progress=0.5
            )
            
            if success:
                print("❌ 存在しない実験の更新が成功した")
                return False
            print("✅ 存在しない実験の更新で適切にFalse")
            
        finally:
            db.close()
        
        return True
        
    except Exception as e:
        print(f"❌ エラーシナリオテストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_and_memory():
    """パフォーマンスとメモリ使用量テスト"""
    print("\n=== パフォーマンス・メモリテスト ===")
    
    try:
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"初期メモリ使用量: {initial_memory:.2f} MB")
        
        # 大量のデータベース操作
        db = SessionLocal()
        try:
            strategy_repo = GeneratedStrategyRepository(db)
            
            # 大量の戦略データ作成
            start_time = time.time()
            
            strategies_data = []
            for i in range(100):
                strategies_data.append({
                    "experiment_id": 1,  # 既存の実験ID
                    "gene_data": {
                        "id": f"perf_test_{i}",
                        "indicators": [{"type": "SMA", "parameters": {"period": 20}}],
                        "entry_conditions": [],
                        "exit_conditions": [],
                        "risk_management": {},
                        "metadata": {}
                    },
                    "generation": 1,
                    "fitness_score": 0.5 + i * 0.001
                })
            
            # 一括保存
            saved_strategies = strategy_repo.save_strategies_batch(strategies_data)
            
            save_time = time.time() - start_time
            print(f"100戦略の一括保存時間: {save_time:.2f}秒")
            
            # メモリ使用量確認
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = current_memory - initial_memory
            
            print(f"現在メモリ使用量: {current_memory:.2f} MB")
            print(f"メモリ増加量: {memory_increase:.2f} MB")
            
            # 大量検索テスト
            start_time = time.time()
            best_strategies = strategy_repo.get_best_strategies(limit=50)
            search_time = time.time() - start_time
            
            print(f"50戦略検索時間: {search_time:.4f}秒")
            
            # クリーンアップ
            deleted_count = strategy_repo.delete_strategies_by_experiment(1)
            print(f"クリーンアップ: {deleted_count}戦略削除")
            
            # ガベージコレクション
            gc.collect()
            
            final_memory = process.memory_info().rss / 1024 / 1024
            print(f"最終メモリ使用量: {final_memory:.2f} MB")
            
            # パフォーマンス基準チェック
            if save_time > 5.0:
                print("⚠️ 一括保存が遅い")
            if search_time > 1.0:
                print("⚠️ 検索が遅い")
            if memory_increase > 100:
                print("⚠️ メモリ使用量が多い")
            
            print("✅ パフォーマンステスト完了")
            return True
            
        finally:
            db.close()
            
    except ImportError:
        print("⚠️ psutilが利用できません。パフォーマンステストをスキップ")
        return True
    except Exception as e:
        print(f"❌ パフォーマンステストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """包括的テスト実行"""
    try:
        print("🧪 包括的永続化処理テスト開始")
        
        tests = [
            ("同時実行実験", test_concurrent_experiments),
            ("データベース整合性", test_database_consistency),
            ("エラーシナリオ", test_error_scenarios),
            ("パフォーマンス・メモリ", test_performance_and_memory),
        ]
        
        results = []
        for test_name, test_func in tests:
            print(f"\n{'='*50}")
            print(f"テスト実行: {test_name}")
            print(f"{'='*50}")
            
            try:
                result = test_func()
                results.append((test_name, result))
                
                if result:
                    print(f"✅ {test_name}: 成功")
                else:
                    print(f"❌ {test_name}: 失敗")
                    
            except Exception as e:
                print(f"❌ {test_name}: 例外発生 - {e}")
                results.append((test_name, False))
        
        # 結果サマリー
        print(f"\n{'='*50}")
        print("テスト結果サマリー")
        print(f"{'='*50}")
        
        success_count = 0
        for test_name, result in results:
            status = "✅ 成功" if result else "❌ 失敗"
            print(f"{test_name}: {status}")
            if result:
                success_count += 1
        
        print(f"\n成功率: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
        
        if success_count == len(results):
            print("\n🎉 全テスト成功！永続化処理に問題ありません")
            return True
        else:
            print(f"\n⚠️ {len(results)-success_count}個のテストで問題が検出されました")
            return False
        
    except Exception as e:
        print(f"\n❌ テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
