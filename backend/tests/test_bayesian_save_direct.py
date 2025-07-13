"""
ベイジアン最適化のDB保存を直接テストするスクリプト
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import logging
from datetime import datetime

from database.connection import SessionLocal
from database.repositories.bayesian_optimization_repository import BayesianOptimizationRepository
from app.api.bayesian_optimization import MLOptimizationRequest, optimize_ml_hyperparameters
from app.core.services.optimization import BayesianOptimizer

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_direct_repository_save():
    """リポジトリ直接保存テスト"""
    print("\n=== リポジトリ直接保存テスト ===")
    
    db = SessionLocal()
    repo = BayesianOptimizationRepository(db)
    
    try:
        # 保存前のレコード数
        initial_count = len(repo.get_all_results())
        print(f"保存前のレコード数: {initial_count}")
        
        # テストデータを作成
        result = repo.create_optimization_result(
            profile_name="direct_test_profile",
            optimization_type="bayesian_ml",
            model_type="LightGBM",
            best_params={"n_estimators": 100, "learning_rate": 0.1},
            best_score=0.85,
            total_evaluations=30,
            optimization_time=120.0,
            convergence_info={"converged": True},
            optimization_history=[
                {"iteration": 1, "score": 0.75, "params": {"n_estimators": 50}},
                {"iteration": 2, "score": 0.85, "params": {"n_estimators": 100}}
            ],
            description="直接保存テスト",
            target_model_type="LightGBM"
        )
        
        print(f"✅ 直接保存成功: ID={result.id}, Name={result.profile_name}")
        
        # 保存後のレコード数
        after_count = len(repo.get_all_results())
        print(f"保存後のレコード数: {after_count}")
        
        # 保存されたデータを確認
        saved_data = repo.get_by_id(result.id)
        if saved_data:
            print(f"✅ データ確認成功: {saved_data.profile_name}")
        else:
            print("❌ 保存されたデータが見つかりません")
        
        return result.id
        
    except Exception as e:
        print(f"❌ 直接保存エラー: {e}")
        return None
    finally:
        db.close()

def test_bayesian_optimizer_direct():
    """ベイジアン最適化エンジン直接テスト"""
    print("\n=== ベイジアン最適化エンジン直接テスト ===")
    
    try:
        optimizer = BayesianOptimizer()
        
        def dummy_objective(params):
            """ダミー目的関数"""
            import random
            return random.uniform(0.5, 0.9)
        
        print("ベイジアン最適化実行中...")
        optimization_result = optimizer.optimize_ml_hyperparameters(
            model_type="LightGBM",
            objective_function=dummy_objective,
            n_calls=5
        )
        
        print(f"✅ 最適化完了:")
        print(f"   ベストスコア: {optimization_result.best_score:.4f}")
        print(f"   ベストパラメータ: {optimization_result.best_params}")
        print(f"   評価回数: {optimization_result.total_evaluations}")
        print(f"   最適化時間: {optimization_result.optimization_time:.2f}秒")
        
        # 結果をDBに保存
        db = SessionLocal()
        repo = BayesianOptimizationRepository(db)
        
        try:
            saved_result = repo.create_optimization_result(
                profile_name="optimizer_direct_test",
                optimization_type="bayesian_ml",
                model_type="LightGBM",
                best_params=optimization_result.best_params,
                best_score=optimization_result.best_score,
                total_evaluations=optimization_result.total_evaluations,
                optimization_time=optimization_result.optimization_time,
                convergence_info=optimization_result.convergence_info,
                optimization_history=optimization_result.optimization_history,
                description="最適化エンジン直接テスト",
                target_model_type="LightGBM"
            )
            
            print(f"✅ 最適化結果DB保存成功: ID={saved_result.id}")
            return saved_result.id
            
        except Exception as e:
            print(f"❌ 最適化結果DB保存エラー: {e}")
            return None
        finally:
            db.close()
        
    except Exception as e:
        print(f"❌ ベイジアン最適化エラー: {e}")
        return None

async def test_api_endpoint_with_save():
    """APIエンドポイントでの保存テスト"""
    print("\n=== APIエンドポイント保存テスト ===")
    
    # プロファイル保存ありのリクエスト
    request = MLOptimizationRequest(
        model_type="LightGBM",
        n_calls=5,
        save_as_profile=True,
        profile_name="api_endpoint_test",
        profile_description="APIエンドポイント保存テスト"
    )
    
    db = SessionLocal()
    
    try:
        print("APIエンドポイント実行中...")
        print(f"リクエスト内容:")
        print(f"  model_type: {request.model_type}")
        print(f"  n_calls: {request.n_calls}")
        print(f"  save_as_profile: {request.save_as_profile}")
        print(f"  profile_name: {request.profile_name}")
        
        # 保存前のレコード数
        repo = BayesianOptimizationRepository(db)
        initial_count = len(repo.get_all_results())
        print(f"保存前のレコード数: {initial_count}")
        
        # APIエンドポイントを直接呼び出し
        result = await optimize_ml_hyperparameters(request, db)
        
        print(f"✅ APIエンドポイント実行完了:")
        print(f"   成功: {result.get('success', False)}")
        print(f"   メッセージ: {result.get('message', 'N/A')}")
        
        if result.get('success') and 'result' in result:
            api_result = result['result']
            print(f"   ベストスコア: {api_result.get('best_score', 'N/A')}")
            
            if 'saved_profile_id' in api_result:
                print(f"   保存されたプロファイルID: {api_result['saved_profile_id']}")
            else:
                print("   ⚠️ saved_profile_idが結果に含まれていません")
        
        # 保存後のレコード数
        after_count = len(repo.get_all_results())
        print(f"保存後のレコード数: {after_count}")
        
        if after_count > initial_count:
            print("✅ APIエンドポイント経由でDB保存成功")
            
            # 保存されたプロファイルを確認
            saved_profile = repo.get_by_profile_name("api_endpoint_test")
            if saved_profile:
                print(f"   保存されたプロファイル: {saved_profile.profile_name}")
                return saved_profile.id
        else:
            print("❌ APIエンドポイント経由でDB保存されませんでした")
        
        return None
        
    except Exception as e:
        print(f"❌ APIエンドポイントテストエラー: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        db.close()

def cleanup_test_data(test_ids):
    """テストデータのクリーンアップ"""
    print("\n=== テストデータクリーンアップ ===")
    
    db = SessionLocal()
    repo = BayesianOptimizationRepository(db)
    
    try:
        for test_id in test_ids:
            if test_id:
                result = repo.get_by_id(test_id)
                if result:
                    db.delete(result)
                    print(f"削除: ID={test_id}, Name={result.profile_name}")
        
        db.commit()
        print("✅ クリーンアップ完了")
        
    except Exception as e:
        print(f"❌ クリーンアップエラー: {e}")
        db.rollback()
    finally:
        db.close()

async def main():
    """メイン実行関数"""
    print("ベイジアン最適化DB保存テスト開始")
    print("=" * 50)
    
    test_ids = []
    
    # 1. リポジトリ直接保存テスト
    direct_id = test_direct_repository_save()
    if direct_id:
        test_ids.append(direct_id)
    
    # 2. ベイジアン最適化エンジン直接テスト
    optimizer_id = test_bayesian_optimizer_direct()
    if optimizer_id:
        test_ids.append(optimizer_id)
    
    # 3. APIエンドポイント保存テスト
    api_id = await test_api_endpoint_with_save()
    if api_id:
        test_ids.append(api_id)
    
    # 最終的なDB状況確認
    print("\n=== 最終DB状況確認 ===")
    db = SessionLocal()
    repo = BayesianOptimizationRepository(db)
    try:
        all_results = repo.get_all_results()
        print(f"総レコード数: {len(all_results)}")
        for result in all_results:
            print(f"  ID: {result.id}, Name: {result.profile_name}, Model: {result.model_type}")
    finally:
        db.close()
    
    # クリーンアップ
    cleanup_test_data(test_ids)
    
    print("\n" + "=" * 50)
    print("テスト完了")

if __name__ == "__main__":
    asyncio.run(main())
