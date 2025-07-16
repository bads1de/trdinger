"""
ベイジアン最適化のDB保存テスト

ベイジアン最適化の結果がデータベースに正しく保存されるかをテストします。
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pytest
import json
import sqlite3
from datetime import datetime
from sqlalchemy.orm import Session
from fastapi.testclient import TestClient

from database.connection import get_db, SessionLocal
from database.models import BayesianOptimizationResult
from database.repositories.bayesian_optimization_repository import BayesianOptimizationRepository
from app.api.bayesian_optimization import MLOptimizationRequest


@pytest.fixture
def db_session():
    """テスト用データベースセッション"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def test_database_connection():
    """データベース接続テスト"""
    print("\n=== データベース接続テスト ===")
    
    # SQLiteデータベースに直接接続
    db_path = "C:/Users/buti3/trading/backend/trdinger.db"
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # テーブル一覧を取得
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        print(f"データベース内のテーブル: {[table[0] for table in tables]}")
        
        # bayesian_optimization_resultsテーブルの存在確認
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='bayesian_optimization_results'")
        table_exists = cursor.fetchone()
        
        if table_exists:
            print("✅ bayesian_optimization_resultsテーブルが存在します")
            
            # テーブル構造を確認
            cursor.execute("PRAGMA table_info(bayesian_optimization_results)")
            columns = cursor.fetchall()
            print("テーブル構造:")
            for col in columns:
                print(f"  {col[1]} {col[2]}")
            
            # 既存データを確認
            cursor.execute("SELECT COUNT(*) FROM bayesian_optimization_results")
            count = cursor.fetchone()[0]
            print(f"既存レコード数: {count}")
            
            if count > 0:
                cursor.execute("SELECT id, profile_name, model_type, best_score, created_at FROM bayesian_optimization_results LIMIT 5")
                records = cursor.fetchall()
                print("既存レコード（最新5件）:")
                for record in records:
                    print(f"  ID: {record[0]}, Name: {record[1]}, Model: {record[2]}, Score: {record[3]}, Created: {record[4]}")
        else:
            print("❌ bayesian_optimization_resultsテーブルが存在しません")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ データベース接続エラー: {e}")


def test_repository_save_functionality(db_session):
    """リポジトリの保存機能テスト"""
    print("\n=== リポジトリ保存機能テスト ===")
    
    repo = BayesianOptimizationRepository(db_session)
    
    # テスト用データ
    test_data = {
        "profile_name": "test_save_functionality",
        "optimization_type": "bayesian_ml",
        "model_type": "LightGBM",
        "best_params": {"n_estimators": 100, "learning_rate": 0.1},
        "best_score": 0.85,
        "total_evaluations": 30,
        "optimization_time": 120.5,
        "convergence_info": {"converged": True, "iterations": 25},
        "optimization_history": [
            {"iteration": 1, "score": 0.75, "params": {"n_estimators": 50}},
            {"iteration": 2, "score": 0.80, "params": {"n_estimators": 75}},
            {"iteration": 3, "score": 0.85, "params": {"n_estimators": 100}}
        ],
        "description": "リポジトリ保存機能テスト",
        "target_model_type": "LightGBM",
        "is_default": False
    }
    
    try:
        # データを保存
        result = repo.create_optimization_result(**test_data)
        
        print(f"✅ データ保存成功: ID={result.id}")
        print(f"   プロファイル名: {result.profile_name}")
        print(f"   モデルタイプ: {result.model_type}")
        print(f"   ベストスコア: {result.best_score}")
        
        # 保存されたデータを確認
        saved_result = repo.get_by_id(result.id)
        assert saved_result is not None
        assert saved_result.profile_name == test_data["profile_name"]
        assert saved_result.model_type == test_data["model_type"]
        assert saved_result.best_score == test_data["best_score"]
        
        print("✅ データ取得確認成功")
        
        # クリーンアップ
        db_session.delete(result)
        db_session.commit()
        print("✅ テストデータクリーンアップ完了")
        
    except Exception as e:
        print(f"❌ リポジトリ保存エラー: {e}")
        raise


def test_api_save_with_profile_flag():
    """APIでプロファイル保存フラグを使った保存テスト"""
    print("\n=== API プロファイル保存テスト ===")
    
    from app.api.bayesian_optimization import optimize_ml_hyperparameters
    from database.connection import get_db
    
    # テスト用リクエスト（プロファイル保存あり）
    request_with_save = MLOptimizationRequest(
        model_type="LightGBM",
        n_calls=5,  # 短時間で完了するように少なく設定
        save_as_profile=True,
        profile_name="api_test_profile_with_save",
        profile_description="API保存テスト用プロファイル"
    )
    
    # テスト用リクエスト（プロファイル保存なし）
    request_without_save = MLOptimizationRequest(
        model_type="LightGBM",
        n_calls=5,
        save_as_profile=False
    )
    
    db = next(get_db())
    repo = BayesianOptimizationRepository(db)
    
    try:
        # 保存前のレコード数を確認
        initial_count = len(repo.get_all_results())
        print(f"保存前のレコード数: {initial_count}")
        
        # プロファイル保存ありでテスト
        print("プロファイル保存ありでテスト実行中...")
        import asyncio
        result_with_save = asyncio.run(optimize_ml_hyperparameters(request_with_save, db))
        
        print(f"保存ありテスト結果: {result_with_save}")
        
        # 保存後のレコード数を確認
        after_save_count = len(repo.get_all_results())
        print(f"保存後のレコード数: {after_save_count}")
        
        if after_save_count > initial_count:
            print("✅ プロファイル保存ありの場合、DBに保存されました")
            
            # 保存されたプロファイルを確認
            saved_profile = repo.get_by_profile_name("api_test_profile_with_save")
            if saved_profile:
                print(f"   保存されたプロファイル: {saved_profile.profile_name}")
                print(f"   ベストスコア: {saved_profile.best_score}")
                
                # クリーンアップ
                db.delete(saved_profile)
                db.commit()
        else:
            print("❌ プロファイル保存ありでもDBに保存されませんでした")
        
        # プロファイル保存なしでテスト
        print("プロファイル保存なしでテスト実行中...")
        result_without_save = asyncio.run(optimize_ml_hyperparameters(request_without_save, db))
        
        print(f"保存なしテスト結果: {result_without_save}")
        
        # レコード数が変わらないことを確認
        final_count = len(repo.get_all_results())
        print(f"最終レコード数: {final_count}")
        
        if final_count == initial_count:
            print("✅ プロファイル保存なしの場合、DBに保存されませんでした（正常）")
        else:
            print("❌ プロファイル保存なしでもDBに保存されました（異常）")
        
    except Exception as e:
        print(f"❌ API保存テストエラー: {e}")
        raise
    finally:
        db.close()


def test_manual_optimization_and_save():
    """手動でベイジアン最適化を実行してDB保存をテスト"""
    print("\n=== 手動最適化・保存テスト ===")
    
    from app.core.services.optimization import BayesianOptimizer
    
    db = SessionLocal()
    repo = BayesianOptimizationRepository(db)
    
    try:
        # ベイジアン最適化を手動実行
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
        
        print(f"最適化完了: ベストスコア={optimization_result.best_score:.4f}")
        
        # 結果をDBに保存
        saved_result = repo.create_optimization_result(
            profile_name="manual_test_profile",
            optimization_type="bayesian_ml",
            model_type="LightGBM",
            best_params=optimization_result.best_params,
            best_score=optimization_result.best_score,
            total_evaluations=optimization_result.total_evaluations,
            optimization_time=optimization_result.optimization_time,
            convergence_info=optimization_result.convergence_info,
            optimization_history=optimization_result.optimization_history,
            description="手動最適化テスト",
            target_model_type="LightGBM"
        )
        
        print(f"✅ 手動保存成功: ID={saved_result.id}")
        
        # 保存されたデータを確認
        retrieved = repo.get_by_id(saved_result.id)
        assert retrieved is not None
        print(f"✅ 保存データ確認成功: {retrieved.profile_name}")
        
        # クリーンアップ
        db.delete(saved_result)
        db.commit()
        print("✅ クリーンアップ完了")
        
    except Exception as e:
        print(f"❌ 手動最適化・保存テストエラー: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    print("ベイジアン最適化DB保存テスト開始")
    
    test_database_connection()
    
    # pytest実行時以外は個別テストを実行
    import sys
    if 'pytest' not in sys.modules:
        db = SessionLocal()
        try:
            test_repository_save_functionality(db)
            test_manual_optimization_and_save()
        finally:
            db.close()
        
        print("\n=== テスト完了 ===")
