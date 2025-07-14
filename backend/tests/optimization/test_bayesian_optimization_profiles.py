"""
ベイジアン最適化プロファイル統合テスト

optimization_profilesテーブルを削除し、bayesian_optimization_resultsに一元化した後の
プロファイル機能のテストを行います。
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pytest
import json
from datetime import datetime
from sqlalchemy.orm import Session
from fastapi.testclient import TestClient

from database.connection import get_db, SessionLocal
from database.models import BayesianOptimizationResult
from database.repositories.bayesian_optimization_repository import BayesianOptimizationRepository
from app.main import app


@pytest.fixture
def db_session():
    """テスト用データベースセッション"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def client():
    """テスト用FastAPIクライアント"""
    with TestClient(app) as client:
        yield client


@pytest.fixture
def sample_optimization_result(db_session):
    """テスト用ベイジアン最適化結果"""
    repo = BayesianOptimizationRepository(db_session)
    
    result = repo.create_optimization_result(
        profile_name="test_profile_lightgbm",
        optimization_type="bayesian_ml",
        model_type="LightGBM",
        best_params={
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 6
        },
        best_score=0.85,
        total_evaluations=30,
        optimization_time=120.5,
        convergence_info={"converged": True, "iterations": 25},
        optimization_history=[
            {"iteration": 1, "score": 0.75, "params": {"n_estimators": 50}},
            {"iteration": 2, "score": 0.80, "params": {"n_estimators": 75}},
            {"iteration": 3, "score": 0.85, "params": {"n_estimators": 100}}
        ],
        description="LightGBMのテスト用最適化結果",
        target_model_type="LightGBM",
        is_default=True
    )
    
    yield result
    
    # クリーンアップ
    db_session.delete(result)
    db_session.commit()


class TestBayesianOptimizationRepository:
    """BayesianOptimizationRepositoryのテスト"""

    def test_create_optimization_result_with_profile_fields(self, db_session):
        """プロファイルフィールドを含む最適化結果の作成テスト"""
        repo = BayesianOptimizationRepository(db_session)
        
        result = repo.create_optimization_result(
            profile_name="test_create_profile",
            optimization_type="bayesian_ml",
            model_type="RandomForest",
            best_params={"n_estimators": 200},
            best_score=0.90,
            total_evaluations=50,
            optimization_time=300.0,
            convergence_info={"converged": True},
            optimization_history=[],
            target_model_type="RandomForest",
            is_default=False,
            description="作成テスト用"
        )
        
        assert result is not None
        assert result.profile_name == "test_create_profile"
        assert result.target_model_type == "RandomForest"
        assert result.is_default is False
        assert result.description == "作成テスト用"
        
        # クリーンアップ
        db_session.delete(result)
        db_session.commit()

    def test_get_default_profile(self, db_session, sample_optimization_result):
        """デフォルトプロファイル取得テスト"""
        repo = BayesianOptimizationRepository(db_session)
        
        default_profile = repo.get_default_profile("LightGBM")
        
        assert default_profile is not None
        assert default_profile.target_model_type == "LightGBM"
        assert default_profile.is_default is True
        assert default_profile.profile_name == "test_profile_lightgbm"

    def test_set_default_profile(self, db_session):
        """デフォルトプロファイル設定テスト"""
        repo = BayesianOptimizationRepository(db_session)
        
        # 2つのプロファイルを作成
        profile1 = repo.create_optimization_result(
            profile_name="profile1_xgboost",
            optimization_type="bayesian_ml",
            model_type="XGBoost",
            best_params={"n_estimators": 100},
            best_score=0.80,
            total_evaluations=30,
            optimization_time=100.0,
            convergence_info={"converged": True},
            optimization_history=[],
            target_model_type="XGBoost",
            is_default=True
        )
        
        profile2 = repo.create_optimization_result(
            profile_name="profile2_xgboost",
            optimization_type="bayesian_ml",
            model_type="XGBoost",
            best_params={"n_estimators": 200},
            best_score=0.85,
            total_evaluations=40,
            optimization_time=150.0,
            convergence_info={"converged": True},
            optimization_history=[],
            target_model_type="XGBoost",
            is_default=False
        )
        
        # profile2をデフォルトに設定
        success = repo.set_default_profile(profile2.id, "XGBoost")
        assert success is True
        
        # 確認
        db_session.refresh(profile1)
        db_session.refresh(profile2)
        
        assert profile1.is_default is False
        assert profile2.is_default is True
        
        # クリーンアップ
        db_session.delete(profile1)
        db_session.delete(profile2)
        db_session.commit()

    def test_get_profiles_by_model_type(self, db_session):
        """モデルタイプ別プロファイル取得テスト"""
        repo = BayesianOptimizationRepository(db_session)
        
        # 複数のプロファイルを作成
        profiles = []
        for i in range(3):
            profile = repo.create_optimization_result(
                profile_name=f"test_profile_catboost_{i}",
                optimization_type="bayesian_ml",
                model_type="CatBoost",
                best_params={"iterations": 100 + i * 50},
                best_score=0.80 + i * 0.05,
                total_evaluations=30,
                optimization_time=100.0,
                convergence_info={"converged": True},
                optimization_history=[],
                target_model_type="CatBoost",
                is_default=(i == 0)
            )
            profiles.append(profile)
        
        # 取得テスト
        retrieved_profiles = repo.get_profiles_by_model_type("CatBoost")
        
        assert len(retrieved_profiles) == 3
        # デフォルトプロファイルが最初に来ることを確認
        assert retrieved_profiles[0].is_default is True
        
        # クリーンアップ
        for profile in profiles:
            db_session.delete(profile)
        db_session.commit()


class TestBayesianOptimizationIntegration:
    """ベイジアン最適化統合テスト（APIテストなし）"""

    def test_profile_integration_workflow(self, db_session):
        """プロファイル統合ワークフローテスト"""
        repo = BayesianOptimizationRepository(db_session)

        # 1. プロファイルを作成
        profile = repo.create_optimization_result(
            profile_name="integration_test_profile",
            optimization_type="bayesian_ml",
            model_type="LightGBM",
            best_params={"n_estimators": 150, "learning_rate": 0.05},
            best_score=0.88,
            total_evaluations=25,
            optimization_time=180.0,
            convergence_info={"converged": True, "iterations": 20},
            optimization_history=[],
            target_model_type="LightGBM",
            is_default=True,
            description="統合テスト用プロファイル"
        )

        assert profile is not None
        assert profile.profile_name == "integration_test_profile"
        assert profile.is_default is True

        # 2. デフォルトプロファイルとして取得
        default_profile = repo.get_default_profile("LightGBM")
        assert default_profile is not None
        assert default_profile.id == profile.id

        # 3. プロファイルを更新
        updated_profile = repo.update_optimization_result(
            profile.id,
            description="更新された統合テスト用プロファイル",
            best_score=0.90
        )

        assert updated_profile.description == "更新された統合テスト用プロファイル"
        assert updated_profile.best_score == 0.90

        # 4. モデルタイプ別プロファイル取得
        profiles = repo.get_profiles_by_model_type("LightGBM")
        assert len(profiles) >= 1
        assert any(p.id == profile.id for p in profiles)

        # クリーンアップ
        db_session.delete(profile)
        db_session.commit()

        print("✅ プロファイル統合ワークフローテスト完了")

    def test_multiple_model_types_workflow(self, db_session):
        """複数モデルタイプのワークフローテスト"""
        repo = BayesianOptimizationRepository(db_session)

        model_types = ["LightGBM", "XGBoost", "RandomForest"]
        created_profiles = []

        # 各モデルタイプのプロファイルを作成
        for i, model_type in enumerate(model_types):
            profile = repo.create_optimization_result(
                profile_name=f"multi_model_test_{model_type.lower()}",
                optimization_type="bayesian_ml",
                model_type=model_type,
                best_params={"n_estimators": 100 + i * 50},
                best_score=0.80 + i * 0.05,
                total_evaluations=30,
                optimization_time=120.0,
                convergence_info={"converged": True},
                optimization_history=[],
                target_model_type=model_type,
                is_default=True
            )
            created_profiles.append(profile)

        # 各モデルタイプのデフォルトプロファイルを確認
        for model_type in model_types:
            default_profile = repo.get_default_profile(model_type)
            assert default_profile is not None
            assert default_profile.target_model_type == model_type
            assert default_profile.is_default is True

        # クリーンアップ
        for profile in created_profiles:
            db_session.delete(profile)
        db_session.commit()

        print("✅ 複数モデルタイプワークフローテスト完了")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
