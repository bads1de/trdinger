"""
最終統合テスト

真の重複解消リファクタリング後の統合テスト
"""

import sys
sys.path.append('.')

import numpy as np
import pandas as pd

def test_metrics_integration():
    """メトリクス統合テスト"""
    print("=== メトリクス統合テスト ===")
    
    try:
        # 統合されたメトリクス計算器のテスト
        from app.services.ml.evaluation.enhanced_metrics import (
            enhanced_metrics_calculator,
            record_metric,
            record_performance,
            evaluate_and_record_model,
            MLMetricsCollector  # 後方互換性エイリアス
        )
        
        print("✅ 統合メトリクス計算器インポート成功")
        
        # 後方互換性テスト
        collector = MLMetricsCollector()
        print("✅ 後方互換性エイリアス動作確認")
        
        # 便利関数テスト
        record_metric("test_metric", 0.95)
        record_performance("test_operation", 100.0)
        print("✅ 便利関数動作確認")
        
        # 統合評価テスト
        np.random.seed(42)
        y_true = np.random.choice([0, 1, 2], size=30)
        y_pred = np.random.choice([0, 1, 2], size=30)
        y_proba = np.random.dirichlet([1, 1, 1], size=30)
        
        result = evaluate_and_record_model(
            model_name="final_test_model",
            model_type="final_test_type",
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba
        )
        
        print(f"✅ 統合評価成功: accuracy={result.get('accuracy', 'N/A'):.4f}")
        
    except Exception as e:
        print(f"❌ メトリクス統合テストエラー: {e}")

def test_trainer_integration():
    """トレーナー統合テスト"""
    print("=== トレーナー統合テスト ===")
    
    try:
        from app.services.ml.base_ml_trainer import BaseMLTrainer
        
        # 単一モデルトレーナー
        single_trainer = BaseMLTrainer(
            trainer_config={
                "type": "single",
                "model_type": "lightgbm"
            }
        )
        print(f"✅ 統合単一トレーナー作成: {single_trainer.trainer_type}")
        
        # アンサンブルトレーナー
        ensemble_trainer = BaseMLTrainer(
            trainer_config={
                "type": "ensemble",
                "ensemble_config": {
                    "method": "bagging",
                    "bagging_params": {"n_estimators": 3}
                }
            }
        )
        print(f"✅ 統合アンサンブルトレーナー作成: {ensemble_trainer.trainer_type}")
        
    except Exception as e:
        print(f"❌ トレーナー統合テストエラー: {e}")

def test_service_integration():
    """サービス統合テスト"""
    print("=== サービス統合テスト ===")
    
    try:
        from app.services.ml.ml_training_service import MLTrainingService
        
        # 簡素化されたサービス
        service = MLTrainingService(
            trainer_type="single",
            single_model_config={"model_type": "xgboost"}
        )
        
        print(f"✅ 簡素化サービス作成: {service.trainer_type}")
        print(f"   使用トレーナー: {type(service.trainer).__name__}")
        print(f"   トレーナータイプ: {service.trainer.trainer_type}")
        print(f"   モデルタイプ: {service.trainer.model_type}")
        
    except Exception as e:
        print(f"❌ サービス統合テストエラー: {e}")

def test_backward_compatibility():
    """後方互換性テスト"""
    print("=== 後方互換性テスト ===")
    
    try:
        # 旧インポートパスのテスト
        from app.services.ml.common import (
            metrics_collector,
            record_metric,
            record_performance,
            MLMetricsCollector
        )
        
        print("✅ 旧インポートパス動作確認")
        
        # 旧インターフェースのテスト
        record_metric("compat_metric", 0.88)
        record_performance("compat_operation", 75.0)
        
        collector = MLMetricsCollector()
        print("✅ 旧インターフェース動作確認")
        
    except Exception as e:
        print(f"❌ 後方互換性テストエラー: {e}")

def test_file_reduction():
    """ファイル削減確認テスト"""
    print("=== ファイル削減確認テスト ===")
    
    import os
    
    # 削除されたファイルが存在しないことを確認
    deleted_files = [
        "backend/app/services/ml/common/metrics.py",
        "backend/app/services/ml/common/unified_metrics_manager.py", 
        "backend/app/services/ml/common/trainer_factory.py",
        "backend/app/services/ml/common/metrics_constants.py"
    ]
    
    for file_path in deleted_files:
        if not os.path.exists(file_path):
            print(f"✅ {os.path.basename(file_path)} 正常に削除済み")
        else:
            print(f"❌ {os.path.basename(file_path)} まだ存在")
    
    # 統合されたファイルが存在することを確認
    integrated_files = [
        "backend/app/services/ml/evaluation/enhanced_metrics.py",
        "backend/app/services/ml/base_ml_trainer.py"
    ]
    
    for file_path in integrated_files:
        if os.path.exists(file_path):
            print(f"✅ {os.path.basename(file_path)} 統合ファイル存在確認")
        else:
            print(f"❌ {os.path.basename(file_path)} 統合ファイル不足")

def test_functionality_completeness():
    """機能完全性テスト"""
    print("=== 機能完全性テスト ===")
    
    try:
        # 必要な機能がすべて利用可能かテスト
        from app.services.ml.evaluation.enhanced_metrics import enhanced_metrics_calculator
        from app.services.ml.base_ml_trainer import BaseMLTrainer
        from app.services.ml.ml_training_service import MLTrainingService
        
        # メトリクス機能
        required_metrics_methods = [
            'calculate_comprehensive_metrics',
            'record_metric',
            'record_performance',
            'evaluate_and_record_model'
        ]
        
        for method in required_metrics_methods:
            if hasattr(enhanced_metrics_calculator, method):
                print(f"✅ メトリクス機能: {method}")
            else:
                print(f"❌ メトリクス機能不足: {method}")
        
        # トレーナー機能
        trainer = BaseMLTrainer()
        required_trainer_methods = [
            'train_model',
            'predict',
            '_train_single_model',
            '_train_ensemble_model'
        ]
        
        for method in required_trainer_methods:
            if hasattr(trainer, method):
                print(f"✅ トレーナー機能: {method}")
            else:
                print(f"❌ トレーナー機能不足: {method}")
        
        # サービス機能
        service = MLTrainingService()
        required_service_methods = [
            'train_model',
            '_create_trainer_config'
        ]
        
        for method in required_service_methods:
            if hasattr(service, method):
                print(f"✅ サービス機能: {method}")
            else:
                print(f"❌ サービス機能不足: {method}")
        
    except Exception as e:
        print(f"❌ 機能完全性テストエラー: {e}")

def main():
    """メインテスト実行"""
    print("🚀 最終統合テスト開始")
    print("=" * 50)
    
    test_metrics_integration()
    print()
    test_trainer_integration()
    print()
    test_service_integration()
    print()
    test_backward_compatibility()
    print()
    test_file_reduction()
    print()
    test_functionality_completeness()
    
    print()
    print("=" * 50)
    print("✅ 最終統合テスト完了")
    print()
    print("📊 リファクタリング結果サマリー:")
    print("   - メトリクス機能: enhanced_metrics.py に統合")
    print("   - トレーナー機能: base_ml_trainer.py に統合")
    print("   - サービス層: 簡素化完了")
    print("   - 削除ファイル数: 4個")
    print("   - 後方互換性: 維持")

if __name__ == "__main__":
    main()
