"""
トレーナー統合テスト
"""

import sys
sys.path.append('.')

from app.services.ml.base_ml_trainer import BaseMLTrainer

def test_unified_trainer():
    """統合トレーナーのテスト"""
    print("=== 統合トレーナーテスト ===")
    
    # 単一モデル設定
    single_config = {
        "type": "single",
        "model_type": "lightgbm"
    }
    
    trainer_single = BaseMLTrainer(trainer_config=single_config)
    print(f"✅ 単一モデルトレーナー作成: {trainer_single.trainer_type}, {trainer_single.model_type}")
    
    # アンサンブル設定
    ensemble_config = {
        "type": "ensemble",
        "model_type": "bagging",
        "ensemble_config": {
            "method": "bagging",
            "bagging_params": {
                "n_estimators": 3,
                "bootstrap_fraction": 0.8,
                "base_model_type": "lightgbm"
            }
        }
    }
    
    trainer_ensemble = BaseMLTrainer(trainer_config=ensemble_config)
    print(f"✅ アンサンブルトレーナー作成: {trainer_ensemble.trainer_type}, {trainer_ensemble.ensemble_config.get('method')}")

def test_trainer_backward_compatibility():
    """トレーナー後方互換性テスト"""
    print("=== トレーナー後方互換性テスト ===")
    
    # 従来の初期化方法
    trainer_legacy = BaseMLTrainer()
    print(f"✅ レガシー初期化: {trainer_legacy.trainer_type}, {trainer_legacy.model_type}")
    
    # AutoML設定のみ
    automl_config = {"feature_selection": {"enabled": True}}
    trainer_automl = BaseMLTrainer(automl_config=automl_config)
    print(f"✅ AutoML設定初期化: {trainer_automl.use_automl}")

def test_trainer_methods():
    """トレーナーメソッドテスト"""
    print("=== トレーナーメソッドテスト ===")
    
    trainer = BaseMLTrainer()
    
    # 必要なメソッドが存在することを確認
    required_methods = [
        '_train_model_impl',
        '_train_single_model',
        '_train_ensemble_model',
        '_evaluate_model_with_unified_metrics'
    ]
    
    for method_name in required_methods:
        if hasattr(trainer, method_name):
            print(f"✅ {method_name} メソッド存在確認")
        else:
            print(f"❌ {method_name} メソッド不足")
    
    # 必要な属性が存在することを確認
    required_attributes = [
        'trainer_type',
        'model_type',
        'ensemble_config',
        'models'
    ]
    
    for attr_name in required_attributes:
        if hasattr(trainer, attr_name):
            print(f"✅ {attr_name} 属性存在確認")
        else:
            print(f"❌ {attr_name} 属性不足")

def test_trainer_configuration():
    """トレーナー設定テスト"""
    print("=== トレーナー設定テスト ===")
    
    # 様々な設定パターンをテスト
    configs = [
        # 単一モデル - LightGBM
        {
            "type": "single",
            "model_type": "lightgbm"
        },
        # 単一モデル - XGBoost
        {
            "type": "single", 
            "model_type": "xgboost"
        },
        # アンサンブル - Bagging
        {
            "type": "ensemble",
            "ensemble_config": {
                "method": "bagging",
                "bagging_params": {"n_estimators": 3}
            }
        },
        # アンサンブル - Stacking
        {
            "type": "ensemble",
            "ensemble_config": {
                "method": "stacking",
                "stacking_params": {
                    "base_models": ["lightgbm", "xgboost"],
                    "meta_model": "lightgbm"
                }
            }
        }
    ]
    
    for i, config in enumerate(configs):
        try:
            trainer = BaseMLTrainer(trainer_config=config)
            print(f"✅ 設定{i+1} 成功: {config.get('type')} - {config.get('model_type', config.get('ensemble_config', {}).get('method'))}")
        except Exception as e:
            print(f"❌ 設定{i+1} 失敗: {e}")

def main():
    """メインテスト実行"""
    print("🚀 トレーナー統合テスト開始")
    
    test_unified_trainer()
    test_trainer_backward_compatibility()
    test_trainer_methods()
    test_trainer_configuration()
    
    print("✅ トレーナー統合テスト完了")

if __name__ == "__main__":
    main()
