"""
サービス層簡素化テスト
"""

import sys
sys.path.append('.')

from app.services.ml.ml_training_service import MLTrainingService

def test_simplified_service():
    """簡素化されたMLTrainingServiceのテスト"""
    print("=== 簡素化サービステスト ===")
    
    # アンサンブルサービス
    try:
        service_ensemble = MLTrainingService(
            trainer_type="ensemble",
            ensemble_config={
                "method": "bagging",
                "bagging_params": {"n_estimators": 3}
            }
        )
        print(f"✅ アンサンブルサービス作成成功: {service_ensemble.trainer_type}")
        print(f"   トレーナータイプ: {service_ensemble.trainer.trainer_type}")
        print(f"   モデルタイプ: {service_ensemble.trainer.model_type}")
    except Exception as e:
        print(f"❌ アンサンブルサービス作成失敗: {e}")
    
    # 単一モデルサービス
    try:
        service_single = MLTrainingService(
            trainer_type="single",
            single_model_config={"model_type": "xgboost"}
        )
        print(f"✅ 単一モデルサービス作成成功: {service_single.trainer_type}")
        print(f"   トレーナータイプ: {service_single.trainer.trainer_type}")
        print(f"   モデルタイプ: {service_single.trainer.model_type}")
    except Exception as e:
        print(f"❌ 単一モデルサービス作成失敗: {e}")

def test_service_backward_compatibility():
    """サービス後方互換性テスト"""
    print("=== サービス後方互換性テスト ===")
    
    # デフォルト設定
    try:
        service_default = MLTrainingService()
        print(f"✅ デフォルトサービス作成成功: {service_default.trainer_type}")
        print(f"   トレーナータイプ: {service_default.trainer.trainer_type}")
    except Exception as e:
        print(f"❌ デフォルトサービス作成失敗: {e}")
    
    # AutoML設定
    try:
        automl_config = {"feature_selection": {"enabled": True}}
        service_automl = MLTrainingService(
            trainer_type="single",
            automl_config=automl_config
        )
        print(f"✅ AutoMLサービス作成成功: {service_automl.trainer.use_automl}")
    except Exception as e:
        print(f"❌ AutoMLサービス作成失敗: {e}")

def test_service_methods():
    """サービスメソッドテスト"""
    print("=== サービスメソッドテスト ===")
    
    service = MLTrainingService()
    
    # 必要なメソッドが存在することを確認
    required_methods = [
        'train_model',
        '_create_trainer_config'
    ]
    
    for method_name in required_methods:
        if hasattr(service, method_name):
            print(f"✅ {method_name} メソッド存在確認")
        else:
            print(f"❌ {method_name} メソッド不足")
    
    # 必要な属性が存在することを確認
    required_attributes = [
        'trainer',
        'trainer_type',
        'config'
    ]
    
    for attr_name in required_attributes:
        if hasattr(service, attr_name):
            print(f"✅ {attr_name} 属性存在確認")
        else:
            print(f"❌ {attr_name} 属性不足")

def test_service_integration():
    """サービス統合テスト"""
    print("=== サービス統合テスト ===")
    
    # 統合されたトレーナーが正しく使用されているかテスト
    service = MLTrainingService(trainer_type="single")
    
    # BaseMLTrainerが使用されていることを確認
    trainer_class_name = type(service.trainer).__name__
    print(f"使用されているトレーナークラス: {trainer_class_name}")
    
    if trainer_class_name == "BaseMLTrainer":
        print("✅ 統合されたBaseMLTrainerが正しく使用されています")
    else:
        print(f"❌ 期待されるBaseMLTrainerではなく{trainer_class_name}が使用されています")
    
    # トレーナーの設定が正しく渡されているかテスト
    if hasattr(service.trainer, 'trainer_type'):
        print(f"✅ トレーナー設定が正しく渡されています: {service.trainer.trainer_type}")
    else:
        print("❌ トレーナー設定が正しく渡されていません")

def test_configuration_variations():
    """設定バリエーションテスト"""
    print("=== 設定バリエーションテスト ===")
    
    # 様々な設定パターンをテスト
    configs = [
        # 基本アンサンブル
        {
            "trainer_type": "ensemble",
            "description": "基本アンサンブル"
        },
        # カスタムアンサンブル
        {
            "trainer_type": "ensemble",
            "ensemble_config": {
                "method": "stacking",
                "stacking_params": {
                    "base_models": ["lightgbm", "xgboost"],
                    "meta_model": "lightgbm"
                }
            },
            "description": "カスタムアンサンブル"
        },
        # 基本単一モデル
        {
            "trainer_type": "single",
            "description": "基本単一モデル"
        },
        # カスタム単一モデル
        {
            "trainer_type": "single",
            "single_model_config": {"model_type": "catboost"},
            "description": "カスタム単一モデル"
        }
    ]
    
    for i, config in enumerate(configs):
        try:
            description = config.pop("description")
            service = MLTrainingService(**config)
            print(f"✅ 設定{i+1} ({description}) 成功")
        except Exception as e:
            print(f"❌ 設定{i+1} ({description}) 失敗: {e}")

def main():
    """メインテスト実行"""
    print("🚀 サービス層簡素化テスト開始")
    
    test_simplified_service()
    test_service_backward_compatibility()
    test_service_methods()
    test_service_integration()
    test_configuration_variations()
    
    print("✅ サービス層簡素化テスト完了")

if __name__ == "__main__":
    main()
