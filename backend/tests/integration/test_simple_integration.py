"""
シンプルなアンサンブル学習統合テスト

EnsembleTrainerの直接テストとMLTrainingServiceの基本動作確認
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.services.ml.ensemble.ensemble_trainer import EnsembleTrainer
from app.core.services.ml.ml_training_service import MLTrainingService


def test_ensemble_trainer_direct():
    """EnsembleTrainerの直接テスト"""
    print("\n=== EnsembleTrainer直接テスト ===")
    
    try:
        # テストデータを作成
        X, y = make_classification(
            n_samples=300,
            n_features=10,
            n_informative=8,
            n_redundant=2,
            n_classes=3,
            random_state=42
        )
        
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name="target")
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y_series, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"データ準備完了: 学習={X_train.shape}, テスト={X_test.shape}")
        
        # バギングアンサンブル
        ensemble_config = {
            "method": "bagging",
            "bagging_params": {
                "n_estimators": 3,
                "bootstrap_fraction": 0.8,
                "base_model_type": "lightgbm"
            }
        }
        
        trainer = EnsembleTrainer(ensemble_config=ensemble_config)
        
        print("バギングアンサンブル学習を開始...")
        result = trainer._train_model_impl(X_train, X_test, y_train, y_test)
        
        print(f"学習完了!")
        print(f"精度: {result.get('accuracy', 'N/A'):.4f}")
        print(f"アンサンブル手法: {result.get('ensemble_method', 'N/A')}")
        print(f"モデルタイプ: {result.get('model_type', 'N/A')}")
        
        # 予測テスト
        predictions = trainer.predict(X_test)
        print(f"予測形状: {predictions.shape}")
        print(f"予測サンプル: {predictions[:3]}")
        
        return result.get('accuracy', 0) > 0.5
        
    except Exception as e:
        print(f"EnsembleTrainer直接テストでエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ml_training_service_basic():
    """MLTrainingServiceの基本動作テスト"""
    print("\n=== MLTrainingService基本動作テスト ===")
    
    try:
        # アンサンブル設定
        ensemble_config = {
            "method": "bagging",
            "bagging_params": {
                "n_estimators": 3,
                "bootstrap_fraction": 0.8,
                "base_model_type": "lightgbm"
            }
        }
        
        # MLTrainingServiceを作成
        ml_service = MLTrainingService(
            trainer_type="ensemble",
            ensemble_config=ensemble_config
        )
        
        print(f"MLTrainingService作成完了")
        print(f"トレーナータイプ: {ml_service.trainer_type}")
        print(f"アンサンブル設定: {ml_service.ensemble_config}")
        print(f"トレーナークラス: {type(ml_service.trainer).__name__}")
        
        # 設定検証
        assert ml_service.trainer_type == "ensemble"
        assert ml_service.ensemble_config is not None
        assert hasattr(ml_service.trainer, 'ensemble_config')
        
        print("✅ MLTrainingService設定検証成功")
        return True
        
    except Exception as e:
        print(f"MLTrainingService基本動作テストでエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ensemble_config_validation():
    """アンサンブル設定の検証テスト"""
    print("\n=== アンサンブル設定検証テスト ===")
    
    try:
        # バギング設定テスト
        bagging_config = {
            "method": "bagging",
            "bagging_params": {
                "n_estimators": 5,
                "bootstrap_fraction": 0.8,
                "base_model_type": "lightgbm"
            }
        }
        
        trainer1 = EnsembleTrainer(ensemble_config=bagging_config)
        print(f"✅ バギング設定検証成功: {trainer1.ensemble_method}")
        
        # スタッキング設定テスト
        stacking_config = {
            "method": "stacking",
            "stacking_params": {
                "base_models": ["lightgbm", "random_forest"],
                "meta_model": "logistic_regression",
                "cv_folds": 3,
                "use_probas": True
            }
        }
        
        trainer2 = EnsembleTrainer(ensemble_config=stacking_config)
        print(f"✅ スタッキング設定検証成功: {trainer2.ensemble_method}")
        
        # MLTrainingService設定テスト
        ml_service1 = MLTrainingService(trainer_type="ensemble", ensemble_config=bagging_config)
        ml_service2 = MLTrainingService(trainer_type="ensemble", ensemble_config=stacking_config)
        
        print(f"✅ MLTrainingService設定検証成功")
        print(f"   バギング: {ml_service1.trainer.ensemble_method}")
        print(f"   スタッキング: {ml_service2.trainer.ensemble_method}")
        
        return True
        
    except Exception as e:
        print(f"アンサンブル設定検証テストでエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_global_instance():
    """グローバルインスタンスのテスト"""
    print("\n=== グローバルインスタンステスト ===")
    
    try:
        from app.core.services.ml.ml_training_service import ml_training_service
        
        print(f"グローバルインスタンス取得成功")
        print(f"トレーナータイプ: {ml_training_service.trainer_type}")
        print(f"トレーナークラス: {type(ml_training_service.trainer).__name__}")
        
        # デフォルトでアンサンブルになっているか確認
        assert ml_training_service.trainer_type == "ensemble"
        assert hasattr(ml_training_service.trainer, 'ensemble_config')
        
        print("✅ グローバルインスタンスがアンサンブル設定になっています")
        return True
        
    except Exception as e:
        print(f"グローバルインスタンステストでエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_config_compatibility():
    """API設定との互換性テスト"""
    print("\n=== API設定互換性テスト ===")
    
    try:
        from app.api.ml_training import EnsembleConfig, BaggingParamsConfig, StackingParamsConfig
        
        # API設定モデルを作成
        ensemble_config = EnsembleConfig(
            enabled=True,
            method="bagging",
            bagging_params=BaggingParamsConfig(
                n_estimators=5,
                bootstrap_fraction=0.8
            ),
            stacking_params=StackingParamsConfig(
                base_models=["lightgbm", "random_forest"],
                meta_model="logistic_regression",
                cv_folds=5,
                use_probas=True
            )
        )
        
        print(f"✅ API設定モデル作成成功")
        print(f"   有効: {ensemble_config.enabled}")
        print(f"   手法: {ensemble_config.method}")
        print(f"   バギングn_estimators: {ensemble_config.bagging_params.n_estimators}")
        print(f"   スタッキングベースモデル: {ensemble_config.stacking_params.base_models}")
        
        # 辞書形式に変換
        config_dict = ensemble_config.dict()
        print(f"✅ 辞書変換成功: {len(config_dict)}項目")
        
        return True
        
    except Exception as e:
        print(f"API設定互換性テストでエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """メイン関数"""
    print("🚀 シンプルなアンサンブル学習統合テストを開始")
    print("=" * 70)
    
    results = []
    
    # 各テストを実行
    results.append(("EnsembleTrainer直接", test_ensemble_trainer_direct()))
    results.append(("MLTrainingService基本", test_ml_training_service_basic()))
    results.append(("アンサンブル設定検証", test_ensemble_config_validation()))
    results.append(("グローバルインスタンス", test_global_instance()))
    results.append(("API設定互換性", test_api_config_compatibility()))
    
    # 結果をまとめ
    print("\n" + "=" * 70)
    print("=== シンプル統合テスト結果まとめ ===")
    for test_name, success in results:
        status = "✅ 成功" if success else "❌ 失敗"
        print(f"{test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    if all_passed:
        print("\n🎉 全てのシンプル統合テストが成功しました！")
        print("アンサンブル学習の基本機能が正しく動作しています。")
        print("LightGBMオンリーからアンサンブル学習への移行が正常に完了しました。")
    else:
        print("\n⚠️ 一部のシンプル統合テストが失敗しました。")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
