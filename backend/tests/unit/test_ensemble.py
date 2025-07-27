"""
アンサンブル学習の動作確認テスト

基本的な動作確認を行うためのテストスクリプト
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.ml.ensemble.bagging import BaggingEnsemble
from app.services.ml.ensemble.stacking import StackingEnsemble
from app.services.ml.ensemble.ensemble_trainer import EnsembleTrainer


def create_test_data():
    """テスト用データを作成"""
    print("テストデータを作成中...")
    
    # 分類データセットを生成
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        random_state=42
    )
    
    # DataFrameに変換
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="target")
    
    # 学習・テストデータに分割
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"学習データ: {X_train.shape}, テストデータ: {X_test.shape}")
    print(f"クラス分布: {y_series.value_counts().to_dict()}")
    
    return X_train, X_test, y_train, y_test


def test_bagging_ensemble():
    """バギングアンサンブルのテスト"""
    print("\n=== バギングアンサンブルテスト ===")
    
    try:
        X_train, X_test, y_train, y_test = create_test_data()
        
        # バギング設定
        bagging_config = {
            "n_estimators": 3,
            "bootstrap_fraction": 0.8,
            "base_model_type": "lightgbm",
            "random_state": 42
        }
        
        # バギングアンサンブルを作成
        bagging = BaggingEnsemble(config=bagging_config)
        
        # 学習
        print("バギングアンサンブルを学習中...")
        result = bagging.fit(X_train, y_train, X_test, y_test)
        
        # 予測
        print("予測を実行中...")
        predictions = bagging.predict(X_test)
        probabilities = bagging.predict_proba(X_test)
        
        print(f"バギング学習完了!")
        print(f"精度: {result.get('accuracy', 'N/A'):.4f}")
        print(f"予測形状: {predictions.shape}")
        print(f"確率形状: {probabilities.shape}")
        
        return True
        
    except Exception as e:
        print(f"バギングテストでエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_stacking_ensemble():
    """スタッキングアンサンブルのテスト"""
    print("\n=== スタッキングアンサンブルテスト ===")
    
    try:
        X_train, X_test, y_train, y_test = create_test_data()
        
        # スタッキング設定
        stacking_config = {
            "base_models": ["lightgbm", "random_forest"],
            "meta_model": "logistic_regression",
            "cv_folds": 3,
            "use_probas": True,
            "random_state": 42
        }
        
        # スタッキングアンサンブルを作成
        stacking = StackingEnsemble(config=stacking_config)
        
        # 学習
        print("スタッキングアンサンブルを学習中...")
        result = stacking.fit(X_train, y_train, X_test, y_test)
        
        # 予測
        print("予測を実行中...")
        predictions = stacking.predict(X_test)
        probabilities = stacking.predict_proba(X_test)
        
        print(f"スタッキング学習完了!")
        print(f"精度: {result.get('accuracy', 'N/A'):.4f}")
        print(f"予測形状: {predictions.shape}")
        print(f"確率形状: {probabilities.shape}")
        
        return True
        
    except Exception as e:
        print(f"スタッキングテストでエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ensemble_trainer():
    """EnsembleTrainerのテスト"""
    print("\n=== EnsembleTrainerテスト ===")
    
    try:
        X_train, X_test, y_train, y_test = create_test_data()
        
        # アンサンブル設定（バギング）
        ensemble_config = {
            "method": "bagging",
            "bagging_params": {
                "n_estimators": 3,
                "bootstrap_fraction": 0.8,
                "base_model_type": "lightgbm"
            }
        }
        
        # EnsembleTrainerを作成
        trainer = EnsembleTrainer(ensemble_config=ensemble_config)
        
        # 学習
        print("EnsembleTrainer（バギング）を学習中...")
        result = trainer._train_model_impl(X_train, X_test, y_train, y_test)
        
        # 予測
        print("予測を実行中...")
        predictions = trainer.predict(X_test)
        
        print(f"EnsembleTrainer学習完了!")
        print(f"精度: {result.get('accuracy', 'N/A'):.4f}")
        print(f"予測形状: {predictions.shape}")
        print(f"アンサンブル手法: {result.get('ensemble_method', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"EnsembleTrainerテストでエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """メイン関数"""
    print("アンサンブル学習動作確認テストを開始")
    
    results = []
    
    # 各テストを実行
    results.append(("バギング", test_bagging_ensemble()))
    results.append(("スタッキング", test_stacking_ensemble()))
    results.append(("EnsembleTrainer", test_ensemble_trainer()))
    
    # 結果をまとめ
    print("\n=== テスト結果まとめ ===")
    for test_name, success in results:
        status = "✅ 成功" if success else "❌ 失敗"
        print(f"{test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    if all_passed:
        print("\n🎉 全てのテストが成功しました！")
    else:
        print("\n⚠️ 一部のテストが失敗しました。")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
