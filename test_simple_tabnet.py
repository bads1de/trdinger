#!/usr/bin/env python3
"""
TabNetの特徴量重要度機能を直接テストするスクリプト
"""

import sys
import os
import pandas as pd
import numpy as np
import requests

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def test_tabnet_direct():
    """TabNetモデルを直接テスト"""
    try:
        print("🧪 TabNet直接テスト開始")
        
        # テストデータを作成
        np.random.seed(42)
        n_samples = 1000
        n_features = 10
        
        # 特徴量データ
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # ターゲットデータ（3クラス分類）
        y = pd.Series(np.random.choice([0, 1, 2], size=n_samples))
        
        # 訓練・テストデータに分割
        split_idx = int(n_samples * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"データ形状: X_train={X_train.shape}, y_train={y_train.shape}")
        print(f"クラス分布: {dict(y_train.value_counts())}")
        
        # TabNetモデルを直接作成
        from backend.app.services.ml.models.tabnet_wrapper import TabNetModel
        
        model = TabNetModel()
        
        # モデル学習
        print("TabNet学習中...")
        training_result = model._train_model_impl(X_train, X_test, y_train, y_test)
        print(f"学習完了: {training_result}")
        
        # 特徴量重要度を取得
        print("特徴量重要度を取得中...")
        feature_importance = model.get_feature_importance(top_n=5)
        
        if feature_importance:
            print(f"✅ 特徴量重要度取得成功 (Top 5):")
            for feature, importance in feature_importance.items():
                print(f"  {feature}: {importance:.4f}")
        else:
            print("❌ 特徴量重要度が取得できませんでした")
        
        # 予測テスト
        print("予測テスト中...")
        predictions = model.predict_proba(X_test.head(5))
        print(f"予測結果形状: {predictions.shape}")
        print(f"予測例: {predictions[0]}")
        
        return {
            "success": True,
            "feature_importance": feature_importance,
            "prediction_shape": predictions.shape
        }
        
    except Exception as e:
        print(f"❌ TabNet直接テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

def test_model_manager_save_load():
    """モデル保存・読み込みテスト"""
    try:
        print("\n🗄️ モデル保存・読み込みテスト開始")
        
        # テストデータを作成
        np.random.seed(42)
        n_samples = 500
        n_features = 8
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        y = pd.Series(np.random.choice([0, 1, 2], size=n_samples))
        
        split_idx = int(n_samples * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # TabNetモデルを学習
        from backend.app.services.ml.models.tabnet_wrapper import TabNetModel
        
        model = TabNetModel()
        training_result = model._train_model_impl(X_train, X_test, y_train, y_test)
        print(f"学習完了: {training_result}")
        
        # 特徴量重要度を取得
        feature_importance = model.get_feature_importance(top_n=100)
        print(f"特徴量重要度: {len(feature_importance)}個")
        
        # モデルマネージャーを使用して保存
        from backend.app.services.ml.model_manager import model_manager
        
        metadata = {
            "model_type": "tabnet",
            "trainer_type": "direct_test",
            "feature_count": len(X.columns),
            "feature_importance": feature_importance
        }
        
        model_path = model_manager.save_model(
            model=model.model,
            model_name="test_tabnet_direct",
            metadata=metadata,
            feature_columns=X.columns.tolist()
        )
        
        print(f"✅ モデル保存完了: {model_path}")
        
        # 保存されたモデルを読み込み
        model_data = model_manager.load_model(model_path)
        if model_data and "metadata" in model_data:
            saved_feature_importance = model_data["metadata"].get("feature_importance", {})
            print(f"✅ 保存された特徴量重要度: {len(saved_feature_importance)}個")
            
            if saved_feature_importance:
                print("保存された特徴量重要度 (Top 3):")
                sorted_importance = sorted(
                    saved_feature_importance.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:3]
                for feature, importance in sorted_importance:
                    print(f"  {feature}: {importance:.4f}")
        
        return {
            "success": True,
            "model_path": model_path,
            "saved_feature_importance_count": len(saved_feature_importance)
        }
        
    except Exception as e:
        print(f"❌ モデル保存・読み込みテストエラー: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

def test_api_with_saved_model():
    """保存されたモデルでAPIテスト"""
    try:
        print("\n🌐 API特徴量重要度テスト")
        
        # APIサーバーが起動しているかチェック
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code != 200:
                print("❌ APIサーバーが起動していません")
                return False
        except requests.exceptions.RequestException:
            print("❌ APIサーバーに接続できません")
            return False
        
        # 特徴量重要度APIをテスト
        response = requests.get("http://localhost:8000/api/ml/feature-importance?top_n=5", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            feature_importance = data.get("feature_importance", {})
            
            if feature_importance:
                print("✅ API経由取得成功:")
                for feature, importance in feature_importance.items():
                    print(f"  {feature}: {importance:.4f}")
                return True
            else:
                print("❌ API応答に特徴量重要度が含まれていません")
                print(f"応答データ: {data}")
                return False
        else:
            print(f"❌ API呼び出し失敗: {response.status_code}")
            print(f"応答: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ APIテストエラー: {e}")
        return False

if __name__ == "__main__":
    print("🚀 TabNet特徴量重要度簡易テスト開始")
    
    # 1. TabNet直接テスト
    direct_result = test_tabnet_direct()
    
    # 2. モデル保存・読み込みテスト
    save_load_result = test_model_manager_save_load()
    
    # 3. APIテスト
    api_success = test_api_with_saved_model()
    
    # 結果サマリー
    print(f"\n{'='*50}")
    print("🏁 テスト結果サマリー")
    print(f"{'='*50}")
    
    print(f"✅ TabNet直接テスト: {direct_result.get('success', False)}")
    if direct_result.get('feature_importance'):
        print(f"   特徴量重要度: {len(direct_result['feature_importance'])}個")
    
    print(f"✅ モデル保存・読み込み: {save_load_result.get('success', False)}")
    if save_load_result.get('saved_feature_importance_count'):
        print(f"   保存された特徴量重要度: {save_load_result['saved_feature_importance_count']}個")
    
    print(f"✅ API経由取得: {api_success}")
    
    if (direct_result.get('success') and 
        save_load_result.get('success') and 
        api_success):
        print("\n🎉 全てのテストが成功しました！")
    else:
        print("\n⚠️  一部のテストが失敗しました")
    
    print(f"\n🎯 テスト完了!")
