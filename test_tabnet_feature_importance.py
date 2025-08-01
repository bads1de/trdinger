#!/usr/bin/env python3
"""
TabNetの特徴量重要度機能をテストするスクリプト
"""

import sys
import os
import pandas as pd
import numpy as np
import requests
import time
from typing import Dict, Any

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def create_test_data():
    """テスト用のOHLCVデータを作成"""
    np.random.seed(42)
    n_samples = 1000
    
    # 基本的な価格データを生成
    base_price = 50000
    returns = np.random.normal(0, 0.02, n_samples)
    prices = [base_price]
    
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(new_price)
    
    # OHLCV データを作成
    data = []
    for i in range(n_samples):
        close = prices[i]
        high = close * (1 + abs(np.random.normal(0, 0.01)))
        low = close * (1 - abs(np.random.normal(0, 0.01)))
        open_price = low + (high - low) * np.random.random()
        volume = np.random.randint(1000, 10000)
        
        data.append({
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume,
            'timestamp': pd.Timestamp('2024-01-01') + pd.Timedelta(hours=i)
        })
    
    return pd.DataFrame(data)

def test_tabnet_training_and_feature_importance():
    """TabNetの学習と特徴量重要度取得をテスト"""
    try:
        print("🧪 TabNet特徴量重要度テスト開始")
        
        # テストデータを作成
        df = create_test_data()
        print(f"テストデータ作成完了: {df.shape}")
        print(f"カラム: {list(df.columns)}")
        
        # SingleModelTrainerでTabNetを学習
        from backend.app.services.ml.single_model.single_model_trainer import SingleModelTrainer
        
        print("\n1. SingleModelTrainerでTabNet学習")
        trainer = SingleModelTrainer(model_type="tabnet")
        
        # 特徴量とターゲットを準備
        feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        X = df[feature_columns]
        
        # 簡単なターゲット作成（価格上昇/下降/横ばい）
        price_change = df['Close'].pct_change().fillna(0)
        y = pd.cut(price_change, bins=[-np.inf, -0.01, 0.01, np.inf], labels=[0, 1, 2])
        y = y.astype(int)
        
        # 訓練・テストデータに分割
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"学習データ: X_train={X_train.shape}, y_train={y_train.shape}")
        print(f"クラス分布: {dict(y_train.value_counts())}")
        
        # モデル学習
        print("TabNet学習開始...")
        training_result = trainer.train_model(X_train, X_test, y_train, y_test)
        print(f"学習完了: {training_result}")
        
        # 特徴量重要度を直接取得
        print("\n2. 直接特徴量重要度取得")
        feature_importance = trainer.get_feature_importance(top_n=5)
        if feature_importance:
            print("✅ 直接取得成功:")
            for feature, importance in feature_importance.items():
                print(f"  {feature}: {importance:.4f}")
        else:
            print("❌ 直接取得失敗")
        
        # モデル保存
        print("\n3. モデル保存（特徴量重要度含む）")
        model_path = trainer.save_model("test_tabnet_model")
        print(f"モデル保存完了: {model_path}")
        
        # MLOrchestratorを使用した取得テスト
        print("\n4. MLOrchestrator経由での特徴量重要度取得")
        from backend.app.services.auto_strategy.services.ml_orchestrator import MLOrchestrator
        
        orchestrator = MLOrchestrator()
        api_feature_importance = orchestrator.get_feature_importance(top_n=5)
        
        if api_feature_importance:
            print("✅ MLOrchestrator取得成功:")
            for feature, importance in api_feature_importance.items():
                print(f"  {feature}: {importance:.4f}")
        else:
            print("❌ MLOrchestrator取得失敗")
        
        return {
            "direct_success": bool(feature_importance),
            "direct_count": len(feature_importance) if feature_importance else 0,
            "api_success": bool(api_feature_importance),
            "api_count": len(api_feature_importance) if api_feature_importance else 0,
            "model_path": model_path
        }
        
    except Exception as e:
        print(f"❌ テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

def test_api_endpoint():
    """APIエンドポイントをテスト"""
    try:
        print("\n5. APIエンドポイントテスト")
        
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
    print("🚀 TabNet特徴量重要度統合テスト開始")
    
    # 1. 学習と特徴量重要度取得テスト
    result = test_tabnet_training_and_feature_importance()
    
    # 2. APIエンドポイントテスト
    api_success = test_api_endpoint()
    
    # 結果サマリー
    print(f"\n{'='*50}")
    print("🏁 テスト結果サマリー")
    print(f"{'='*50}")
    
    if "error" not in result:
        print(f"✅ 直接取得: {result['direct_success']} ({result['direct_count']}個)")
        print(f"✅ MLOrchestrator: {result['api_success']} ({result['api_count']}個)")
        print(f"✅ API経由: {api_success}")
        print(f"📁 保存モデル: {result.get('model_path', 'N/A')}")
        
        if result['direct_success'] and result['api_success'] and api_success:
            print("\n🎉 全てのテストが成功しました！")
        else:
            print("\n⚠️  一部のテストが失敗しました")
    else:
        print(f"❌ テスト実行エラー: {result['error']}")
    
    print(f"\n🎯 テスト完了!")
