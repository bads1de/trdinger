#!/usr/bin/env python3
"""
特徴量重要度付きMLモデル学習テストスクリプト

新しいモデルを学習して、特徴量重要度が正しく保存されるかテストします。
"""

import sys
import os
import logging
import pandas as pd
import numpy as np

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.ml.ml_training_service import MLTrainingService
from app.services.ml.model_manager import model_manager

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_data():
    """サンプルデータを作成"""
    np.random.seed(42)
    
    # 100行のサンプルOHLCVデータを作成
    dates = pd.date_range('2024-01-01', periods=100, freq='1H')
    
    # 基本的なOHLCVデータ（大文字のカラム名を使用）
    data = {
        'timestamp': dates,
        'Open': np.random.uniform(40000, 50000, 100),
        'High': np.random.uniform(50000, 55000, 100),
        'Low': np.random.uniform(35000, 40000, 100),
        'Close': np.random.uniform(40000, 50000, 100),
        'Volume': np.random.uniform(100, 1000, 100),
    }
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    # Closeの変化に基づいてターゲットを作成
    df['price_change'] = df['Close'].pct_change()
    df['target'] = 0  # range
    df.loc[df['price_change'] > 0.02, 'target'] = 1  # up
    df.loc[df['price_change'] < -0.02, 'target'] = 2  # down
    
    return df

def test_ml_training_with_feature_importance():
    """特徴量重要度付きMLモデル学習のテスト"""
    try:
        print("=" * 60)
        print("特徴量重要度付きMLモデル学習テスト")
        print("=" * 60)
        
        # 1. サンプルデータを作成
        print("\n1. サンプルデータの作成:")
        training_data = create_sample_data()
        print(f"  データサイズ: {len(training_data)}行")
        print(f"  カラム: {list(training_data.columns)}")
        print(f"  ターゲット分布: {training_data['target'].value_counts().to_dict()}")
        
        # 2. MLTrainingServiceを初期化
        print("\n2. MLTrainingServiceの初期化:")
        ml_service = MLTrainingService(trainer_type="ensemble")
        print(f"  トレーナータイプ: {ml_service.trainer_type}")
        
        # 3. モデルを学習
        print("\n3. モデル学習の実行:")
        training_params = {
            'test_size': 0.3,
            'random_state': 42,
        }
        
        result = ml_service.train_model(
            training_data=training_data,
            save_model=True,
            model_name="test_feature_importance_model",
            **training_params
        )
        
        print(f"  学習結果: {result.get('success', False)}")
        print(f"  精度: {result.get('accuracy', 0.0):.4f}")
        print(f"  F1スコア: {result.get('f1_score', 0.0):.4f}")
        
        # 4. 特徴量重要度の確認
        print("\n4. 学習結果の特徴量重要度:")
        feature_importance = result.get('feature_importance', {})
        if not feature_importance:
            print("  学習結果に特徴量重要度が含まれていません")
        else:
            print(f"  特徴量重要度の数: {len(feature_importance)}")
            sorted_importance = sorted(
                feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
            
            print("\n  上位10個の特徴量重要度:")
            for i, (feature, importance) in enumerate(sorted_importance, 1):
                print(f"    {i:2d}. {feature}: {importance:.6f}")
        
        # 5. 保存されたモデルから特徴量重要度を確認
        print("\n5. 保存されたモデルの特徴量重要度:")
        # 最新のモデル（パターンを広く指定）
        latest_model_path = model_manager.get_latest_model("*")
        if latest_model_path:
            print(f"  最新モデル: {os.path.basename(latest_model_path)}")
            
            model_data = model_manager.load_model(latest_model_path)
            if model_data:
                metadata = model_data.get("metadata", {})
                saved_feature_importance = metadata.get("feature_importance", {})
                
                if not saved_feature_importance:
                    print("  保存されたモデルに特徴量重要度が含まれていません")
                else:
                    print(f"  保存された特徴量重要度の数: {len(saved_feature_importance)}")
                    sorted_saved = sorted(
                        saved_feature_importance.items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )[:5]
                    
                    print("\n  保存された上位5個の特徴量重要度:")
                    for i, (feature, importance) in enumerate(sorted_saved, 1):
                        print(f"    {i:2d}. {feature}: {importance:.6f}")
        else:
            print("  最新モデルが見つかりません")
        
        print("\n" + "=" * 60)
        print("テスト完了")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ml_training_with_feature_importance()
