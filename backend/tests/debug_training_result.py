"""
学習結果のデバッグスクリプト

training_resultにどのような指標が含まれているかを詳細に調査し、
AUC指標が正しく含まれているかを確認します。
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.services.ml.ml_training_service import MLTrainingService

logger = logging.getLogger(__name__)


def create_sample_training_data():
    """サンプル学習データを作成"""
    np.random.seed(42)
    
    # 1000サンプルのOHLCVデータを作成
    n_samples = 1000
    
    data = {
        "timestamp": pd.date_range("2023-01-01", periods=n_samples, freq="h"),
        "Open": 50000 + np.random.randn(n_samples) * 1000,
        "High": 50000 + np.random.randn(n_samples) * 1000 + 500,
        "Low": 50000 + np.random.randn(n_samples) * 1000 - 500,
        "Close": 50000 + np.random.randn(n_samples) * 1000,
        "Volume": np.random.uniform(100, 1000, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # 価格の整合性を保つ
    for i in range(len(df)):
        prices = [
            df.loc[i, "Open"],
            df.loc[i, "High"],
            df.loc[i, "Low"],
            df.loc[i, "Close"],
        ]
        df.loc[i, "High"] = max(prices)
        df.loc[i, "Low"] = min(prices)
    
    return df


def debug_training_result():
    """学習結果を詳細にデバッグ"""
    print("=" * 80)
    print("学習結果デバッグ")
    print("=" * 80)
    
    try:
        # MLTrainingServiceを初期化
        ml_service = MLTrainingService(trainer_type="ensemble")
        
        # サンプルデータを作成
        print("1. サンプル学習データを作成中...")
        training_data = create_sample_training_data()
        print(f"   データサイズ: {len(training_data)}行")
        
        # モデルを学習（保存はしない）
        print("\n2. モデル学習を開始...")
        result = ml_service.train_model(
            training_data=training_data,
            save_model=False,  # 保存はしない
            model_name=None
        )
        
        print("✅ モデル学習完了")
        
        # 学習結果の詳細分析
        print("\n3. 学習結果の詳細分析:")
        print("-" * 60)
        
        print(f"学習結果のキー: {list(result.keys())}")
        
        # 各キーの値を詳細表示
        for key, value in result.items():
            print(f"\n{key}:")
            if isinstance(value, dict):
                print(f"  型: 辞書 (キー数: {len(value)})")
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float)):
                        print(f"    {sub_key}: {sub_value:.4f}")
                    else:
                        print(f"    {sub_key}: {type(sub_value)} - {sub_value}")
            elif isinstance(value, (int, float)):
                print(f"  値: {value:.4f} (型: {type(value)})")
            elif isinstance(value, np.ndarray):
                print(f"  型: numpy配列 (形状: {value.shape})")
            elif isinstance(value, list):
                print(f"  型: リスト (長さ: {len(value)})")
            else:
                print(f"  値: {value} (型: {type(value)})")
        
        # 重要な指標の確認
        print("\n4. 重要指標の確認:")
        print("-" * 60)
        
        important_metrics = [
            'accuracy', 'precision', 'recall', 'f1_score',
            'balanced_accuracy', 'matthews_corrcoef', 'cohen_kappa',
            'auc_roc', 'auc_pr', 'roc_auc', 'pr_auc',  # 異なる名前の可能性
            'specificity', 'sensitivity', 'npv', 'ppv',
            'log_loss', 'brier_score'
        ]
        
        found_metrics = {}
        for metric in important_metrics:
            if metric in result:
                found_metrics[metric] = result[metric]
                print(f"✅ {metric}: {result[metric]}")
            else:
                print(f"❌ {metric}: 見つかりません")
        
        # AUC指標の特別調査
        print("\n5. AUC指標の特別調査:")
        print("-" * 60)
        
        auc_related_keys = [k for k in result.keys() if 'auc' in k.lower() or 'roc' in k.lower()]
        if auc_related_keys:
            print("AUC関連のキー:")
            for key in auc_related_keys:
                print(f"  {key}: {result[key]}")
        else:
            print("AUC関連のキーが見つかりません")
        
        # 予測確率の確認
        print("\n6. 予測確率の確認:")
        print("-" * 60)
        
        # 少量のテストデータで予測確率を確認
        test_data = training_data.head(10)
        try:
            # 特徴量を計算
            features_df = ml_service.trainer._calculate_features(test_data)
            print(f"特徴量形状: {features_df.shape}")
            
            # 予測確率を取得
            if hasattr(ml_service.trainer, 'model') and ml_service.trainer.model:
                if hasattr(ml_service.trainer.model, 'predict_proba'):
                    pred_proba = ml_service.trainer.model.predict_proba(features_df)
                    print(f"予測確率形状: {pred_proba.shape}")
                    print(f"予測確率サンプル:\n{pred_proba[:3]}")
                else:
                    print("モデルにpredict_probaメソッドがありません")
            else:
                print("学習済みモデルが見つかりません")
        except Exception as e:
            print(f"予測確率確認エラー: {e}")
        
        return result
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("学習結果デバッグ開始")
    print("=" * 80)
    
    result = debug_training_result()
    
    print("\n" + "=" * 80)
    if result:
        print("✅ デバッグ完了")
    else:
        print("❌ デバッグ失敗")
    print("=" * 80)
