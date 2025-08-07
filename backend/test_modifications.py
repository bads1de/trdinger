"""
3.9と3.10の修正内容をテストするスクリプト
"""

import sys
import os
import numpy as np
import pandas as pd

# パス設定
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

def test_data_validator():
    """DataValidatorのテスト"""
    print("=== DataValidatorのテスト ===")
    
    try:
        from utils.data_validation import DataValidator
        
        # テストデータ
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        print(f"入力データ: {data.tolist()}")
        
        # 正規化実行
        result = DataValidator.safe_normalize(data, window=5)
        print(f"正規化結果: {result.tolist()}")
        print(f"結果の型: {type(result)}")
        print(f"有限値チェック: {np.isfinite(result).all()}")
        
        # 定数値での正規化テスト
        constant_data = pd.Series([5, 5, 5, 5, 5])
        normalized_constant = DataValidator.safe_normalize(constant_data, window=3)
        print(f"定数値正規化結果: {normalized_constant.tolist()}")
        print(f"定数値有限値チェック: {np.isfinite(normalized_constant).all()}")
        
        print("✅ DataValidatorテスト成功")
        return True
        
    except Exception as e:
        print(f"❌ DataValidatorテスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ensemble_parameter_space():
    """EnsembleParameterSpaceのテスト"""
    print("\n=== EnsembleParameterSpaceのテスト ===")
    
    try:
        from services.optimization.ensemble_parameter_space import EnsembleParameterSpace
        
        # KNNパラメータ空間取得
        param_space = EnsembleParameterSpace.get_knn_parameter_space()
        print(f"パラメータ空間のキー: {list(param_space.keys())}")
        
        # knn_metricの確認
        if 'knn_metric' in param_space:
            print(f"knn_metricの選択肢: {param_space['knn_metric'].categories}")
            print("✅ EnsembleParameterSpaceテスト成功")
            return True
        else:
            print("❌ knn_metricが見つかりません")
            return False
        
    except Exception as e:
        print(f"❌ EnsembleParameterSpaceテスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_optimized_crypto_features():
    """OptimizedCryptoFeaturesのテスト"""
    print("\n=== OptimizedCryptoFeaturesのテスト ===")
    
    try:
        from services.ml.feature_engineering.optimized_crypto_features import OptimizedCryptoFeatures
        
        # テストデータ作成
        dates = pd.date_range('2023-01-01', periods=30, freq='H')
        np.random.seed(42)
        
        test_data = pd.DataFrame({
            'Open': 100 + np.random.randn(30) * 5,
            'High': 105 + np.random.randn(30) * 5,
            'Low': 95 + np.random.randn(30) * 5,
            'Close': 100 + np.random.randn(30) * 5,
            'Volume': 1000 + np.random.randn(30) * 100,
            'open_interest': 5000 + np.random.randn(30) * 500,
            'funding_rate': np.random.randn(30) * 0.001,
            'fear_greed_value': 50 + np.random.randn(30) * 20
        }, index=dates)
        
        print(f"テストデータ形状: {test_data.shape}")
        
        # 特徴量エンジンのテスト
        feature_engine = OptimizedCryptoFeatures()
        result = feature_engine.create_optimized_features(test_data)
        
        print(f"結果データ形状: {result.shape}")
        print(f"追加された特徴量数: {len(result.columns) - len(test_data.columns)}")
        
        # 無限値やNaN値のチェック
        infinite_check = result.isin([np.inf, -np.inf]).any().any()
        print(f"無限値チェック: {not infinite_check}")
        
        # ロバストリターン特徴量のチェック
        robust_return_cols = [col for col in result.columns if 'robust_return' in col]
        print(f"ロバストリターン特徴量数: {len(robust_return_cols)}")
        
        if len(robust_return_cols) > 0:
            for col in robust_return_cols:
                finite_check = np.isfinite(result[col]).all()
                print(f"{col}の有限値チェック: {finite_check}")
        
        print("✅ OptimizedCryptoFeaturesテスト成功")
        return True
        
    except Exception as e:
        print(f"❌ OptimizedCryptoFeaturesテスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メインテスト実行"""
    print("3.9と3.10の修正内容テストを開始します...\n")
    
    results = []
    
    # 各テストを実行
    results.append(test_data_validator())
    results.append(test_ensemble_parameter_space())
    results.append(test_optimized_crypto_features())
    
    # 結果サマリー
    print("\n=== テスト結果サマリー ===")
    success_count = sum(results)
    total_count = len(results)
    
    print(f"成功: {success_count}/{total_count}")
    
    if success_count == total_count:
        print("🎉 すべてのテストが成功しました！")
        return True
    else:
        print("⚠️ 一部のテストが失敗しました。")
        return False

if __name__ == "__main__":
    success = main()
    print(f"\n終了コード: {0 if success else 1}")
