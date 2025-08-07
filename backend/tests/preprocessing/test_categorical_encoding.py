#!/usr/bin/env python3
"""
カテゴリカル変数エンコーディング機能のテストスクリプト

このスクリプトは、fear_greed_classification カラムの文字列値が
正しく数値にエンコーディングされることを確認します。
"""

import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import logging
from app.utils.data_processing import DataProcessor

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_fear_greed_classification_encoding():
    """Fear & Greed Classification のエンコーディングテスト"""
    logger.info("=== Fear & Greed Classification エンコーディングテスト開始 ===")
    
    try:
        # テストデータを作成
        test_data = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104],
            'Volume': [1000, 1100, 1200, 1300, 1400],
            'fear_greed_value': [25, 35, 50, 65, 85],
            'fear_greed_classification': ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
        })
        
        logger.info("テストデータ（エンコーディング前）:")
        logger.info(f"\n{test_data}")
        logger.info(f"データ型:\n{test_data.dtypes}")
        
        # DataProcessorを初期化
        processor = DataProcessor()
        
        # カテゴリカル変数エンコーディングを実行
        encoded_data = processor._encode_categorical_variables(test_data)
        
        logger.info("エンコーディング後のデータ:")
        logger.info(f"\n{encoded_data}")
        logger.info(f"データ型:\n{encoded_data.dtypes}")
        
        # 結果検証
        expected_mapping = {
            'Extreme Fear': 0,
            'Fear': 1,
            'Neutral': 2,
            'Greed': 3,
            'Extreme Greed': 4
        }
        
        # fear_greed_classification が数値になっていることを確認
        assert 'fear_greed_classification' in encoded_data.columns
        assert encoded_data['fear_greed_classification'].dtype in ['int64', 'float64']
        
        # エンコーディング結果が期待値と一致することを確認
        expected_values = [0, 1, 2, 3, 4]
        actual_values = encoded_data['fear_greed_classification'].tolist()
        
        assert actual_values == expected_values, f"期待値: {expected_values}, 実際: {actual_values}"
        
        logger.info("✅ Fear & Greed Classification エンコーディングテスト成功")
        return True
        
    except Exception as e:
        logger.error(f"❌ テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_missing_values_handling():
    """欠損値処理のテスト"""
    logger.info("=== 欠損値処理テスト開始 ===")
    
    try:
        # 欠損値を含むテストデータを作成
        test_data = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104],
            'Volume': [1000, 1100, 1200, 1300, 1400],
            'fear_greed_value': [25, 35, np.nan, 65, 85],
            'fear_greed_classification': ['Extreme Fear', None, 'Neutral', np.nan, 'Extreme Greed']
        })
        
        logger.info("テストデータ（欠損値あり）:")
        logger.info(f"\n{test_data}")
        
        # DataProcessorを初期化
        processor = DataProcessor()
        
        # カテゴリカル変数エンコーディングを実行
        encoded_data = processor._encode_categorical_variables(test_data)
        
        logger.info("エンコーディング後のデータ:")
        logger.info(f"\n{encoded_data}")
        
        # 欠損値が適切に処理されていることを確認
        assert not encoded_data['fear_greed_classification'].isna().any(), "欠損値が残っています"
        
        # 欠損値が 'Neutral' (2) にエンコーディングされていることを確認
        expected_values = [0, 2, 2, 2, 4]  # None と np.nan は 'Neutral' (2) になる
        actual_values = encoded_data['fear_greed_classification'].tolist()
        
        assert actual_values == expected_values, f"期待値: {expected_values}, 実際: {actual_values}"
        
        logger.info("✅ 欠損値処理テスト成功")
        return True
        
    except Exception as e:
        logger.error(f"❌ テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_preprocess_features_integration():
    """preprocess_features メソッドの統合テスト"""
    logger.info("=== preprocess_features 統合テスト開始 ===")
    
    try:
        # 実際のMLトレーニングで使用されるようなデータを作成
        test_data = pd.DataFrame({
            'Close': [100.0, 101.5, 99.8, 103.2, 102.1],
            'Volume': [1000.0, 1100.0, 1200.0, 1300.0, 1400.0],
            'RSI': [45.2, 52.1, 38.9, 67.3, 55.8],
            'MACD': [0.5, -0.2, 0.8, -0.3, 0.1],
            'fear_greed_value': [25.0, 35.0, 50.0, 65.0, 85.0],
            'fear_greed_classification': ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
        })
        
        logger.info("統合テストデータ:")
        logger.info(f"\n{test_data}")
        logger.info(f"データ型:\n{test_data.dtypes}")
        
        # DataProcessorを初期化
        processor = DataProcessor()
        
        # 包括的前処理を実行
        processed_data = processor.preprocess_features(
            test_data,
            imputation_strategy="median",
            scale_features=False,  # スケーリングは無効にしてテストを簡単に
            remove_outliers=False
        )
        
        logger.info("前処理後のデータ:")
        logger.info(f"\n{processed_data}")
        logger.info(f"データ型:\n{processed_data.dtypes}")
        
        # すべてのカラムが数値型になっていることを確認
        for col in processed_data.columns:
            assert processed_data[col].dtype in ['int64', 'float64'], f"カラム {col} が数値型ではありません: {processed_data[col].dtype}"
        
        # fear_greed_classification が正しくエンコーディングされていることを確認
        expected_fg_values = [0, 1, 2, 3, 4]
        actual_fg_values = processed_data['fear_greed_classification'].tolist()
        assert actual_fg_values == expected_fg_values, f"Fear & Greed エンコーディングが不正: {actual_fg_values}"
        
        logger.info("✅ preprocess_features 統合テスト成功")
        return True
        
    except Exception as e:
        logger.error(f"❌ テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メインテスト実行"""
    logger.info("カテゴリカル変数エンコーディング機能のテストを開始します")
    
    tests = [
        test_fear_greed_classification_encoding,
        test_missing_values_handling,
        test_preprocess_features_integration
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        print("\n" + "="*50)
        if test_func():
            passed += 1
        else:
            failed += 1
    
    print("\n" + "="*50)
    logger.info(f"テスト結果: 成功 {passed}件, 失敗 {failed}件")
    
    if failed == 0:
        logger.info("🎉 すべてのテストが成功しました！")
        logger.info("カテゴリカル変数エンコーディング機能は正常に動作しています。")
    else:
        logger.error("❌ 一部のテストが失敗しました。")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
