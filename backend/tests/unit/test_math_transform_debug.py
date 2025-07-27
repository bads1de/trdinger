#!/usr/bin/env python3
"""
数学変換系指標のNaN問題デバッグ用テストスクリプト

このスクリプトは数学系指標でNaNが発生する問題を再現し、
修正後の動作を検証するために使用します。
"""

import sys
import os
import numpy as np
import logging

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from app.services.indicators.technical_indicators.math_transform import MathTransformIndicators

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_acos_nan_issue():
    """ACOS関数のNaN問題をテスト"""
    print("\n=== ACOS関数テスト ===")
    
    # 正常な範囲のデータ（[-1, 1]）
    normal_data = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    print(f"正常データ: {normal_data}")
    
    try:
        result = MathTransformIndicators.acos(normal_data)
        print(f"正常データ結果: {result}")
        print(f"NaN数: {np.sum(np.isnan(result))}")
    except Exception as e:
        print(f"正常データエラー: {e}")
    
    # 範囲外のデータ（[-1, 1]外）
    invalid_data = np.array([-2.0, -1.5, 0.0, 1.5, 2.0])
    print(f"\n範囲外データ: {invalid_data}")
    
    try:
        result = MathTransformIndicators.acos(invalid_data)
        print(f"範囲外データ結果: {result}")
        print(f"NaN数: {np.sum(np.isnan(result))}")
    except Exception as e:
        print(f"範囲外データエラー: {e}")

def test_asin_nan_issue():
    """ASIN関数のNaN問題をテスト"""
    print("\n=== ASIN関数テスト ===")
    
    # 範囲外のデータ（[-1, 1]外）
    invalid_data = np.array([-2.0, -1.5, 0.0, 1.5, 2.0])
    print(f"範囲外データ: {invalid_data}")
    
    try:
        result = MathTransformIndicators.asin(invalid_data)
        print(f"範囲外データ結果: {result}")
        print(f"NaN数: {np.sum(np.isnan(result))}")
    except Exception as e:
        print(f"範囲外データエラー: {e}")

def test_ln_nan_issue():
    """LN関数のNaN問題をテスト"""
    print("\n=== LN関数テスト ===")
    
    # 負の値と0を含むデータ
    invalid_data = np.array([-1.0, 0.0, 0.5, 1.0, 2.0])
    print(f"負の値・0を含むデータ: {invalid_data}")
    
    try:
        result = MathTransformIndicators.ln(invalid_data)
        print(f"結果: {result}")
        print(f"NaN数: {np.sum(np.isnan(result))}")
        print(f"無限大数: {np.sum(np.isinf(result))}")
    except Exception as e:
        print(f"エラー: {e}")

def test_log10_nan_issue():
    """LOG10関数のNaN問題をテスト"""
    print("\n=== LOG10関数テスト ===")
    
    # 負の値と0を含むデータ
    invalid_data = np.array([-1.0, 0.0, 0.1, 1.0, 10.0])
    print(f"負の値・0を含むデータ: {invalid_data}")
    
    try:
        result = MathTransformIndicators.log10(invalid_data)
        print(f"結果: {result}")
        print(f"NaN数: {np.sum(np.isnan(result))}")
        print(f"無限大数: {np.sum(np.isinf(result))}")
    except Exception as e:
        print(f"エラー: {e}")

def test_sqrt_nan_issue():
    """SQRT関数のNaN問題をテスト"""
    print("\n=== SQRT関数テスト ===")
    
    # 負の値を含むデータ
    invalid_data = np.array([-4.0, -1.0, 0.0, 1.0, 4.0])
    print(f"負の値を含むデータ: {invalid_data}")
    
    try:
        result = MathTransformIndicators.sqrt(invalid_data)
        print(f"結果: {result}")
        print(f"NaN数: {np.sum(np.isnan(result))}")
    except Exception as e:
        print(f"エラー: {e}")

def test_realistic_price_data():
    """実際の価格データに近いテストケース"""
    print("\n=== 実際の価格データテスト ===")
    
    # 実際の価格データを模擬（正規化されていない）
    price_data = np.array([100.0, 105.0, 98.0, 102.0, 110.0, 95.0, 108.0])
    print(f"価格データ: {price_data}")
    
    # 正規化（-1から1の範囲に変換）
    normalized_data = (price_data - np.mean(price_data)) / np.std(price_data)
    print(f"正規化データ: {normalized_data}")
    print(f"範囲: [{np.min(normalized_data):.3f}, {np.max(normalized_data):.3f}]")
    
    # ACOS関数でテスト
    try:
        result = MathTransformIndicators.acos(normalized_data)
        print(f"ACOS結果: {result}")
        print(f"NaN数: {np.sum(np.isnan(result))}")
    except Exception as e:
        print(f"ACOSエラー: {e}")

def main():
    """メイン実行関数"""
    print("数学変換系指標のNaN問題デバッグテスト開始")
    print("=" * 50)
    
    test_acos_nan_issue()
    test_asin_nan_issue()
    test_ln_nan_issue()
    test_log10_nan_issue()
    test_sqrt_nan_issue()
    test_realistic_price_data()
    
    print("\n" + "=" * 50)
    print("テスト完了")

if __name__ == "__main__":
    main()
