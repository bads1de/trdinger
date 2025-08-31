#!/usr/bin/env python3
"""
修正された指標のテストスクリプト
VWMA, ROC系指標, LINREG系の函数マッピング修正を確認する
"""
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# バックエンドパスを追加
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

try:
    from app.services.indicators.config import indicator_registry
    from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService
    print("モジュールのインポートに成功しました")
except ImportError as e:
    print(f"モジュールのインポートに失敗しました: {e}")
    sys.exit(1)

def test_fixed_indicators():
    """修正された指標のテスト"""
    print("=" * 60)
    print("修正された指標のテスト開始")
    print("=" * 60)

    # テストデータ生成
    np.random.seed(42)  # 固定シード
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    data = {
        "open": 100 + np.random.randn(100) * 5,
        "high": 105 + np.random.randn(100) * 5,
        "low": 95 + np.random.randn(100) * 5,
        "close": 100 + np.random.randn(100) * 5,
        "volume": np.random.randint(1000, 10000, 100),
    }
    df = pd.DataFrame(data, index=dates)

    print(f"テストデータ作成完了: {len(df)}行")
    print(f"カラム: {list(df.columns)}")

    # テストする指標リスト
    test_indicators = [
        "VWMA",      # 修正済み
        "ROC",       # 修正済み
        "ROCP",      # 修正済み
        "ROCR",      # 修正済み
        "ROCR100",   # 修正済み
        "LINREG",    # 修正済み
        "LINREG_SLOPE",    # 修正済み
        "LINREG_INTERCEPT", # 修正済み
        "LINREG_ANGLE",     # 修正済み
    ]

    calculation_service = TechnicalIndicatorService()

    success_count = 0
    fail_count = 0

    for indicator_name in test_indicators:
        print(f"\n--- {indicator_name} のテスト ---")
        try:
            # レジストリから設定を取得
            config = indicator_registry.get_indicator_config(indicator_name)
            if config is None:
                print(f"[ERROR] {indicator_name}: 設定が見つかりません")
                fail_count += 1
                continue

            print(f"設定取得成功: {config.adapter_function.__name__}")
            print(f"param_map: {config.param_map}")

            # パラメータを取得（デフォルト値を使用）
            params = {}
            if hasattr(config, 'parameters') and config.parameters:
                if isinstance(config.parameters, dict):
                    # dict形式の場合（実際の設定に基づく）
                    params = config.parameters.copy()
                    print(f"  パラメータ (dict): {params}")
                elif isinstance(config.parameters, list):
                    # list形式の場合
                    for param in config.parameters:
                        if hasattr(param, 'name') and hasattr(param, 'default_value'):
                            params[param.name] = param.default_value
                            print(f"  パラメータ {param.name}: {param.default_value}")
                        else:
                            print(f"  未対応のパラメータ形式: {type(param)}")
                else:
                    print(f"  未対応のパラメータ構造: {type(config.parameters)}")
            else:
                print(f"  パラメータなし")
            print(f"  設定されたパラメータ数: {len(params)}")

            # 指標計算 - 直接関数呼び出し
            try:
                # 関数を直接呼び出し
                indicator_function = config.adapter_function
                mapped_params = {}

                # パラメータマッピング
                if hasattr(config, 'param_map') and config.param_map:
                    for config_param, func_param in config.param_map.items():
                        if func_param == 'data':
                            if config_param in df.columns:
                                mapped_params[func_param] = df[config_param]
                        elif len(func_param) > 1 and func_param.endswith('_'):
                            # 特別なパラメータマッピング（例: open_）
                            if func_param[:-1] in df.columns:
                                mapped_params[func_param] = df[func_param[:-1]]
                        elif config_param in df.columns:
                            mapped_params[func_param] = df[config_param]
                        elif config_param in params:
                            mapped_params[func_param] = params[config_param]

                # パラメータマッピングの修正
                if hasattr(config, 'param_map') and config.param_map:
                    for param_key, func_param in config.param_map.items():
                        if func_param == 'data':
                            if param_key in df.columns:
                                mapped_params[func_param] = df[param_key]
                        elif func_param in params:
                            mapped_params[func_param] = params[func_param]
                        elif param_key in params:
                            mapped_params[func_param] = params[param_key]

                # パラメータが欠落している場合はデフォルト値を設定
                if not mapped_params and hasattr(config, 'parameters') and config.parameters:
                    for param_name, param_config in config.parameters.items():
                        if hasattr(param_config, 'default_value'):
                            mapped_params[param_name] = param_config.default_value

                print(f"mapped_params: {list(mapped_params.keys())}")
                result = indicator_function(**mapped_params)

            except Exception as calc_error:
                print(f"関数呼び出しエラー: {str(calc_error)}")
                result = None

            if result is not None and not result.empty:
                print("[OK] 計算成功")
                if isinstance(result, pd.Series):
                    print(f"   結果サイズ: {len(result)}")
                else:
                    print(f"   結果サイズ: {result.shape}")
                    if hasattr(result, 'shape') and len(result.shape) > 1:
                        print(f"   出力数: {result.shape[1] if len(result.shape) > 1 else 1}")
                success_count += 1
            else:
                print("[FAIL] 計算失敗または結果が空です")
                fail_count += 1

        except Exception as e:
            print(f"[ERROR] {indicator_name}: 計算中にエラーが発生しました")
            print(f"   エラー: {str(e)}")
            import traceback
            traceback.print_exc()
            fail_count += 1

    print("\n" + "=" * 60)
    print("テスト結果まとめ")
    print("=" * 60)
    print(f"[OK] 成功: {success_count}")
    print(f"[FAIL] 失敗: {fail_count}")
    print(f"成功率: {100.0 * success_count / (success_count + fail_count):.1f}%")

    if fail_count > 0:
        print(f"\n[WARNING] 一部の指標で修正が不完全である可能性があります。{fail_count}個失敗")
        return False
    else:
        print("\n[SUCCESS] すべての修正された指標で計算が成功しました！")
        return True

if __name__ == "__main__":
    success = test_fixed_indicators()
    sys.exit(0 if success else 1)