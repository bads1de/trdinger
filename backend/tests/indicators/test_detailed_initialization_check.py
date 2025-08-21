"""
インジケータの初期化テスト詳細確認
各インジケータの初期化成功/失敗を詳細に確認するテスト
"""

import numpy as np
import pandas as pd
import pytest
from typing import Dict, List

from app.services.indicators import TechnicalIndicatorService
from app.services.indicators.config import indicator_registry


def make_test_df(n=300):
    """テスト用データフレーム作成"""
    idx = pd.date_range("2024-01-01", periods=n, freq="H")
    # 緩やかなトレンドと妥当な範囲のダミーデータ
    open_ = np.linspace(100, 120, n)
    high = open_ + 1.0
    low = open_ - 1.0
    close = open_ + 0.2
    volume = np.full(n, 1000, dtype=float)
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=idx,
    )


def _default_params(config):
    """デフォルトパラメータ生成"""
    params = {}
    for name, pconf in config.parameters.items():
        params[name] = pconf.default_value
    return params


def test_detailed_initialization_analysis():
    """詳細な初期化分析テスト"""
    df = make_test_df()
    svc = TechnicalIndicatorService()

    # 全インジケータを取得
    all_indicators = []
    for name, config in indicator_registry._configs.items():
        if config.adapter_function:
            all_indicators.append((name, config))

    results = {
        'success': [],
        'failed': [],
        'error_details': {}
    }

    print(f"\n=== インジケータ初期化詳細分析 ===")
    print(f"総インジケータ数: {len(all_indicators)}")

    for name, config in all_indicators:
        try:
            params = _default_params(config)
            result = svc.calculate_indicator(df, name, params)

            if result is not None:
                results['success'].append(name)
                print(f"PASS {name}: 成功")
            else:
                results['failed'].append(name)
                results['error_details'][name] = "結果がNone"
                print(f"FAIL {name}: 結果がNone")

        except Exception as e:
            results['failed'].append(name)
            results['error_details'][name] = str(e)
            print(f"FAIL {name}: {str(e)[:100]}...")

    # 結果サマリー
    print(f"\n=== 結果サマリー ===")
    print(f"成功: {len(results['success'])}個")
    print(f"失敗: {len(results['failed'])}個")
    print(f"成功率: {len(results['success']) / len(all_indicators) * 100:.1f}%")

    print(f"\n=== 成功したインジケータ ===")
    for name in results['success']:
        print(f"  {name}")

    print(f"\n=== 失敗したインジケータ ===")
    for name in results['failed']:
        print(f"  {name}: {results['error_details'][name][:80]}...")

    # アサーション
    success_rate = len(results['success']) / len(all_indicators)
    print(f"\n成功率: {success_rate * 100:.1f}%")

    # 少なくとも80%は成功すべき
    assert success_rate >= 0.8, f"成功率が80%未満: {success_rate * 100:.1f}%"

    # 主要インジケータは必ず成功すべき
    critical_indicators = ['RSI', 'SMA', 'EMA', 'MACD', 'BB', 'ATR']
    for indicator in critical_indicators:
        assert indicator in results['success'], f"重要なインジケータ {indicator} が失敗"

    return results


def test_specific_indicator_debug():
    """特定のインジケータをデバッグ"""
    df = make_test_df()
    svc = TechnicalIndicatorService()

    # デバッグしたいインジケータを指定
    debug_indicators = ['CHOP', 'VORTEX', 'RSI_EMA_CROSS', 'RVI']

    for name in debug_indicators:
        print(f"\n=== {name} デバッグ ===")
        try:
            config = indicator_registry.get_indicator_config(name)
            if not config:
                print(f"  設定が見つかりません")
                continue

            params = _default_params(config)
            print(f"  パラメータ: {params}")
            print(f"  必要なデータ: {config.required_data}")
            print(f"  param_map: {getattr(config, 'param_map', None)}")

            result = svc.calculate_indicator(df, name, params)
            print(f"  結果: {type(result)}, shape: {getattr(result, 'shape', 'N/A')}")

        except Exception as e:
            print(f"  エラー: {str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    test_detailed_initialization_analysis()
    test_specific_indicator_debug()