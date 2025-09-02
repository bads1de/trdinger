#!/usr/bin/env python3
"""
全17指標の初期化と計算検証テストスクリプト
"""
import pandas as pd
import numpy as np
import logging
from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService

# 全17指標リスト
MAIN_INDICATORS = [
    'RSI', 'STOCH', 'STOCHRSI', 'KDJ', 'QQE', 'CMO', 'TRIX', 'ULTOSC',
    'BOP', 'APO', 'PPO', 'MFI', 'ADX', 'CCI', 'WILLR', 'MACD', 'WILLIAMS_PERCENT_R'
]

def create_sample_data(periods=100):
    """サンプルデータを生成"""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=periods, freq='D')

    # 基本価格データ
    close = 100 + np.cumsum(np.random.randn(periods) * 2)
    high = close + np.abs(np.random.randn(periods) * 5)
    low = close - np.abs(np.random.randn(periods) * 5)
    open_price = close + np.random.randn(periods) * 2
    volume = np.abs(np.random.randn(periods) * 1000) + 1000

    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })

    return df

def test_indicator_initialization():
    """指標初期化テスト"""
    print("[TEST] 全17指標初期化テスト開始")
    service = TechnicalIndicatorService()

    # 利用可能指標確認
    available_indicators = service.get_supported_indicators()
    print(f"[INFO] 利用可能指標数: {len(available_indicators)}")

    # テスト対象指標の利用可能性確認
    test_results = []
    for indicator in MAIN_INDICATORS:
        available = indicator in available_indicators
        test_results.append({
            'indicator': indicator,
            'available': available
        })
        status = "[OK]" if available else "[NG]"
        print(f"{status} {indicator}: {'利用可能' if available else '未利用可能'}")

    # 利用不可能な指標
    unavailable = [r for r in test_results if not r['available']]
    if unavailable:
        print(f"\n[WARN] 利用不可能な指標: {', '.join([u['indicator'] for u in unavailable])}")
    else:
        print("\n[SUCCESS] 全指標利用可能です！")

    return test_results

def test_indicator_calculation(sample_df):
    """指標計算テスト"""
    print("\n[CALC] 指標計算テスト開始")
    service = TechnicalIndicatorService()

    calc_results = []
    error_details = []

    for indicator in MAIN_INDICATORS:
        try:
            # デフォルトパラメータで計算テスト
            params = {}  # デフォルトパラメータ
            result = service.calculate_indicator(sample_df, indicator, params)

            if result is None:
                status = "[FAIL] None"
                has_nan = True
                error_msg = "result is None"
            elif isinstance(result, tuple):
                status = f"[OK] tuple({len(result)})"
                # NaNチェック
                has_nan = any(pd.isna(pd.Series(r)).any() for r in result if hasattr(r, '__iter__'))
                error_msg = "NaN detected" if has_nan else None
            else:
                status = "[OK] single"
                has_nan = pd.isna(result).any() if hasattr(result, '__iter__') else False
                error_msg = "NaN detected" if has_nan else None

            calc_results.append({
                'indicator': indicator,
                'status': status,
                'nan_detected': has_nan,
                'error': error_msg,
                'sample_value': None  # 後で追加
            })

            print(f"{status} {indicator}: {'OK' if not has_nan else 'NaN detected'}")

            # サンプル値取得（最初の有効値）
            try:
                if isinstance(result, tuple):
                    calc_results[-1]['sample_value'] = str(tuple(float(r.iloc[20]) for r in result if len(r) > 20))
                else:
                    calc_results[-1]['sample_value'] = float(result.iloc[20]) if len(result) > 20 else None
            except Exception:
                pass

        except Exception as e:
            error_details.append({
                'indicator': indicator,
                'error_type': type(e).__name__,
                'error_msg': str(e)
            })
            calc_results.append({
                'indicator': indicator,
                'status': f"[ERROR] {type(e).__name__}",
                'nan_detected': True,
                'error': str(e),
                'sample_value': None
            })
            print(f"[ERROR] {indicator}: {type(e).__name__} - {str(e)[:50]}...")

    # エラーダンプ
    if error_details:
        print(f"\n[ERRORS] エラーレポート ({len(error_details)} indicators)")
        for error in error_details:
            print(f"   {error['indicator']}: {error['error_type']} - {error['error_msg'][:100]}...")

    return calc_results, error_details

def generate_summary(initialization_results, calculation_results, error_details):
    """詳細結果まとめ"""
    print("\n[SUMMARY] 総合サマリーレポート")
    print("=" * 50)

    # 初期化統計
    total_indicators = len(MAIN_INDICATORS)
    available_count = sum(1 for r in initialization_results if r['available'])
    print(f"全指標数: {total_indicators}")
    print(f"利用可能指標数: {available_count}")
    print(".1f")

    # 計算統計
    successful_calcs = sum(1 for r in calculation_results if not r['nan_detected'] and '[ERROR]' not in r['status'])
    nan_detected = sum(1 for r in calculation_results if r['nan_detected'])
    error_count = sum(1 for r in calculation_results if '[ERROR]' in r['status'])

    print(f"計算成功指標数: {successful_calcs}")
    print(f"NaN検出指標数: {nan_detected}")
    print(f"エラー発生指標数: {error_count}")

    # 詳細結果表示
    print("\n[RESULTS] 詳細結果:")
    for result in calculation_results:
        status_icon = "[OK]" if not result['nan_detected'] and '[ERROR]' not in result['status'] else "[FAIL]"
        print(f"  {status_icon} {result['indicator']}: {result['status']}")
        if result['sample_value'] and not result['nan_detected']:
            print(".6f")

    return {
        'total_indicators': total_indicators,
        'available_count': available_count,
        'successful_calcs': successful_calcs,
        'nan_detected': nan_detected,
        'error_count': error_count,
        'error_details': error_details
    }

def main():
    """メイン関数"""
    logging.basicConfig(level=logging.WARNING)  # エラーログのみ

    # 初期化テスト
    init_results = test_indicator_initialization()

    # サンプルデータ作成
    print("\n[DATA] サンプルデータ作成中...")
    sample_data = create_sample_data(100)
    print(f"   データ期間: {len(sample_data)}日（OHLCV）")
    print(f"   価格範囲: {sample_data['close'].min():.2f} - {sample_data['close'].max():.2f}")

    # 計算テスト
    calc_results, error_details = test_indicator_calculation(sample_data)

    # まとめ
    summary = generate_summary(init_results, calc_results, error_details)

    print("\n[COMPLETE] テスト完了")
    print(f"検証完了指標: {summary['available_count'] - summary['error_count']} / {summary['total_indicators']}")
    print(f"正常計算指標: {summary['successful_calcs']} / {summary['available_count']}")

    return summary

if __name__ == "__main__":
    result = main()