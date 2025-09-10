#!/usr/bin/env python3
"""
pandas-ta設定検証スクリプト

保持指標リストとPANDAS_TA_CONFIGを比較し、欠けている項目を表示する。
"""

import sys
import os

# バックエンドディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.indicators.config.indicator_definitions import PANDAS_TA_CONFIG, POSITIONAL_DATA_FUNCTIONS

def main():
    # 計画の保持指標リスト
    keep_momentum = ['RSI', 'MACD', 'STOCH', 'WILLR', 'CCI', 'ROC', 'MOM', 'ADX', 'QQE', 'SQUEEZE']
    keep_trend = ['SMA', 'EMA', 'WMA', 'DEMA', 'TEMA', 'T3', 'KAMA', 'SAR']
    keep_volatility = ['ATR', 'BBANDS', 'KELTNER', 'DONCHIAN', 'SUPERTREND', 'ACCBANDS', 'UI']
    keep_volume = ['OBV', 'AD', 'ADOSC', 'CMF', 'EFI', 'MFI', 'VWAP']

    keep_all = keep_momentum + keep_trend + keep_volatility + keep_volume

    print("=" * 50)
    print("pandas-ta設定検証結果")
    print("=" * 50)

    print("\n1. 保持指標リスト:")
    print(f"総数: {len(keep_all)}")
    print(f"Momentum: {len(keep_momentum)} - {keep_momentum}")
    print(f"Trend: {len(keep_trend)} - {keep_trend}")
    print(f"Volatility: {len(keep_volatility)} - {keep_volatility}")
    print(f"Volume: {len(keep_volume)} - {keep_volume}")

    print("\n2. PANDAS_TA_CONFIG内容:")
    config_items = list(PANDAS_TA_CONFIG.keys())
    print(f"総数: {len(config_items)}")
    print(f"項目: {config_items}")

    print("\n3. POSITIONAL_DATA_FUNCTIONS内容:")
    print(f"総数: {len(POSITIONAL_DATA_FUNCTIONS)}")
    print(f"項目: {POSITIONAL_DATA_FUNCTIONS}")

    # 欠けている項目の検出
    missing_from_config = [item for item in keep_all if item not in PANDAS_TA_CONFIG]
    extra_in_config = [item for item in PANDAS_TA_CONFIG.keys() if item not in keep_all]

    print("\n4. 比較結果:")
    print(f"計画保持リストからの欠如 (PANDAS_TA_CONFIG): {missing_from_config}")
    print(f"計画保持リストを超える項目 (PANDAS_TA_CONFIG): {extra_in_config}")

    # POSITIONAL_DATA_FUNCTIONSの検証
    config_functions = set(PANDAS_TA_CONFIG[item].get("function", "") for item in PANDAS_TA_CONFIG)
    missing_pos_functions = config_functions - set(POSITIONAL_DATA_FUNCTIONS)
    extra_pos_functions = set(POSITIONAL_DATA_FUNCTIONS) - config_functions

    print("\n5. POSITIONAL_DATA_FUNCTIONS詳細:")
    print(f"PANDAS_TA_CONFIGの関数: {config_functions}")
    print(f"欠如 (POSITIONAL_DATA_FUNCTIONS): {missing_pos_functions}")
    print(f"超過 (POSITIONAL_DATA_FUNCTIONS): {extra_pos_functions}")

    if missing_from_config or missing_pos_functions:
        print("\n" + "=" * 50)
        print("WARNING: 問題点:")
        if missing_from_config:
            print(f"- PANDAS_TA_CONFIGから欠けている保持指標: {missing_from_config}")
        if missing_pos_functions:
            print(f"- POSITIONAL_DATA_FUNCTIONSから欠けている関数: {missing_pos_functions}")
        if extra_in_config:
            print(f"- 計画を超える項目 (削除されたはずの指標): {extra_in_config}")
        print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("SUCCESS: 設定が正しく更新されています!")
        print("=" * 50)

    # SQUEEZEおよびMFIの特別チェック
    print("\n6. 特別確認:")
    critical_items = ['SQUEEZE', 'MFI']
    for item in critical_items:
        if item in PANDAS_TA_CONFIG:
            func = PANDAS_TA_CONFIG[item].get("function", "unknown")
            print(f"OK: {item} 存在 (function: {func})")
        else:
            print(f"ERROR: {item} 欠如")

    print("\n検証完了!")

if __name__ == "__main__":
    main()