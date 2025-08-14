#!/usr/bin/env python3
"""
momentum.pyの一括簡素化スクリプト

シンプルなメソッドを一括で簡素化します。
"""

import re


def simplify_single_input_method(method_text):
    """単一入力メソッドの簡素化"""
    # @handle_pandas_ta_errorsデコレータを削除
    method_text = re.sub(r"@handle_pandas_ta_errors\s*\n\s*", "", method_text)

    # ensure_series_minimal_conversionを簡素化
    method_text = re.sub(
        r"series = ensure_series_minimal_conversion\(data\)",
        "series = pd.Series(data) if isinstance(data, np.ndarray) else data",
        method_text,
    )

    # validate_series_dataを削除
    method_text = re.sub(r"\s*validate_series_data\([^)]+\)\s*\n", "", method_text)

    # validate_indicator_parametersを削除
    method_text = re.sub(
        r"\s*validate_indicator_parameters\([^)]+\)\s*\n", "", method_text
    )

    return method_text


def simplify_multi_input_method(method_text):
    """複数入力メソッドの簡素化"""
    # @handle_pandas_ta_errorsデコレータを削除
    method_text = re.sub(r"@handle_pandas_ta_errors\s*\n\s*", "", method_text)

    # ensure_series_minimal_conversionを簡素化
    patterns = [
        (
            r"high_series = ensure_series_minimal_conversion\(high\)",
            "high_series = pd.Series(high) if isinstance(high, np.ndarray) else high",
        ),
        (
            r"low_series = ensure_series_minimal_conversion\(low\)",
            "low_series = pd.Series(low) if isinstance(low, np.ndarray) else low",
        ),
        (
            r"close_series = ensure_series_minimal_conversion\(close\)",
            "close_series = pd.Series(close) if isinstance(close, np.ndarray) else close",
        ),
        (
            r"open_series = ensure_series_minimal_conversion\(open_\)",
            "open_series = pd.Series(open_) if isinstance(open_, np.ndarray) else open_",
        ),
        (
            r"volume_series = ensure_series_minimal_conversion\(volume\)",
            "volume_series = pd.Series(volume) if isinstance(volume, np.ndarray) else volume",
        ),
    ]

    for pattern, replacement in patterns:
        method_text = re.sub(pattern, replacement, method_text)

    # validate_series_dataを削除
    method_text = re.sub(r"\s*validate_series_data\([^)]+\)\s*\n", "", method_text)

    return method_text


# 簡素化対象のメソッド名リスト
simple_methods = [
    "cmo",
    "roc",
    "rocp",
    "rocr",
    "rocr100",
    "mom",
    "trix",
    "apo",
    "qqe",
    "tsi",
    "cfo",
    "cti",
    "rmi",
    "dpo",
]

multi_input_methods = [
    "willr",
    "cci",
    "adx",
    "aroon",
    "aroonosc",
    "mfi",
    "ultosc",
    "bop",
    "ao",
]

print("momentum.pyの一括簡素化対象メソッド:")
print(f"単一入力メソッド: {simple_methods}")
print(f"複数入力メソッド: {multi_input_methods}")
print(f"合計: {len(simple_methods) + len(multi_input_methods)}個のメソッド")
