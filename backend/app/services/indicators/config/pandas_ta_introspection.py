"""
pandas-ta イントロスペクションモジュール

pandas-taのソースコードおよびドキュメントを解析して、各指標のメタデータを動的に抽出します。
手動での定数メンテナンスを排除し、ライブラリの更新に自動追従します。
"""

import inspect
import logging
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional

import pandas_ta as ta

logger = logging.getLogger(__name__)


# =============================================================================
# min_length 抽出
# =============================================================================


@lru_cache(maxsize=256)
def extract_min_length_expression(indicator_name: str) -> Optional[str]:
    """
    pandas-taのソースコードからmin_length計算式を抽出します。
    """
    func = getattr(ta, indicator_name.lower(), None)

    if func is None or not callable(func):
        return None

    try:
        source = inspect.getsource(func)

        # verify_series(..., <min_length>) パターンを探します
        for line in source.split("\n"):
            if "verify_series" in line and "," in line:
                # verify_series(xxx, yyy) から yyy を抽出
                match = re.search(
                    r"verify_series\s*\(\s*\w+\s*,\s*(.+?)\s*\)\s*$", line.strip()
                )

                if match:
                    expr = match.group(1).strip()

                    # _length のような中間変数の場合、その定義をソース内から探します
                    if expr.startswith("_"):
                        var_def_pattern = rf"{expr}\s*=\s*([^\n]+)"
                        var_defs = re.findall(var_def_pattern, source)

                        if var_defs:
                            expr = var_defs[0].strip()

                    return expr

        return None

    except Exception as e:
        logger.debug(f"Failed to extract min_length for {indicator_name}: {e}")
        return None


def _get_indicator_defaults(indicator_name: str) -> Dict[str, Any]:
    """
    pandas-taの指標からデフォルトパラメータ値を取得します。
    """
    func = getattr(ta, indicator_name.lower(), None)

    if func is None:
        return {}

    try:
        source = inspect.getsource(func)
        defaults = {}

        # 抽出用パターンリスト
        patterns = [
            r"(\w+)\s*=\s*int\s*\(\s*\1\s*\)\s+if\s+\1\s+and\s+\1\s*>\s*0\s+else\s+(\d+)",
            r"(\w+)\s*=\s*\1\s+if\s+\1\s+and\s+\1\s*>\s*0\s+else\s+(\d+)",
            r"(\w+)\s*=\s*float\s*\(\s*\1\s*\)\s+if\s+\1(?:\s+and|\s*\)).*?else\s+(\d+(?:\.\d+)?)"
        ]

        for pattern in patterns:
            matches = re.findall(pattern, source)

            for param_name, default_value in matches:
                if param_name not in defaults:
                    try:
                        if "." in default_value:
                            defaults[param_name] = float(default_value)
                        else:
                            defaults[param_name] = int(default_value)
                    except ValueError:
                        pass

        return defaults

    except Exception:
        return {}


def _evaluate_expression(expr: str, params: Dict[str, Any]) -> Optional[int]:
    """
    min_length式を評価して数値を返します。
    """
    try:
        # 安全な評価のため、許可された関数のみを使用
        safe_globals = {"max": max, "min": min}

        # パラメータを数値に変換
        safe_locals = {
            k: int(v) for k, v in params.items() if isinstance(v, (int, float))
        }

        result = eval(expr, safe_globals, safe_locals)

        return int(result) if isinstance(result, (int, float)) else None

    except Exception:
        return None


def calculate_min_length(indicator_name: str, params: Dict[str, Any]) -> int:
    """
    指標の最小必要データ長を計算します。
    """
    # デフォルト値を取得してマージ
    defaults = _get_indicator_defaults(indicator_name)
    merged_params = {**defaults, **params}

    # min_length式を取得
    expr = extract_min_length_expression(indicator_name)

    if expr:
        result = _evaluate_expression(expr, merged_params)
        if result is not None:
            return result

    # フォールバック: lengthまたはperiodパラメータを使用
    return merged_params.get("length", merged_params.get("period", 1))


# =============================================================================
# 戻り値タイプ検出
# =============================================================================


@lru_cache(maxsize=256)
def is_multi_column_indicator(indicator_name: str) -> bool:
    """
    指標が複数カラム（DataFrame）を返すかどうかを判定します。
    """
    name_lower = indicator_name.lower()
    func = getattr(ta, name_lower, None)

    if func is None:
        return False

    # 1. docstringから型情報を優先的に取得
    doc = getattr(func, "__doc__", "")

    if doc:
        # Returns セクションの pd.DataFrame を探す
        if re.search(r"Returns:\s+pd\.DataFrame", doc, re.IGNORECASE):
            return True
        if re.search(r"Returns:\s+pd\.Series", doc, re.IGNORECASE):
            return False

    # 2. docstringで見つからない場合、ソースコードから推測
    try:
        source = inspect.getsource(func)

        # docstringとコメントを除去
        source_clean = re.sub(
            r'("""[\s\S]*?"""|"""[\s\S]*?"""|#.*$)',
            "",
            source,
            flags=re.MULTILINE
        )

        # DataFrameを返す特徴的なキーワードを検索
        indicators_returning_df = [
            r"return\s+DataFrame\(",
            r"return\s+pd\.concat\(",
            r"return\s+\w+df\b"
        ]

        return any(re.search(p, source_clean) for p in indicators_returning_df)

    except Exception:
        return False


@lru_cache(maxsize=256)
def get_return_column_count(indicator_name: str) -> int:
    """
    指標が返すカラム数を取得します。
    """
    if not is_multi_column_indicator(indicator_name):
        return 1

    # 正確なカラム名を取得してカウントを試みる
    names = get_return_column_names(indicator_name)

    if names:
        return len(names)

    # 推測失敗時のデフォルト
    return 2


@lru_cache(maxsize=256)
def get_return_column_names(indicator_name: str) -> Optional[List[str]]:
    """
    指標が返すカラム名のリストを取得します。
    """
    import numpy as np
    import pandas as pd

    name_lower = indicator_name.lower()

    if not is_multi_column_indicator(name_lower):
        return None

    func = getattr(ta, name_lower, None)

    if func is None:
        return None

    try:
        # サンプルデータでの実行により実際のカラム名を特定
        np.random.seed(42)
        n = 100

        df = pd.DataFrame({
            "open": np.random.uniform(100, 110, n),
            "high": np.random.uniform(105, 115, n),
            "low": np.random.uniform(95, 105, n),
            "close": np.random.uniform(100, 110, n),
            "volume": np.random.uniform(1000, 5000, n),
        })

        # high > low を保証
        df["high"] = np.maximum(df["high"], df["low"] + 1)

        # 指標関数のシグネチャを確認して引数を準備
        sig = inspect.signature(func)
        kwargs = {}

        for p in sig.parameters.keys():
            if p in df.columns:
                kwargs[p] = df[p]
            elif p == "open_":
                kwargs[p] = df["open"]

        # 実際に実行
        result = func(**kwargs)

        if isinstance(result, pd.DataFrame):
            return list(result.columns)

    except Exception:
        pass

    return None


# =============================================================================
# カテゴリ・全指標スキャン
# =============================================================================


@lru_cache(maxsize=256)
def get_indicator_category(indicator_name: str) -> Optional[str]:
    """
    指標のカテゴリ名 (momentum, trend, volatility, etc.) を取得します。
    """
    name_lower = indicator_name.lower()

    for cat, items in ta.Category.items():
        if name_lower in items:
            return cat

    return None


def extract_default_parameters(indicator_name: str) -> Dict[str, Any]:
    """
    指標のデフォルトパラメータを抽出します。
    """
    func = getattr(ta, indicator_name.lower(), None)

    if func is None:
        return {}

    try:
        # シグネチャからの抽出
        sig = inspect.signature(func)
        defaults = {
            k: v.default
            for k, v in sig.parameters.items()
            if v.default != inspect.Parameter.empty
        }

        # ソースコードからの補完（シグネチャにデフォルト値がない場合）
        source_defaults = _get_indicator_defaults(indicator_name)

        for k, v in source_defaults.items():
            if k not in defaults or defaults[k] is None:
                defaults[k] = v

        return defaults

    except Exception:
        return {}

def get_all_pandas_ta_indicators() -> List[str]:
    """
    pandas-taの全指標名を動的に取得します。
    """
    indicators = []

    for items in ta.Category.values():
        indicators.extend(items)

    # 重複を除去してソート
    return sorted(list(set(indicators)))