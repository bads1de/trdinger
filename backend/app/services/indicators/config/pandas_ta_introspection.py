"""
pandas-ta イントロスペクションモジュール

pandas-taのソースコードを解析して、各指標のメタデータを動的に抽出します。
これにより、手動での定数メンテナンスを大幅に削減できます。

主な機能:
- min_length (最小必要データ長) の計算式を抽出
- 戻り値のカラム数/タイプを検出
- インジケーターカテゴリを取得
- デフォルトパラメータを抽出
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
    pandas-taのソースコードからmin_length計算式を抽出

    Args:
        indicator_name: 指標名（小文字）

    Returns:
        min_length計算式の文字列、または None
    """
    func = getattr(ta, indicator_name.lower(), None)
    if func is None or not callable(func):
        return None

    try:
        source = inspect.getsource(func)

        # verify_series(..., <min_length>) パターンを探す
        # ネストした括弧を正しく処理するため、手動でパースする
        for line in source.split("\n"):
            if "verify_series" in line and "," in line:
                # verify_series(xxx, yyy) から yyy を抽出
                match = re.search(
                    r"verify_series\s*\(\s*\w+\s*,\s*(.+?)\s*\)\s*$", line.strip()
                )
                if match:
                    expr = match.group(1).strip()

                    # _length のような中間変数の場合、その定義を探す
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
    """pandas-taの指標からデフォルトパラメータ値を取得"""
    func = getattr(ta, indicator_name.lower(), None)
    if func is None:
        return {}

    try:
        source = inspect.getsource(func)
        defaults = {}

        # パターン1: length = int(length) if length and length > 0 else 14
        # 例: fast = int(fast) if fast and fast > 0 else 12
        pattern1 = r"(\w+)\s*=\s*int\s*\(\s*\1\s*\)\s+if\s+\1\s+and\s+\1\s*>\s*0\s+else\s+(\d+)"
        matches = re.findall(pattern1, source)
        for param_name, default_value in matches:
            try:
                defaults[param_name] = int(default_value)
            except ValueError:
                defaults[param_name] = float(default_value)

        # パターン2: k = k if k and k > 0 else 14 (int()なし)
        pattern2 = r"(\w+)\s*=\s*\1\s+if\s+\1\s+and\s+\1\s*>\s*0\s+else\s+(\d+)"
        matches2 = re.findall(pattern2, source)
        for param_name, default_value in matches2:
            if param_name not in defaults:
                try:
                    defaults[param_name] = int(default_value)
                except ValueError:
                    defaults[param_name] = float(default_value)

        # パターン3: length = float(length) if length and length > 0 else 100
        pattern3 = r"(\w+)\s*=\s*float\s*\(\s*\1\s*\)\s+if\s+\1(?:\s+and|\s*\)).*?else\s+(\d+(?:\.\d+)?)"
        matches3 = re.findall(pattern3, source)
        for param_name, default_value in matches3:
            if param_name not in defaults:
                try:
                    defaults[param_name] = int(default_value)
                except ValueError:
                    defaults[param_name] = float(default_value)

        return defaults

    except Exception:
        return {}


def _evaluate_expression(expr: str, params: Dict[str, Any]) -> Optional[int]:
    """
    min_length式を評価して数値を返す

    Args:
        expr: 評価する式（例: "max(fast, slow, signal)" or "length"）
        params: パラメータ辞書

    Returns:
        計算されたmin_length値
    """
    try:
        # 安全な評価のため、許可された関数のみを使用
        safe_globals = {"max": max, "min": min}

        # パラメータを数値に変換
        safe_locals = {}
        for key, value in params.items():
            if isinstance(value, (int, float)):
                safe_locals[key] = int(value)

        result = eval(expr, safe_globals, safe_locals)

        if isinstance(result, (int, float)):
            return int(result)

        return None

    except Exception:
        return None


def calculate_min_length(indicator_name: str, params: Dict[str, Any]) -> int:
    """
    指標の最小必要データ長を計算

    Args:
        indicator_name: 指標名
        params: パラメータ辞書

    Returns:
        最小必要データ長
    """
    # デフォルト値を取得してマージ
    defaults = _get_indicator_defaults(indicator_name)
    merged_params = {**defaults, **params}

    # min_length式を取得
    expr = extract_min_length_expression(indicator_name)

    if expr is None:
        # フォールバック: lengthまたはperiodパラメータを使用
        return merged_params.get("length", merged_params.get("period", 1))

    # 式を評価
    result = _evaluate_expression(expr, merged_params)

    if result is not None:
        return result

    # フォールバック
    return merged_params.get("length", merged_params.get("period", 1))


# =============================================================================
# 戻り値タイプ検出
# =============================================================================


# 複数カラムを返す指標のキャッシュ
_MULTI_COLUMN_CACHE: Dict[str, bool] = {}


def is_multi_column_indicator(indicator_name: str) -> bool:
    """
    指標が複数カラム（DataFrame）を返すかどうかを判定

    Args:
        indicator_name: 指標名

    Returns:
        True if multi-column, False otherwise
    """
    name_lower = indicator_name.lower()

    if name_lower in _MULTI_COLUMN_CACHE:
        return _MULTI_COLUMN_CACHE[name_lower]

    # 既知の複合指標リスト（pandas-taの仕様に基づく）
    # これらは DataFrame を返す指標
    known_multi_column = {
        "macd",
        "bbands",
        "stoch",
        "supertrend",
        "kc",
        "donchian",
        "accbands",
        "aroon",
        "tsi",
        "fisher",
        "kst",
        "pvo",
        "stochrsi",
        "squeeze",
        "vortex",
        "kama",
        "adosc",
        "trix",
        "ppo",
        "apo",
        "midprice",
        "psar",
        "ichimoku",
        "dema",
        "ema",
        "sma",
        "wma",
        "aberration",
        "amat",
        "aobv",
        "brar",
        "cksp",
        "decay",
        "decreasing",
        "dpo",
        "ebsw",
        "increasing",
        "long_run",
        "psar",
        "qstick",
        "short_run",
        "td_seq",
        "ttm_trend",
        "vortext",
        "xsignals",
    }

    # 既知の単一カラム指標リスト
    known_single_column = {
        "rsi",
        "sma",
        "ema",
        "atr",
        "adx",
        "cci",
        "mfi",
        "obv",
        "roc",
        "mom",
        "willr",
        "cmf",
        "cmo",
        "chop",
        "dema",
        "wma",
        "tema",
        "trima",
        "hma",
        "zlma",
        "kama",
        "cg",
        "coppock",
        "ao",
        "bop",
        "uo",
        "psl",
        "cti",
        "massi",
        "pgo",
        "rvgi",
        "er",
        "ui",
        "natr",
        "trange",
        "pdist",
    }

    if name_lower in known_single_column:
        _MULTI_COLUMN_CACHE[name_lower] = False
        return False

    if name_lower in known_multi_column:
        _MULTI_COLUMN_CACHE[name_lower] = True
        return True

    # 未知の指標の場合、ソースコードから推測
    func = getattr(ta, name_lower, None)
    if func is None:
        return False

    try:
        source = inspect.getsource(func)

        # docstringとコメントを除去してから判定
        source_no_docstring = re.sub(r'"""[\s\S]*?"""', "", source)
        source_no_docstring = re.sub(r"'''[\s\S]*?'''", "", source_no_docstring)
        source_no_comments = re.sub(
            r"#.*$", "", source_no_docstring, flags=re.MULTILINE
        )

        # return 文で DataFrame を直接返しているかチェック（より厳密）
        has_dataframe = (
            re.search(r"return\s+DataFrame\s*\(", source_no_comments) is not None
            or re.search(r"return\s+pd\.concat\s*\(", source_no_comments) is not None
            or re.search(r"return\s+\w+df\b", source_no_comments, re.IGNORECASE)
            is not None
        )

        _MULTI_COLUMN_CACHE[name_lower] = has_dataframe
        return has_dataframe

    except Exception:
        return False


@lru_cache(maxsize=256)
def get_return_column_count(indicator_name: str) -> int:
    """
    指標が返すカラム数を取得

    Args:
        indicator_name: 指標名

    Returns:
        カラム数（単一の場合は1）
    """
    # 既知のカラム数（pandas-taの仕様に基づく）
    known_column_counts = {
        "macd": 3,  # MACD, Signal, Histogram
        "bbands": 5,  # BBL, BBM, BBU, BBB, BBP
        "stoch": 2,  # STOCHk, STOCHd
        "supertrend": 4,  # SUPERT, SUPERTd, SUPERTl, SUPERTs
        "kc": 3,  # KCL, KCm, KCU
        "donchian": 3,  # DCL, DCM, DCU
        "accbands": 3,  # ACCBL, ACCBM, ACCBU
        "aroon": 3,  # AROOND, AROONU, AROONOSC
        "tsi": 2,  # TSI, TSIs
        "fisher": 2,  # FISHERT, FISHERTs
        "kst": 2,  # KST, KSTs
        "pvo": 3,  # PVO, PVOs, PVOh
        "stochrsi": 2,  # STOCHRSIk, STOCHRSId
        "squeeze": 4,  # SQZ, SQZ_ON, SQZ_OFF, SQZ_NO
        "vortex": 2,  # VTXP, VTXM
        "adosc": 1,  # 単一
    }

    name_lower = indicator_name.lower()

    if name_lower in known_column_counts:
        return known_column_counts[name_lower]

    if is_multi_column_indicator(indicator_name):
        # ソースコードから推測を試みる
        # （正確な値を得るにはサンプルデータで実行が必要だが、起動時コストを避ける）
        return 2  # デフォルトで2を返す

    return 1


# 戻り値カラム名のキャッシュ
_RETURN_COLUMN_NAMES_CACHE: Dict[str, Optional[List[str]]] = {}


def get_return_column_names(indicator_name: str) -> Optional[List[str]]:
    """
    指標が返すカラム名のリストを取得

    サンプルデータで指標を実行し、出力カラム名を取得します。
    単一値を返す指標の場合は None を返します。

    Args:
        indicator_name: 指標名

    Returns:
        カラム名のリスト、または単一値指標の場合は None
    """
    import numpy as np
    import pandas as pd

    name_lower = indicator_name.lower()

    if name_lower in _RETURN_COLUMN_NAMES_CACHE:
        return _RETURN_COLUMN_NAMES_CACHE[name_lower]

    # 単一値指標の場合は None を返す
    if not is_multi_column_indicator(indicator_name):
        _RETURN_COLUMN_NAMES_CACHE[name_lower] = None
        return None

    func = getattr(ta, name_lower, None)
    if func is None:
        _RETURN_COLUMN_NAMES_CACHE[name_lower] = None
        return None

    try:
        # サンプルデータを生成
        np.random.seed(42)
        n = 100
        sample_data = {
            "open": np.random.uniform(100, 110, n),
            "high": np.random.uniform(105, 115, n),
            "low": np.random.uniform(95, 105, n),
            "close": np.random.uniform(100, 110, n),
            "volume": np.random.uniform(1000, 5000, n),
        }
        # high > low を保証
        sample_data["high"] = np.maximum(sample_data["high"], sample_data["low"] + 1)
        df = pd.DataFrame(sample_data)

        # 指標を実行
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())

        # 必要な引数を準備
        kwargs = {}
        if "close" in params:
            kwargs["close"] = df["close"]
        if "high" in params:
            kwargs["high"] = df["high"]
        if "low" in params:
            kwargs["low"] = df["low"]
        if "open" in params or "open_" in params:
            key = "open_" if "open_" in params else "open"
            kwargs[key] = df["open"]
        if "volume" in params:
            kwargs["volume"] = df["volume"]

        result = func(**kwargs)

        if result is None:
            _RETURN_COLUMN_NAMES_CACHE[name_lower] = None
            return None

        if isinstance(result, pd.DataFrame):
            cols = list(result.columns)
            _RETURN_COLUMN_NAMES_CACHE[name_lower] = cols
            return cols
        elif isinstance(result, pd.Series):
            _RETURN_COLUMN_NAMES_CACHE[name_lower] = None
            return None

    except Exception as e:
        logger.debug(f"Failed to get return column names for {indicator_name}: {e}")
        _RETURN_COLUMN_NAMES_CACHE[name_lower] = None
        return None

    return None


# =============================================================================
# カテゴリ抽出
# =============================================================================


@lru_cache(maxsize=256)
def get_indicator_category(indicator_name: str) -> Optional[str]:
    """
    pandas-taから指標のカテゴリを取得

    Args:
        indicator_name: 指標名

    Returns:
        カテゴリ名 (momentum, trend, volatility, volume, overlap, etc.)
    """
    name_lower = indicator_name.lower()
    func = getattr(ta, name_lower, None)

    if func is None:
        return None

    try:
        # モジュール名からカテゴリを抽出
        # 例: pandas_ta.momentum.rsi -> momentum
        module_name = func.__module__
        parts = module_name.split(".")

        if len(parts) >= 2 and parts[0] == "pandas_ta":
            return parts[1]

        return None

    except Exception:
        return None


# =============================================================================
# デフォルトパラメータ抽出
# =============================================================================


def extract_default_parameters(indicator_name: str) -> Dict[str, Any]:
    """
    pandas-taの指標からデフォルトパラメータを抽出

    Args:
        indicator_name: 指標名

    Returns:
        パラメータ名とデフォルト値の辞書
    """
    func = getattr(ta, indicator_name.lower(), None)
    if func is None:
        return {}

    try:
        sig = inspect.signature(func)
        defaults = {}

        for param_name, param in sig.parameters.items():
            if param.default != inspect.Parameter.empty:
                defaults[param_name] = param.default

        # ソースコードからも抽出（シグネチャがNoneの場合）
        source_defaults = _get_indicator_defaults(indicator_name)
        for key, value in source_defaults.items():
            if key not in defaults or defaults[key] is None:
                defaults[key] = value

        return defaults

    except Exception:
        return {}


# =============================================================================
# 全指標スキャン
# =============================================================================


def get_all_pandas_ta_indicators() -> List[str]:
    """
    pandas-taの全指標名を取得

    Returns:
        指標名のリスト
    """
    excluded = {
        "cdl_pattern",
        "cdl_z",
        "ha",
        "tsignals",
        "above",
        "below",
        "cross",
        "cross_value",
        "fibonacci",
        "verify_series",
        "progress_bar",
        "imports",
        "category",
        "utils",
        "strategy",
        "log",
        "datetime",
        "version",
        "indicators",
        "categories",
        "cores",
        "help",
        "get_drift",
        "get_offset",
        "non_zero_range",
        "signed_series",
        "unsigned_differences",
    }

    indicators = []

    for name in dir(ta):
        if name.startswith("_"):
            continue
        if name in excluded:
            continue

        func = getattr(ta, name, None)
        if func is not None and callable(func):
            # モジュール名がpandas_ta.*で始まるものだけ
            module = getattr(func, "__module__", "")
            if module.startswith("pandas_ta"):
                indicators.append(name)

    return indicators
