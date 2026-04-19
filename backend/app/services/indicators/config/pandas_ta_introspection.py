"""
pandas-ta イントロスペクションモジュール

pandas-taのソースコードおよびドキュメントを解析して、各指標のメタデータを動的に抽出します。
手動での定数メンテナンスを排除し、ライブラリの更新に自動追従します。
"""

import inspect
import logging
import re
import warnings
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import pandas_ta_classic as ta

logger = logging.getLogger(__name__)


# サンプルDataFrameキャッシュ（同じパラメータなら同じDataFrameを再利用）
_sample_frame_cache: Dict[tuple, pd.DataFrame] = {}


def _build_sample_ohlcv_frame(
    rows: int,
    *,
    walk_close: bool = False,
    with_datetime_index: bool = False,
) -> pd.DataFrame:
    """
    pandas-ta の検証用サンプル OHLCV DataFrame を作る

    インジケーターの検証やイントロスペクション用のサンプルデータを生成します。
    同じパラメータでの呼び出しはキャッシュから返されます。

    Args:
        rows: 行数（データポイント数）
        walk_close: Trueの場合、close価格をランダムウォークで生成（デフォルト: False）
        with_datetime_index: Trueの場合、datetimeインデックスを設定（デフォルト: False）

    Returns:
        Any: サンプルOHLCV DataFrame

    Note:
        walk_close=Falseの場合、乱数で生成された価格を使用します。
        walk_close=Trueの場合、累積和でトレンドを持つ価格を生成します。
    """
    cache_key = (rows, walk_close, with_datetime_index)
    if cache_key in _sample_frame_cache:
        return _sample_frame_cache[cache_key]

    import numpy as np
    import pandas as pd

    np.random.seed(42)

    if walk_close:
        close = np.random.randn(rows).cumsum() + 100
        open_ = close - 1
        high = close + 1
        low = close - 1
        volume = np.random.randint(100, 1000, rows).astype(float)
    else:
        open_ = np.random.uniform(100, 110, rows)  # type: ignore[assignment]
        high = np.random.uniform(105, 115, rows)  # type: ignore[assignment]
        low = np.random.uniform(95, 105, rows)  # type: ignore[assignment]
        close = np.random.uniform(100, 110, rows)  # type: ignore[assignment]
        volume = np.random.uniform(1000, 5000, rows)

    frame = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )

    if not walk_close:
        # 高値/安値の関係を安定させる
        frame["high"] = np.maximum(frame["high"], frame["low"] + 1)

    if with_datetime_index:
        frame.index = pd.date_range("2024-01-01", periods=rows, freq="h")

    _sample_frame_cache[cache_key] = frame
    return frame


def _build_indicator_call_kwargs(
    func: Any,
    frame: Any,
    default_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    検証用の DataFrame から、関数呼び出し用 kwargs を作る

    関数のシグネチャを解析して、DataFrameから適切なデータカラムを抽出し、
    関数呼び出し用のkwargs辞書を構築します。

    Args:
        func: インジケーター関数
        frame: OHLCV DataFrame
        default_params: デフォルトパラメータ（オプション）

    Returns:
        Dict[str, Any]: 関数呼び出し用kwargs辞書

    マッピングルール:
        - close/high/low/open/volume → 対応するカラム
        - open_ → openカラム
        - data/series → closeカラム
        - その他 → default_paramsから取得
    """
    sig = inspect.signature(func)
    kwargs: Dict[str, Any] = {}
    default_params = default_params or {}

    for p_name in sig.parameters:
        p_lower = p_name.lower()
        if p_lower == "close":
            kwargs[p_name] = frame["close"]
        elif p_lower == "high":
            kwargs[p_name] = frame["high"]
        elif p_lower == "low":
            kwargs[p_name] = frame["low"]
        elif p_lower == "open":
            kwargs[p_name] = frame["open"]
        elif p_lower == "open_":
            kwargs[p_name] = frame["open"]
        elif p_lower == "volume":
            kwargs[p_name] = frame["volume"]
        elif p_lower in ["data", "series"]:
            kwargs[p_name] = frame["close"]
        elif p_name in default_params:
            kwargs[p_name] = default_params[p_name]

    return kwargs


# サンプル実行結果キャッシュ（同じ関数/パラメータなら同じ結果を再利用）
_sample_run_cache: Dict[tuple, Any] = {}


def _run_indicator_on_sample_frame(
    func: Callable[..., Any],
    *,
    rows: int,
    default_params: Optional[Dict[str, Any]] = None,
    walk_close: bool = False,
    with_datetime_index: bool = False,
) -> Any:
    """
    サンプル OHLCV DataFrame で指標を実行する共通処理

    サンプルDataFrameを生成し、インジケーター関数を実行して結果を返します。
    同じ関数/パラメータでの呼び出しはキャッシュから返されます。

    Args:
        func: インジケーター関数
        rows: サンプルデータの行数
        default_params: デフォルトパラメータ（オプション）
        walk_close: close価格をランダムウォークで生成するか（デフォルト: False）
        with_datetime_index: datetimeインデックスを設定するか（デフォルト: False）

    Returns:
        Any: インジケーターの実行結果

    Note:
        FutureWarningを抑制して実行します。
    """
    # キャッシュキーを生成（funcはhashableではないので名前ベース）
    func_id = getattr(func, "__name__", None) or getattr(func, "__qualname__", id(func))
    params_key = tuple(sorted(default_params.items())) if default_params else ()
    cache_key = (func_id, rows, params_key, walk_close, with_datetime_index)

    if cache_key in _sample_run_cache:
        return _sample_run_cache[cache_key]

    sample_frame = _build_sample_ohlcv_frame(
        rows,
        walk_close=walk_close,
        with_datetime_index=with_datetime_index,
    )
    kwargs = _build_indicator_call_kwargs(func, sample_frame, default_params)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        result = func(**kwargs)

    _sample_run_cache[cache_key] = result
    return result


# =============================================================================
# min_length 抽出
# =============================================================================


@lru_cache(maxsize=256)
def extract_min_length_expression(indicator_name: str) -> Optional[str]:
    """
    pandas-taのソースコードからmin_length計算式を抽出します

    インジケーターのソースコードを解析して、verify_series関数の
    第二引数（最小データ長）の計算式を抽出します。

    Args:
        indicator_name: インジケーター名

    Returns:
        Optional[str]: min_length計算式、抽出失敗時はNone

    Note:
        中間変数（_length等）を使用している場合、その定義も抽出します。
    """
    func: Any = getattr(ta, indicator_name.lower(), None)

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
    pandas-taの指標からデフォルトパラメータ値を取得します

    ソースコードを解析して、条件分岐によるデフォルト値を抽出します。
    シグネチャに含まれないデフォルト値を補完するために使用します。

    Args:
        indicator_name: インジケーター名

    Returns:
        Dict[str, Any]: デフォルトパラメータ辞書

    抽出パターン:
        - int(param) if param and param > 0 else default
        - param if param and param > 0 else default
        - float(param) if param ... else default
    """
    func: Any = getattr(ta, indicator_name.lower(), None)

    if func is None:
        return {}

    try:
        source = inspect.getsource(func)
        defaults = {}

        # 抽出用パターンリスト
        patterns = [
            r"(\w+)\s*=\s*int\s*\(\s*\1\s*\)\s+if\s+\1\s+and\s+\1\s*>\s*0\s+else\s+(\d+)",
            r"(\w+)\s*=\s*\1\s+if\s+\1\s+and\s+\1\s*>\s*0\s+else\s+(\d+)",
            r"(\w+)\s*=\s*float\s*\(\s*\1\s*\)\s+if\s+\1(?:\s+and|\s*\)).*?else\s+(\d+(?:\.\d+)?)",
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
    min_length式を評価して数値を返します

    抽出されたmin_length計算式をパラメータ値で評価します。

    Args:
        expr: 計算式（例: 'max(length, period)'）
        params: パラメータ辞書

    Returns:
        Optional[int]: 評価結果、失敗時はNone

    Note:
        安全な評価のため、許可された関数（max, min）のみ使用可能です。
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
    指標の最小必要データ長を計算します

    ソースコードから抽出したmin_length式を評価して、
    インジケーターの最小必要データ長を計算します。

    Args:
        indicator_name: インジケーター名
        params: パラメータ辞書

    Returns:
        int: 最小必要データ長

    Note:
        式の抽出に失敗した場合、lengthまたはperiodパラメータを返します。
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
    指標が複数カラム（DataFrame）を返すかどうかを判定します

    docstringとソースコードを解析して、インジケーターが
    複数カラムを返すかどうかを判定します。

    Args:
        indicator_name: インジケーター名

    Returns:
        bool: 複数カラムを返す場合はTrue、単一カラムの場合はFalse

    判定方法:
        1. 既知のマルチカラムインジケーターのホワイトリスト
        2. docstringのReturnsセクション
        3. ソースコードのDataFrame返却パターン
    """
    name_lower = indicator_name.lower()
    func: Any = getattr(ta, name_lower, None)

    if func is None:
        return False

    # 既知のマルチカラムインジケーター（解析漏れ対策）
    # ソースコード解析で検知できない場合のためのホワイトリスト
    if name_lower in ["fisher", "aroon", "adx", "supertrend"]:
        return True

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
            r'("""[\s\S]*?"""|"""[\s\S]*?"""|#.*$)', "", source, flags=re.MULTILINE
        )

        # DataFrameを返す特徴的なキーワードを検索
        indicators_returning_df = [
            r"return\s+DataFrame\(",
            r"return\s+pd\.concat\(",
            r"return\s+\w+df\b",
        ]

        return any(re.search(p, source_clean) for p in indicators_returning_df)

    except Exception:
        return False


@lru_cache(maxsize=256)
def get_return_column_count(indicator_name: str) -> int:
    """
    指標が返すカラム数を取得します

    マルチカラムインジケーターのカラム数を取得します。

    Args:
        indicator_name: インジケーター名

    Returns:
        int: カラム数

    Note:
        推測失敗時のデフォルトは2です。
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
    指標が返すカラム名のリストを取得します

    サンプル実行により、マルチカラムインジケーターの
    実際のカラム名を取得します。

    Args:
        indicator_name: インジケーター名

    Returns:
        Optional[List[str]]: カラム名リスト、単一カラムまたは失敗時はNone

    Note:
        サンプル実行に失敗した場合はNoneを返します。
    """
    import pandas as pd

    name_lower = indicator_name.lower()

    if not is_multi_column_indicator(name_lower):
        return None

    func: Any = getattr(ta, name_lower, None)

    if func is None:
        return None

    try:
        # サンプルデータでの実行により実際のカラム名を特定
        result = _run_indicator_on_sample_frame(func, rows=100)

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
    指標のカテゴリ名 (momentum, trend, volatility, etc.) を取得します

    pandas-taのCategory辞書からインジケーターのカテゴリを取得します。

    Args:
        indicator_name: インジケーター名

    Returns:
        Optional[str]: カテゴリ名、見つからない場合はNone

    カテゴリ例:
        - momentum: モメンタム系
        - trend: トレンド系
        - volatility: ボラティリティ系
        - volume: ボリューム系
        - overlap: オーバーラップ系
    """
    name_lower = indicator_name.lower()

    for cat, items in ta.Category.items():
        if name_lower in items:
            return cat

    return None


def extract_default_parameters(indicator_name: str) -> Dict[str, Any]:
    """
    指標のデフォルトパラメータを抽出します

    シグネチャとソースコードからデフォルトパラメータを抽出します。

    Args:
        indicator_name: インジケーター名

    Returns:
        Dict[str, Any]: デフォルトパラメータ辞書

    抽出方法:
        1. シグネチャからのデフォルト値抽出
        2. ソースコードからの補完（条件分岐によるデフォルト値）
    """
    func: Any = getattr(ta, indicator_name.lower(), None)

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
    pandas-taの全指標名を動的に取得します

    pandas-taのCategory辞書からすべてのインジケーター名を取得します。

    Returns:
        List[str]: 全インジケーター名のリスト（ソート済み、重複なし）

    Note:
        ライブラリの更新に自動追従します。
    """
    indicators = []

    for items in ta.Category.values():
        indicators.extend(items)

    # 重複を除去してソート
    return sorted(list(set(indicators)))
