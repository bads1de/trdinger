"""
データバリデーション・エラーハンドリングモジュール

テクニカル指標計算に必要なバリデーション機能を統合的に提供します。

主な機能:
- データ長検証と最小要件チェック
- 入力データのバリデーション
- エラーハンドリングデコレーター
- NaN結果生成
"""

import logging
from functools import wraps
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, cast

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PandasTAError(Exception):
    """pandas-ta関連のエラー"""


def _get_indicator_config(indicator_type: str):
    """indicator_registry から設定を取得する共通 helper."""
    from .config.indicator_config import indicator_registry

    return indicator_registry.get_indicator_config(indicator_type.upper())


def _validate_positive_length(length: Optional[int]) -> None:
    """length 引数の正当性を共通チェックする。"""
    if length is not None and length <= 0:
        raise ValueError(f"length must be positive: {length}")


def _return_nan_series_if_needed(
    series: pd.Series,
    min_data_length: int,
) -> Optional[pd.Series]:
    """空系列または最小長不足の場合に NaN Series を返す。"""
    if len(series) == 0:
        return create_nan_series_like(series)

    if min_data_length > 0 and len(series) < min_data_length:
        return create_nan_series_like(series)

    return None


def _validate_series_collection(
    series_items: Mapping[str, pd.Series],
    *,
    length: Optional[int] = None,
    min_data_length: int = 0,
) -> Optional[pd.Series]:
    """単一/複数 Series の共通検証を行う。"""
    if not series_items:
        raise ValueError("series_dict cannot be empty")

    reference_series: Optional[pd.Series] = None
    reference_name: Optional[str] = None

    for name, series in series_items.items():
        if not isinstance(series, pd.Series):
            raise TypeError(f"{name} must be pandas Series")

        if reference_series is None:
            reference_series = series
            reference_name = name
            continue

        if len(series) != len(reference_series):
            raise ValueError(
                f"All series must have the same length. "
                f"{reference_name}={len(reference_series)}, {name}={len(series)}"
            )

    _validate_positive_length(length)

    if reference_series is not None:
        return _return_nan_series_if_needed(reference_series, min_data_length)

    return None


def _create_nan_array(length: int, width: int = 1) -> np.ndarray:
    """NaN で埋まった 1D/2D 配列を作る。"""
    return np.full((length,) if width == 1 else (length, width), np.nan)


def _is_missing_indicator_result(result: Any) -> bool:
    """pandas-ta の失敗結果や空結果を判定する。"""
    if result is None:
        return True

    if isinstance(result, pd.Series):
        return result.empty or bool(result.isna().all())

    if isinstance(result, pd.DataFrame):
        return result.empty or bool(result.isna().all().all())

    if isinstance(result, np.ndarray):
        if result.size == 0:
            return True
        try:
            return bool(np.isnan(result).all())
        except (TypeError, ValueError):
            return False

    if isinstance(result, tuple):
        return len(result) == 0 or all(item is None for item in result)

    return False


def _run_indicator_with_validation(
    validation: Optional[pd.Series],
    result_factory: Callable[[], Any],
    *,
    fallback_factory: Optional[Callable[[], Any]] = None,
    reference_series: Optional[pd.Series] = None,
) -> Any:
    """検証と NaN フォールバックをまとめて扱う。"""
    if validation is not None:
        return fallback_factory() if fallback_factory is not None else validation

    result = result_factory()
    if _is_missing_indicator_result(result):
        if fallback_factory is not None:
            return fallback_factory()
        if reference_series is not None:
            return create_nan_series_like(reference_series)
    return result


def run_series_indicator(
    data: pd.Series,
    length: Optional[int],
    result_factory: Callable[[], Any],
    *,
    min_data_length: int = 0,
    fallback_factory: Optional[Callable[[], Any]] = None,
) -> Any:
    """単一 Series 入力の指標計算を検証付きで実行する。"""
    validation = validate_series_params(data, length, min_data_length=min_data_length)
    return _run_indicator_with_validation(
        validation,
        result_factory,
        fallback_factory=fallback_factory,
        reference_series=data,
    )


def run_multi_series_indicator(
    series_dict: Mapping[str, pd.Series],
    length: Optional[int],
    result_factory: Callable[[], Any],
    *,
    min_data_length: int = 0,
    fallback_factory: Optional[Callable[[], Any]] = None,
) -> Any:
    """複数 Series 入力の指標計算を検証付きで実行する。"""
    validation = validate_multi_series_params(
        dict(series_dict), length, min_data_length=min_data_length
    )
    reference_series = next(iter(series_dict.values()))
    return _run_indicator_with_validation(
        validation,
        result_factory,
        fallback_factory=fallback_factory,
        reference_series=reference_series,
    )


def normalize_non_finite(series: pd.Series, fill_value: Any = np.nan) -> pd.Series:
    """inf/-inf を NaN 経由で指定値に揃える。"""
    return series.replace([np.inf, -np.inf], np.nan).fillna(fill_value)


def create_nan_series_like(
    reference: pd.Series,
    fill_value: Any = np.nan,
    name: Optional[str] = None,
) -> pd.Series:
    """参照 Series と同じ index を持つ定数 Series を作る。"""
    return pd.Series(
        np.full(len(reference), fill_value),
        index=reference.index,
        name=name if name is not None else reference.name,
    )


def create_nan_series_bundle(
    reference: pd.Series,
    count: int,
    fill_value: Any = np.nan,
) -> tuple[pd.Series, ...]:
    """同じ形の Series を複数個まとめて作る。"""
    base = create_nan_series_like(reference, fill_value=fill_value)
    return tuple(base.copy() for _ in range(count))


def create_nan_series_map(
    reference: pd.Series,
    keys: list[str],
    fill_value: Any = np.nan,
) -> Dict[str, pd.Series]:
    """指定キーに対応する NaN Series の辞書を作る。"""
    return {
        key: create_nan_series_like(reference, fill_value=fill_value, name=key)
        for key in keys
    }


def nan_result_for(
    data: Any,
    count: int,
    *,
    to_numpy: bool = False,
    fallback_factory: Optional[Callable[[], Any]] = None,
) -> tuple[Any, ...]:
    """入力に応じて NaN の tuple 結果を生成する。"""
    try:
        if isinstance(data, tuple):
            if to_numpy:
                return tuple(np.asarray(item) for item in data)
            return data

        if isinstance(data, dict):
            reference = next(iter(data.values()))
        elif isinstance(data, pd.DataFrame):
            if data.empty:
                raise ValueError("dataframe is empty")
            reference = data.iloc[:, 0]
        elif isinstance(data, pd.Series):
            reference = data
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        result = create_nan_series_bundle(reference, count)
        if to_numpy:
            return tuple(series.to_numpy() for series in result)
        return result
    except Exception:
        if fallback_factory is not None:
            return fallback_factory()
        raise


def extract_tuple_result(
    result: Any,
    count: int,
    *,
    by_index: bool = False,
    column_names: Optional[list[str]] = None,
    to_numpy: bool = False,
    fallback_factory: Optional[Callable[[], Any]] = None,
) -> tuple[Any, ...]:
    """DataFrame や tuple の結果を tuple に正規化する。"""
    try:
        if isinstance(result, tuple):
            if to_numpy:
                return tuple(np.asarray(item) for item in result)
            return result

        if isinstance(result, pd.DataFrame):
            if column_names is not None:
                missing = [name for name in column_names if name not in result.columns]
                if missing:
                    raise KeyError(f"Missing columns: {missing}")
                selected_columns = list(column_names)
            else:
                if count <= 0:
                    raise ValueError("count must be positive")
                if by_index:
                    if len(result.columns) < count:
                        raise IndexError(
                            f"DataFrame has {len(result.columns)} columns but count={count}"
                        )
                    selected_columns = list(result.columns[:count])
                elif len(result.columns) >= count:
                    selected_columns = list(result.columns[:count])
                else:
                    raise IndexError(
                        f"DataFrame has {len(result.columns)} columns but count={count}"
                    )

            extracted = tuple(result[col] for col in selected_columns)
            if to_numpy:
                return tuple(series.to_numpy() for series in extracted)
            return extracted

        if isinstance(result, pd.Series):
            extracted = (result,)
            if to_numpy:
                return (result.to_numpy(),)
            return extracted

        if fallback_factory is not None:
            return fallback_factory()

        raise TypeError(f"Unsupported result type: {type(result)}")
    except Exception:
        if fallback_factory is not None:
            return fallback_factory()
        raise


def get_param_value(params: Dict[str, Any], keys: list, default: Any) -> Any:
    """パラメータ名がlengthまたはwindowの場合の値取得をサポート"""
    for key in keys:
        if key in params:
            return params[key]
    return default


def get_minimum_data_length(indicator_type: str, params: Dict[str, Any]) -> int:
    """
    指標の種類とパラメータから最小必要データ長を取得

    Args:
        indicator_type: 指標タイプ
        params: パラメータ辞書

    Returns:
        最小必要データ長
    """
    config = _get_indicator_config(indicator_type)
    if config and config.min_length_func:
        return config.min_length_func(params)

    # フォールバック：デフォルト値 - lengthまたはwindowパラメータをサポート
    if config and config.default_values:
        length_value = get_param_value(config.default_values, ["length", "window"], 14)
        return length_value

    return 1  # 最低1つのデータ点


def get_absolute_minimum_length(indicator_type: str) -> int:
    """
    各指標の絶対的最小データ長を取得
    """
    config = _get_indicator_config(indicator_type)
    if (
        config
        and hasattr(config, "absolute_min_length")
        and config.absolute_min_length is not None
    ):
        return int(config.absolute_min_length)
    return 1


def validate_data_length_with_fallback(
    df: pd.DataFrame, indicator_type: str, params: Dict[str, Any]
) -> Tuple[bool, int]:
    """
    データ長検証を強化し、フォールバック可能な最小データ長を返す

    Args:
        df: OHLCV価格データ
        indicator_type: 指標タイプ
        params: パラメータ辞書

    Returns:
        (データ長が十分かどうか, フォールバック可能な最小データ長)
    """
    standard_length = get_minimum_data_length(indicator_type, params) or 1
    absolute_minimum = get_absolute_minimum_length(indicator_type.upper()) or 1

    # パラメータに基づく長さと絶対的な最小長さの大きい方を採用
    required_length = max(int(standard_length), int(absolute_minimum))

    data_length = len(df)

    if data_length >= required_length:
        return True, required_length

    # フォールバック処理のための緩和された最小長を計算
    min_required = max(int(absolute_minimum), int(standard_length) // 3)

    if data_length >= min_required:
        logger.info(
            "データ長不足時は、この関数でフォールバック加工を実行。"
            f"{indicator_type}: 必要データ長 {required_length} が {data_length} 不足のため、"
            f"最小データ長 {min_required} にフォールしNaNフィルタを適用"
        )
        return True, data_length

    logger.warning(
        "この関数で運用データでの警告出力を強化。"
        f"{indicator_type}: 必要なデータ長 {required_length} 以上、"
        f"最低データ長 {min_required} が {data_length} 不足のためNaNフィルタを適用"
    )
    return False, min_required


def create_nan_result(df: pd.DataFrame, indicator_type: str) -> np.ndarray:
    """
    データ長不足時のNaN結果生成

    Args:
        df: 元のデータフレーム
        indicator_type: 指標タイプ

    Returns:
        NaN配列
    """
    config = _get_indicator_config(indicator_type)
    data_length = len(df)

    if not config:
        return _create_nan_array(data_length)

    if config.returns == "single":
        return _create_nan_array(data_length)

    return_cols = config.return_cols or ["Result"]
    return _create_nan_array(data_length, len(return_cols))


def validate_input(data: object, period: int) -> None:
    """
    入力データの基本検証（pandas.Series専用）

    Args:
        data: 検証対象のデータ（pandas.Series を想定するが、検証前は任意のオブジェクトを受ける）
        period: 期間パラメータ

    Raises:
        PandasTAError: 入力データが無効な場合
    """
    if data is None:
        raise PandasTAError("入力データがNoneです")

    if not isinstance(data, pd.Series):
        raise PandasTAError(
            f"入力データはpandas.Seriesである必要があります。実際の型: {type(data)}"
        )

    series = cast(pd.Series, data)

    if len(series) == 0:
        raise PandasTAError("入力データが空です")

    if period <= 0:
        raise PandasTAError(f"期間は正の整数である必要があります: {period}")

    if len(series) < period:
        raise PandasTAError(
            f"データ長({len(series)})が期間({period})より短いです"
        )

    # NaNや無限大の値をチェック (pandas.Series専用)
    if bool(series.isna().any()):
        logger.warning("入力データにNaN値が含まれています")
    if np.isinf(series).any():
        raise PandasTAError("入力データに無限大の値が含まれています")


def validate_series_params(
    data: pd.Series, length: Optional[int] = None, min_data_length: int = 0
) -> Optional[pd.Series]:
    """
    指標計算用のパラメータ検証（共通化用）

    Args:
        data: 入力データ
        length: 期間（オプション）
        min_data_length: 最小必要データ長（オプション）

    Returns:
        pd.Series: データが空または不足している場合のNaNシリーズ（計算不要）
        None: 検証OK、計算続行

    Raises:
        TypeError: データ型が無効な場合
        ValueError: 期間が無効な場合
    """
    return _validate_series_collection(
        {"data": data}, length=length, min_data_length=min_data_length
    )


def validate_multi_series_params(
    series_dict: dict,
    length: Optional[int] = None,
    min_data_length: int = 0,
) -> Optional[pd.Series]:
    """
    複数のSeriesパラメータを検証（共通化用）

    Args:
        series_dict: 検証する名前付きシリーズの辞書。例: {"high": high, "low": low}
        length: 期間（オプション）
        min_data_length: 最小必要データ長（オプション）

    Returns:
        pd.Series: データが空または不足している場合のNaNシリーズ（計算不要）
        None: 検証OK、計算続行

    Raises:
        TypeError: データ型が無効な場合
        ValueError: 期間が無効な場合、またはシリーズ長が不一致な場合
    """
    return _validate_series_collection(
        series_dict, length=length, min_data_length=min_data_length
    )


# =============================================================================
# エラーハンドリングデコレーター
# =============================================================================


def handle_pandas_ta_errors(func):
    """
    pandas-taエラーハンドリングデコレーター

    重要な異常ケースのみをチェックし、パフォーマンスを重視。
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)

            # 重要な異常ケースのみチェック
            if result is None:
                raise PandasTAError(f"{func.__name__}: 計算結果がNoneです")

            # numpy配列の検証（簡素化）
            if isinstance(result, np.ndarray):
                if len(result) == 0:
                    raise PandasTAError(f"{func.__name__}: 計算結果が空です")
                # 全NaNチェック（重要）
                if len(result) > 0 and np.all(np.isnan(result)):
                    raise PandasTAError(f"{func.__name__}: 計算結果が全てNaNです")

            # tupleの場合（MACD等）
            elif isinstance(result, tuple):
                if len(result) > 0 and all(
                    hasattr(arr, "__len__") and len(arr) == 0 for arr in result
                ):
                    return result

                for i, arr in enumerate(result):
                    if arr is None or (hasattr(arr, "__len__") and len(arr) == 0):
                        raise PandasTAError(f"{func.__name__}: 結果[{i}]が無効です")

            return result

        except PandasTAError:
            # 既にPandasTAErrorの場合は再発生
            raise
        except (TypeError, ValueError):
            # バリデーションエラーは再発生
            raise
        except Exception as e:
            # その他のエラーは簡潔に処理
            raise PandasTAError(f"{func.__name__} 計算エラー: {e}")

    return wrapper
