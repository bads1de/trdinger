"""
fitness 値の抽出を共通化するユーティリティ。
"""

from __future__ import annotations


def normalize_fitness_values(
    values: object,
    *,
    default: float = 0.0,
) -> float | tuple[float, ...]:
    """
    fitness 値を公開向けの表現へ正規化する。

    単一目的なら `float`、多目的なら `tuple[float, ...]` を返します。
    評価値が取得できない場合は `default` を返します。
    """
    if isinstance(values, (tuple, list)):
        if not values:
            return float(default)

        try:
            normalized = tuple(float(value) for value in values)
        except (TypeError, ValueError):
            return float(default)

        if len(normalized) == 1:
            return normalized[0]
        return normalized

    if isinstance(values, (int, float)):
        return float(values)

    return float(default)


def extract_individual_primary_fitness(
    individual: object,
    *,
    default: float = float("-inf"),
) -> float:
    """
    個体から主 fitness を取り出す

    個体のfitnessオブジェクトからwvaluesまたはvaluesを取得し、
    最初の要素（主フィットネス）を返します。

    Args:
        individual: 個体オブジェクト
        default: デフォルト値（デフォルト: -inf）

    Returns:
        float: 主フィットネス値、取得失敗時はデフォルト値

    Note:
        wvaluesを優先的に取得し、次にvaluesを取得します。
    """
    fitness = getattr(individual, "fitness", None)
    if fitness is None:
        return float(default)

    for attr_name in ("wvalues", "values"):
        values = getattr(fitness, attr_name, ())
        if not isinstance(values, (tuple, list)) or not values:
            continue
        try:
            return float(values[0])
        except (TypeError, ValueError):
            continue
    return float(default)


def extract_primary_fitness_from_result(
    result: object,
    *,
    default: float = 0.0,
) -> float:
    """
    評価結果から主 fitness を抽出する

    評価結果からフィットネス値を抽出します。
    タプル/リストの場合は最初の要素、数値の場合はそのまま返します。

    Args:
        result: 評価結果
        default: デフォルト値（デフォルト: 0.0）

    Returns:
        float: フィットネス値、取得失敗時はデフォルト値
    """
    if isinstance(result, (tuple, list)) and result:
        try:
            return float(result[0])
        except (TypeError, ValueError):
            return float(default)
    if isinstance(result, (int, float)):
        return float(result)
    return float(default)


def extract_result_fitness(individual: object) -> float | tuple[float, ...]:
    """
    結果出力用に個体の fitness を整形して返す。

    単一目的なら float、多目的なら tuple[float, ...] として返します。

    Args:
        individual: 個体オブジェクト

    Returns:
        float | tuple[float, ...]: fitness 値。取得できない場合は 0.0。
    """
    fitness = getattr(individual, "fitness", None)
    values = getattr(fitness, "values", ()) if fitness is not None else ()
    return normalize_fitness_values(values)
