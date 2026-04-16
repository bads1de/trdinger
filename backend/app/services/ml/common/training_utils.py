"""
ML 学習用ユーティリティ

学習/評価オーケストレーションで繰り返し使う
分割比率の決定ロジックを共通化します。
"""

from __future__ import annotations

from typing import Optional


def resolve_holdout_test_size(
    *,
    test_size: Optional[float] = None,
    train_test_split: Optional[float] = None,
    validation_split: Optional[float] = None,
    default_train_split: float = 0.8,
    default_validation_split: float = 0.2,
) -> float:
    """
    ホールドアウト分割比率を決定する。

    複数のパラメータ（test_size、train_test_split、validation_split）から
    テストデータの分割比率を決定します。優先順位は以下の通りです：
    1. test_size（直接指定）
    2. train_test_split（1 - train_test_split）
    3. validation_split（直接使用）
    4. default_train_split（1 - default_train_split）

    Args:
        test_size: テストデータの分割比率（0-1の範囲）
        train_test_split: 学習データの分割比率（0-1の範囲）
        validation_split: 検証データの分割比率（0-1の範囲）
        default_train_split: デフォルトの学習データ分割比率（デフォルト: 0.8）
        default_validation_split: デフォルトの検証データ分割比率（デフォルト: 0.2）

    Returns:
        float: テストデータの分割比率（0-1の範囲）

    Note:
        train_test_splitがデフォルト値から変更されている場合、
        validation_splitよりも優先されます。
    """
    if test_size is not None:
        return float(test_size)

    effective_train_split = (
        default_train_split if train_test_split is None else float(train_test_split)
    )
    effective_validation_split = (
        default_validation_split
        if validation_split is None
        else float(validation_split)
    )

    # 既存の train_test_split を優先し、未変更なら validation_split を使う
    if effective_train_split != default_train_split:
        return 1 - effective_train_split
    if effective_validation_split != default_validation_split:
        return effective_validation_split
    return 1 - effective_train_split
