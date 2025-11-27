"""
ML共通ユーティリティ関数

データ検証、ログ出力など、ML処理で頻繁に使用される共通ロジックを提供します。
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def validate_training_inputs(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame | None = None,
    y_test: pd.Series | None = None,
    log_info: bool = True,
) -> None:
    """
    学習用データの検証を行う共通関数

    Args:
        X_train: 学習用特徴量
        y_train: 学習用ターゲット
        X_test: テスト用特徴量（オプション）
        y_test: テスト用ターゲット（オプション）
        log_info: データサイズをログ出力するか

    Raises:
        ValueError: データが無効な場合
    """
    # 入力データの検証
    if X_train is None or X_train.empty:
        raise ValueError("学習用特徴量データが空です")
    if y_train is None or len(y_train) == 0:
        raise ValueError("学習用ターゲットデータが空です")
    if len(X_train) != len(y_train):
        raise ValueError("特徴量とターゲットの長さが一致しません")

    # 情報ログ
    if log_info:
        logger.info(f"学習データサイズ: {len(X_train)}行, {len(X_train.columns)}特徴量")
        logger.info(f"ターゲット分布: {y_train.value_counts().to_dict()}")

        if X_test is not None:
            logger.info(f"テストデータサイズ: {len(X_test)}行")
