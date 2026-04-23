"""
MLトレーニング設定のバリデーション

トレーニング設定の妥当性を検証するためのサブバリデーターを集約します。
"""

from typing import Any

from app.utils.datetime_utils import parse_datetime_range_optional


def validate_date_range(start_date: str, end_date: str) -> None:
    """
    日付範囲のバリデーション

    Args:
        start_date: 開始日
        end_date: 終了日

    Raises:
        ValueError: 日付範囲が不正な場合
    """
    date_range = parse_datetime_range_optional(start_date, end_date)
    if date_range is None:
        raise ValueError("開始日は終了日より前である必要があります")

    start_dt, end_dt = date_range
    if (end_dt - start_dt).days < 7:
        raise ValueError("トレーニング期間は最低7日間必要です")


def validate_split_ratios(
    train_test_split: float,
    validation_split: float,
) -> None:
    """
    データ分割比率のバリデーション

    Args:
        train_test_split: 訓練・テスト分割比率
        validation_split: バリデーション分割比率

    Raises:
        ValueError: 分割比率が不正な場合
    """
    if not 0 < train_test_split < 1:
        raise ValueError(
            "train_test_split は 0 より大きく 1 未満である必要があります"
        )
    if not 0 < validation_split < 1:
        raise ValueError(
            "validation_split は 0 より大きく 1 未満である必要があります"
        )


def validate_model_parameters(
    prediction_horizon: int,
    cross_validation_folds: int,
) -> None:
    """
    モデルパラメータのバリデーション

    Args:
        prediction_horizon: 予測期間
        cross_validation_folds: クロスバリデーションのフォールド数

    Raises:
        ValueError: モデルパラメータが不正な場合
    """
    if prediction_horizon < 1:
        raise ValueError("prediction_horizon は 1 以上である必要があります")
    if cross_validation_folds < 1:
        raise ValueError(
            "cross_validation_folds は 1 以上である必要があります"
        )


def validate_task_type(task_type: str) -> None:
    """
    タスクタイプのバリデーション

    Args:
        task_type: タスクタイプ文字列

    Raises:
        ValueError: タスクタイプがサポートされていない場合
    """
    if task_type != "volatility_regression":
        raise ValueError(
            "現在サポートしている task_type は volatility_regression のみです"
        )


def validate_target_kind(target_kind: str) -> None:
    """
    ターゲット種類のバリデーション

    Args:
        target_kind: ターゲット種類文字列

    Raises:
        ValueError: ターゲット種類がサポートされていない場合
    """
    if target_kind != "log_realized_vol":
        raise ValueError(
            "現在サポートしている target_kind は log_realized_vol のみです"
        )


def validate_ensemble_config(
    task_type: str,
    ensemble_config: Any,
) -> None:
    """
    アンサンブル設定のバリデーション

    Args:
        task_type: タスクタイプ
        ensemble_config: アンサンブル設定オブジェクト

    Raises:
        ValueError: アンサンブル設定が不正な場合
    """
    if task_type == "volatility_regression" and ensemble_config:
        if getattr(ensemble_config, "enabled", False):
            raise ValueError(
                "volatility_regression では ensemble_config はサポートしていません"
            )


def validate_training_config(config) -> None:
    """
    トレーニング設定の包括的バリデーション（エントリーポイント）

    各サブバリデーション関数を呼び出して、トレーニング設定全体の
    妥当性を検証します。

    Args:
        config: MLTrainingConfigオブジェクト

    Raises:
        ValueError: 設定値が不正な場合
    """
    validate_date_range(config.start_date, config.end_date)
    validate_split_ratios(config.train_test_split, config.validation_split)
    validate_model_parameters(
        config.prediction_horizon, config.cross_validation_folds
    )
    validate_task_type(config.task_type)
    validate_target_kind(config.target_kind)
    validate_ensemble_config(config.task_type, config.ensemble_config)
