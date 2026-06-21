"""
MLトレーニング設定バリデーターのテスト

training_config_validator モジュールの7つのバリデーション関数を
正常系・異常系・境界値でカバーします。
"""

from __future__ import annotations

import pytest

from app.services.ml.orchestration.training_config_validator import (
    validate_date_range,
    validate_ensemble_config,
    validate_model_parameters,
    validate_split_ratios,
    validate_target_kind,
    validate_task_type,
    validate_training_config,
)


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def mock_config():
    """validate_training_config 用のモック設定オブジェクト"""

    class _MockEnsembleConfig:
        def __init__(self, enabled: bool = True):
            self.enabled = enabled

    class _MockConfig:
        start_date = "2024-01-01"
        end_date = "2024-02-01"
        train_test_split = 0.8
        validation_split = 0.2
        prediction_horizon = 4
        cross_validation_folds = 5
        task_type = "volatility_regression"
        target_kind = "log_realized_vol"
        ensemble_config = _MockEnsembleConfig(enabled=False)

    return _MockConfig()


@pytest.fixture
def ensemble_config():
    """アンサンブル設定オブジェクト"""

    class _EnsembleConfig:
        def __init__(self, enabled: bool = True):
            self.enabled = enabled

    return _EnsembleConfig


# ======================================================================
# validate_date_range
# ======================================================================


class TestValidateDateRange:
    """日付範囲バリデーションのテスト"""

    def test_valid_date_range(self):
        """有効な日付範囲（7日以上）はエラーにならないこと"""
        validate_date_range("2024-01-01", "2024-01-10")  # 9日間
        validate_date_range("2024-01-01", "2024-01-08")  # 丁度7日間
        validate_date_range("2024-01-01", "2024-12-31")  # 長期間

    def test_start_date_after_end_date(self):
        """開始日が終了日より後の場合はValueError"""
        with pytest.raises(ValueError, match="開始日は終了日より前である必要があります"):
            validate_date_range("2024-02-01", "2024-01-01")

    def test_start_date_equals_end_date(self):
        """開始日と終了日が同じ日付の場合はエラー"""
        with pytest.raises(ValueError, match="開始日は終了日より前である必要があります"):
            validate_date_range("2024-01-01", "2024-01-01")

    def test_less_than_7_days(self):
        """期間が7日未満の場合はValueError"""
        with pytest.raises(ValueError, match="トレーニング期間は最低7日間必要です"):
            validate_date_range("2024-01-01", "2024-01-06")  # 5日間

        with pytest.raises(ValueError, match="トレーニング期間は最低7日間必要です"):
            validate_date_range("2024-01-01", "2024-01-03")  # 2日間

    def test_exactly_7_days(self):
        """丁度7日間の場合は正常終了すること（境界値）"""
        # 2024-01-01 00:00:00 〜 2024-01-08 00:00:00 で丁度7日間
        validate_date_range("2024-01-01", "2024-01-08")

    def test_invalid_date_format(self):
        """不正な日付形式の場合はValueError"""
        with pytest.raises(ValueError, match="開始日は終了日より前である必要があります"):
            validate_date_range("invalid-date", "2024-01-10")

    def test_none_date(self):
        """日付がNoneの場合はValueError"""
        with pytest.raises(ValueError):
            validate_date_range(None, "2024-01-10")  # type: ignore[arg-type]

    def test_timezone_aware_dates(self):
        """タイムゾーン付き日付も正しく処理されること"""
        validate_date_range("2024-01-01T00:00:00Z", "2024-01-10T00:00:00Z")
        validate_date_range(
            "2024-01-01T00:00:00+09:00", "2024-01-10T00:00:00+09:00"
        )


# ======================================================================
# validate_split_ratios
# ======================================================================


class TestValidateSplitRatios:
    """データ分割比率バリデーションのテスト"""

    def test_valid_ratios(self):
        """有効な分割比率（0 < ratio < 1）はエラーにならないこと"""
        validate_split_ratios(0.8, 0.2)
        validate_split_ratios(0.5, 0.5)
        validate_split_ratios(0.99, 0.01)
        validate_split_ratios(0.01, 0.99)

    @pytest.mark.parametrize(
        "train_test_split, validation_split",
        [
            (0.0, 0.2),  # train_test_split == 0
            (1.0, 0.2),  # train_test_split == 1
            (-0.1, 0.2),  # train_test_split < 0
            (1.5, 0.2),  # train_test_split > 1
            (0.8, 0.0),  # validation_split == 0
            (0.8, 1.0),  # validation_split == 1
            (0.8, -0.1),  # validation_split < 0
            (0.8, 1.5),  # validation_split > 1
        ],
    )
    def test_invalid_ratios(self, train_test_split: float, validation_split: float):
        """無効な分割比率の場合はValueError"""
        with pytest.raises(ValueError):
            validate_split_ratios(train_test_split, validation_split)

    def test_boundary_values(self):
        """境界値（0と1に近い値）のテスト"""
        with pytest.raises(ValueError, match="0 より大きく 1 未満"):
            validate_split_ratios(0.0, 0.2)
        with pytest.raises(ValueError, match="0 より大きく 1 未満"):
            validate_split_ratios(1.0, 0.2)
        with pytest.raises(ValueError, match="0 より大きく 1 未満"):
            validate_split_ratios(0.8, 0.0)
        with pytest.raises(ValueError, match="0 より大きく 1 未満"):
            validate_split_ratios(0.8, 1.0)

    def test_both_invalid(self):
        """両方の比率が無効な場合もValueErrorとなること"""
        with pytest.raises(ValueError):
            validate_split_ratios(0.0, 0.0)
        with pytest.raises(ValueError):
            validate_split_ratios(2.0, 2.0)


# ======================================================================
# validate_model_parameters
# ======================================================================


class TestValidateModelParameters:
    """モデルパラメータバリデーションのテスト"""

    def test_valid_parameters(self):
        """有効なパラメータはエラーにならないこと"""
        validate_model_parameters(1, 1)
        validate_model_parameters(100, 10)
        validate_model_parameters(4, 5)

    @pytest.mark.parametrize(
        "prediction_horizon, cross_validation_folds, match",
        [
            (0, 5, "prediction_horizon は 1 以上"),
            (-1, 5, "prediction_horizon は 1 以上"),
            (4, 0, "cross_validation_folds は 1 以上"),
            (4, -1, "cross_validation_folds は 1 以上"),
        ],
    )
    def test_invalid_parameters(
        self, prediction_horizon: int, cross_validation_folds: int, match: str
    ):
        """無効なパラメータの場合はValueError"""
        with pytest.raises(ValueError, match=match):
            validate_model_parameters(prediction_horizon, cross_validation_folds)

    def test_zero_values(self):
        """0は無効（境界値）"""
        with pytest.raises(ValueError, match="prediction_horizon は 1 以上"):
            validate_model_parameters(0, 5)
        with pytest.raises(ValueError, match="cross_validation_folds は 1 以上"):
            validate_model_parameters(4, 0)

    def test_negative_values(self):
        """負の値は無効"""
        with pytest.raises(ValueError):
            validate_model_parameters(-1, 5)
        with pytest.raises(ValueError):
            validate_model_parameters(4, -5)

    def test_large_values(self):
        """大きな値でも正常に動作すること"""
        validate_model_parameters(9999, 100)


# ======================================================================
# validate_task_type
# ======================================================================


class TestValidateTaskType:
    """タスクタイプバリデーションのテスト"""

    def test_valid_task_type(self):
        """volatility_regression はエラーにならないこと"""
        validate_task_type("volatility_regression")

    def test_invalid_task_type(self):
        """volatility_regression 以外はValueError"""
        with pytest.raises(
            ValueError, match="サポートしている task_type は volatility_regression のみ"
        ):
            validate_task_type("classification")

    @pytest.mark.parametrize(
        "invalid_type",
        [
            "classification",
            "regression",
            "binary_classification",
            "multi_class",
            "",
            None,
        ],
    )
    def test_various_invalid_types(self, invalid_type: str | None):
        """様々な無効なタスクタイプ"""
        with pytest.raises(ValueError):
            validate_task_type(invalid_type)  # type: ignore[arg-type]

    def test_empty_string(self):
        """空文字列も無効"""
        with pytest.raises(ValueError):
            validate_task_type("")


# ======================================================================
# validate_target_kind
# ======================================================================


class TestValidateTargetKind:
    """ターゲット種類バリデーションのテスト"""

    def test_valid_target_kind(self):
        """log_realized_vol はエラーにならないこと"""
        validate_target_kind("log_realized_vol")

    def test_invalid_target_kind(self):
        """log_realized_vol 以外はValueError"""
        with pytest.raises(
            ValueError,
            match="現在サポートしている target_kind は log_realized_vol のみです",
        ):
            validate_target_kind("classification_label")

    @pytest.mark.parametrize(
        "invalid_kind",
        [
            "classification_label",
            "price_direction",
            "regression_target",
            "",
            None,
        ],
    )
    def test_various_invalid_kinds(self, invalid_kind: str | None):
        """様々な無効なターゲット種類"""
        with pytest.raises(ValueError):
            validate_target_kind(invalid_kind)  # type: ignore[arg-type]

    def test_empty_string(self):
        """空文字列も無効"""
        with pytest.raises(ValueError):
            validate_target_kind("")


# ======================================================================
# validate_ensemble_config
# ======================================================================


class TestValidateEnsembleConfig:
    """アンサンブル設定バリデーションのテスト"""

    def test_no_ensemble_config(self):
        """ensemble_config=None の場合はエラーにならないこと"""
        validate_ensemble_config("volatility_regression", None)

    def test_ensemble_disabled(self):
        """ensemble_config.enabled=False の場合はエラーにならないこと"""

        class _Config:
            enabled = False

        validate_ensemble_config("volatility_regression", _Config())

    def test_ensemble_enabled_with_volatility_regression(self):
        """volatility_regression + ensemble enabled はValueError"""

        class _Config:
            enabled = True

        with pytest.raises(
            ValueError,
            match="volatility_regression では ensemble_config はサポートしていません",
        ):
            validate_ensemble_config("volatility_regression", _Config())

    def test_ensemble_enabled_with_other_task_type(self):
        """volatility_regression以外 + ensemble enabled はエラーにならないこと"""

        class _Config:
            enabled = True

        validate_ensemble_config("classification", _Config())

    def test_ensemble_config_without_enabled_attribute(self):
        """enabled属性を持たないオブジェクトはエラーにならないこと"""
        validate_ensemble_config("volatility_regression", object())  # enabled=Falseとみなされる


# ======================================================================
# validate_training_config（統合テスト）
# ======================================================================


class TestValidateTrainingConfig:
    """validate_training_config 統合テスト"""

    def test_valid_config(self, mock_config):
        """有効な設定はエラーにならないこと"""
        validate_training_config(mock_config)

    def test_invalid_date_range(self, mock_config):
        """無効な日付範囲の場合はValueError"""
        mock_config.start_date = "2024-02-01"
        mock_config.end_date = "2024-01-01"
        with pytest.raises(ValueError, match="開始日は終了日より前"):
            validate_training_config(mock_config)

    def test_invalid_split_ratios(self, mock_config):
        """無効な分割比率の場合はValueError"""
        mock_config.train_test_split = 0.0
        with pytest.raises(ValueError, match="train_test_split は 0 より大きく"):
            validate_training_config(mock_config)

    def test_invalid_model_parameters(self, mock_config):
        """無効なモデルパラメータの場合はValueError"""
        mock_config.prediction_horizon = 0
        with pytest.raises(ValueError, match="prediction_horizon は 1 以上"):
            validate_training_config(mock_config)

    def test_invalid_task_type(self, mock_config):
        """無効なタスクタイプの場合はValueError"""
        mock_config.task_type = "classification"
        with pytest.raises(ValueError, match="サポートしている task_type"):
            validate_training_config(mock_config)

    def test_invalid_target_kind(self, mock_config):
        """無効なターゲット種類の場合はValueError"""
        mock_config.target_kind = "invalid"
        with pytest.raises(ValueError, match="サポートしている target_kind"):
            validate_training_config(mock_config)

    def test_ensemble_enabled_for_vol_reg(self, mock_config):
        """volatility_regression で ensemble enabled はValueError"""

        class _EnsembleConfig:
            enabled = True

        mock_config.ensemble_config = _EnsembleConfig()
        with pytest.raises(ValueError, match="ensemble_config はサポートしていません"):
            validate_training_config(mock_config)

    def test_multiple_errors_first_one_reported(self, mock_config):
        """複数のバリデーションエラーがある場合、最初のエラーが報告されること"""
        mock_config.start_date = "2024-02-01"
        mock_config.end_date = "2024-01-01"
        mock_config.train_test_split = 0.0
        # start_date > end_date が先にチェックされる
        with pytest.raises(ValueError, match="開始日は終了日より前"):
            validate_training_config(mock_config)


# ======================================================================
# エッジケース
# ======================================================================


class TestEdgeCases:
    """境界値・エッジケースのテスト"""

    def test_split_ratios_close_to_boundaries(self):
        """分割比率が境界値ぎりぎりのケース"""
        # 有効
        validate_split_ratios(0.001, 0.999)
        validate_split_ratios(0.999, 0.001)

    def test_prediction_horizon_minimum(self):
        """prediction_horizonの最小値（1）"""
        validate_model_parameters(1, 5)

    def test_cross_validation_folds_minimum(self):
        """cross_validation_foldsの最小値（1）"""
        validate_model_parameters(4, 1)

    def test_validate_training_config_with_empty_ensemble(self):
        """ensemble_configが空オブジェクトの場合"""

        class _EmptyEnsemble:
            pass

        class _Config:
            start_date = "2024-01-01"
            end_date = "2024-01-10"
            train_test_split = 0.8
            validation_split = 0.2
            prediction_horizon = 4
            cross_validation_folds = 5
            task_type = "volatility_regression"
            target_kind = "log_realized_vol"
            ensemble_config = _EmptyEnsemble()

        config = _Config()
        # enabled属性がないのでFalse扱いになりエラーにならない
        validate_training_config(config)
