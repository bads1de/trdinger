"""
exceptions モジュールのユニットテスト
"""


from app.services.ml.common.exceptions import (
    MLBaseError,
    MLDataError,
    MLFeatureError,
    MLModelError,
    MLPredictionError,
    MLTrainingError,
    MLValidationError,
)


class TestMLBaseError:
    def test_is_exception(self):
        assert issubclass(MLBaseError, Exception)

    def test_message(self):
        err = MLBaseError("test message")
        assert str(err) == "test message"
        assert err.message == "test message"
        assert err.error_code is None

    def test_with_error_code(self):
        err = MLBaseError("msg", error_code="CODE")
        assert err.error_code == "CODE"


class TestMLDataError:
    def test_is_ml_base_error(self):
        assert issubclass(MLDataError, MLBaseError)

    def test_default_data_info(self):
        err = MLDataError("data error")
        assert err.data_info == {}
        assert err.error_code == "ML_DATA_ERROR"

    def test_with_data_info(self):
        err = MLDataError("data error", data_info={"rows": 100})
        assert err.data_info == {"rows": 100}


class TestMLValidationError:
    def test_is_ml_base_error(self):
        assert issubclass(MLValidationError, MLBaseError)

    def test_default_validation_details(self):
        err = MLValidationError("val error")
        assert err.validation_details == {}
        assert err.error_code == "ML_VALIDATION_ERROR"


class TestMLModelError:
    def test_is_ml_base_error(self):
        assert issubclass(MLModelError, MLBaseError)

    def test_default_model_info(self):
        err = MLModelError("model error")
        assert err.model_info == {}
        assert err.error_code == "ML_MODEL_ERROR"

    def test_custom_error_code(self):
        err = MLModelError("err", error_code="CUSTOM")
        assert err.error_code == "CUSTOM"


class TestMLTrainingError:
    def test_is_ml_base_error(self):
        assert issubclass(MLTrainingError, MLBaseError)

    def test_default_training_info(self):
        err = MLTrainingError("train error")
        assert err.training_info == {}
        assert err.error_code == "ML_TRAINING_ERROR"


class TestMLPredictionError:
    def test_is_ml_base_error(self):
        assert issubclass(MLPredictionError, MLBaseError)

    def test_default_prediction_info(self):
        err = MLPredictionError("pred error")
        assert err.prediction_info == {}
        assert err.error_code == "ML_PREDICTION_ERROR"


class TestMLFeatureError:
    def test_is_ml_base_error(self):
        assert issubclass(MLFeatureError, MLBaseError)

    def test_default_feature_info(self):
        err = MLFeatureError("feat error")
        assert err.feature_info == {}
        assert err.error_code == "ML_FEATURE_ERROR"
