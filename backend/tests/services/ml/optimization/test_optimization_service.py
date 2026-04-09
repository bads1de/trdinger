"""
OptimizationService のテスト

app/services/ml/optimization/optimization_service.py のテストモジュール
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import pandas as pd
import numpy as np

from app.services.ml.optimization.optimization_service import (
    OptimizationService,
    OptimizationSettings,
)


class TestOptimizationSettings:
    """OptimizationSettings クラスのテスト"""

    def test_initialization_with_defaults(self):
        """デフォルト値での初期化"""
        settings = OptimizationSettings()
        assert settings.enabled is False
        assert settings.n_calls == 50
        assert settings.parameter_space == {}

    def test_initialization_with_custom_values(self):
        """カスタム値での初期化"""
        custom_space = {"learning_rate": {"type": "real", "low": 0.001, "high": 0.1}}
        settings = OptimizationSettings(
            enabled=True, n_calls=100, parameter_space=custom_space
        )
        assert settings.enabled is True
        assert settings.n_calls == 100
        assert settings.parameter_space == custom_space


class TestOptimizationServiceInitialization:
    """OptimizationService 初期化のテスト"""

    def test_service_initialization(self):
        """サービスの初期化"""
        service = OptimizationService()
        assert service.optimizer is not None


class TestOptimizeParameters:
    """optimize_parameters メソッドのテスト"""

    @pytest.fixture
    def mock_trainer(self):
        """モックトレーナー"""
        trainer = MagicMock()
        trainer.ensemble_config = {"method": "stacking", "models": ["lightgbm"]}
        return trainer

    @pytest.fixture
    def sample_data(self):
        """サンプルデータ"""
        return pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
                "target": np.random.randint(0, 2, 100),
            }
        )

    @pytest.fixture
    def optimization_settings(self):
        """最適化設定"""
        return OptimizationSettings(enabled=True, n_calls=5)

    def test_optimize_parameters_disabled(self, mock_trainer, sample_data):
        """最適化が無効な場合のテスト"""
        settings = OptimizationSettings(enabled=False)
        service = OptimizationService()

        # 無効な場合は早期リターンまたはエラー
        # 実装に応じて調整
        result = service.optimize_parameters(
            trainer=mock_trainer,
            training_data=sample_data,
            optimization_settings=settings,
        )
        # 結果の構造を確認
        assert isinstance(result, dict)

    @patch("app.services.ml.optimization.optimization_service.EnsembleTrainer")
    def test_optimize_parameters_success(
        self, mock_ensemble_trainer, mock_trainer, sample_data, optimization_settings
    ):
        """最適化成功時のテスト"""
        # モックの設定
        mock_temp_trainer = MagicMock()
        mock_temp_trainer.train_model.return_value = {
            "f1_score": 0.8,
            "classification_report": {"macro avg": {"f1-score": 0.85}},
        }
        mock_ensemble_trainer.return_value = mock_temp_trainer

        service = OptimizationService()

        result = service.optimize_parameters(
            trainer=mock_trainer,
            training_data=sample_data,
            optimization_settings=optimization_settings,
        )

        # 結果の構造を確認
        assert "method" in result
        assert "best_params" in result
        assert "best_score" in result
        assert "total_evaluations" in result
        assert "optimization_time" in result

    def test_optimize_parameters_with_additional_data(
        self, mock_trainer, sample_data, optimization_settings
    ):
        """追加データ（funding_rate, open_interest）を使用したテスト"""
        funding_data = pd.DataFrame(
            {"funding_rate": np.random.randn(100) * 0.01}
        )
        oi_data = pd.DataFrame({"open_interest": np.random.randn(100) * 1000})

        service = OptimizationService()

        with patch("app.services.ml.optimization.optimization_service.EnsembleTrainer"):
            result = service.optimize_parameters(
                trainer=mock_trainer,
                training_data=sample_data,
                optimization_settings=optimization_settings,
                funding_rate_data=funding_data,
                open_interest_data=oi_data,
            )

            assert isinstance(result, dict)


class TestPrepareParameterSpace:
    """_prepare_parameter_space メソッドのテスト"""

    @pytest.fixture
    def service(self):
        return OptimizationService()

    def test_with_custom_parameter_space(self, service):
        """カスタムパラメータ空間を使用する場合"""
        mock_trainer = MagicMock()
        custom_space = {
            "learning_rate": {"type": "real", "low": 0.001, "high": 0.1}
        }
        settings = OptimizationSettings(enabled=True, parameter_space=custom_space)

        result = service._prepare_parameter_space(mock_trainer, settings)

        assert isinstance(result, dict)
        assert "learning_rate" in result

    def test_with_ensemble_config(self, service):
        """EnsembleTrainerの設定を使用する場合"""
        mock_trainer = MagicMock()
        mock_trainer.ensemble_config = {"method": "stacking", "models": ["lightgbm"]}
        settings = OptimizationSettings(enabled=True)

        with patch.object(
            service.optimizer, "get_ensemble_parameter_space"
        ) as mock_get_space:
            mock_get_space.return_value = {"param1": "value1"}

            result = service._prepare_parameter_space(mock_trainer, settings)

            mock_get_space.assert_called_once()

    def test_with_default_space(self, service):
        """デフォルトパラメータ空間を使用する場合"""
        mock_trainer = MagicMock()
        # ensemble_configを持たないトレーナー
        del mock_trainer.ensemble_config
        settings = OptimizationSettings(enabled=True)

        with patch.object(service.optimizer, "get_default_parameter_space"):
            result = service._prepare_parameter_space(mock_trainer, settings)


class TestConvertParameterSpaceConfig:
    """_convert_parameter_space_config メソッドのテスト"""

    @pytest.fixture
    def service(self):
        return OptimizationService()

    def test_convert_real_parameter(self, service):
        """real型パラメータの変換"""
        config = {"learning_rate": {"type": "real", "low": 0.001, "high": 0.1}}
        result = service._convert_parameter_space_config(config)

        assert "learning_rate" in result
        assert result["learning_rate"].type == "real"
        assert result["learning_rate"].low == 0.001
        assert result["learning_rate"].high == 0.1

    def test_convert_integer_parameter(self, service):
        """integer型パラメータの変換"""
        config = {"n_estimators": {"type": "integer", "low": 10, "high": 1000}}
        result = service._convert_parameter_space_config(config)

        assert "n_estimators" in result
        assert result["n_estimators"].type == "integer"
        assert result["n_estimators"].low == 10
        assert result["n_estimators"].high == 1000

    def test_convert_categorical_parameter(self, service):
        """categorical型パラメータの変換"""
        config = {"boosting_type": {"type": "categorical", "categories": ["gbdt", "dart"]}}
        result = service._convert_parameter_space_config(config)

        assert "boosting_type" in result
        assert result["boosting_type"].type == "categorical"
        assert result["boosting_type"].categories == ["gbdt", "dart"]


class TestCreateObjectiveFunction:
    """_create_objective_function メソッドのテスト"""

    @pytest.fixture
    def service(self):
        return OptimizationService()

    @pytest.fixture
    def mock_trainer(self):
        trainer = MagicMock()
        trainer.ensemble_config = {"method": "stacking"}
        return trainer

    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
                "target": np.random.randint(0, 2, 100),
            }
        )

    def test_objective_function_returns_score(
        self, service, mock_trainer, sample_data
    ):
        """目的関数がスコアを返すこと"""
        settings = OptimizationSettings(enabled=True, n_calls=5)

        with patch.object(
            service, "_create_temp_trainer"
        ) as mock_create_temp:
            mock_temp_trainer = MagicMock()
            mock_temp_trainer.train_model.return_value = {
                "f1_score": 0.8,
                "classification_report": {"macro avg": {"f1-score": 0.85}},
            }
            mock_create_temp.return_value = mock_temp_trainer

            objective = service._create_objective_function(
                trainer=mock_trainer,
                training_data=sample_data,
                optimization_settings=settings,
            )

            score = objective({"learning_rate": 0.1})

            assert isinstance(score, float)
            assert 0 <= score <= 1

    def test_objective_function_handles_exception(
        self, service, mock_trainer, sample_data
    ):
        """目的関数が例外を処理すること"""
        settings = OptimizationSettings(enabled=True, n_calls=5)

        with patch.object(service, "_create_temp_trainer") as mock_create_temp:
            mock_temp_trainer = MagicMock()
            mock_temp_trainer.train_model.side_effect = Exception("Test error")
            mock_create_temp.return_value = mock_temp_trainer

            objective = service._create_objective_function(
                trainer=mock_trainer,
                training_data=sample_data,
                optimization_settings=settings,
            )

            score = objective({"learning_rate": 0.1})

            # エラー時は0を返す
            assert score == 0.0


class TestCreateTempTrainer:
    """_create_temp_trainer メソッドのテスト"""

    @pytest.fixture
    def service(self):
        return OptimizationService()

    def test_creates_temp_trainer_from_ensemble_config(self, service):
        """EnsembleTrainerから一時トレーナーを作成"""
        original_trainer = MagicMock()
        original_trainer.ensemble_config = {
            "method": "stacking",
            "models": ["lightgbm"],
            "stacking_params": {"cv_folds": 5},
        }

        with patch(
            "app.services.ml.optimization.optimization_service.EnsembleTrainer"
        ) as mock_ensemble:
            service._create_temp_trainer(original_trainer, {})

            # 呼び出しを確認
            mock_ensemble.assert_called_once()
            call_kwargs = mock_ensemble.call_args.kwargs
            temp_config = call_kwargs["ensemble_config"]

            # CV foldsが減らされていることを確認
            assert temp_config["stacking_params"]["cv_folds"] == 3

    def test_fallback_to_default_config(self, service):
        """デフォルト設定にフォールバック"""
        original_trainer = MagicMock()
        # ensemble_configを持たない
        del original_trainer.ensemble_config

        with patch(
            "app.services.ml.optimization.optimization_service.EnsembleTrainer"
        ) as mock_ensemble:
            service._create_temp_trainer(original_trainer, {})

            mock_ensemble.assert_called_once_with(
                ensemble_config={"method": "stacking"}
            )


class TestOptimizeFullPipeline:
    """optimize_full_pipeline メソッドのテスト"""

    @pytest.fixture
    def service(self):
        return OptimizationService()

    @pytest.fixture
    def sample_superset(self):
        """サンプル特徴量スーパーセット"""
        dates = pd.date_range("2023-01-01", periods=500, freq="h")
        data = {
            f"feature_{d}": np.random.randn(500) for d in [0.3, 0.4, 0.5, 0.6]
        }
        return pd.DataFrame(data, index=dates)

    @pytest.fixture
    def sample_labels(self):
        """サンプルラベル"""
        return pd.Series(np.random.randint(0, 2, 500))

    @pytest.fixture
    def sample_ohlcv(self):
        """サンプルOHLCVデータ"""
        dates = pd.date_range("2023-01-01", periods=500, freq="h")
        return pd.DataFrame(
            {
                "open": np.random.randn(500) * 10 + 100,
                "high": np.random.randn(500) * 10 + 105,
                "low": np.random.randn(500) * 10 + 95,
                "close": np.random.randn(500) * 10 + 100,
                "volume": np.random.randn(500) * 1000 + 5000,
            },
            index=dates,
        )

    def test_optimize_full_pipeline_basic(
        self, service, sample_superset, sample_labels, sample_ohlcv
    ):
        """パイプライン最適化の基本テスト"""
        with patch.object(
            service, "_generate_pipeline_labels"
        ) as mock_generate_labels:
            mock_generate_labels.return_value = sample_labels

            with patch.object(
                service, "_evaluate_selected_model_pipeline"
            ) as mock_evaluate:
                mock_evaluate.return_value = (0.8, 10)

                result = service.optimize_full_pipeline(
                    feature_superset=sample_superset,
                    labels=sample_labels,
                    ohlcv_data=sample_ohlcv,
                    n_trials=2,  # テスト用に少なく
                )

                # 結果の構造を確認
                assert isinstance(result, dict)
                assert "best_params" in result
                assert "best_score" in result
                assert "test_score" in result

    def test_optimize_full_pipeline_with_fixed_params(
        self, service, sample_superset, sample_labels, sample_ohlcv
    ):
        """固定パラメータを使用したパイプライン最適化"""
        fixed_params = {"tbm_horizon": 24}

        with patch.object(
            service, "_generate_pipeline_labels"
        ) as mock_generate_labels:
            mock_generate_labels.return_value = sample_labels

            with patch.object(
                service, "_evaluate_selected_model_pipeline"
            ) as mock_evaluate:
                mock_evaluate.return_value = (0.8, 10)

                result = service.optimize_full_pipeline(
                    feature_superset=sample_superset,
                    labels=sample_labels,
                    ohlcv_data=sample_ohlcv,
                    n_trials=2,
                    fixed_label_params=fixed_params,
                )

                # 固定パラメータが結果に含まれることを確認
                assert result["best_params"]["tbm_horizon"] == 24


class TestHelperMethods:
    """ヘルパーメソッドのテスト"""

    @pytest.fixture
    def service(self):
        return OptimizationService()

    def test_align_feature_superset_and_labels(self, service):
        """特徴量とラベルのアライメントテスト"""
        superset = pd.DataFrame(
            {"feature1": [1, 2, 3]}, index=pd.date_range("2023-01-01", periods=3)
        )
        labels = pd.Series(
            [0, 1, 0], index=pd.date_range("2023-01-01", periods=3)
        )

        X_aligned, y_aligned = service._align_feature_superset_and_labels(
            superset, labels
        )

        assert len(X_aligned) == len(y_aligned)
        assert len(X_aligned) == 3

    def test_split_by_date(self, service):
        """日付による分割テスト"""
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        X = pd.DataFrame({"feature1": np.random.randn(100)}, index=dates)
        y = pd.Series(np.random.randint(0, 2, 100), index=dates)
        split_date = dates[70]

        X_train, y_train, X_test, y_test = service._split_by_date(X, y, split_date)

        assert len(X_train) == 70
        assert len(X_test) == 30
        assert len(y_train) == 70
        assert len(y_test) == 30

    def test_get_pipeline_parameter_space(self, service):
        """パイプラインパラメータ空間の取得テスト"""
        d_values = [0.3, 0.4, 0.5]

        space = service._get_pipeline_parameter_space(d_values)

        assert "frac_diff_d" in space
        assert "selection_method" in space
        assert "learning_rate" in space
        assert "num_leaves" in space

    def test_get_pipeline_parameter_space_with_fixed_params(self, service):
        """固定パラメータを使用したパイプラインパラメータ空間の取得"""
        d_values = [0.3, 0.4, 0.5]
        fixed_params = {"tbm_horizon": 24}

        space = service._get_pipeline_parameter_space(d_values, fixed_params)

        # 固定パラメータは空間に含まれない
        assert "tbm_horizon" not in space
        # 他のパラメータは含まれる
        assert "frac_diff_d" in space
        assert "tbm_pt" in space  # 固定されていないので含まれる


class TestOptimizeMetaModelWithOOF:
    """optimize_meta_model_with_oof メソッドのテスト"""

    # 注: このメソッドは複雑な依存関係（TimeSeriesSplit、実際のパイプラインfitなど）があり、
    # ユニットテストで適切にモックするのが困難です。統合テストでカバーすることを推奨します。

    @pytest.mark.skip(reason="複雑すぎてユニットテストで適切にモックできない")
    def test_optimize_meta_model_with_oof_success(self):
        """メタモデル最適化の成功テスト"""
        pass

    @pytest.mark.skip(reason="複雑すぎてユニットテストで適切にモックできない")
    def test_optimize_meta_model_with_oof_insufficient_samples(self):
        """サンプル数が不足している場合のテスト"""
        pass

    @pytest.mark.skip(reason="複雑すぎてユニットテストで適切にモックできない")
    def test_optimize_meta_model_with_oof_exception_handling(self):
        """例外処理のテスト"""
        pass
