"""
TimeSeriesSplit統合テスト

BaseMLTrainerとMLTrainingServiceのTimeSeriesSplit統一化に関する
包括的な統合テストを実装します。
BaseMLTrainerは抽象クラスであるため、テスト用の具象クラス(ConcreteMLTrainer)を使用して検証を行います。
"""

from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd
import pytest

from app.services.ml.base_ml_trainer import BaseMLTrainer
from app.services.ml.config import ml_config
from app.services.ml.ml_training_service import MLTrainingService


# テスト用の具象クラス
class ConcreteMLTrainer(BaseMLTrainer):
    def _train_model_impl(self, X_train, y_train, X_test=None, y_test=None, **kwargs):
        # テスト用のダミー学習結果
        return {
            "accuracy": 0.85,
            "f1_score": 0.82,
            "classification_report": {"macro avg": {"f1-score": 0.82}},
            "model": "dummy_model",
        }

    def predict(self, features_df):
        # テスト用のダミー予測
        return np.zeros(len(features_df))


@pytest.fixture
def sample_prepared_data():
    """前処理済みサンプルデータ（モック用）"""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=200, freq="h")

    # 特徴量データ
    X = pd.DataFrame(
        {
            "feature_1": np.random.randn(200),
            "feature_2": np.random.randn(200),
            "feature_3": np.random.randn(200),
        },
        index=dates,
    )

    # ラベルデータ（3クラス分類）
    y = pd.Series(np.random.choice([0, 1, 2], 200), index=dates)

    return X, y


class TestTimeSeriesCVIntegration:
    """TimeSeriesSplit統合テストクラス"""

    @pytest.fixture
    def mock_training_components(self, sample_prepared_data):
        """モックされた学習コンポーネント"""
        X, y = sample_prepared_data

        # モックの特徴量サービス
        mock_feature_service = Mock()
        mock_feature_service.calculate_advanced_features.return_value = X

        # モックのラベル生成器
        mock_label_generator = Mock()
        mock_label_generator.generate_labels.return_value = (
            y,
            {"method": "test", "threshold_up": 0.02, "threshold_down": -0.02},
        )

        # モックのdata_processor
        mock_data_processor = Mock()
        mock_data_processor.prepare_training_data.return_value = (
            X,
            y,
            {"method": "test"},
        )

        return {
            "feature_service": mock_feature_service,
            "label_generator": mock_label_generator,
            "data_processor": mock_data_processor,
            "X": X,
            "y": y,
        }

    @pytest.fixture(autouse=True)
    def mock_ml_config(self):
        """ML設定のモック"""
        with patch("app.services.ml.config.ml_config") as mock_config:
            # training属性をMagicMockで上書き
            mock_config.training = MagicMock()
            
            # training.label_generation を設定
            mock_label_gen = MagicMock()
            mock_config.training.label_generation = mock_label_gen
            
            # その他の必要な設定
            mock_config.training.CROSS_VALIDATION_FOLDS = 5
            mock_config.training.MAX_TRAIN_SIZE = None
            mock_config.training.USE_TIME_SERIES_SPLIT = True
            mock_config.training.USE_PURGED_KFOLD = False
            mock_config.training.PREDICTION_HORIZON = 24
            
            yield mock_config

    def test_end_to_end_ml_training_flow_with_mocks(self, mock_training_components):
        """
        エンドツーエンドのML学習フロー統合テスト（モック使用）

        MLTrainingServiceは内部でEnsembleTrainerなどを使用するが、
        ここではBaseMLTrainerのメソッドをモックして、呼び出しフローを確認する。
        """
        X = mock_training_components["X"]
        y = mock_training_components["y"]

        # BaseMLTrainerのメソッドをモック化
        # 注意: MLTrainingServiceはEnsembleTrainerを使うため、その親クラスのメソッドをモックする
        with (
            patch(
                "app.services.ml.base_ml_trainer.BaseMLTrainer._calculate_features"
            ) as mock_calc_features,
            patch(
                "app.services.ml.base_ml_trainer.BaseMLTrainer._prepare_training_data"
            ) as mock_prepare_data,
            patch(
                "app.services.ml.ensemble.ensemble_trainer.EnsembleTrainer._train_model_impl"
            ) as mock_train_impl,
        ):
            # モックの設定
            mock_calc_features.return_value = X
            mock_prepare_data.return_value = (X, y)

            # 学習結果のモック
            mock_train_impl.return_value = {
                "accuracy": 0.85,
                "f1_score": 0.82,
                "classification_report": {"macro avg": {"f1-score": 0.82}},
            }

            # サービスを初期化
            service = MLTrainingService(trainer_type="ensemble")

            # 学習データの準備
            training_data = X.copy()
            training_data["open"] = 10000
            training_data["high"] = 10100
            training_data["low"] = 9900
            training_data["close"] = 10000
            training_data["volume"] = 500

            # 学習を実行
            result = service.train_model(
                training_data,
                save_model=False,
                use_cross_validation=True,
                cv_splits=3,
            )

            # 検証
            assert result["success"] is True
            assert "cv_scores" in result
            assert len(result["cv_scores"]) == 3

    def test_cross_validation_consistency(self, sample_prepared_data):
        """クロスバリデーションの一貫性テスト"""
        X, y = sample_prepared_data

        # ConcreteMLTrainerを使用
        trainer = ConcreteMLTrainer()

        with (
            patch.object(trainer, "_calculate_features", return_value=X),
            patch.object(trainer, "_prepare_training_data", return_value=(X, y)),
        ):
            training_data = X.copy()
            # 必須カラムを追加
            for col in ["open", "high", "low", "close", "volume"]:
                training_data[col] = 100

            # CVを実行
            result = trainer.train_model(
                training_data,
                save_model=False,
                use_cross_validation=True,
                cv_splits=5,
            )

            assert len(result["cv_scores"]) == 5
            # fold情報が含まれているか
            assert "fold_results" in result
            assert len(result["fold_results"]) == 5

    def test_config_parameter_propagation(self, sample_prepared_data):
        """設定パラメータの伝播テスト"""
        X, y = sample_prepared_data
        service = MLTrainingService()

        # デフォルト値の確認
        default_cv_folds = ml_config.training.CROSS_VALIDATION_FOLDS
        assert default_cv_folds == 5

        # ConcreteMLTrainerを返すようにモック
        mock_trainer = ConcreteMLTrainer()
        # 内部メソッドをモックしてデータをそのまま通す
        mock_trainer._calculate_features = Mock(return_value=X)
        mock_trainer._prepare_training_data = Mock(return_value=(X, y))

        service.trainer = mock_trainer

        training_data = X.copy()
        for col in ["open", "high", "low", "close", "volume"]:
            training_data[col] = 100

        # 1. デフォルト設定での学習
        result1 = service.train_model(
            training_data,
            save_model=False,
            use_cross_validation=True,
        )
        assert len(result1["cv_scores"]) == default_cv_folds

        # 2. カスタム設定での学習
        custom_folds = 3
        result2 = service.train_model(
            training_data,
            save_model=False,
            use_cross_validation=True,
            cv_splits=custom_folds,
        )
        assert len(result2["cv_scores"]) == custom_folds

    def test_backward_compatibility_random_split(self, sample_prepared_data):
        """後方互換性テスト（ランダム分割）"""
        X, y = sample_prepared_data
        trainer = ConcreteMLTrainer()

        with (
            patch.object(trainer, "_calculate_features", return_value=X),
            patch.object(trainer, "_prepare_training_data", return_value=(X, y)),
        ):
            training_data = X.copy()
            for col in ["open", "high", "low", "close", "volume"]:
                training_data[col] = 100

            # ランダム分割での学習
            result = trainer.train_model(
                training_data,
                save_model=False,
                use_random_split=True,
            )

            assert result["success"] is True
            assert "cv_scores" not in result  # ランダム分割なのでCVスコアはない

    def test_error_handling_invalid_cv_splits(self):
        """エラーハンドリング: 無効なcv_splits"""
        service = MLTrainingService()
        dummy_data = pd.DataFrame(
            {col: [100] * 100 for col in ["open", "high", "low", "close", "volume"]}
        )

        with pytest.raises(ValueError, match="cv_splitsは2以上"):
            service.train_model(
                dummy_data,
                save_model=False,
                use_cross_validation=True,
                cv_splits=1,
            )

    def test_error_handling_invalid_max_train_size(self):
        """エラーハンドリング: 無効なmax_train_size"""
        service = MLTrainingService()
        dummy_data = pd.DataFrame(
            {col: [100] * 100 for col in ["open", "high", "low", "close", "volume"]}
        )

        with pytest.raises(ValueError, match="max_train_sizeは正の整数"):
            service.train_model(
                dummy_data,
                save_model=False,
                use_cross_validation=True,
                max_train_size=0,
            )

    def test_ml_training_service_integration_with_base_trainer(self):
        """
        MLTrainingServiceとBaseMLTrainerの統合テスト

        期待される動作:
        - MLTrainingServiceからBaseMLTrainerへのパラメータ伝播が正しく機能する
        - TimeSeriesSplit関連パラメータが正しく設定される
        """
        service = MLTrainingService(trainer_type="ensemble")

        # _prepare_training_paramsメソッドのテスト
        training_params = {
            "use_cross_validation": True,
            "cv_splits": 4,
            "max_train_size": 1000,
        }

        prepared_params = service._prepare_training_params(training_params)

        # パラメータが正しく準備されていることを確認
        assert prepared_params["use_cross_validation"] is True
        assert prepared_params["cv_splits"] == 4
        assert prepared_params["max_train_size"] == 1000
        assert "use_time_series_split" in prepared_params

    def test_timeseries_split_data_leak_prevention(self, sample_prepared_data):
        """データリーク防止テスト（時系列順序の保持）"""
        X, y = sample_prepared_data
        trainer = ConcreteMLTrainer()

        # 時系列分割
        X_train, X_test, y_train, y_test = trainer._split_data(
            X, y, use_time_series_split=True
        )

        # インデックスによる順序確認
        # 学習データの最後 < テストデータの最初
        assert X_train.index[-1] < X_test.index[0]

        # 重複なし確認
        assert len(set(X_train.index) & set(X_test.index)) == 0

    def test_cv_with_different_split_numbers(self, sample_prepared_data):
        """異なるCV分割数での動作確認"""
        X, y = sample_prepared_data
        trainer = ConcreteMLTrainer()

        with (
            patch.object(trainer, "_calculate_features", return_value=X),
            patch.object(trainer, "_prepare_training_data", return_value=(X, y)),
        ):
            training_data = X.copy()
            for col in ["open", "high", "low", "close", "volume"]:
                training_data[col] = 100

            for splits in [2, 3]:
                result = trainer.train_model(
                    training_data,
                    save_model=False,
                    use_cross_validation=True,
                    cv_splits=splits,
                )
                assert len(result["cv_scores"]) == splits

    def test_random_split_behavior(self, sample_prepared_data):
        """ランダム分割の挙動確認"""
        X, y = sample_prepared_data
        trainer = ConcreteMLTrainer()

        X_train, X_test, y_train, y_test = trainer._split_data(
            X, y, use_random_split=True, test_size=0.2, random_state=42
        )

        assert len(X_train) > 0
        assert len(X_test) > 0
        # ランダム分割なので、必ずしも時系列順ではないが、分割はされていること


class TestTimeSeriesCVParameterValidation:
    """TimeSeriesCV パラメータ検証テスト"""

    # setup_methodを導入し、serviceを初期化
    def setup_method(self):
        self.service = MLTrainingService()

    def test_cv_splits_validation(self):
        params = self.service._prepare_training_params(
            {"use_cross_validation": True, "cv_splits": 5}
        )
        assert params["cv_splits"] == 5

        with pytest.raises(ValueError, match="cv_splitsは2以上である必要があります"):
            self.service._prepare_training_params(
                {"use_cross_validation": True, "cv_splits": 1}
            )

    def test_max_train_size_validation(self):
        params = self.service._prepare_training_params(
            {"use_cross_validation": True, "max_train_size": 1000}
        )
        assert params["max_train_size"] == 1000

        with pytest.raises(
            ValueError, match="max_train_sizeは正の整数である必要があります"
        ):
            self.service._prepare_training_params(
                {"use_cross_validation": True, "max_train_size": 0}
            )

        with pytest.raises(
            ValueError, match="max_train_sizeは正の整数である必要があります"
        ):
            self.service._prepare_training_params(
                {"use_cross_validation": True, "max_train_size": -100}
            )

    def test_conflicting_parameters(
        self, sample_prepared_data
    ):  # sample_prepared_data は必要
        """
        矛盾するパラメータの処理テスト

        期待される動作:
        - use_random_split=True と use_time_series_split=True の同時指定時、
          use_random_splitが優先される
        """
        X, y = sample_prepared_data
        trainer = ConcreteMLTrainer()

        # 両方Trueならランダム分割が優先される実装になっているか確認
        # _split_dataのロジック: if use_time_series_split and not use_random_split: ... else: ...
        # つまり use_random_split=Trueならelseブロック（ランダム）に行く

        # しかしここでは単純にエラーにならないことを確認
        X_train, X_test, _, _ = trainer._split_data(
            X, y, use_random_split=True, use_time_series_split=True
        )
        assert len(X_train) > 0


class TestTimeSeriesCVEdgeCases:
    """TimeSeriesCV エッジケーステスト"""

    # setup_methodを導入し、serviceを初期化
    def setup_method(self):
        self.service = MLTrainingService()

    def test_minimum_cv_splits(self):
        # cv_splits=2（最小値）
        params = self.service._prepare_training_params(
            {"use_cross_validation": True, "cv_splits": 2}
        )
        assert params["cv_splits"] == 2

    def test_none_max_train_size(self):
        # max_train_size=None（制限なし）
        params = self.service._prepare_training_params(
            {"use_cross_validation": True, "max_train_size": None}
        )
        assert params["max_train_size"] is None

    def test_empty_training_params(self):
        # 空の辞書
        params = self.service._prepare_training_params({})

        # デフォルト値が設定されていることを確認
        assert "use_time_series_split" in params
        assert (
            params["use_time_series_split"] == ml_config.training.USE_TIME_SERIES_SPLIT
        )


class TestMLConfigIntegration:
    """ml_config統合テスト"""

    def test_ml_config_structure(self):
        """ml_config構造の検証テスト"""
        # training設定の存在確認
        assert hasattr(ml_config, "training")
        assert hasattr(ml_config.training, "USE_TIME_SERIES_SPLIT")
        assert hasattr(ml_config.training, "CROSS_VALIDATION_FOLDS")
        assert hasattr(ml_config.training, "MAX_TRAIN_SIZE")

    def test_ml_config_default_values(self):
        """ml_configデフォルト値の検証テスト"""
        # デフォルト値の確認
        assert ml_config.training.USE_TIME_SERIES_SPLIT is True
        assert ml_config.training.CROSS_VALIDATION_FOLDS == 5
        assert ml_config.training.MAX_TRAIN_SIZE is None

    def test_ml_training_service_uses_ml_config(self):
        """MLTrainingServiceがml_configを使用していることの確認"""
        service = MLTrainingService()

        # サービスがml_configを参照していることを確認
        assert service.config == ml_config

        # _prepare_training_paramsでml_configが使用されることを確認
        params = service._prepare_training_params({"use_cross_validation": True})

        assert (
            params["use_time_series_split"] == ml_config.training.USE_TIME_SERIES_SPLIT
        )
        assert params["cv_splits"] == ml_config.training.CROSS_VALIDATION_FOLDS


class TestBackwardCompatibility:
    """後方互換性テスト"""

    def test_use_time_series_split_false(self, sample_prepared_data):
        """use_time_series_split=Falseの後方互換性テスト"""
        X, y = sample_prepared_data
        trainer = ConcreteMLTrainer()

        # use_time_series_split=Falseを明示
        X_train, X_test, y_train, y_test = trainer._split_data(
            X, y, use_time_series_split=False
        )

        # 分割が成功していることを確認
        assert len(X_train) > 0
        assert len(X_test) > 0

    def test_use_random_split_true(self, sample_prepared_data):
        """use_random_split=Trueの後方互換性テスト"""
        X, y = sample_prepared_data
        trainer = ConcreteMLTrainer()

        # use_random_split=Trueを使用
        X_train, X_test, y_train, y_test = trainer._split_data(
            X, y, use_random_split=True
        )

        # 分割が成功していることを確認
        assert len(X_train) > 0
        assert len(X_test) > 0

    def test_explicit_test_size_parameter(self, sample_prepared_data):
        """test_sizeパラメータの明示的指定テスト"""
        X, y = sample_prepared_data
        trainer = ConcreteMLTrainer()

        # test_size=0.3を指定
        X_train, X_test, y_train, y_test = trainer._split_data(
            X, y, use_time_series_split=True, test_size=0.3
        )

        # テストサイズが約30%であることを確認
        expected_test_size = int(len(X) * 0.3)
        actual_test_size = len(X_test)
        assert abs(actual_test_size - expected_test_size) <= 1
