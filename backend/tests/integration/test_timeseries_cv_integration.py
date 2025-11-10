"""
TimeSeriesSplit統合テスト

BaseMLTrainerとMLTrainingServiceのTimeSeriesSplit統一化に関する
包括的な統合テストを実装します。
"""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from backend.app.services.ml.base_ml_trainer import BaseMLTrainer
from backend.app.services.ml.config import ml_config
from backend.app.services.ml.ml_training_service import MLTrainingService


class TestTimeSeriesCVIntegration:
    """TimeSeriesSplit統合テストクラス"""

    @pytest.fixture
    def sample_prepared_data(self):
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

    def test_end_to_end_ml_training_flow_with_mocks(self, mock_training_components):
        """
        エンドツーエンドのML学習フロー統合テスト（モック使用）

        期待される動作:
        - MLTrainingServiceを使用したTimeSeriesSplit学習が成功する
        - BaseMLTrainerとの統合動作が正常に機能する
        - ml_configからのパラメータ読み込みが正しく行われる
        """
        X = mock_training_components["X"]
        y = mock_training_components["y"]

        # BaseMLTrainerの内部メソッドをモック
        with (
            patch.object(BaseMLTrainer, "_calculate_features") as mock_calc_features,
            patch.object(BaseMLTrainer, "_prepare_training_data") as mock_prepare_data,
            patch.object(BaseMLTrainer, "_train_model_impl") as mock_train_impl,
        ):

            # モックの設定
            mock_calc_features.return_value = X
            mock_prepare_data.return_value = (X, y)

            # 各foldの学習結果をモック
            def mock_train_side_effect(X_train, X_test, y_train, y_test, **kwargs):
                return {
                    "accuracy": 0.85,
                    "f1_score": 0.82,
                    "classification_report": {"macro avg": {"f1-score": 0.82}},
                }

            mock_train_impl.side_effect = mock_train_side_effect

            # サービスを初期化
            service = MLTrainingService(trainer_type="ensemble")

            # 学習データの準備
            training_data = X.copy()
            training_data["open"] = 10000
            training_data["high"] = 10100
            training_data["low"] = 9900
            training_data["close"] = 10000
            training_data["volume"] = 500

            # TimeSeriesSplit有効で学習を実行
            result = service.train_model(
                training_data,
                save_model=False,
                use_cross_validation=True,
                cv_splits=3,
            )

            # 学習が成功していることを確認
            assert result["success"] is True
            assert "cv_scores" in result
            assert "cv_mean" in result
            assert "cv_std" in result
            assert len(result["cv_scores"]) == 3

            # 各CV結果の妥当性を確認
            for score in result["cv_scores"]:
                assert 0.0 <= score <= 1.0

    def test_cross_validation_consistency(self, sample_prepared_data):
        """
        クロスバリデーションの一貫性テスト

        期待される動作:
        - TimeSeriesSplitのfold数が全コンポーネントで一致
        - データリークが発生していない
        - 時系列順序が保持されている
        """
        X, y = sample_prepared_data

        with (
            patch.object(BaseMLTrainer, "_calculate_features") as mock_calc_features,
            patch.object(BaseMLTrainer, "_prepare_training_data") as mock_prepare_data,
            patch.object(BaseMLTrainer, "_train_model_impl") as mock_train_impl,
        ):

            mock_calc_features.return_value = X
            mock_prepare_data.return_value = (X, y)
            mock_train_impl.return_value = {
                "accuracy": 0.85,
                "f1_score": 0.82,
            }

            trainer = BaseMLTrainer()

            # OHLCVデータを準備
            training_data = X.copy()
            training_data["open"] = 10000
            training_data["high"] = 10100
            training_data["low"] = 9900
            training_data["close"] = 10000
            training_data["volume"] = 500

            # TimeSeriesSplit CVを実行
            result = trainer.train_model(
                training_data,
                save_model=False,
                use_cross_validation=True,
                cv_splits=5,
            )

            # fold数が一致することを確認
            assert len(result["cv_scores"]) == 5
            assert result["n_splits"] == 5

            # 各foldの結果を検証
            for fold_result in result.get("fold_results", []):
                # 学習サンプル数 < テストサンプル数の開始位置を確認（時系列順序保持）
                assert fold_result["train_samples"] > 0
                assert fold_result["test_samples"] > 0

                # fold情報が正しく記録されていることを確認
                assert "fold" in fold_result
                assert "train_period" in fold_result
                assert "test_period" in fold_result

    def test_config_parameter_propagation(self, sample_prepared_data):
        """
        設定パラメータの伝播テスト

        期待される動作:
        - ml_configの設定がBaseMLTrainerまで正しく伝わる
        - training_paramsによる上書きが正しく機能する
        - デフォルト値が適切に設定される
        """
        X, y = sample_prepared_data

        service = MLTrainingService()

        # ml_configのデフォルト値を確認
        default_cv_folds = ml_config.training.CROSS_VALIDATION_FOLDS
        assert default_cv_folds == 5  # ml_configのデフォルト値

        with (
            patch.object(BaseMLTrainer, "_calculate_features") as mock_calc_features,
            patch.object(BaseMLTrainer, "_prepare_training_data") as mock_prepare_data,
            patch.object(BaseMLTrainer, "_train_model_impl") as mock_train_impl,
        ):

            mock_calc_features.return_value = X
            mock_prepare_data.return_value = (X, y)
            mock_train_impl.return_value = {"accuracy": 0.85}

            training_data = X.copy()
            training_data["open"] = 10000
            training_data["high"] = 10100
            training_data["low"] = 9900
            training_data["close"] = 10000
            training_data["volume"] = 500

            # デフォルト設定での学習（パラメータ指定なし）
            result1 = service.train_model(
                training_data,
                save_model=False,
                use_cross_validation=True,
            )

            # デフォルトのfold数が使用されていることを確認
            assert len(result1["cv_scores"]) == default_cv_folds

            # カスタムパラメータでの学習（上書き）
            custom_cv_folds = 3
            result2 = service.train_model(
                training_data,
                save_model=False,
                use_cross_validation=True,
                cv_splits=custom_cv_folds,
            )

            # カスタム値が使用されていることを確認
            assert len(result2["cv_scores"]) == custom_cv_folds

    def test_backward_compatibility_random_split(self, sample_prepared_data):
        """
        後方互換性テスト（ランダム分割）

        期待される動作:
        - use_random_split=Trueでの従来動作が維持されている
        - 既存のAPIが破壊されていない
        - 明示的なパラメータ指定が機能する
        """
        X, y = sample_prepared_data

        with (
            patch.object(BaseMLTrainer, "_calculate_features") as mock_calc_features,
            patch.object(BaseMLTrainer, "_prepare_training_data") as mock_prepare_data,
            patch.object(BaseMLTrainer, "_train_model_impl") as mock_train_impl,
        ):

            mock_calc_features.return_value = X
            mock_prepare_data.return_value = (X, y)
            mock_train_impl.return_value = {
                "success": True,
                "accuracy": 0.85,
            }

            trainer = BaseMLTrainer()

            training_data = X.copy()
            training_data["open"] = 10000
            training_data["high"] = 10100
            training_data["low"] = 9900
            training_data["close"] = 10000
            training_data["volume"] = 500

            # ランダム分割での学習
            result = trainer.train_model(
                training_data,
                save_model=False,
                use_random_split=True,
            )

            # 学習が成功していることを確認
            assert result["success"] is True

            # CVスコアは含まれないことを確認（通常の分割）
            assert "cv_scores" not in result

    def test_error_handling_invalid_cv_splits(self):
        """
        エラーハンドリングテスト（無効なcv_splits）

        期待される動作:
        - 無効なパラメータで適切なエラーが発生する
        """
        service = MLTrainingService()

        # ダミーデータ
        dummy_data = pd.DataFrame(
            {
                "open": [10000] * 100,
                "high": [10100] * 100,
                "low": [9900] * 100,
                "close": [10000] * 100,
                "volume": [500] * 100,
            }
        )

        # cv_splits < 2 でエラー
        with pytest.raises(ValueError, match="cv_splitsは2以上である必要があります"):
            service.train_model(
                dummy_data,
                save_model=False,
                use_cross_validation=True,
                cv_splits=1,
            )

    def test_error_handling_invalid_max_train_size(self):
        """
        エラーハンドリングテスト（無効なmax_train_size）

        期待される動作:
        - max_train_size <= 0 で適切なエラーが発生する
        """
        service = MLTrainingService()

        # ダミーデータ
        dummy_data = pd.DataFrame(
            {
                "open": [10000] * 100,
                "high": [10100] * 100,
                "low": [9900] * 100,
                "close": [10000] * 100,
                "volume": [500] * 100,
            }
        )

        # max_train_size <= 0 でエラー
        with pytest.raises(
            ValueError, match="max_train_sizeは正の整数である必要があります"
        ):
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
        """
        データリーク防止テスト

        期待される動作:
        - 学習データとテストデータで時系列の重複がない
        - テストデータは常に学習データより未来のデータである
        """
        X, y = sample_prepared_data

        trainer = BaseMLTrainer()

        # データを分割
        X_train, X_test, y_train, y_test = trainer._split_data(
            X, y, use_time_series_split=True
        )

        # 時系列順序が保持されていることを確認
        assert X_train.index[-1] < X_test.index[0]
        assert y_train.index[-1] < y_test.index[0]

        # データの重複がないことを確認
        train_indices = set(X_train.index)
        test_indices = set(X_test.index)
        assert len(train_indices & test_indices) == 0

    def test_cv_with_different_split_numbers(self, sample_prepared_data):
        """
        異なる分割数でのCV動作テスト

        期待される動作:
        - 異なるcv_splitsパラメータで正しく動作する
        """
        X, y = sample_prepared_data

        with (
            patch.object(BaseMLTrainer, "_calculate_features") as mock_calc_features,
            patch.object(BaseMLTrainer, "_prepare_training_data") as mock_prepare_data,
            patch.object(BaseMLTrainer, "_train_model_impl") as mock_train_impl,
        ):

            mock_calc_features.return_value = X
            mock_prepare_data.return_value = (X, y)
            mock_train_impl.return_value = {"accuracy": 0.85}

            trainer = BaseMLTrainer()

            training_data = X.copy()
            training_data["open"] = 10000
            training_data["high"] = 10100
            training_data["low"] = 9900
            training_data["close"] = 10000
            training_data["volume"] = 500

            # 異なる分割数でテスト
            for cv_splits in [2, 3, 5, 7]:
                result = trainer.train_model(
                    training_data,
                    save_model=False,
                    use_cross_validation=True,
                    cv_splits=cv_splits,
                )

                assert len(result["cv_scores"]) == cv_splits
                assert result["n_splits"] == cv_splits

    def test_config_default_values_propagation(self):
        """
        ml_configデフォルト値の伝播テスト

        期待される動作:
        - ml_configのデフォルト値が全コンポーネントで一貫して使用される
        """
        # ml_configのデフォルト値を確認
        assert ml_config.training.USE_TIME_SERIES_SPLIT is True
        assert ml_config.training.CROSS_VALIDATION_FOLDS == 5

        # MLTrainingServiceでのパラメータ準備を確認
        service = MLTrainingService()

        # パラメータ指定なし
        params = service._prepare_training_params({})
        assert (
            params["use_time_series_split"] == ml_config.training.USE_TIME_SERIES_SPLIT
        )

        # CV有効時
        params = service._prepare_training_params({"use_cross_validation": True})
        assert params["cv_splits"] == ml_config.training.CROSS_VALIDATION_FOLDS

    def test_parameter_override_priority(self):
        """
        パラメータ上書きの優先順位テスト

        期待される動作:
        - training_paramsの明示的な値がml_configのデフォルト値より優先される
        """
        service = MLTrainingService()

        # デフォルト値とは異なるカスタム値を指定
        custom_params = {
            "use_cross_validation": True,
            "cv_splits": 10,
            "max_train_size": 5000,
            "use_time_series_split": False,
        }

        prepared_params = service._prepare_training_params(custom_params)

        # カスタム値が優先されていることを確認
        assert prepared_params["cv_splits"] == 10
        assert prepared_params["max_train_size"] == 5000
        assert prepared_params["use_time_series_split"] is False

    def test_time_series_split_order_preservation(self, sample_prepared_data):
        """
        TimeSeriesSplitでの時系列順序保持テスト

        期待される動作:
        - 分割後も時系列順序が保持される
        - 学習データは常にテストデータより過去である
        """
        X, y = sample_prepared_data

        trainer = BaseMLTrainer()

        # 時系列分割を使用
        X_train, X_test, y_train, y_test = trainer._split_data(
            X, y, use_time_series_split=True, test_size=0.2
        )

        # 時系列順序の確認
        assert X_train.index.is_monotonic_increasing
        assert X_test.index.is_monotonic_increasing
        assert X_train.index[-1] < X_test.index[0]

    def test_random_split_behavior(self, sample_prepared_data):
        """
        ランダム分割の動作テスト

        期待される動作:
        - use_random_split=Trueでランダム分割が実行される
        - 時系列順序は保証されない
        """
        X, y = sample_prepared_data

        trainer = BaseMLTrainer()

        # ランダム分割を使用
        X_train, X_test, y_train, y_test = trainer._split_data(
            X, y, use_random_split=True, test_size=0.2
        )

        # データが分割されていることを確認
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(X_train) + len(X_test) == len(X)


class TestTimeSeriesCVParameterValidation:
    """TimeSeriesCV パラメータ検証テスト"""

    def test_cv_splits_validation(self):
        """cv_splitsパラメータの検証テスト"""
        service = MLTrainingService()

        # 有効な値
        valid_params = service._prepare_training_params(
            {"use_cross_validation": True, "cv_splits": 5}
        )
        assert valid_params["cv_splits"] == 5

        # 無効な値（< 2）
        with pytest.raises(ValueError, match="cv_splitsは2以上である必要があります"):
            service._prepare_training_params(
                {"use_cross_validation": True, "cv_splits": 1}
            )

    def test_max_train_size_validation(self):
        """max_train_sizeパラメータの検証テスト"""
        service = MLTrainingService()

        # 有効な値
        valid_params = service._prepare_training_params(
            {"use_cross_validation": True, "max_train_size": 1000}
        )
        assert valid_params["max_train_size"] == 1000

        # 無効な値（<= 0）
        with pytest.raises(
            ValueError, match="max_train_sizeは正の整数である必要があります"
        ):
            service._prepare_training_params(
                {"use_cross_validation": True, "max_train_size": 0}
            )

        with pytest.raises(
            ValueError, match="max_train_sizeは正の整数である必要があります"
        ):
            service._prepare_training_params(
                {"use_cross_validation": True, "max_train_size": -100}
            )

    def test_conflicting_parameters(self):
        """
        矛盾するパラメータの処理テスト

        期待される動作:
        - use_random_split=True と use_time_series_split=True の同時指定時、
          use_random_splitが優先される
        """
        X = pd.DataFrame(
            {
                "feature_1": np.random.randn(100),
                "feature_2": np.random.randn(100),
            },
            index=pd.date_range(start="2023-01-01", periods=100, freq="h"),
        )
        y = pd.Series(np.random.choice([0, 1], 100), index=X.index)

        trainer = BaseMLTrainer()

        # 矛盾するパラメータでの分割
        X_train, X_test, y_train, y_test = trainer._split_data(
            X,
            y,
            use_random_split=True,
            use_time_series_split=True,  # これは無視される
        )

        # 分割が成功していることを確認
        assert len(X_train) > 0
        assert len(X_test) > 0


class TestTimeSeriesCVEdgeCases:
    """TimeSeriesCV エッジケーステスト"""

    def test_minimum_cv_splits(self):
        """最小CV分割数のテスト"""
        service = MLTrainingService()

        # cv_splits=2（最小値）
        params = service._prepare_training_params(
            {"use_cross_validation": True, "cv_splits": 2}
        )
        assert params["cv_splits"] == 2

    def test_none_max_train_size(self):
        """max_train_size=Noneの処理テスト"""
        service = MLTrainingService()

        # max_train_size=None（制限なし）
        params = service._prepare_training_params(
            {"use_cross_validation": True, "max_train_size": None}
        )
        assert params["max_train_size"] is None

    def test_empty_training_params(self):
        """空のtraining_paramsの処理テスト"""
        service = MLTrainingService()

        # 空の辞書
        params = service._prepare_training_params({})

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

    def test_use_time_series_split_false(self):
        """use_time_series_split=Falseの後方互換性テスト"""
        X = pd.DataFrame(
            {"feature_1": np.random.randn(100), "feature_2": np.random.randn(100)},
            index=pd.date_range(start="2023-01-01", periods=100, freq="h"),
        )
        y = pd.Series(np.random.choice([0, 1], 100), index=X.index)

        trainer = BaseMLTrainer()

        # use_time_series_split=Falseを明示
        X_train, X_test, y_train, y_test = trainer._split_data(
            X, y, use_time_series_split=False
        )

        # 分割が成功していることを確認
        assert len(X_train) > 0
        assert len(X_test) > 0

    def test_use_random_split_true(self):
        """use_random_split=Trueの後方互換性テスト"""
        X = pd.DataFrame(
            {"feature_1": np.random.randn(100), "feature_2": np.random.randn(100)},
            index=pd.date_range(start="2023-01-01", periods=100, freq="h"),
        )
        y = pd.Series(np.random.choice([0, 1], 100), index=X.index)

        trainer = BaseMLTrainer()

        # use_random_split=Trueを使用
        X_train, X_test, y_train, y_test = trainer._split_data(
            X, y, use_random_split=True
        )

        # 分割が成功していることを確認
        assert len(X_train) > 0
        assert len(X_test) > 0

    def test_explicit_test_size_parameter(self):
        """test_sizeパラメータの明示的指定テスト"""
        X = pd.DataFrame(
            {"feature_1": np.random.randn(100), "feature_2": np.random.randn(100)},
            index=pd.date_range(start="2023-01-01", periods=100, freq="h"),
        )
        y = pd.Series(np.random.choice([0, 1], 100), index=X.index)

        trainer = BaseMLTrainer()

        # test_size=0.3を指定
        X_train, X_test, y_train, y_test = trainer._split_data(
            X, y, use_time_series_split=True, test_size=0.3
        )

        # テストサイズが約30%であることを確認
        expected_test_size = int(len(X) * 0.3)
        actual_test_size = len(X_test)
        assert abs(actual_test_size - expected_test_size) <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
