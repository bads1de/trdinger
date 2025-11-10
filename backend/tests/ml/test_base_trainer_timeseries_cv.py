"""
BaseMLTrainerのTimeSeriesSplit対応テスト

TDDアプローチにより、TimeSeriesSplitをデフォルトのCV手法とする
新機能をテストファーストで実装します。
"""

import numpy as np
import pandas as pd
import pytest

from backend.app.services.ml.base_ml_trainer import BaseMLTrainer


class TestBaseMLTrainerTimeSeriesCV:
    """BaseMLTrainerのTimeSeriesSplit対応テスト"""

    @pytest.fixture
    def sample_timeseries_data(self):
        """時系列データのサンプル（200サンプル）"""
        np.random.seed(42)
        dates = pd.date_range(start="2023-01-01", periods=200, freq="H")

        data = pd.DataFrame(
            {
                "open": 10000 + np.random.randn(200) * 100,
                "high": 10100 + np.random.randn(200) * 100,
                "low": 9900 + np.random.randn(200) * 100,
                "close": 10000 + np.random.randn(200) * 100,
                "volume": 500 + np.random.randint(100, 500, 200),
            },
            index=dates,
        )

        return data

    def test_default_split_is_timeseries(self, sample_timeseries_data):
        """
        デフォルトでTimeSeriesSplitが使用されることをテスト

        期待される動作:
        - use_time_series_split パラメータを指定しない場合、
          デフォルトでTimeSeriesSplitが使用される
        """
        trainer = BaseMLTrainer()

        # use_time_series_splitを指定せずに学習
        result = trainer.train_model(
            sample_timeseries_data,
            save_model=False,
        )

        # 学習が成功していることを確認
        assert result["success"] is True
        assert "accuracy" in result or "f1_score" in result

        # ログ出力で時系列分割が使用されたことを確認（間接的）
        # 実際のログ確認は統合テストで行う

    def test_timeseries_split_parameter_explicit(self, sample_timeseries_data):
        """
        use_time_series_split=Trueを明示的に指定した場合のテスト

        期待される動作:
        - use_time_series_split=True を指定した場合、
          TimeSeriesSplitが使用される
        """
        trainer = BaseMLTrainer()

        result = trainer.train_model(
            sample_timeseries_data,
            save_model=False,
            use_time_series_split=True,
        )

        assert result["success"] is True

    def test_random_split_with_flag(self, sample_timeseries_data):
        """
        use_random_split=Trueでランダム分割が使用されることをテスト

        期待される動作:
        - use_random_split=True を指定した場合、
          従来のtrain_test_splitが使用される（下位互換性）
        """
        trainer = BaseMLTrainer()

        result = trainer.train_model(
            sample_timeseries_data,
            save_model=False,
            use_random_split=True,
        )

        assert result["success"] is True

    def test_cross_validation_with_timeseries(self, sample_timeseries_data):
        """
        use_cross_validation=TrueでTimeSeriesSplitのCVが実行されることをテスト

        期待される動作:
        - use_cross_validation=True を指定した場合、
          TimeSeriesSplitによるクロスバリデーションが実行される
        - CV結果にcv_scores、cv_mean、cv_stdが含まれる
        """
        trainer = BaseMLTrainer()

        result = trainer.train_model(
            sample_timeseries_data,
            save_model=False,
            use_cross_validation=True,
            cv_splits=3,  # 3分割
        )

        # CV結果が含まれていることを確認
        assert result["success"] is True
        assert "cv_scores" in result
        assert "cv_mean" in result
        assert "cv_std" in result
        assert len(result["cv_scores"]) == 3

    def test_cv_splits_parameter(self, sample_timeseries_data):
        """
        cv_splitsパラメータが正しく動作することをテスト

        期待される動作:
        - cv_splits パラメータで分割数を指定できる
        """
        trainer = BaseMLTrainer()

        result = trainer.train_model(
            sample_timeseries_data,
            save_model=False,
            use_cross_validation=True,
            cv_splits=5,  # 5分割
        )

        assert len(result["cv_scores"]) == 5

    def test_max_train_size_parameter(self, sample_timeseries_data):
        """
        max_train_sizeパラメータが正しく動作することをテスト

        期待される動作:
        - max_train_size パラメータで学習データの最大サイズを制限できる
        """
        trainer = BaseMLTrainer()

        result = trainer.train_model(
            sample_timeseries_data,
            save_model=False,
            use_cross_validation=True,
            cv_splits=3,
            max_train_size=100,  # 学習データを最大100サンプルに制限
        )

        assert result["success"] is True

    def test_config_integration(self, sample_timeseries_data):
        """
        ml_configからTimeSeriesSplit関連設定が読み込まれることをテスト

        期待される動作:
        - ml_config.training.CROSS_VALIDATION_FOLDS が
          デフォルトのcv_splitsとして使用される
        """
        trainer = BaseMLTrainer()

        # デフォルト設定での学習
        result = trainer.train_model(
            sample_timeseries_data,
            save_model=False,
            use_cross_validation=True,
        )

        # ml_config.training.CROSS_VALIDATION_FOLDS（デフォルト5）が使用される
        assert len(result["cv_scores"]) == 5

    def test_backward_compatibility_use_time_series_split_false(
        self, sample_timeseries_data
    ):
        """
        下位互換性: use_time_series_split=Falseでランダム分割が使用されることをテスト

        期待される動作:
        - use_time_series_split=False を指定した場合、
          従来のランダム分割が使用される
        """
        trainer = BaseMLTrainer()

        result = trainer.train_model(
            sample_timeseries_data,
            save_model=False,
            use_time_series_split=False,
        )

        assert result["success"] is True

    def test_split_data_method_default_timeseries(self):
        """
        _split_dataメソッドがデフォルトで時系列分割を使用することをテスト

        期待される動作:
        - _split_data メソッドを直接呼び出した場合、
          デフォルトでTimeSeriesSplitが使用される
        """
        trainer = BaseMLTrainer()

        # サンプルデータ作成
        dates = pd.date_range(start="2023-01-01", periods=100, freq="H")
        X = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
            },
            index=dates,
        )
        y = pd.Series(np.random.choice([0, 1, 2], 100), index=dates)

        # デフォルト設定で分割
        X_train, X_test, y_train, y_test = trainer._split_data(X, y)

        # 時系列順序が保持されていることを確認
        assert X_train.index[-1] < X_test.index[0]
        assert y_train.index[-1] < y_test.index[0]

    def test_split_data_method_with_random_flag(self):
        """
        _split_dataメソッドでuse_random_split=Trueの場合のテスト

        期待される動作:
        - use_random_split=True を指定した場合、
          ランダム分割が使用される
        """
        trainer = BaseMLTrainer()

        # サンプルデータ作成
        dates = pd.date_range(start="2023-01-01", periods=100, freq="H")
        X = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
            },
            index=dates,
        )
        y = pd.Series(np.random.choice([0, 1, 2], 100), index=dates)

        # ランダム分割を使用
        X_train, X_test, y_train, y_test = trainer._split_data(
            X, y, use_random_split=True
        )

        # データが分割されていることを確認（順序の検証はしない）
        assert len(X_train) > 0
        assert len(X_test) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
