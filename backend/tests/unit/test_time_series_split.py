"""
時系列分割機能のテスト

BaseMLTrainerの時系列分割機能が正しく動作することを確認します。
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from backend.app.services.ml.base_ml_trainer import BaseMLTrainer


class MockMLTrainer(BaseMLTrainer):
    """テスト用のMLトレーナー実装"""

    def __init__(self):
        super().__init__()
        self.model = Mock()
        self.model_type = "MockModel"

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """モック予測"""
        return np.random.rand(len(features_df))

    def _train_model_impl(self, X_train, X_test, y_train, y_test, **training_params):
        """モック学習実装"""
        return {
            "accuracy": 0.65,
            "precision": 0.63,
            "recall": 0.67,
            "f1_score": 0.65,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        }


@pytest.fixture
def sample_time_series_data():
    """時系列テストデータを生成"""
    # 1000時間分のデータを生成
    start_time = datetime(2023, 1, 1)
    timestamps = [start_time + timedelta(hours=i) for i in range(1000)]

    # OHLCVデータを生成
    np.random.seed(42)
    base_price = 100.0
    prices = []

    for i in range(1000):
        # ランダムウォーク的な価格変動
        change = np.random.normal(0, 0.02)
        base_price *= 1 + change

        high = base_price * (1 + abs(np.random.normal(0, 0.01)))
        low = base_price * (1 - abs(np.random.normal(0, 0.01)))
        volume = np.random.uniform(1000, 10000)

        prices.append(
            {
                "timestamp": timestamps[i],
                "Open": base_price,
                "High": high,
                "Low": low,
                "Close": base_price,
                "Volume": volume,
            }
        )

    df = pd.DataFrame(prices)
    df.set_index("timestamp", inplace=True)
    return df


class TestTimeSeriesSplit:
    """時系列分割のテストクラス"""

    def test_time_series_split_basic(self, sample_time_series_data):
        """基本的な時系列分割のテスト"""
        trainer = MockMLTrainer()

        # 時系列分割でデータを分割
        X = sample_time_series_data[["Open", "High", "Low", "Close", "Volume"]]
        y = pd.Series(np.random.randint(0, 3, len(X)), index=X.index)

        X_train, X_test, y_train, y_test = trainer._split_data(
            X, y, use_time_series_split=True, test_size=0.2
        )

        # 分割結果の検証
        assert len(X_train) == 800  # 80%が学習データ
        assert len(X_test) == 200  # 20%がテストデータ
        assert len(y_train) == 800
        assert len(y_test) == 200

        # 時間順序の検証
        assert (
            X_train.index[-1] < X_test.index[0]
        )  # 学習データの最後 < テストデータの最初

        # インデックスの連続性確認
        assert X_train.index.equals(X.index[:800])
        assert X_test.index.equals(X.index[800:])

    def test_random_split_fallback(self, sample_time_series_data):
        """ランダム分割のフォールバック機能テスト"""
        trainer = MockMLTrainer()

        X = sample_time_series_data[["Open", "High", "Low", "Close", "Volume"]]
        y = pd.Series(np.random.randint(0, 3, len(X)), index=X.index)

        X_train, X_test, y_train, y_test = trainer._split_data(
            X,
            y,
            use_time_series_split=False,  # ランダム分割を強制
            test_size=0.2,
            random_state=42,
        )

        # 分割結果の検証
        assert len(X_train) == 800
        assert len(X_test) == 200
        assert len(y_train) == 800
        assert len(y_test) == 200

        # ランダム分割では時間順序が保持されない
        # （学習データの最後がテストデータの最初より後になる可能性がある）
        train_max_time = X_train.index.max()
        test_min_time = X_test.index.min()
        # ランダム分割では必ずしも時間順序が保持されない

    def test_time_series_cross_validation(self, sample_time_series_data):
        """時系列クロスバリデーションのテスト"""
        trainer = MockMLTrainer()

        X = sample_time_series_data[["Open", "High", "Low", "Close", "Volume"]]
        y = pd.Series(np.random.randint(0, 3, len(X)), index=X.index)

        # クロスバリデーションを実行
        cv_result = trainer._time_series_cross_validate(
            X, y, cv_splits=3, test_size=0.2
        )

        # 結果の検証
        assert "cv_scores" in cv_result
        assert "cv_mean" in cv_result
        assert "cv_std" in cv_result
        assert "fold_results" in cv_result

        assert len(cv_result["cv_scores"]) == 3
        assert len(cv_result["fold_results"]) == 3
        assert cv_result["n_splits"] == 3

        # 各フォールドの結果を確認
        for i, fold_result in enumerate(cv_result["fold_results"]):
            assert fold_result["fold"] == i + 1
            assert "train_samples" in fold_result
            assert "test_samples" in fold_result
            assert "accuracy" in fold_result

    @patch("backend.app.services.ml.base_ml_trainer.logger")
    def test_logging_output(self, mock_logger, sample_time_series_data):
        """ログ出力のテスト"""
        trainer = MockMLTrainer()

        X = sample_time_series_data[["Open", "High", "Low", "Close", "Volume"]]
        y = pd.Series(np.random.randint(0, 3, len(X)), index=X.index)

        # 時系列分割を実行
        trainer._split_data(X, y, use_time_series_split=True, test_size=0.2)

        # ログが正しく出力されているか確認
        mock_logger.info.assert_any_call("🕒 時系列分割を使用（データリーク防止）")

        # ランダム分割の警告ログ
        trainer._split_data(X, y, use_time_series_split=False, test_size=0.2)
        mock_logger.warning.assert_any_call(
            "⚠️ ランダム分割を使用（時系列データには非推奨）"
        )

    def test_label_distribution_logging(self, sample_time_series_data):
        """ラベル分布ログのテスト"""
        trainer = MockMLTrainer()

        X = sample_time_series_data[["Open", "High", "Low", "Close", "Volume"]]
        # 不均衡なラベル分布を作成
        y = pd.Series([0] * 700 + [1] * 200 + [2] * 100, index=X.index)

        with patch(
            "backend.app.services.ml.base_ml_trainer.logger"
        ) as mock_logger:
            trainer._split_data(X, y, use_time_series_split=True, test_size=0.2)

            # ラベル分布のログが出力されているか確認
            info_calls = [call.args[0] for call in mock_logger.info.call_args_list]

            # 学習データとテストデータのラベル分布ログが含まれているか確認
            assert any("学習データのラベル分布:" in call for call in info_calls)
            assert any("テストデータのラベル分布:" in call for call in info_calls)

    def test_edge_cases(self, sample_time_series_data):
        """エッジケースのテスト"""
        trainer = MockMLTrainer()

        # 小さなデータセット
        small_data = sample_time_series_data.head(10)
        X_small = small_data[["Open", "High", "Low", "Close", "Volume"]]
        y_small = pd.Series(np.random.randint(0, 3, len(X_small)), index=X_small.index)

        X_train, X_test, y_train, y_test = trainer._split_data(
            X_small, y_small, use_time_series_split=True, test_size=0.2
        )

        # 最小限のデータでも分割できることを確認
        assert len(X_train) == 8  # 80%
        assert len(X_test) == 2  # 20%

        # 単一ラベルのケース
        y_single = pd.Series([1] * len(X_small), index=X_small.index)

        X_train, X_test, y_train, y_test = trainer._split_data(
            X_small, y_single, use_time_series_split=True, test_size=0.2
        )

        # 単一ラベルでも分割できることを確認
        assert len(X_train) == 8
        assert len(X_test) == 2
        assert all(y_train == 1)
        assert all(y_test == 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
