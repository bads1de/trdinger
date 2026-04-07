import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from app.services.ml.trainers.base_ml_trainer import BaseMLTrainer


# テスト用の具象クラス
class MockTrainer(BaseMLTrainer):
    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        # future_log_realized_vol を返す
        n = len(features_df)
        return np.full(n, 0.7)

    def _train_model_impl(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        **training_params,
    ) -> dict:
        self._model = MagicMock()
        return {
            "qlike": 0.12,
            "rmse_log_rv": 0.08,
            "mae_log_rv": 0.05,
            "gate_cutoff_log_rv": 0.6,
            "gate_cutoff_vol": float(np.exp(0.6)),
        }


class TestBaseMLTrainer:
    @pytest.fixture
    def trainer(self):
        return MockTrainer()

    @pytest.fixture
    def sample_data(self):
        dates = pd.date_range(start="2023-01-01", periods=150, freq="h")
        df = pd.DataFrame(
            {
                "open": np.random.randn(150) + 100,
                "high": np.random.randn(150) + 101,
                "low": np.random.randn(150) + 99,
                "close": np.random.randn(150) + 100,
                "volume": np.random.rand(150) * 1000,
            },
            index=dates,
        )
        return df

    def test_train_model_insufficient_data(self, trainer):
        """データ不足の場合のテスト"""
        short_data = pd.DataFrame(
            np.random.randn(50, 5), columns=["open", "high", "low", "close", "volume"]
        )
        # @safe_ml_operation によって例外はキャッチされ、デフォルト値が返される
        result = trainer.train_model(short_data)
        assert result["success"] is False

    def test_train_model_success(self, trainer, sample_data):
        """正常な学習フローのテスト"""
        with patch.object(
            trainer.feature_service,
            "calculate_advanced_features",
            return_value=pd.DataFrame(
                np.random.randn(150, 10), index=sample_data.index
            ),
        ):
            with patch.object(
                trainer.target_service,
                "prepare_targets",
                return_value=(
                    pd.DataFrame(np.random.randn(140, 10)),
                    pd.Series(np.random.randn(140)),
                ),
            ):
                with patch(
                    "app.services.ml.trainers.base_ml_trainer.model_manager.save_model",
                    return_value="/path/to/model",
                ):
                    result = trainer.train_model(sample_data, save_model=True)

                    assert result["success"] is True
                    assert trainer.is_trained is True
                    assert "qlike" in result
                    assert result["model_path"] == "/path/to/model"

    def test_predict_volatility_not_trained(self, trainer, sample_data):
        """未学習状態での予測"""
        # 未学習時はデフォルト値を返すべき
        result = trainer.predict_volatility(sample_data)
        assert "forecast_log_rv" in result
        assert "gate_open" in result

    def test_predict_volatility_success(self, trainer, sample_data):
        """学習後の予測シグナル取得"""
        trainer.is_trained = True
        trainer.feature_columns = ["feat1", "feat2"]
        trainer._model = MagicMock()
        trainer.current_model_metadata = {"gate_cutoff_log_rv": 0.6}

        features = pd.DataFrame(
            np.random.randn(10, 2),
            columns=["feat1", "feat2"],
            index=sample_data.index[:10],
        )

        with patch(
            "app.services.ml.trainers.base_ml_trainer.prepare_data_for_prediction",
            return_value=features,
        ):
            result = trainer.predict_volatility(features)
            assert "forecast_log_rv" in result
            assert result["forecast_log_rv"] == 0.7
            assert result["gate_open"] is True

    def test_predict_volatility_uses_latest_1d_prediction(self, trainer, sample_data):
        """1次元の予測配列でも最新値を使うことを確認"""
        trainer.is_trained = True
        trainer.feature_columns = ["feat1", "feat2"]
        trainer.current_model_metadata = {"gate_cutoff_log_rv": 0.6}
        trainer.predict = MagicMock(return_value=np.array([0.1, 0.2, 1.0]))

        features = pd.DataFrame(
            np.random.randn(3, 2),
            columns=["feat1", "feat2"],
            index=sample_data.index[:3],
        )

        with patch(
            "app.services.ml.trainers.base_ml_trainer.prepare_data_for_prediction",
            return_value=features,
        ):
            result = trainer.predict_volatility(features)

        assert result["forecast_log_rv"] == 1.0
        assert result["gate_open"] is True

    def test_load_model_failure(self, trainer):
        """モデル読み込み失敗"""
        with patch(
            "app.services.ml.trainers.base_ml_trainer.model_manager.load_model",
            return_value=None,
        ):
            result = trainer.load_model("/invalid/path")
            assert result is False
            assert trainer.is_trained is False

    def test_load_model_success(self, trainer):
        """モデル読み込み成功"""
        model_data = {
            "model": MagicMock(),
            "scaler": MagicMock(),
            "feature_columns": ["f1", "f2"],
            "metadata": {
                "task_type": "volatility_regression",
                "target_kind": "log_realized_vol",
            },
        }
        with patch(
            "app.services.ml.trainers.base_ml_trainer.model_manager.load_model",
            return_value=model_data,
        ):
            result = trainer.load_model("/path/to/model")
            assert result is True
            assert trainer.is_trained is True
            assert trainer.feature_columns == ["f1", "f2"]

    def test_load_model_rejects_incompatible_metadata_without_mutating_state(self, trainer):
        """互換性のないモデルは拒否し、内部状態を汚さない"""
        trainer._model = None
        trainer.feature_columns = None
        trainer.is_trained = False

        model_data = {
            "model": MagicMock(),
            "scaler": MagicMock(),
            "feature_columns": ["f1", "f2"],
            "metadata": {
                "task_type": "classification",
                "target_kind": "classification_label",
            },
        }
        with patch(
            "app.services.ml.trainers.base_ml_trainer.model_manager.load_model",
            return_value=model_data,
        ):
            result = trainer.load_model("/path/to/model")

        assert result is False
        assert trainer._model is None
        assert trainer.feature_columns is None
        assert trainer.is_trained is False

    def test_load_model_rejects_missing_task_metadata(self, trainer):
        """task_type/target_kind が欠落した旧モデルも拒否する"""
        model_data = {
            "model": MagicMock(),
            "scaler": MagicMock(),
            "feature_columns": ["f1", "f2"],
            "metadata": {},
        }
        with patch(
            "app.services.ml.trainers.base_ml_trainer.model_manager.load_model",
            return_value=model_data,
        ):
            result = trainer.load_model("/path/to/model")

        assert result is False
        assert trainer._model is None
        assert trainer.is_trained is False

    def test_calculate_features_fallback(self, trainer, sample_data):
        """特徴量計算エラー時のフォールバック"""
        with patch.object(
            trainer.feature_service,
            "calculate_advanced_features",
            side_effect=Exception("Calc error"),
        ):
            # エラー時は元のデータをコピーして返すべき
            result = trainer._calculate_features(sample_data)
            pd.testing.assert_frame_equal(result, sample_data)

    def test_cross_validation_flow(self, trainer, sample_data):
        """クロスバリデーションのフロー確認"""
        X = pd.DataFrame(np.random.randn(150, 5), index=sample_data.index)
        y = pd.Series(np.random.randn(150), index=sample_data.index)

        with patch(
            "app.services.ml.trainers.base_ml_trainer.PurgedKFold"
        ) as mock_kfold:
            mock_kfold.return_value.split.return_value = [
                (np.arange(100), np.arange(100, 120)),
                (np.arange(20, 120), np.arange(120, 140)),
            ]

            # 各フォールドの結果をシミュレート
            trainer._train_model_impl = MagicMock(return_value={"qlike": 0.2})

            result = trainer._time_series_cross_validate(X, y, cv_splits=2)

            assert "cv_scores" in result
            assert len(result["cv_scores"]) == 2
            assert result["mean_score"] == 0.2

    def test_cross_validation_uses_training_params_for_t1(self, trainer, sample_data):
        """CV用のt1計算が学習パラメータを使うことを確認"""
        X = pd.DataFrame(np.random.randn(150, 5), index=sample_data.index)
        y = pd.Series(np.random.randn(150), index=sample_data.index)

        with patch("app.services.ml.cross_validation.get_t1_series") as mock_t1:
            mock_t1.return_value = pd.Series(X.index, index=X.index)
            with patch(
                "app.services.ml.trainers.base_ml_trainer.PurgedKFold"
            ) as mock_kfold:
                mock_kfold.return_value.split.return_value = [
                    (np.arange(100), np.arange(100, 120)),
                ]
                trainer._train_model_impl = MagicMock(return_value={"qlike": 0.2})

                result = trainer._time_series_cross_validate(
                    X, y, cv_splits=2, horizon_n=12, timeframe="1h"
                )

        mock_t1.assert_called_once_with(X.index, 12, timeframe="1h")
        assert "cv_scores" in result
        assert len(result["cv_scores"]) == 1

    def test_split_data_prefers_train_test_split_over_validation_split(
        self, trainer, sample_data
    ):
        """明示された train_test_split を優先して分割することを確認"""
        X = pd.DataFrame(
            np.random.randn(20, 3),
            columns=["feat1", "feat2", "feat3"],
            index=sample_data.index[:20],
        )
        y = pd.Series(np.random.randn(20), index=sample_data.index[:20])

        X_train, X_test, y_train, y_test = trainer._split_data(
            X,
            y,
            train_test_split=0.75,
            validation_split=0.2,
        )

        assert len(X_train) == 15
        assert len(X_test) == 5
        assert len(y_train) == 15
        assert len(y_test) == 5

    def test_train_model_runs_cv_before_feature_selection(self, trainer, sample_data):
        """CVが特徴量選択より先に実行され、学習データだけを使うことを確認"""
        X_all = pd.DataFrame(
            np.arange(150 * 4, dtype=float).reshape(150, 4),
            columns=["feat1", "feat2", "feat3", "feat4"],
            index=sample_data.index,
        )
        y = pd.Series(np.random.randn(150), index=sample_data.index)

        X_tr = X_all.iloc[:120].copy()
        X_te = X_all.iloc[120:].copy()
        y_tr = y.iloc[:120].copy()
        y_te = y.iloc[120:].copy()

        feature_selector = MagicMock()
        feature_selector.get_feature_names_out.return_value = ["feat1", "feat2"]
        feature_selector.transform.side_effect = lambda X: X[
            ["feat1", "feat2"]
        ].to_numpy()
        trainer.feature_selector = feature_selector

        call_order = []

        def cv_side_effect(X_cv, y_cv, **kwargs):
            call_order.append("cv")
            pd.testing.assert_frame_equal(X_cv, X_tr)
            pd.testing.assert_series_equal(y_cv, y_tr)
            return {"cv_scores": [0.2], "mean_score": 0.2}

        def fit_side_effect(X_fit, y_fit):
            call_order.append("fit")
            pd.testing.assert_frame_equal(X_fit, X_tr)
            pd.testing.assert_series_equal(y_fit, y_tr)
            return feature_selector

        feature_selector.fit.side_effect = fit_side_effect

        with patch.object(trainer, "_calculate_features", return_value=X_all):
            with patch.object(
                trainer,
                "_prepare_training_data",
                return_value=(X_all, y),
            ):
                with patch.object(
                    trainer,
                    "_split_data",
                    return_value=(X_tr, X_te, y_tr, y_te),
                ):
                    with patch.object(
                        trainer,
                        "_time_series_cross_validate",
                        side_effect=cv_side_effect,
                    ) as mock_cv:
                        with patch.object(
                            trainer,
                            "_train_model_impl",
                            return_value={"accuracy": 0.8},
                        ):
                            with patch(
                                "app.services.ml.trainers.base_ml_trainer.model_manager.save_model",
                                return_value="/path/to/model",
                            ):
                                result = trainer.train_model(
                                    sample_data,
                                    save_model=True,
                                    use_cross_validation=True,
                                )

        assert call_order == ["cv", "fit"]
        mock_cv.assert_called_once()
        assert result["success"] is True
        assert result["model_path"] == "/path/to/model"

    def test_train_model_logs_warning_when_result_recombine_fails(
        self, trainer, sample_data, caplog
    ):
        """学習後の全量再構築に失敗しても警告を残して学習結果を返す"""
        X_all = pd.DataFrame(
            np.arange(150 * 4, dtype=float).reshape(150, 4),
            columns=["feat1", "feat2", "feat3", "feat4"],
            index=sample_data.index,
        )
        y = pd.Series(np.random.randn(150), index=sample_data.index)

        X_tr = X_all.iloc[:120].copy()
        X_te = X_all.iloc[120:].copy()
        y_tr = y.iloc[:120].copy()
        y_te = y.iloc[120:].copy()

        with patch.object(trainer, "_calculate_features", return_value=X_all):
            with patch.object(trainer, "_prepare_training_data", return_value=(X_all, y)):
                with patch.object(
                    trainer,
                    "_split_data",
                    return_value=(X_tr, X_te, y_tr, y_te),
                ):
                    with patch.object(
                        trainer,
                        "_train_model_impl",
                        return_value={"accuracy": 0.8},
                    ):
                        with patch(
                            "app.services.ml.trainers.base_ml_trainer.pd.concat",
                            side_effect=TypeError("concat failed"),
                        ):
                            with caplog.at_level("WARNING"):
                                result = trainer.train_model(
                                    sample_data,
                                    save_model=False,
                                )

        assert result["success"] is True
        assert result["total_samples"] == len(X_tr)
        assert "学習結果の全データ再構築に失敗" in caplog.text

    def test_cleanup_resources(self, trainer):
        """リソースクリーンアップのテスト"""
        trainer.is_trained = True
        trainer._model = MagicMock()

        trainer.cleanup_resources()

        assert trainer._model is None
        assert trainer.is_trained is False

    # ------------------------------------------------------------------
    # save_model
    # ------------------------------------------------------------------

    def test_save_model_raises_when_not_trained(self, trainer):
        """未学習状態で save_model を呼ぶとエラー"""
        from app.services.ml.common.exceptions import MLModelError

        with pytest.raises(MLModelError, match="学習済みモデルがありません"):
            trainer.save_model("test_model")

    def test_save_model_success(self, trainer):
        """学習済みモデルの保存成功"""
        trainer.is_trained = True
        trainer.feature_columns = ["f1", "f2"]
        trainer._model = MagicMock()

        with patch(
            "app.services.ml.trainers.base_ml_trainer.model_manager.save_model",
            return_value="/saved/path",
        ):
            path = trainer.save_model("my_model", metadata={"acc": 0.9})
            assert path == "/saved/path"

    # ------------------------------------------------------------------
    # get_feature_importance
    # ------------------------------------------------------------------

    def test_get_feature_importance_not_trained(self, trainer):
        """未学習時は空辞書を返す"""
        assert trainer.get_feature_importance() == {}

    def test_get_feature_importance_trained(self, trainer):
        """学習済みの重要度取得"""
        import numpy as np

        trainer.is_trained = True
        trainer.feature_columns = ["f1", "f2"]
        mock_model = MagicMock()
        mock_model.feature_importances_ = np.array([0.8, 0.2])
        trainer._model = mock_model

        imp = trainer.get_feature_importance(top_n=5)
        assert imp["f1"] == 0.8

    # ------------------------------------------------------------------
    # _split_data
    # ------------------------------------------------------------------

    def test_split_data_default(self, trainer, sample_data):
        """デフォルトの時系列分割"""
        X = pd.DataFrame(np.random.randn(150, 3), index=sample_data.index)
        y = pd.Series(np.random.randint(0, 2, 150), index=sample_data.index)

        X_tr, X_te, y_tr, y_te = trainer._split_data(X, y, test_size=0.2)

        assert len(X_tr) == 120
        assert len(X_te) == 30
        # 時系列順序の検証
        assert X_tr.index[-1] < X_te.index[0]

    def test_split_data_custom_size(self, trainer, sample_data):
        """カスタムサイズの分割"""
        X = pd.DataFrame(np.random.randn(150, 3), index=sample_data.index)
        y = pd.Series(np.random.randint(0, 2, 150), index=sample_data.index)

        X_tr, X_te, y_tr, y_te = trainer._split_data(X, y, test_size=0.3)
        assert len(X_tr) == 105
        assert len(X_te) == 45

    # ------------------------------------------------------------------
    # _preprocess_data
    # ------------------------------------------------------------------

    def test_preprocess_data_scaling(self, trainer):
        """スケーリングのテスト"""
        X_tr = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
        X_te = pd.DataFrame({"a": [4.0, 5.0], "b": [40.0, 50.0]})

        X_tr_s, X_te_s = trainer._preprocess_data(X_tr, X_te)

        assert isinstance(X_tr_s, pd.DataFrame)
        assert isinstance(X_te_s, pd.DataFrame)
        # スケーリング後は平均≈0
        assert abs(X_tr_s["a"].mean()) < 1e-10

    # ------------------------------------------------------------------
    # _format_training_result
    # ------------------------------------------------------------------

    def test_format_training_result(self, trainer):
        """学習結果の整形"""
        trainer.feature_columns = ["f1", "f2", "f3"]
        X = pd.DataFrame(np.random.randn(10, 3))
        y = pd.Series(np.random.randint(0, 2, 10))

        res = trainer._format_training_result({"accuracy": 0.9}, X, y)

        assert res["success"] is True
        assert res["feature_count"] == 3
        assert res["total_samples"] == 10
        assert res["accuracy"] == 0.9

    # ------------------------------------------------------------------
    # _apply_feature_selection
    # ------------------------------------------------------------------

    def test_apply_feature_selection_fallback_on_error(self, trainer):
        """特徴量選択エラー時は全特徴量をそのまま使用"""
        X_tr = pd.DataFrame({"f1": [1, 2, 3], "f2": [4, 5, 6]})
        X_te = pd.DataFrame({"f1": [7, 8], "f2": [9, 10]})
        y_tr = pd.Series([0, 1, 0])

        mock_selector = MagicMock()
        mock_selector.fit.side_effect = Exception("fit error")
        trainer.feature_selector = mock_selector

        X_tr_out, X_te_out = trainer._apply_feature_selection(X_tr, X_te, y_tr)

        assert list(X_tr_out.columns) == ["f1", "f2"]
        assert list(X_te_out.columns) == ["f1", "f2"]

    # ------------------------------------------------------------------
    # _prepare_training_data
    # ------------------------------------------------------------------

    def test_prepare_training_data_raises_on_error(self, trainer, sample_data):
        """ターゲット生成エラー時は DataError"""
        from app.utils.error_handler import DataError

        with patch.object(
            trainer.target_service,
            "prepare_targets",
            side_effect=Exception("target error"),
        ):
            with pytest.raises(DataError, match="準備に失敗"):
                trainer._prepare_training_data(sample_data, sample_data)

    # ------------------------------------------------------------------
    # _cleanup_models
    # ------------------------------------------------------------------

    def test_cleanup_models_clears_state(self, trainer):
        """モデルクリーンアップで状態がリセットされる"""
        from app.services.ml.common.base_resource_manager import CleanupLevel

        trainer.is_trained = True
        trainer._model = MagicMock()
        trainer.feature_columns = ["f1"]
        trainer.current_model_path = "/some/path"

        trainer._cleanup_models(CleanupLevel.THOROUGH)

        assert trainer._model is None
        assert trainer.is_trained is False
        assert trainer.feature_columns is None

    # ------------------------------------------------------------------
    # predict_volatility 異常系
    # ------------------------------------------------------------------

    def test_predict_volatility_returns_default_on_error(self, trainer, sample_data):
        """予測エラー時はデフォルト値を返す"""
        trainer.is_trained = True
        trainer.feature_columns = ["f1", "f2"]
        trainer.predict = MagicMock(side_effect=RuntimeError("inference error"))

        features = pd.DataFrame(
            np.random.randn(5, 2), columns=["f1", "f2"], index=sample_data.index[:5]
        )

        with patch(
            "app.services.ml.trainers.base_ml_trainer.prepare_data_for_prediction",
            return_value=features,
        ):
            result = trainer.predict_volatility(features)
            assert "forecast_log_rv" in result
