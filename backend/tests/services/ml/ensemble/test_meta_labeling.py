import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from app.services.ml.ensemble.meta_labeling import MetaLabelingService


class TestMetaLabelingService:
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n = 100
        dates = pd.date_range("2023-01-01", periods=n)
        X = pd.DataFrame(np.random.randn(n, 2), index=dates, columns=["f1", "f2"])
        y = pd.Series(np.random.randint(0, 2, n), index=dates)
        primary_proba = pd.Series(np.random.rand(n), index=dates)
        base_probs = pd.DataFrame(
            {"m1": np.random.rand(n), "m2": np.random.rand(n)}, index=dates
        )
        return X, y, primary_proba, base_probs

    def test_create_meta_labels(self, sample_data):
        """メタラベル生成ロジックのテスト"""
        X, y, primary_proba, base_probs = sample_data
        service = MetaLabelingService()

        # 一時的に閾値を 0.5 に設定
        mask, y_meta = service.create_meta_labels(primary_proba, y, threshold=0.5)

        # マスクの数は元データと同じ
        assert len(mask) == len(y)
        # y_meta のインデックスはマスクが True の場所のみ
        assert y_meta.index.equals(y.index[mask])
        # メタラベリングの定義: 正解ラベルがそのままメタラベルになる（一次予測が1の場合のみ抽出するため）
        assert y_meta.equals(y.loc[y_meta.index])

    def test_train_success(self, sample_data):
        """学習の成功フロー（内部モデルをモック化）"""
        X, y, primary_proba, base_probs = sample_data
        service = MetaLabelingService(model_type="random_forest")

        # 実際に学習させる代わりに、内部モデルの初期化をモック化
        with patch.object(service, "_init_model") as mock_init:
            mock_model = MagicMock()
            mock_init.return_value = mock_model

            # 50サンプル以上ヒットするように調整
            primary_proba = pd.Series(0.6, index=primary_proba.index)

            result = service.train(X, y, primary_proba, base_probs)

            assert result["status"] == "success"
            assert service.is_trained is True
            # メタ特徴量が構築されて fit に渡されたか
            assert mock_model.fit.called
            args, kwargs = mock_model.fit.call_args
            X_meta = args[0]
            # 特徴量が少なくとも1つ以上存在することを確認
            assert len(X_meta.columns) > 0
            assert "primary_proba" in X_meta.columns

    def test_predict_flow(self, sample_data):
        """予測フローのテスト"""
        X, y, primary_proba, base_probs = sample_data
        service = MetaLabelingService(model_type="random_forest")

        # 学習済み状態をシミュレート
        service.is_trained = True
        service.model = MagicMock()
        service.model.predict.return_value = np.ones(10)  # マスクされた10件分
        service.base_model_names = ["m1", "m2"]

        # 10件だけトレンドと予測されるように設定
        primary_proba = pd.Series(0.1, index=primary_proba.index)
        primary_proba.iloc[:10] = 0.9

        final_pred = service.predict(X, primary_proba, base_probs)

        assert len(final_pred) == len(X)
        assert final_pred.sum() == 10
        assert (final_pred.iloc[:10] == 1).all()

    def test_insufficient_data_skip(self, sample_data):
        """データ不足時の学習スキップ"""
        X, y, primary_proba, base_probs = sample_data
        service = MetaLabelingService()

        # 閾値を高くしてヒット数を減らす
        primary_proba = pd.Series(0.1, index=primary_proba.index)

        result = service.train(X, y, primary_proba, base_probs, threshold=0.9)
        assert result["status"] == "skipped"
        assert result["reason"] == "insufficient_data"

    def test_add_base_model_statistics(self):
        """統計量追加のテスト"""
        service = MetaLabelingService()
        X_meta = pd.DataFrame({"feat": [1, 2]})
        base_probs = pd.DataFrame({"m1": [0.1, 0.9], "m2": [0.3, 0.7]})

        res = service._add_base_model_statistics(X_meta, base_probs)
        assert "base_prob_mean" in res.columns
        assert res.loc[0, "base_prob_mean"] == pytest.approx(0.2)
        assert res.loc[1, "base_prob_mean"] == pytest.approx(0.8)
