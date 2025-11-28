import pytest
import pandas as pd
import numpy as np
from app.services.ml.ensemble.meta_labeling import MetaLabelingService


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n_samples = 200  # サンプル数を増やす
    X = pd.DataFrame(
        np.random.rand(n_samples, 5),
        columns=[f"feature_{i}" for i in range(5)],
        index=pd.RangeIndex(n_samples),
    )
    y = pd.Series(
        np.random.randint(0, 2, n_samples),
        name="target",
        index=pd.RangeIndex(n_samples),
    )

    # 一次モデルの予測確率 (ランダムだが、少し高めに設定してTrend判定を増やす)
    primary_proba = np.random.uniform(0.4, 1.0, n_samples)
    primary_proba = pd.Series(primary_proba, index=X.index, name="primary_proba")

    # ベースモデルの予測確率 (複数)
    base_probs = pd.DataFrame(
        {"lgbm": np.random.rand(n_samples), "xgb": np.random.rand(n_samples)},
        index=X.index,
    )

    return X, y, primary_proba, base_probs


def test_create_meta_labels(sample_data):
    X, y, primary_proba, base_probs = sample_data
    service = MetaLabelingService()

    threshold = 0.5
    trend_mask, y_meta = service.create_meta_labels(primary_proba, y, threshold)

    # 一次モデルがトレンドと予測した数と一致するか
    expected_count = (primary_proba >= threshold).sum()
    assert len(y_meta) == expected_count
    assert len(trend_mask) == len(primary_proba)
    assert trend_mask.sum() == expected_count

    # y_metaは元のyと同じ値であるはず（インデックスが合っていれば）
    assert y_meta.equals(y.loc[y_meta.index])


def test_train_and_predict(sample_data):
    X, y, primary_proba, base_probs = sample_data
    service = MetaLabelingService()

    # 学習
    result = service.train(X, y, primary_proba, base_probs, threshold=0.5)
    assert result["status"] == "success"
    assert service.is_trained

    # 予測
    final_pred = service.predict(X, primary_proba, base_probs, threshold=0.5)

    assert len(final_pred) == len(X)
    # Pass(0)かExecute(1)のいずれか
    assert set(final_pred.unique()).issubset({0, 1})

    # 一次モデルがRange(確率 < 0.5)と予測したものは必ず0になるはず
    range_mask = primary_proba < 0.5
    assert (final_pred[range_mask] == 0).all()


def test_evaluate(sample_data):
    X, y, primary_proba, base_probs = sample_data
    service = MetaLabelingService()

    service.train(X, y, primary_proba, base_probs, threshold=0.5)

    metrics = service.evaluate(X, y, primary_proba, base_probs, threshold=0.5)

    assert "meta_accuracy" in metrics
    assert "meta_precision" in metrics
    assert "improvement_precision" in metrics
    assert metrics["meta_precision"] >= 0.0
    assert metrics["meta_precision"] <= 1.0
