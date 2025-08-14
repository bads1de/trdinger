import numpy as np
import pandas as pd
import pytest

from app.utils.data_processing import (
    DataProcessor,
    OutlierRemovalTransformer,
    create_outlier_removal_pipeline,
)


def test_outlier_removal_transformer_isolation_forest():
    """IsolationForestを使用した外れ値除去テスト"""
    df = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 1000],  # 1000 should be an outlier
            "B": [10, 11, 12, 11, 10],  # no outliers
        }
    )

    transformer = OutlierRemovalTransformer(
        method="isolation_forest", contamination=0.2
    )

    # フィットと変換
    transformer.fit(df)
    result = transformer.transform(df)

    # 外れ値がNaNに置き換えられていることを確認
    assert result.isna().sum().sum() > 0, "外れ値がNaNに置き換わっていません"


def test_outlier_removal_transformer_local_outlier_factor():
    """LocalOutlierFactorを使用した外れ値除去テスト"""
    df = pd.DataFrame({"A": [0.0, 0.1, -0.2, 0.05, 100.0]})

    transformer = OutlierRemovalTransformer(
        method="local_outlier_factor", contamination=0.2
    )

    # フィットと変換
    transformer.fit(df)
    result = transformer.transform(df)

    # 外れ値がNaNに置き換えられていることを確認
    assert result.isna().sum().sum() > 0, "外れ値がNaNに置き換わっていません"


def test_outlier_removal_pipeline():
    """外れ値除去パイプラインテスト"""
    df = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 1000],
            "B": [10, 11, 12, 11, 10],
        }
    )

    pipeline = create_outlier_removal_pipeline(
        method="isolation_forest", contamination=0.2
    )

    result = pipeline.fit_transform(df)

    # パイプラインが正常に動作することを確認
    assert result.shape[0] == df.shape[0]
    assert result.shape[1] == df.shape[1]
    # 欠損値が補完されていることを確認（パイプラインにはImputerが含まれる）
    assert not pd.isna(result).any().any(), "パイプライン処理後に欠損値が残っています"


def test_mixed_data_types_handling():
    """混在データ型の処理テスト"""
    df = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 1000],
            "C": ["x", "y", "z", "w", "v"],  # non-numeric
        }
    )

    # DataProcessorのPipelineベース処理を使用
    dp = DataProcessor()
    result = dp.preprocess_with_pipeline(
        df,
        pipeline_name="test_mixed",
        fit_pipeline=True,
        remove_outliers=True,
        outlier_method="isolation_forest",
        outlier_threshold=0.2,
    )

    # 処理が正常に完了することを確認
    assert result is not None
    assert len(result) == len(df)


def test_pipeline_integration_complete_flow():
    """完全なパイプライン統合テスト"""
    df = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 1000.0]})

    dp = DataProcessor()

    # 効率的なデータ処理を使用
    result = dp.process_data_efficiently(
        df,
        pipeline_name="integration_test",
        remove_outliers=True,
        outlier_method="isolation_forest",
        outlier_threshold=0.2,
        numeric_strategy="median",
    )

    # 処理が完了し、欠損値が補完されていることを確認
    assert result is not None
    assert not pd.isna(result).any().any(), "統合処理後に欠損値が残っています"


def test_transformer_error_handling():
    """Transformerのエラーハンドリングテスト"""
    transformer = OutlierRemovalTransformer()

    df = pd.DataFrame({"A": [1, 2, 3, 4, 5]})

    # fit前のtransformでエラーが発生することを確認
    with pytest.raises(ValueError, match="fit\\(\\)を先に実行してください"):
        transformer.transform(df)

    # 無効なメソッドでエラーが発生することを確認
    invalid_transformer = OutlierRemovalTransformer(method="invalid_method")
    with pytest.raises(ValueError, match="未対応の外れ値検出方法"):
        invalid_transformer.fit(df)
