import pandas as pd
import numpy as np
import pytest
from app.services.ml.feature_engineering.feature_engineering_service import (
    FeatureEngineeringService,
)


class TestFeatureLeak:
    def test_imputation_leakage(self):
        """欠損値補完におけるデータリークを検証"""
        # データ準備
        # 前半: 0付近の値
        # 後半: 1000付近の値
        # 前半に欠損値を入れる
        dates = pd.date_range(start="2024-01-01", periods=20, freq="D")

        # 再現性のためシード固定
        np.random.seed(42)

        data = {
            "open": np.concatenate(
                [np.random.randn(10) * 1, np.random.randn(10) * 1 + 1000]
            ),
            "high": np.concatenate(
                [np.random.randn(10) * 1 + 1, np.random.randn(10) * 1 + 1001]
            ),
            "low": np.concatenate(
                [np.random.randn(10) * 1 - 1, np.random.randn(10) * 1 + 999]
            ),
            "close": np.concatenate(
                [np.random.randn(10) * 1, np.random.randn(10) * 1 + 1000]
            ),
            "volume": np.random.rand(20) * 100,
        }
        df = pd.DataFrame(data, index=dates)

        # 前半の最後の値を欠損させる
        # 注意: calculate_advanced_features内でコピーされるので元データは変更されないが、
        # ここでは入力データとして欠損値を含むものを渡す
        df.iloc[9, df.columns.get_loc("close")] = np.nan

        service = FeatureEngineeringService()

        # 特徴量計算（内部で欠損値補完が行われる）
        # 警告を抑制
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result_df = service.calculate_advanced_features(df, profile="research")

        # 補完された値を確認
        imputed_value = result_df.iloc[9]["close"]

        print(f"Index 8 (before): {result_df.iloc[8]['close']}")
        print(f"Index 9 (imputed): {result_df.iloc[9]['close']}")
        print(f"Index 10 (after): {result_df.iloc[10]['close']}")

        # 全体の中央値を使うと、後半の1000の影響を受けて500くらいになるはず
        # 正しい実装（過去のみ）なら、前半の中央値（0付近）になるはず

        # 閾値を設定（例えば100以上ならリークとみなす）
        assert (
            imputed_value < 100
        ), f"Imputed value {imputed_value} is too high, suggesting leakage from future data (global median imputation)"
