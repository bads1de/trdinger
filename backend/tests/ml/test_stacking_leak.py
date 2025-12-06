import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

from app.services.ml.ensemble.stacking import StackingEnsemble


from app.services.ml.cross_validation.purged_kfold import PurgedKFold


class TestStackingLeak:
    def test_stacking_uses_purged_kfold(self):
        """StackingEnsembleがPurgedKFold（時系列分割）を使用しているか検証"""
        # データ準備
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        X = pd.DataFrame(
            np.random.randn(100, 5),
            index=dates,
            columns=[f"feat_{i}" for i in range(5)],
        )

        config = {
            "base_models": ["lightgbm"],
            "meta_model": "logistic_regression",
            "cv_folds": 5,
            # cv_strategy を指定しない（デフォルト: purged_kfold）
        }

        ensemble = StackingEnsemble(config)

        # _create_cv_splitterメソッドを直接テスト
        cv = ensemble._create_cv_splitter(X)

        # PurgedKFoldであることを検証
        assert isinstance(
            cv, PurgedKFold
        ), f"PurgedKFold must be used for time series data. Got: {type(cv)}"
        assert not isinstance(
            cv, StratifiedKFold
        ), "StratifiedKFold causes data leakage in time series!"

    def test_stacking_uses_kfold_when_specified(self):
        """cv_strategy='kfold'の場合、KFoldが使用されることを検証"""
        from sklearn.model_selection import KFold

        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        X = pd.DataFrame(
            np.random.randn(100, 5),
            index=dates,
            columns=[f"feat_{i}" for i in range(5)],
        )

        config = {
            "base_models": ["lightgbm"],
            "meta_model": "logistic_regression",
            "cv_folds": 5,
            "cv_strategy": "kfold",
        }

        ensemble = StackingEnsemble(config)
        cv = ensemble._create_cv_splitter(X)

        assert isinstance(
            cv, KFold
        ), f"KFold must be used when specified. Got: {type(cv)}"

    def test_stacking_uses_stratified_kfold_when_specified(self):
        """cv_strategy='stratified_kfold'の場合、StratifiedKFoldが使用されることを検証"""
        from sklearn.model_selection import StratifiedKFold

        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        X = pd.DataFrame(
            np.random.randn(100, 5),
            index=dates,
            columns=[f"feat_{i}" for i in range(5)],
        )

        config = {
            "base_models": ["lightgbm"],
            "meta_model": "logistic_regression",
            "cv_folds": 5,
            "cv_strategy": "stratified_kfold",
        }

        ensemble = StackingEnsemble(config)
        cv = ensemble._create_cv_splitter(X)

        assert isinstance(
            cv, StratifiedKFold
        ), f"StratifiedKFold must be used when specified. Got: {type(cv)}"
