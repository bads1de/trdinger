import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from sklearn.model_selection import StratifiedKFold

from app.services.ml.ensemble.stacking import StackingEnsemble


from app.services.ml.cross_validation.purged_kfold import PurgedKFold


class TestStackingLeak:
    def test_stacking_uses_timeseries_split(self):
        """StackingEnsembleが時系列分割を使用しているか検証"""
        # データ準備
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        X = pd.DataFrame(
            np.random.randn(100, 5),
            index=dates,
            columns=[f"feat_{i}" for i in range(5)],
        )
        y = pd.Series(np.random.randint(0, 2, 100), index=dates)

        config = {
            "base_models": ["lightgbm"],
            "meta_model": "logistic_regression",
            "cv_folds": 5,
        }

        ensemble = StackingEnsemble(config)

        # StackingClassifierのモック化
        with patch(
            "app.services.ml.ensemble.stacking.StackingClassifier"
        ) as MockStackingClassifier:
            # 内部メソッドのモック化（依存関係を断ち切るため）
            ensemble._create_base_estimators = MagicMock(
                return_value=[("lgb", MagicMock())]
            )
            ensemble._create_base_model = MagicMock(return_value=MagicMock())

            # cross_val_predict のモック化
            # fitメソッド内で呼ばれる cross_val_predict もモックする
            with patch(
                "app.services.ml.ensemble.stacking.cross_val_predict"
            ) as mock_cv_predict:
                # ダミーの戻り値 (n_samples, 2クラス)
                mock_cv_predict.return_value = np.zeros((100, 2))

                # fit実行
                try:
                    ensemble.fit(X, y)
                except Exception as e:
                    with open("test_debug_error.txt", "w") as f:
                        f.write(f"Error during fit: {e}\n")
                    raise e

            # 検証: StackingClassifierの初期化引数 'cv' をチェック
            with open("test_debug_output.txt", "w") as f:
                f.write(f"Mock called: {MockStackingClassifier.called}\n")
                if MockStackingClassifier.called:
                    args, kwargs = MockStackingClassifier.call_args
                    cv_arg = kwargs.get("cv")
                    f.write(f"cv_arg type: {type(cv_arg)}\n")
                    f.write(f"cv_arg: {cv_arg}\n")

            # 現状の実装では StratifiedKFold が使われているため、
            # TimeSeriesSplit であることを期待するこのアサーションは失敗するはず
            assert not isinstance(
                cv_arg, StratifiedKFold
            ), "StratifiedKFold causes data leakage in time series!"
            assert isinstance(
                cv_arg, PurgedKFold
            ), f"PurgedKFold must be used for time series data. Got: {type(cv_arg)}"
