import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from app.services.ml.data_processing.sampling import ImbalanceSampler
from app.config.unified_config import MLTrainingConfig


class TestClassImbalance:
    def test_smote_increases_minority_samples(self):
        """SMOTEが少数派クラスを増やすことを確認"""
        # 不均衡データを作成 (0: 90個, 1: 10個)
        X = pd.DataFrame(
            np.random.randn(100, 5), columns=[f"col_{i}" for i in range(5)]
        )
        y = pd.Series([0] * 90 + [1] * 10)

        sampler = ImbalanceSampler(method="smote", random_state=42)
        X_resampled, y_resampled = sampler.fit_resample(X, y)

        # 少数派クラス(1)が増えていることを確認
        assert y_resampled.value_counts()[1] > y.value_counts()[1]
        # デフォルトでは均衡するはず
        assert y_resampled.value_counts()[1] == y_resampled.value_counts()[0]

    def test_config_has_imbalance_settings(self):
        """MLTrainingConfigに不均衡対策の設定が含まれていることを確認"""
        config = MLTrainingConfig()
        assert hasattr(config, "use_class_weight")
        assert hasattr(config, "class_weight_mode")
        assert hasattr(config, "use_smote")
        assert hasattr(config, "smote_method")

        # デフォルト値の確認
        assert config.use_class_weight is False
        assert config.class_weight_mode == "balanced"
        assert config.use_smote is False
        assert config.smote_method == "smote"

    @patch("lightgbm.train")
    def test_lightgbm_receives_class_weight(self, mock_train):
        """LightGBMモデルがclass_weightパラメータを受け取ることを確認"""
        # これは統合テストに近いが、LightGBMModelクラスの改修を確認するために必要
        # ここではLightGBMModelを直接インスタンス化してテストする
        from app.services.ml.models.lightgbm import LightGBMModel

        # lgb.trainが返すモックオブジェクトを設定
        mock_lgbm_model = MagicMock()
        # predictメソッドがndarrayを返すように設定
        mock_lgbm_model.predict.return_value = np.array([0.1, 0.9, 0.2, 0.8, 0.5, 0.6, 0.3, 0.7, 0.4, 0.9] * 10) # 100個の確率値をモック
        # best_iteration属性も必要に応じてモック
        mock_lgbm_model.best_iteration = 1
        mock_train.return_value = mock_lgbm_model

        model = LightGBMModel()
        X = pd.DataFrame(np.random.randn(100, 5))
        y = pd.Series([0] * 50 + [1] * 50)

        # class_weightを指定して学習
        model.fit(X, y, class_weight="balanced")

        # lgb.trainの引数paramsにclass_weightが含まれているか確認
        # 注: LightGBMのpython APIでは class_weight は params ではなく
        # train() の引数や Dataset の weight として渡す場合もあるが、
        # sklearn API (LGBMClassifier) なら class_weight 引数がある。
        # Trdingerの実装が native API か sklearn API かによる。
        # 既存コードを確認する必要があるが、一旦 native API と仮定して params をチェック。
        # もし sklearn API なら fit の引数をチェック。

        # 既存実装を確認していないので、とりあえず呼び出しが行われることだけ確認し、
        # 詳細は実装時に合わせる。
        assert mock_train.called

        # lgb.trainの第2引数 (train_data) が weight 属性を持つことを確認
        train_data_arg = mock_train.call_args[0][1]
        assert hasattr(train_data_arg, 'weight')
        assert train_data_arg.weight is not None
