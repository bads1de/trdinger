import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from app.services.ml.models.base_rnn_model import BaseRNNModel


# テスト用の具象クラス
class ConcreteRNNModel(BaseRNNModel):
    def _create_model(self):
        return nn.Linear(self.input_dim, 1)  # ダミーモデル


class TestBaseRNNModel:
    @pytest.fixture
    def model(self):
        return ConcreteRNNModel(input_dim=10, seq_len=5, epochs=1, batch_size=2)

    def test_create_sequences(self, model):
        data = np.random.rand(20, 10)
        targets = np.random.randint(0, 2, 20)

        X_seq, y_seq = model._create_sequences(data, targets)

        # seq_len=5 なので、20 - 5 + 1 = 16 サンプル
        assert X_seq.shape == (16, 5, 10)
        assert y_seq.shape == (16,)

        # ターゲットなしの場合
        X_seq_no_target, y_seq_no_target = model._create_sequences(data)
        assert X_seq_no_target.shape == (16, 5, 10)
        assert y_seq_no_target is None

    def test_fit(self, model):
        X = pd.DataFrame(np.random.rand(20, 10))
        y = pd.Series(np.random.randint(0, 2, 20))

        # _create_model はダミーなので、学習ループが回るか確認
        # nn.Linear は forward で (batch, seq, dim) を受け取るとエラーになる可能性があるが
        # BaseRNNModel の学習ループの実装依存。
        # ここでは _create_model をモックして検証する方が安全かもしれないが、
        # 簡易的な実装で学習ループの通過を確認する。

        # ダミーモデルを少し修正してシーケンス入力を扱えるようにする
        class DummyModel(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.fc = nn.Linear(input_dim, 1)

            def forward(self, x):
                # x: (batch, seq, dim) -> (batch, dim) (last step)
                return torch.sigmoid(self.fc(x[:, -1, :]))

        model._create_model = MagicMock(return_value=DummyModel(10).to(model.device))

        model.fit(X, y)

        assert model.model is not None
        model._create_model.assert_called_once()

    def test_predict_proba(self, model):
        X = pd.DataFrame(np.random.rand(20, 10))

        # モデルを設定
        class DummyModel(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.fc = nn.Linear(input_dim, 1)

            def forward(self, x):
                return torch.sigmoid(self.fc(x[:, -1, :]))

        model.model = DummyModel(10).to(model.device)
        model.model.eval()

        probs = model.predict_proba(X)

        assert len(probs) == 20
        assert isinstance(probs, np.ndarray)
        # パディング部分は 0.5
        assert np.all(probs[:4] == 0.5)

    def test_predict(self, model):
        X = pd.DataFrame(np.random.rand(20, 10))
        model.predict_proba = MagicMock(return_value=np.array([0.1, 0.6, 0.4, 0.9]))

        preds = model.predict(X, threshold=0.5)

        assert np.array_equal(preds, np.array([0, 1, 0, 1]))


