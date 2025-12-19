import pytest
import numpy as np
import pandas as pd
import torch
from app.services.ml.models.lstm_model import LSTMModel

class TestLSTMModel:
    @pytest.fixture
    def sample_data(self):
        # 100件、2特徴量
        X = np.random.randn(100, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        return X, y

    def test_create_sequences(self):
        """シーケンス作成ロジックのテスト"""
        X = np.arange(20).reshape(10, 2) # 10 samples
        y = np.arange(10)
        
        # seq_len = 3
        model = LSTMModel(input_dim=2, seq_len=3)
        X_seq, y_seq = model._create_sequences(X, y)
        
        # 10 - 3 + 1 = 8 samples (sliding window)
        assert len(X_seq) == 8
        assert X_seq.shape == (8, 3, 2)
        assert len(y_seq) == 8
        # ラベルはシーケンスの最後（ターゲット）に合わせる
        assert y_seq[0] == 2 # 0, 1, [2]
        assert y_seq[-1] == 9 # 7, 8, [9]

    def test_fit_and_predict(self, sample_data):
        """学習と予測のフルフロー"""
        X, y = sample_data
        model = LSTMModel(input_dim=2, seq_len=10, epochs=2, batch_size=16)
        
        # 1. 学習
        model.fit(X, y)
        assert model.model is not None
        
        # 2. 予測確率
        probs = model.predict_proba(X)
        assert len(probs) == len(X)
        # 前方の seq_len - 1 個はパディング(0.5)
        assert np.all(probs[:9] == 0.5)
        assert np.all((probs >= 0) & (probs <= 1))
        
        # 3. クラス予測
        preds = model.predict(X)
        assert len(preds) == len(X)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_insufficient_data(self):
        """データ不足時の挙動"""
        X = np.random.randn(5, 2)
        y = np.array([0, 1, 0, 1, 0])
        model = LSTMModel(input_dim=2, seq_len=10)
        
        # 学習データが seq_len 未満
        model.fit(X, y)
        assert model.model is None # 学習されない
        
        # 未学習状態で予測を呼ぶと RuntimeError
        with pytest.raises(RuntimeError, match="Model has not been trained yet"):
            model.predict_proba(X)

    def test_device_selection(self):
        """デバイス選択ロジック"""
        model_cpu = LSTMModel(input_dim=2, device="cpu")
        assert model_cpu.device == torch.device("cpu")
        
        # auto の場合は torch の利用可能性に従う
        model_auto = LSTMModel(input_dim=2, device="auto")
        assert isinstance(model_auto.device, torch.device)
