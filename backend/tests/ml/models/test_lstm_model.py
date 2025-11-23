import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from app.services.ml.models.lstm_model import LSTMModel, LSTMClassifier

class TestLSTMModel:
    def test_lstm_classifier_architecture(self):
        """PyTorchモデルアーキテクチャのテスト"""
        input_dim = 10
        hidden_dim = 32
        num_layers = 2
        
        model = LSTMClassifier(input_dim, hidden_dim, num_layers)
        
        # レイヤー構成の確認
        assert isinstance(model.lstm, nn.LSTM)
        assert model.lstm.input_size == input_dim
        assert model.lstm.hidden_size == hidden_dim
        assert model.lstm.num_layers == num_layers
        assert isinstance(model.fc, nn.Linear)
        assert isinstance(model.sigmoid, nn.Sigmoid)

    def test_lstm_forward(self):
        """Forwardパスの動作確認"""
        batch_size = 5
        seq_len = 20
        input_dim = 10
        hidden_dim = 32
        
        model = LSTMClassifier(input_dim, hidden_dim, 1)
        
        # ダミー入力 (batch, seq, features)
        x = torch.randn(batch_size, seq_len, input_dim)
        
        output = model(x)
        
        # 出力形状確認 (batch, 1)
        assert output.shape == (batch_size, 1)
        # 値の範囲確認 (Sigmoid後なので 0~1)
        assert (output >= 0).all() and (output <= 1).all()

    def test_lstm_model_fit_predict(self):
        """ラッパークラスの学習と予測テスト"""
        n_samples = 100
        n_features = 5
        seq_len = 10
        
        # ダミーデータ
        X = pd.DataFrame(np.random.randn(n_samples, n_features), columns=[f"col_{i}" for i in range(n_features)])
        y = pd.Series(np.random.randint(0, 2, n_samples))
        
        params = {
            "input_dim": n_features,
            "hidden_dim": 16,
            "num_layers": 1,
            "seq_len": seq_len,
            "batch_size": 16,
            "epochs": 2,
            "learning_rate": 0.01,
            "dropout": 0.0
        }
        
        model = LSTMModel(**params)
        
        # 学習
        model.fit(X, y)
        
        # 予測 (確率)
        preds = model.predict_proba(X)
        
        # 検証
        assert len(preds) == n_samples
        assert isinstance(preds, np.ndarray)
        assert np.all((preds >= 0) & (preds <= 1))
        assert not np.isnan(preds).any()
