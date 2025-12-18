import logging

import torch.nn as nn

from .base_rnn_model import BaseRNNModel

logger = logging.getLogger(__name__)


class LSTMClassifier(nn.Module):
    """
    LSTMベースの二値分類モデル
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # LSTM Forward (初期状態は自動的にゼロ初期化される)
        out, _ = self.lstm(x)
        # 最後のタイムステップの出力を使用
        return self.sigmoid(self.fc(out[:, -1, :]))


class LSTMModel(BaseRNNModel):
    """
    scikit-learnライクなLSTMモデルラッパー
    """

    def _create_model(self) -> nn.Module:
        return LSTMClassifier(
            self.input_dim, self.hidden_dim, self.num_layers, self.dropout
        )



