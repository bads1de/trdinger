import logging

import torch.nn as nn

from .base_rnn_model import BaseRNNModel

logger = logging.getLogger(__name__)


class GRUClassifier(nn.Module):
    """
    GRUベースの二値分類モデル
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super(GRUClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # GRU Forward (初期状態は自動的にゼロ初期化される)
        out, _ = self.gru(x)
        # 最後のタイムステップの出力を使用
        return self.sigmoid(self.fc(out[:, -1, :]))


class GRUModel(BaseRNNModel):
    """
    scikit-learnライクなGRUモデルラッパー
    """

    def _create_model(self) -> nn.Module:
        return GRUClassifier(
            self.input_dim, self.hidden_dim, self.num_layers, self.dropout
        )



