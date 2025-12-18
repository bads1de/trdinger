import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class BaseRNNModel(ABC):
    """
    RNN系モデルの基底クラス
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        seq_len: int = 24,
        batch_size: int = 64,
        epochs: int = 20,
        learning_rate: float = 0.001,
        dropout: float = 0.2,
        device: str = "auto",
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.dropout = dropout

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = None

    @abstractmethod
    def _create_model(self) -> nn.Module:
        """モデルを作成する抽象メソッド"""

    def _create_sequences(
        self, data: np.ndarray, targets: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """時系列データをシーケンスに変換（ベクトル化版）"""
        if len(data) < self.seq_len:
            return np.array([]), (np.array([]) if targets is not None else None)

        from numpy.lib.stride_tricks import sliding_window_view
        xs = sliding_window_view(data, window_shape=(self.seq_len, data.shape[1])).squeeze(1)
        ys = targets[self.seq_len - 1 :] if targets is not None else None
        return xs, ys

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        eval_set=None,
    ):
        """
        モデルの学習
        """
        X_values = X.values if isinstance(X, pd.DataFrame) else X
        y_values = y.values if isinstance(y, pd.Series) else y

        X_seq, y_seq = self._create_sequences(X_values, y_values)

        if len(X_seq) == 0:
            logger.warning("学習データが少なすぎてシーケンスを生成できませんでした")
            return self

        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.FloatTensor(y_seq).unsqueeze(1).to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = self._create_model().to(self.device)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        return self

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        予測確率を出力
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")

        X_values = X.values if isinstance(X, pd.DataFrame) else X
        X_seq, _ = self._create_sequences(X_values)

        if len(X_seq) == 0:
            return np.zeros(len(X_values)) + 0.5

        X_tensor = torch.FloatTensor(X_seq).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = []
            dataset = TensorDataset(X_tensor)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

            for batch in loader:
                batch_X = batch[0]
                out = self.model(batch_X)
                outputs.append(out.cpu().numpy())

            preds = np.concatenate(outputs).flatten()

        padding_len = len(X_values) - len(preds)
        padding = np.full(padding_len, 0.5)

        full_preds = np.concatenate([padding, preds])

        return full_preds

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray], threshold: float = 0.5
    ) -> np.ndarray:
        """クラス予測"""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)



