import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Tuple, Union

logger = logging.getLogger(__name__)

class LSTMClassifier(nn.Module):
    """
    LSTMベースの二値分類モデル
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        
        # 初期隠れ状態とセル状態
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # LSTM Forward
        # out: (batch_size, seq_len, hidden_dim)
        out, _ = self.lstm(x, (h0, c0))
        
        # 最後のタイムステップの出力を使用
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out

class LSTMModel:
    """
    scikit-learnライクなLSTMモデルラッパー
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
        device: str = "auto"
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
        
    def _create_sequences(self, data: np.ndarray, targets: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        時系列データをスライディングウィンドウでシーケンスに変換
        """
        xs = []
        ys = []
        
        if len(data) < self.seq_len:
            return np.array([]), np.array([]) if targets is not None else None

        for i in range(len(data) - self.seq_len + 1):
            x = data[i:(i + self.seq_len)]
            xs.append(x)
            if targets is not None:
                ys.append(targets[i + self.seq_len - 1])
                
        return np.array(xs), np.array(ys) if targets is not None else None

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], eval_set=None):
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
        
        self.model = LSTMClassifier(
            self.input_dim, 
            self.hidden_dim, 
            self.num_layers, 
            self.dropout
        ).to(self.device)
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
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

    def predict(self, X: Union[pd.DataFrame, np.ndarray], threshold: float = 0.5) -> np.ndarray:
        """クラス予測"""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
