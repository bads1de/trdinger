import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Tuple, Union

logger = logging.getLogger(__name__)

class GRUClassifier(nn.Module):
    """
    GRUベースの二値分類モデル
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(GRUClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
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
        
        # 初期隠れ状態 (num_layers, batch_size, hidden_dim)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # GRU Forward
        # out: (batch_size, seq_len, hidden_dim)
        out, _ = self.gru(x, h0)
        
        # 最後のタイムステップの出力を使用
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out

class GRUModel:
    """
    scikit-learnライクなGRUモデルラッパー
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
            # データが短すぎる場合は空を返すかエラー
            return np.array([]), np.array([]) if targets is not None else None

        for i in range(len(data) - self.seq_len + 1):
            x = data[i:(i + self.seq_len)]
            xs.append(x)
            if targets is not None:
                # ターゲットはシーケンスの最後の時点（の予測したい未来）に対応
                # ここでは「現在のシーケンスから次の足を予測」ではなく、
                # 「現在のシーケンスが表す時点のラベル（アライメント済み）」を使う想定
                # 通常、yは既に shift(-1) されているか、予測対象の時刻にアライメントされている
                ys.append(targets[i + self.seq_len - 1])
                
        return np.array(xs), np.array(ys) if targets is not None else None

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], eval_set=None):
        """
        モデルの学習
        """
        X_values = X.values if isinstance(X, pd.DataFrame) else X
        y_values = y.values if isinstance(y, pd.Series) else y
        
        # シーケンス生成
        X_seq, y_seq = self._create_sequences(X_values, y_values)
        
        if len(X_seq) == 0:
            logger.warning("学習データが少なすぎてシーケンスを生成できませんでした")
            return self

        # Tensor変換
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.FloatTensor(y_seq).unsqueeze(1).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # モデル初期化
        self.model = GRUClassifier(
            self.input_dim, 
            self.hidden_dim, 
            self.num_layers, 
            self.dropout
        ).to(self.device)
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # 学習ループ
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
            
            # 簡易ログ (冗長になるのでコメントアウト可)
            # avg_loss = total_loss / len(loader)
            # logger.info(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")
            
        return self

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        予測確率を出力
        入力データと同じ長さの配列を返す（先頭はパディング）
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")
            
        X_values = X.values if isinstance(X, pd.DataFrame) else X
        
        # シーケンス生成 (ターゲットなし)
        X_seq, _ = self._create_sequences(X_values)
        
        if len(X_seq) == 0:
            # データ不足
            return np.zeros(len(X_values)) + 0.5

        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            # バッチ処理しないとメモリ溢れる可能性あり
            # 簡易的に一括処理（データ量が少なければOK）
            outputs = []
            # 推論時はshuffle=False
            dataset = TensorDataset(X_tensor)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
            
            for batch in loader:
                batch_X = batch[0]
                out = self.model(batch_X)
                outputs.append(out.cpu().numpy())
                
            preds = np.concatenate(outputs).flatten()
            
        # 長さを合わせるためのパディング
        # create_sequencesで (len - seq_len + 1) になっているので、
        # 先頭 (seq_len - 1) 個が足りない
        padding_len = len(X_values) - len(preds)
        padding = np.full(padding_len, 0.5) # 0.5で埋める（不明）
        
        full_preds = np.concatenate([padding, preds])
        
        return full_preds

    def predict(self, X: Union[pd.DataFrame, np.ndarray], threshold: float = 0.5) -> np.ndarray:
        """クラス予測"""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
