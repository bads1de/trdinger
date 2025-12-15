import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

# プロジェクトルートをパスに追加してimport可能にする
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from app.services.ml.models.gru_model import GRUModel, GRUClassifier

class TestGRUModel:
    def test_gru_classifier_architecture(self):
        """PyTorchモデルアーキテクチャのテスト"""
        input_dim = 10
        hidden_dim = 32
        num_layers = 2
        
        model = GRUClassifier(input_dim, hidden_dim, num_layers)
        
        # レイヤー構成の確認
        assert isinstance(model.gru, nn.GRU)
        assert model.gru.input_size == input_dim
        assert model.gru.hidden_size == hidden_dim
        assert model.gru.num_layers == num_layers
        assert isinstance(model.fc, nn.Linear)
        assert isinstance(model.sigmoid, nn.Sigmoid)

    def test_gru_forward(self):
        """Forwardパスの動作確認"""
        batch_size = 5
        seq_len = 20
        input_dim = 10
        hidden_dim = 32
        
        model = GRUClassifier(input_dim, hidden_dim, 1)
        
        # ダミー入力 (batch, seq, features)
        x = torch.randn(batch_size, seq_len, input_dim)
        
        output = model(x)
        
        # 出力形状確認 (batch, 1)
        assert output.shape == (batch_size, 1)
        # 値の範囲確認 (Sigmoid後なので 0~1)
        assert (output >= 0).all() and (output <= 1).all()

    def test_gru_model_fit_predict(self):
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
        
        model = GRUModel(**params)
        
        # 学習
        model.fit(X, y)
        
        # 予測 (確率)
        preds = model.predict_proba(X)
        
        # 検証
        # 出力は入力と同じ長さ（パディング処理済み）であることを期待
        assert len(preds) == n_samples
        assert isinstance(preds, np.ndarray)
        # 確率は 0~1
        assert np.all((preds >= 0) & (preds <= 1))
        
        # 最初の seq_len-1 個は予測不可のため NaN または 0.5 (あるいは何らかのデフォルト値)
        # 実装依存だが、ここでは欠損していないか、あるいは特定の扱いになっているか確認
        # 今回は「後ろに詰める」方式ではなく「インデックスを合わせる」方式を想定
        # つまり最初の数個は予測不可。
        
        # NaNチェック (仕様によるが、使いやすさのため0.5埋めなどを想定)
        assert not np.isnan(preds).any()

    def test_sequence_generation(self):
        """時系列データのシーケンス変換テスト"""
        n_samples = 20
        n_features = 2
        seq_len = 5
        
        X = pd.DataFrame(np.arange(n_samples * n_features).reshape(n_samples, n_features))
        # 0, 1
        # 2, 3
        # ...
        
        model = GRUModel(input_dim=n_features, seq_len=seq_len)
        
        # 内部メソッドのテスト（本来はprivateだが確認のため）
        sequences, targets = model._create_sequences(X.values, np.zeros(n_samples))
        
        # 生成されるシーケンス数 = n_samples - seq_len + 1
        expected_samples = n_samples - seq_len + 1
        assert len(sequences) == expected_samples
        assert sequences.shape == (expected_samples, seq_len, n_features)
        
        # 最初のシーケンスの中身確認
        # X.values[0:5] と一致するはず
        np.testing.assert_array_equal(sequences[0], X.values[0:seq_len])


