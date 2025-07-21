"""
ハイパーパラメータ最適化のエンドツーエンドテスト

フロントエンドからバックエンドまでの最適化ワークフロー全体をテストします。
"""

import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.main import app
from app.core.services.ml.ml_training_service import MLTrainingService, OptimizationSettings
from app.core.services.optimization.optimizer_factory import OptimizerFactory


class TestOptimizationE2E:
    """最適化機能のエンドツーエンドテスト"""

    @pytest.fixture
    def client(self):
        """テストクライアント"""
        return TestClient(app)

    @pytest.fixture
    def sample_training_data(self):
        """サンプル学習データ"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=1000, freq='1H')
        data = pd.DataFrame({
            'timestamp': dates,
            'Open': np.random.uniform(40000, 50000, 1000),
            'High': np.random.uniform(40000, 50000, 1000),
            'Low': np.random.uniform(40000, 50000, 1000),
            'Close': np.random.uniform(40000, 50000, 1000),
            'Volume': np.random.uniform(100, 1000, 1000),
        })
        # HighとLowを調整
        data['High'] = np.maximum(data[['Open', 'Close']].max(axis=1), data['High'])
        data['Low'] = np.minimum(data[['Open', 'Close']].min(axis=1), data['Low'])
        return data

    def test_bayesian_optimization_e2e(self, client, sample_training_data):
        """ベイジアン最適化のエンドツーエンドテスト"""
        # データ準備をモック
        with patch('app.core.services.data.data_service.get_ohlcv_data') as mock_get_data:
            mock_get_data.return_value = sample_training_data
            
            # MLTrainingServiceの学習をモック（実際の学習は時間がかかるため）
            with patch.object(MLTrainingService, 'train_model') as mock_train:
                mock_train.return_value = {
                    "accuracy": 0.85,
                    "f1_score": 0.82,
                    "classification_report": {
                        "macro avg": {"f1-score": 0.82}
                    },
                    "optimization_result": {
                        "method": "bayesian",
                        "best_params": {"num_leaves": 50, "learning_rate": 0.1},
                        "best_score": 0.82,
                        "total_evaluations": 10,
                        "optimization_time": 120.5
                    }
                }

                # APIリクエスト
                request_data = {
                    "symbol": "BTC/USDT:USDT",
                    "timeframe": "1h",
                    "start_date": "2023-01-01",
                    "end_date": "2023-12-31",
                    "save_model": False,
                    "train_test_split": 0.8,
                    "random_state": 42,
                    "optimization_settings": {
                        "enabled": True,
                        "method": "bayesian",
                        "n_calls": 10,
                        "parameter_space": {
                            "num_leaves": {
                                "type": "integer",
                                "low": 10,
                                "high": 100
                            },
                            "learning_rate": {
                                "type": "real",
                                "low": 0.01,
                                "high": 0.3
                            }
                        }
                    }
                }

                response = client.post("/api/ml-training/train", json=request_data)
                
                assert response.status_code == 200
                result = response.json()
                
                # 最適化結果の確認
                assert "optimization_result" in result
                opt_result = result["optimization_result"]
                assert opt_result["method"] == "bayesian"
                assert "best_params" in opt_result
                assert "best_score" in opt_result
                assert opt_result["total_evaluations"] == 10

    def test_grid_search_optimization_e2e(self, client, sample_training_data):
        """グリッドサーチ最適化のエンドツーエンドテスト"""
        with patch('app.core.services.data.data_service.get_ohlcv_data') as mock_get_data:
            mock_get_data.return_value = sample_training_data
            
            with patch.object(MLTrainingService, 'train_model') as mock_train:
                mock_train.return_value = {
                    "accuracy": 0.83,
                    "f1_score": 0.80,
                    "classification_report": {
                        "macro avg": {"f1-score": 0.80}
                    },
                    "optimization_result": {
                        "method": "grid",
                        "best_params": {"num_leaves": 30, "learning_rate": 0.05},
                        "best_score": 0.80,
                        "total_evaluations": 9,
                        "optimization_time": 85.2
                    }
                }

                request_data = {
                    "symbol": "BTC/USDT:USDT",
                    "timeframe": "1h",
                    "start_date": "2023-01-01",
                    "end_date": "2023-12-31",
                    "save_model": False,
                    "train_test_split": 0.8,
                    "random_state": 42,
                    "optimization_settings": {
                        "enabled": True,
                        "method": "grid",
                        "n_calls": 20,
                        "parameter_space": {
                            "num_leaves": {
                                "type": "integer",
                                "low": 20,
                                "high": 40
                            },
                            "learning_rate": {
                                "type": "real",
                                "low": 0.05,
                                "high": 0.15
                            }
                        }
                    }
                }

                response = client.post("/api/ml-training/train", json=request_data)
                
                assert response.status_code == 200
                result = response.json()
                
                assert "optimization_result" in result
                opt_result = result["optimization_result"]
                assert opt_result["method"] == "grid"

    def test_random_search_optimization_e2e(self, client, sample_training_data):
        """ランダムサーチ最適化のエンドツーエンドテスト"""
        with patch('app.core.services.data.data_service.get_ohlcv_data') as mock_get_data:
            mock_get_data.return_value = sample_training_data
            
            with patch.object(MLTrainingService, 'train_model') as mock_train:
                mock_train.return_value = {
                    "accuracy": 0.81,
                    "f1_score": 0.78,
                    "classification_report": {
                        "macro avg": {"f1-score": 0.78}
                    },
                    "optimization_result": {
                        "method": "random",
                        "best_params": {"num_leaves": 75, "learning_rate": 0.12},
                        "best_score": 0.78,
                        "total_evaluations": 15,
                        "optimization_time": 95.8
                    }
                }

                request_data = {
                    "symbol": "BTC/USDT:USDT",
                    "timeframe": "1h",
                    "start_date": "2023-01-01",
                    "end_date": "2023-12-31",
                    "save_model": False,
                    "train_test_split": 0.8,
                    "random_state": 42,
                    "optimization_settings": {
                        "enabled": True,
                        "method": "random",
                        "n_calls": 15,
                        "parameter_space": {
                            "num_leaves": {
                                "type": "integer",
                                "low": 50,
                                "high": 100
                            },
                            "learning_rate": {
                                "type": "real",
                                "low": 0.1,
                                "high": 0.2
                            }
                        }
                    }
                }

                response = client.post("/api/ml-training/train", json=request_data)
                
                assert response.status_code == 200
                result = response.json()
                
                assert "optimization_result" in result
                opt_result = result["optimization_result"]
                assert opt_result["method"] == "random"

    def test_optimization_disabled_e2e(self, client, sample_training_data):
        """最適化無効時のエンドツーエンドテスト"""
        with patch('app.core.services.data.data_service.get_ohlcv_data') as mock_get_data:
            mock_get_data.return_value = sample_training_data
            
            with patch.object(MLTrainingService, 'train_model') as mock_train:
                mock_train.return_value = {
                    "accuracy": 0.75,
                    "f1_score": 0.72,
                    "classification_report": {
                        "macro avg": {"f1-score": 0.72}
                    }
                }

                request_data = {
                    "symbol": "BTC/USDT:USDT",
                    "timeframe": "1h",
                    "start_date": "2023-01-01",
                    "end_date": "2023-12-31",
                    "save_model": False,
                    "train_test_split": 0.8,
                    "random_state": 42,
                    "optimization_settings": {
                        "enabled": False,
                        "method": "bayesian",
                        "n_calls": 50,
                        "parameter_space": {}
                    }
                }

                response = client.post("/api/ml-training/train", json=request_data)
                
                assert response.status_code == 200
                result = response.json()
                
                # 最適化結果が含まれていないことを確認
                assert "optimization_result" not in result

    def test_invalid_optimization_method_e2e(self, client):
        """無効な最適化手法のエンドツーエンドテスト"""
        request_data = {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "save_model": False,
            "train_test_split": 0.8,
            "random_state": 42,
            "optimization_settings": {
                "enabled": True,
                "method": "invalid_method",
                "n_calls": 10,
                "parameter_space": {
                    "num_leaves": {
                        "type": "integer",
                        "low": 10,
                        "high": 100
                    }
                }
            }
        }

        response = client.post("/api/ml-training/train", json=request_data)
        
        # エラーレスポンスを確認
        assert response.status_code == 400 or response.status_code == 422

    def test_optimization_service_integration(self):
        """最適化サービスの統合テスト"""
        # OptimizationSettingsの作成
        settings = OptimizationSettings(
            enabled=True,
            method="bayesian",
            n_calls=5,
            parameter_space={
                "test_param": {
                    "type": "real",
                    "low": 0.1,
                    "high": 1.0
                }
            }
        )

        # MLTrainingServiceの初期化
        service = MLTrainingService()

        # パラメータ空間の準備をテスト
        parameter_space = service._prepare_parameter_space(settings.parameter_space)
        
        assert len(parameter_space) == 1
        assert "test_param" in parameter_space
        assert parameter_space["test_param"].type == "real"
        assert parameter_space["test_param"].low == 0.1
        assert parameter_space["test_param"].high == 1.0

        # 目的関数の作成をテスト（モックデータで）
        mock_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Close': [103, 104, 105],
            'Volume': [1000, 1100, 1200]
        })

        objective_function = service._create_objective_function(
            training_data=mock_data
        )

        # 目的関数が呼び出し可能であることを確認
        assert callable(objective_function)

        # OptimizerFactoryの統合テスト
        optimizer = OptimizerFactory.create_optimizer("bayesian")
        assert optimizer is not None
        assert optimizer.get_method_name() == "Bayesian"
