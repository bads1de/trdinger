"""
MLTrainingServiceとOptunaOptimizerの統合テスト
"""

import pytest
import pandas as pd
import numpy as np
from app.core.services.ml.ml_training_service import (
    MLTrainingService,
    OptimizationSettings,
)


def create_test_ohlcv_data(n_rows: int = 200) -> pd.DataFrame:
    """テスト用のOHLCVデータを作成"""
    np.random.seed(42)

    # 基本価格を生成
    base_price = 50000
    price_changes = np.random.normal(0, 0.02, n_rows)
    prices = [base_price]

    for change in price_changes:
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)

    prices = prices[1:]  # 最初の要素を削除

    # OHLCV データを生成
    data = []
    for i, close in enumerate(prices):
        high = close * (1 + abs(np.random.normal(0, 0.01)))
        low = close * (1 - abs(np.random.normal(0, 0.01)))
        open_price = close * (1 + np.random.normal(0, 0.005))
        volume = np.random.uniform(100, 1000)

        data.append(
            {
                "Open": open_price,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": volume,
            }
        )

    df = pd.DataFrame(data)

    # タイムスタンプを追加
    df.index = pd.date_range(start="2023-01-01", periods=len(df), freq="1H")

    return df


class TestMLOptunaIntegration:
    """MLTrainingServiceとOptunaOptimizerの統合テスト"""

    def test_optuna_ml_training_integration(self):
        """OptunaとMLTrainingServiceの統合テスト"""
        service = MLTrainingService()

        # テストデータを作成
        training_data = create_test_ohlcv_data(200)

        # Optuna最適化設定
        optimization_settings = OptimizationSettings(
            enabled=True,
            n_calls=10,  # テストなので少なめ
        )

        # 学習実行
        result = service.train_model(
            training_data=training_data,
            optimization_settings=optimization_settings,
            save_model=False,  # テストなので保存しない
        )

        # 結果検証
        assert result["success"] is True
        assert "optimization_result" in result
        assert result["optimization_result"]["method"] == "optuna"
        assert result["optimization_result"]["total_evaluations"] <= 10
        assert result["optimization_result"]["optimization_time"] > 0
        assert "best_params" in result["optimization_result"]
        assert "best_score" in result["optimization_result"]

        # 最適化されたパラメータが妥当な範囲内にあることを確認
        best_params = result["optimization_result"]["best_params"]
        assert 10 <= best_params.get("num_leaves", 50) <= 100
        assert 0.01 <= best_params.get("learning_rate", 0.1) <= 0.3

    def test_optuna_with_custom_parameter_space(self):
        """カスタムパラメータ空間でのOptuna最適化テスト"""
        service = MLTrainingService()

        # テストデータを作成
        training_data = create_test_ohlcv_data(150)

        # カスタムパラメータ空間
        custom_parameter_space = {
            "num_leaves": {"type": "integer", "low": 20, "high": 80},
            "learning_rate": {"type": "real", "low": 0.05, "high": 0.2},
        }

        optimization_settings = OptimizationSettings(
            enabled=True,
            n_calls=8,
            parameter_space=custom_parameter_space,
        )

        # 学習実行
        result = service.train_model(
            training_data=training_data,
            optimization_settings=optimization_settings,
            save_model=False,
        )

        # 結果検証
        assert result["success"] is True
        best_params = result["optimization_result"]["best_params"]

        # カスタム範囲内にあることを確認
        assert 20 <= best_params["num_leaves"] <= 80
        assert 0.05 <= best_params["learning_rate"] <= 0.2

    def test_optuna_optimization_disabled(self):
        """最適化無効時のテスト"""
        service = MLTrainingService()

        # テストデータを作成
        training_data = create_test_ohlcv_data(100)

        # 最適化無効設定
        optimization_settings = OptimizationSettings(enabled=False)

        # 学習実行
        result = service.train_model(
            training_data=training_data,
            optimization_settings=optimization_settings,
            save_model=False,
        )

        # 結果検証
        assert result["success"] is True
        # 最適化が無効の場合、optimization_resultは含まれない
        assert "optimization_result" not in result

    def test_optuna_with_funding_rate_data(self):
        """ファンディングレートデータを含むOptuna最適化テスト"""
        service = MLTrainingService()

        # テストデータを作成
        training_data = create_test_ohlcv_data(120)

        # ファンディングレートデータを作成
        funding_rate_data = pd.DataFrame(
            {"funding_rate": np.random.normal(0.0001, 0.0005, len(training_data))},
            index=training_data.index,
        )

        optimization_settings = OptimizationSettings(
            enabled=True,
            n_calls=6,
        )

        # 学習実行
        result = service.train_model(
            training_data=training_data,
            funding_rate_data=funding_rate_data,
            optimization_settings=optimization_settings,
            save_model=False,
        )

        # 結果検証
        assert result["success"] is True
        assert "optimization_result" in result
        assert result["optimization_result"]["method"] == "optuna"

    def test_optuna_performance_metrics(self):
        """Optuna最適化の性能指標テスト"""
        service = MLTrainingService()

        # テストデータを作成
        training_data = create_test_ohlcv_data(180)

        optimization_settings = OptimizationSettings(
            enabled=True,
            n_calls=12,
        )

        # 学習実行
        result = service.train_model(
            training_data=training_data,
            optimization_settings=optimization_settings,
            save_model=False,
        )

        # 性能指標の検証
        assert result["success"] is True
        assert "f1_score" in result
        assert "accuracy" in result
        assert "precision" in result
        assert "recall" in result

        # 最適化結果の検証
        opt_result = result["optimization_result"]
        assert opt_result["best_score"] > 0  # F1スコアは正の値
        assert opt_result["optimization_time"] > 0
        assert opt_result["total_evaluations"] <= 12


if __name__ == "__main__":
    pytest.main([__file__])
