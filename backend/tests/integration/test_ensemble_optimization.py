"""
アンサンブル学習のハイパーパラメータ最適化統合テスト
"""

import pytest
import pandas as pd
import numpy as np
from app.services.ml.ml_training_service import (
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


class TestEnsembleOptimization:
    """アンサンブル学習のハイパーパラメータ最適化テスト"""

    def test_ensemble_bagging_optimization(self):
        """バギングアンサンブルの最適化テスト"""
        # アンサンブル設定
        ensemble_config = {
            "method": "bagging",
            "models": ["lightgbm", "xgboost"],
            "bagging_params": {"base_model_type": "mixed"},
        }

        service = MLTrainingService(
            trainer_type="ensemble", ensemble_config=ensemble_config
        )

        # テストデータを作成
        training_data = create_test_ohlcv_data(150)

        # 最適化設定（少ない試行回数でテスト）
        optimization_settings = OptimizationSettings(
            enabled=True,
            n_calls=8,
        )

        # 学習実行
        result = service.train_model(
            training_data=training_data,
            optimization_settings=optimization_settings,
            save_model=False,
        )

        # 結果検証
        assert result["success"] is True
        assert "optimization_result" in result
        assert result["optimization_result"]["method"] == "optuna"
        assert result["optimization_result"]["total_evaluations"] <= 8

        # アンサンブル固有の最適化パラメータが含まれることを確認
        best_params = result["optimization_result"]["best_params"]

        # LightGBMパラメータの確認
        lgb_params = [k for k in best_params.keys() if k.startswith("lgb_")]
        assert len(lgb_params) > 0, "LightGBMパラメータが最適化されていません"

        # XGBoostパラメータの確認
        xgb_params = [k for k in best_params.keys() if k.startswith("xgb_")]
        assert len(xgb_params) > 0, "XGBoostパラメータが最適化されていません"

        # バギング固有パラメータの確認
        bagging_params = [k for k in best_params.keys() if k.startswith("bagging_")]
        assert len(bagging_params) > 0, "バギングパラメータが最適化されていません"

    def test_ensemble_stacking_optimization(self):
        """スタッキングアンサンブルの最適化テスト"""
        # アンサンブル設定
        ensemble_config = {
            "method": "stacking",
            "models": ["lightgbm", "randomforest"],
            "stacking_params": {"meta_model": "lightgbm"},
        }

        service = MLTrainingService(
            trainer_type="ensemble", ensemble_config=ensemble_config
        )

        # テストデータを作成
        training_data = create_test_ohlcv_data(120)

        # 最適化設定
        optimization_settings = OptimizationSettings(
            enabled=True,
            n_calls=10,
        )

        # 学習実行
        result = service.train_model(
            training_data=training_data,
            optimization_settings=optimization_settings,
            save_model=False,
        )

        # 結果検証
        assert result["success"] is True
        assert "optimization_result" in result

        best_params = result["optimization_result"]["best_params"]

        # LightGBMパラメータの確認
        lgb_params = [k for k in best_params.keys() if k.startswith("lgb_")]
        assert len(lgb_params) > 0, "LightGBMパラメータが最適化されていません"

        # RandomForestパラメータの確認
        rf_params = [k for k in best_params.keys() if k.startswith("rf_")]
        assert len(rf_params) > 0, "RandomForestパラメータが最適化されていません"

        # スタッキング固有パラメータの確認
        stacking_params = [k for k in best_params.keys() if k.startswith("stacking_")]
        assert len(stacking_params) > 0, "スタッキングパラメータが最適化されていません"

    def test_ensemble_with_catboost_tabnet_optimization(self):
        """CatBoostとTabNetを含むアンサンブル最適化テスト"""
        # アンサンブル設定
        ensemble_config = {
            "method": "bagging",
            "models": ["lightgbm", "catboost", "tabnet"],
            "bagging_params": {"base_model_type": "mixed"},
        }

        service = MLTrainingService(
            trainer_type="ensemble", ensemble_config=ensemble_config
        )

        # テストデータを作成
        training_data = create_test_ohlcv_data(100)

        # 最適化設定
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

        # 結果検証
        assert result["success"] is True
        assert "optimization_result" in result

        best_params = result["optimization_result"]["best_params"]

        # LightGBMパラメータの確認
        lgb_params = [k for k in best_params.keys() if k.startswith("lgb_")]
        assert len(lgb_params) > 0, "LightGBMパラメータが最適化されていません"

        # CatBoostパラメータの確認
        cat_params = [k for k in best_params.keys() if k.startswith("cat_")]
        assert len(cat_params) > 0, "CatBoostパラメータが最適化されていません"

        # TabNetパラメータの確認
        tab_params = [k for k in best_params.keys() if k.startswith("tab_")]
        assert len(tab_params) > 0, "TabNetパラメータが最適化されていません"

    def test_ensemble_custom_parameter_space(self):
        """カスタムパラメータ空間でのアンサンブル最適化テスト"""
        # アンサンブル設定
        ensemble_config = {
            "method": "bagging",
            "models": ["lightgbm", "xgboost"],
            "bagging_params": {"base_model_type": "mixed"},
        }

        service = MLTrainingService(
            trainer_type="ensemble", ensemble_config=ensemble_config
        )

        # テストデータを作成
        training_data = create_test_ohlcv_data(100)

        # カスタムパラメータ空間
        custom_parameter_space = {
            "lgb_num_leaves": {"type": "integer", "low": 20, "high": 60},
            "lgb_learning_rate": {"type": "real", "low": 0.05, "high": 0.2},
            "xgb_max_depth": {"type": "integer", "low": 4, "high": 12},
            "xgb_learning_rate": {"type": "real", "low": 0.05, "high": 0.25},
            "bagging_n_estimators": {"type": "integer", "low": 2, "high": 5},
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
        assert 20 <= best_params["lgb_num_leaves"] <= 60
        assert 0.05 <= best_params["lgb_learning_rate"] <= 0.2
        assert 4 <= best_params["xgb_max_depth"] <= 12
        assert 0.05 <= best_params["xgb_learning_rate"] <= 0.25
        assert 2 <= best_params["bagging_n_estimators"] <= 5

    def test_ensemble_optimization_performance_metrics(self):
        """アンサンブル最適化の性能指標テスト"""
        # アンサンブル設定
        ensemble_config = {
            "method": "bagging",
            "models": ["lightgbm", "xgboost"],
            "bagging_params": {"base_model_type": "mixed"},
        }

        service = MLTrainingService(
            trainer_type="ensemble", ensemble_config=ensemble_config
        )

        # テストデータを作成
        training_data = create_test_ohlcv_data(150)

        optimization_settings = OptimizationSettings(
            enabled=True,
            n_calls=10,
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

        # アンサンブル固有の情報
        assert "ensemble_method" in result
        assert result["ensemble_method"] == "bagging"

        # 最適化結果の検証
        opt_result = result["optimization_result"]
        assert opt_result["best_score"] > 0  # F1スコアは正の値
        assert opt_result["optimization_time"] > 0
        assert opt_result["total_evaluations"] <= 10


if __name__ == "__main__":
    pytest.main([__file__])
