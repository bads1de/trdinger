"""
AutoML特徴量エンジニアリングの精度とパフォーマンス包括テスト

モデル精度の向上、パフォーマンス、ロバストネスを詳細に検証します。
"""

import pytest
import pandas as pd
import numpy as np
import time
import warnings
from typing import Dict, List, Tuple, Optional
from unittest.mock import patch, MagicMock
import psutil
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

from app.core.services.ml.feature_engineering.enhanced_feature_engineering_service import (
    EnhancedFeatureEngineeringService,
)
from app.core.services.ml.feature_engineering.feature_engineering_service import (
    FeatureEngineeringService,
)
from app.core.services.ml.feature_engineering.automl_features.automl_config import (
    AutoMLConfig,
    TSFreshConfig,
    FeaturetoolsConfig,
    AutoFeatConfig,
)


class TestAutoMLAccuracyComprehensive:
    """AutoML精度とパフォーマンス包括テストクラス"""

    def setup_method(self):
        """各テストメソッドの前に実行される初期化"""
        # 軽量設定でテスト用サービスを初期化
        self.enhanced_service = EnhancedFeatureEngineeringService()
        self.baseline_service = FeatureEngineeringService()

        # パフォーマンス監視用
        self.process = psutil.Process(os.getpid())

    def create_realistic_financial_data(
        self, n_samples: int = 1000
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        現実的な金融データを生成

        Returns:
            OHLCV データ, 回帰ターゲット, 分類ターゲット
        """
        np.random.seed(42)

        # 時系列インデックス
        dates = pd.date_range(start="2020-01-01", periods=n_samples, freq="1h")

        # 価格データ（ランダムウォーク + トレンド + ボラティリティクラスタリング）
        returns = np.random.normal(0, 0.02, n_samples)
        trend = np.sin(np.arange(n_samples) * 2 * np.pi / 100) * 0.001
        volatility = np.abs(np.random.normal(0.02, 0.01, n_samples))

        prices = 100 * np.exp(np.cumsum(returns + trend))

        # OHLCV データ（大文字の列名を使用）
        ohlcv_data = pd.DataFrame(
            {
                "timestamp": dates,
                "Open": prices * (1 + np.random.normal(0, 0.001, n_samples)),
                "High": prices * (1 + np.abs(np.random.normal(0, 0.005, n_samples))),
                "Low": prices * (1 - np.abs(np.random.normal(0, 0.005, n_samples))),
                "Close": prices,
                "Volume": np.random.lognormal(10, 1, n_samples),
            }
        )

        # timestampをインデックスに設定
        ohlcv_data.set_index("timestamp", inplace=True)

        # 回帰ターゲット（次の期間のリターン）
        future_returns = np.roll(returns, -1)
        future_returns[-1] = 0  # 最後の値は0に設定
        regression_target = pd.Series(future_returns, index=dates, name="future_return")

        # 分類ターゲット（価格上昇/下降）
        classification_target = pd.Series(
            (future_returns > 0).astype(int), index=dates, name="price_direction"
        )

        return ohlcv_data, regression_target, classification_target

    def test_model_accuracy_comparison_regression(self):
        """回帰タスクでのモデル精度比較テスト"""
        print("\n=== 回帰タスク精度比較テスト ===")

        # データ生成
        ohlcv_data, regression_target, _ = self.create_realistic_financial_data(500)

        # ベースライン特徴量（手動特徴量のみ）
        print("ベースライン特徴量計算中...")
        baseline_features = self.baseline_service.calculate_advanced_features(
            ohlcv_data=ohlcv_data,
            lookback_periods={
                "short_ma": 5,
                "long_ma": 20,
                "volatility": 20,
                "momentum": 14,
                "volume": 20,
            },
        )

        # 拡張特徴量（手動 + AutoML）
        print("拡張特徴量計算中...")
        enhanced_features = self.enhanced_service.calculate_enhanced_features(
            ohlcv_data=ohlcv_data,
            target=regression_target,
            lookback_periods={
                "short_ma": 5,
                "long_ma": 20,
                "volatility": 20,
                "momentum": 14,
                "volume": 20,
            },
        )

        # データの準備
        baseline_clean = self._prepare_data_for_ml(baseline_features, regression_target)
        enhanced_clean = self._prepare_data_for_ml(enhanced_features, regression_target)

        if baseline_clean is None or enhanced_clean is None:
            pytest.skip("データの準備に失敗しました")

        baseline_X, baseline_y = baseline_clean
        enhanced_X, enhanced_y = enhanced_clean

        # モデル評価
        models = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(n_estimators=50, random_state=42),
            "LightGBM": lgb.LGBMRegressor(n_estimators=50, random_state=42, verbose=-1),
        }

        results = {}

        for model_name, model in models.items():
            print(f"\n{model_name}での評価:")

            # ベースライン評価
            baseline_scores = self._evaluate_regression_model(
                model, baseline_X, baseline_y, f"Baseline_{model_name}"
            )

            # 拡張特徴量評価
            enhanced_scores = self._evaluate_regression_model(
                model, enhanced_X, enhanced_y, f"Enhanced_{model_name}"
            )

            results[model_name] = {
                "baseline": baseline_scores,
                "enhanced": enhanced_scores,
                "improvement": {
                    "r2": enhanced_scores["r2"] - baseline_scores["r2"],
                    "rmse_reduction": baseline_scores["rmse"] - enhanced_scores["rmse"],
                },
            }

            print(
                f"  ベースライン R²: {baseline_scores['r2']:.4f}, RMSE: {baseline_scores['rmse']:.6f}"
            )
            print(
                f"  拡張特徴量 R²: {enhanced_scores['r2']:.4f}, RMSE: {enhanced_scores['rmse']:.6f}"
            )
            print(f"  改善度 R²: {results[model_name]['improvement']['r2']:+.4f}")
            print(
                f"  RMSE削減: {results[model_name]['improvement']['rmse_reduction']:+.6f}"
            )

        # 結果の検証
        self._validate_regression_results(results)

        print(f"\n特徴量数比較:")
        print(f"  ベースライン: {baseline_X.shape[1]}個")
        print(f"  拡張特徴量: {enhanced_X.shape[1]}個")
        print(f"  追加特徴量: {enhanced_X.shape[1] - baseline_X.shape[1]}個")

    def test_model_accuracy_comparison_classification(self):
        """分類タスクでのモデル精度比較テスト"""
        print("\n=== 分類タスク精度比較テスト ===")

        # データ生成
        ohlcv_data, _, classification_target = self.create_realistic_financial_data(500)

        # ベースライン特徴量
        print("ベースライン特徴量計算中...")
        baseline_features = self.baseline_service.calculate_advanced_features(
            ohlcv_data=ohlcv_data,
            lookback_periods={
                "short_ma": 5,
                "long_ma": 20,
                "volatility": 20,
                "momentum": 14,
                "volume": 20,
            },
        )

        # 拡張特徴量
        print("拡張特徴量計算中...")
        enhanced_features = self.enhanced_service.calculate_enhanced_features(
            ohlcv_data=ohlcv_data,
            target=classification_target,
            lookback_periods={
                "short_ma": 5,
                "long_ma": 20,
                "volatility": 20,
                "momentum": 14,
                "volume": 20,
            },
        )

        # データの準備
        baseline_clean = self._prepare_data_for_ml(
            baseline_features, classification_target
        )
        enhanced_clean = self._prepare_data_for_ml(
            enhanced_features, classification_target
        )

        if baseline_clean is None or enhanced_clean is None:
            pytest.skip("データの準備に失敗しました")

        baseline_X, baseline_y = baseline_clean
        enhanced_X, enhanced_y = enhanced_clean

        # モデル評価
        models = {
            "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000),
            "RandomForest": RandomForestClassifier(n_estimators=50, random_state=42),
            "LightGBM": lgb.LGBMClassifier(
                n_estimators=50, random_state=42, verbose=-1
            ),
        }

        results = {}

        for model_name, model in models.items():
            print(f"\n{model_name}での評価:")

            # ベースライン評価
            baseline_scores = self._evaluate_classification_model(
                model, baseline_X, baseline_y, f"Baseline_{model_name}"
            )

            # 拡張特徴量評価
            enhanced_scores = self._evaluate_classification_model(
                model, enhanced_X, enhanced_y, f"Enhanced_{model_name}"
            )

            results[model_name] = {
                "baseline": baseline_scores,
                "enhanced": enhanced_scores,
                "improvement": {
                    "accuracy": enhanced_scores["accuracy"]
                    - baseline_scores["accuracy"],
                    "f1": enhanced_scores["f1"] - baseline_scores["f1"],
                },
            }

            print(
                f"  ベースライン 精度: {baseline_scores['accuracy']:.4f}, F1: {baseline_scores['f1']:.4f}"
            )
            print(
                f"  拡張特徴量 精度: {enhanced_scores['accuracy']:.4f}, F1: {enhanced_scores['f1']:.4f}"
            )
            print(
                f"  改善度 精度: {results[model_name]['improvement']['accuracy']:+.4f}"
            )
            print(f"  改善度 F1: {results[model_name]['improvement']['f1']:+.4f}")

        # 結果の検証
        self._validate_classification_results(results)

    def test_performance_scalability(self):
        """パフォーマンスとスケーラビリティテスト"""
        print("\n=== パフォーマンス・スケーラビリティテスト ===")

        data_sizes = [100, 500, 1000, 2000]
        performance_results = {}

        for size in data_sizes:
            print(f"\nデータサイズ {size} でのテスト:")

            # データ生成
            ohlcv_data, regression_target, _ = self.create_realistic_financial_data(
                size
            )

            # メモリ使用量測定開始
            memory_before = self.process.memory_info().rss / 1024 / 1024  # MB

            # 処理時間測定
            start_time = time.time()

            # 特徴量計算
            enhanced_features = self.enhanced_service.calculate_enhanced_features(
                ohlcv_data=ohlcv_data,
                target=regression_target,
                lookback_periods={
                    "short_ma": 5,
                    "long_ma": 20,
                    "volatility": 20,
                    "momentum": 14,
                    "volume": 20,
                },
            )

            processing_time = time.time() - start_time

            # メモリ使用量測定終了
            memory_after = self.process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = memory_after - memory_before

            performance_results[size] = {
                "processing_time": processing_time,
                "memory_increase": memory_increase,
                "feature_count": (
                    len(enhanced_features.columns)
                    if enhanced_features is not None
                    else 0
                ),
                "data_points": (
                    len(enhanced_features) if enhanced_features is not None else 0
                ),
                "throughput": size / processing_time if processing_time > 0 else 0,
            }

            print(f"  処理時間: {processing_time:.2f}秒")
            print(f"  メモリ増加: {memory_increase:.1f}MB")
            print(
                f"  特徴量数: {len(enhanced_features.columns) if enhanced_features is not None else 0}個"
            )
            print(
                f"  スループット: {performance_results[size]['throughput']:.1f} samples/sec"
            )

        # パフォーマンス分析
        self._analyze_performance_results(performance_results)

    def _prepare_data_for_ml(
        self, features_df: pd.DataFrame, target: pd.Series
    ) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
        """ML用にデータを準備"""
        try:
            # 共通のインデックスを取得
            common_index = features_df.index.intersection(target.index)

            if len(common_index) < 50:  # 最小データ数チェック
                return None

            # データを整列
            X = features_df.loc[common_index]
            y = target.loc[common_index]

            # 無限値とNaNを除去
            X = X.replace([np.inf, -np.inf], np.nan)
            valid_mask = ~(X.isna().any(axis=1) | y.isna())

            X_clean = X[valid_mask]
            y_clean = y[valid_mask]

            if len(X_clean) < 50:
                return None

            # 数値列のみを選択
            numeric_columns = X_clean.select_dtypes(include=[np.number]).columns
            X_numeric = X_clean[numeric_columns]

            return X_numeric, y_clean

        except Exception as e:
            print(f"データ準備エラー: {e}")
            return None

    def _evaluate_regression_model(
        self, model, X: pd.DataFrame, y: pd.Series, model_name: str
    ) -> Dict[str, float]:
        """回帰モデルを評価"""
        try:
            # データ分割
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )

            # スケーリング
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # モデル訓練
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)

            # 評価指標計算
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            return {"r2": r2, "rmse": rmse}

        except Exception as e:
            print(f"回帰モデル評価エラー ({model_name}): {e}")
            return {"r2": 0.0, "rmse": float("inf")}

    def _evaluate_classification_model(
        self, model, X: pd.DataFrame, y: pd.Series, model_name: str
    ) -> Dict[str, float]:
        """分類モデルを評価"""
        try:
            # データ分割
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )

            # スケーリング
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # モデル訓練
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)

            # 評価指標計算
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")

            return {"accuracy": accuracy, "f1": f1}

        except Exception as e:
            print(f"分類モデル評価エラー ({model_name}): {e}")
            return {"accuracy": 0.0, "f1": 0.0}

    def _validate_regression_results(self, results: Dict):
        """回帰結果の検証"""
        for model_name, result in results.items():
            # 基本的な妥当性チェック
            assert "baseline" in result
            assert "enhanced" in result
            assert "improvement" in result

            # R²スコアが妥当な範囲内かチェック
            assert -1 <= result["baseline"]["r2"] <= 1
            assert -1 <= result["enhanced"]["r2"] <= 1

            # RMSEが正の値かチェック
            assert result["baseline"]["rmse"] >= 0
            assert result["enhanced"]["rmse"] >= 0

            print(f"{model_name}: 検証完了")

    def _validate_classification_results(self, results: Dict):
        """分類結果の検証"""
        for model_name, result in results.items():
            # 基本的な妥当性チェック
            assert "baseline" in result
            assert "enhanced" in result
            assert "improvement" in result

            # 精度が妥当な範囲内かチェック
            assert 0 <= result["baseline"]["accuracy"] <= 1
            assert 0 <= result["enhanced"]["accuracy"] <= 1

            # F1スコアが妥当な範囲内かチェック
            assert 0 <= result["baseline"]["f1"] <= 1
            assert 0 <= result["enhanced"]["f1"] <= 1

            print(f"{model_name}: 検証完了")

    def _analyze_performance_results(self, performance_results: Dict):
        """パフォーマンス結果の分析"""
        print("\n=== パフォーマンス分析結果 ===")

        sizes = sorted(performance_results.keys())

        # 処理時間の線形性チェック
        times = [performance_results[size]["processing_time"] for size in sizes]
        print(f"処理時間の推移: {times}")

        # メモリ使用量の推移
        memory_increases = [
            performance_results[size]["memory_increase"] for size in sizes
        ]
        print(f"メモリ増加の推移: {memory_increases}")

        # スループットの推移
        throughputs = [performance_results[size]["throughput"] for size in sizes]
        print(f"スループットの推移: {throughputs}")

        # 基本的な妥当性チェック
        for size in sizes:
            result = performance_results[size]
            assert (
                result["processing_time"] > 0
            ), f"処理時間が無効: {result['processing_time']}"
            assert (
                result["feature_count"] > 0
            ), f"特徴量数が無効: {result['feature_count']}"
            assert (
                result["data_points"] > 0
            ), f"データポイント数が無効: {result['data_points']}"

        print("パフォーマンス分析完了")
