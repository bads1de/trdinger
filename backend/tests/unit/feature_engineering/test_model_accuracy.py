"""
AutoML特徴量を使用したモデル精度測定テスト

3つのAutoMLライブラリ（TSFresh、Featuretools、AutoFeat）で生成された
特徴量を使用して実際のモデルを訓練し、精度を測定します。
"""

import pytest
import pandas as pd
import numpy as np
import time
import warnings
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression

# LightGBMのインポート
try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBMが利用できません。pip install lightgbm でインストールしてください。")

from app.services.ml.feature_engineering.enhanced_feature_engineering_service import (
    EnhancedFeatureEngineeringService,
)
from app.services.ml.feature_engineering.automl_features.automl_config import (
    AutoMLConfig,
    TSFreshConfig,
    FeaturetoolsConfig,
    AutoFeatConfig,
)


class TestModelAccuracy:
    """AutoML特徴量を使用したモデル精度測定テスト"""

    def setup_method(self):
        """テストセットアップ"""
        # 全AutoMLライブラリを有効にした設定
        tsfresh_config = TSFreshConfig(
            enabled=True,
            feature_selection=False,  # より多くの特徴量を生成
            feature_count_limit=200,
            parallel_jobs=1,
        )

        featuretools_config = FeaturetoolsConfig(
            enabled=True,
            max_depth=2,
            max_features=50,
        )

        autofeat_config = AutoFeatConfig(
            enabled=True,
            max_features=20,
            feateng_steps=2,
            max_gb=1.0,
        )

        self.automl_config = AutoMLConfig(
            tsfresh_config=tsfresh_config,
            featuretools_config=featuretools_config,
            autofeat_config=autofeat_config,
        )

        self.service = EnhancedFeatureEngineeringService(self.automl_config)

    def _generate_realistic_financial_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """リアルな金融時系列データを生成"""
        np.random.seed(42)

        # 基本的な価格データ（ランダムウォーク + トレンド + ボラティリティクラスタリング）
        returns = np.random.normal(0.0005, 0.02, n_samples)  # 日次リターン

        # ボラティリティクラスタリング効果
        volatility = np.ones(n_samples) * 0.02
        for i in range(1, n_samples):
            volatility[i] = (
                0.9 * volatility[i - 1]
                + 0.1 * abs(returns[i - 1])
                + np.random.normal(0, 0.001)
            )
            volatility[i] = max(volatility[i], 0.001)  # 最小値を設定して負の値を防ぐ
            returns[i] = np.random.normal(0.0005, volatility[i])

        # 価格系列を生成
        prices = 100 * np.exp(np.cumsum(returns))

        # OHLCV データを生成
        data = []
        for i in range(n_samples):
            base_price = prices[i]
            daily_volatility = volatility[i] * base_price

            # 日中の価格変動を模擬
            high = base_price + np.random.exponential(daily_volatility * 0.5)
            low = base_price - np.random.exponential(daily_volatility * 0.5)

            if i == 0:
                open_price = base_price
            else:
                open_price = prices[i - 1] * (1 + np.random.normal(0, 0.001))

            close_price = base_price
            volume = np.random.lognormal(15, 1)  # 出来高

            data.append(
                {
                    "Open": open_price,
                    "High": max(open_price, high, close_price),
                    "Low": min(open_price, low, close_price),
                    "Close": close_price,
                    "Volume": volume,
                }
            )

        df = pd.DataFrame(data)

        # タイムスタンプを追加
        df.index = pd.date_range(start="2020-01-01", periods=n_samples, freq="D")

        return df

    def _create_prediction_targets(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """複数の予測ターゲットを作成"""
        targets = {}

        # 1. 次日の終値予測（回帰）
        targets["next_close"] = df["Close"].shift(-1).dropna()

        # 2. 次日のリターン予測（回帰）
        targets["next_return"] = df["Close"].pct_change().shift(-1).dropna()

        # 3. 5日後のリターン予測（回帰）
        targets["return_5d"] = df["Close"].pct_change(5).shift(-5).dropna()

        # 4. 価格方向予測（分類 - 上昇/下降）
        next_return = df["Close"].pct_change().shift(-1)
        targets["direction"] = (next_return > 0).astype(int).dropna()

        return targets

    def _evaluate_model_performance(
        self, X: pd.DataFrame, y: pd.Series, task_type: str = "regression"
    ) -> Dict[str, Any]:
        """複数のモデルで性能を評価"""

        # データを訓練・テストに分割（時系列を考慮）
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # 特徴量を標準化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        results = {}

        if task_type == "regression":
            models = {
                "LinearRegression": LinearRegression(),
                "Ridge": Ridge(alpha=1.0),
                "RandomForest": RandomForestRegressor(
                    n_estimators=100, random_state=42
                ),
                "GradientBoosting": GradientBoostingRegressor(
                    n_estimators=100, random_state=42
                ),
            }

            # LightGBMネイティブAPIを使用（既存実装に合わせる）
            if LIGHTGBM_AVAILABLE:
                models["LightGBM_Native"] = "lightgbm_native"  # 特別なマーカー

            for name, model in models.items():
                start_time = time.time()

                # LightGBMネイティブAPIの特別処理
                if name == "LightGBM_Native" and model == "lightgbm_native":
                    # LightGBMデータセットを作成
                    train_data = lgb.Dataset(X_train_scaled, label=y_train)
                    valid_data = lgb.Dataset(
                        X_test_scaled, label=y_test, reference=train_data
                    )

                    # LightGBMパラメータ（既存実装に合わせる）
                    params = {
                        "objective": "regression",
                        "metric": "rmse",
                        "boosting_type": "gbdt",
                        "num_leaves": 31,
                        "learning_rate": 0.1,
                        "feature_fraction": 0.9,
                        "bagging_fraction": 0.8,
                        "bagging_freq": 5,
                        "verbose": -1,
                        "random_state": 42,
                    }

                    # モデル学習
                    lgb_model = lgb.train(
                        params,
                        train_data,
                        valid_sets=[train_data, valid_data],
                        valid_names=["train", "valid"],
                        num_boost_round=100,
                        callbacks=[
                            lgb.early_stopping(stopping_rounds=20),
                            lgb.log_evaluation(0),  # ログを抑制
                        ],
                    )

                    # 予測
                    y_pred = lgb_model.predict(
                        X_test_scaled, num_iteration=lgb_model.best_iteration
                    )
                else:
                    # 通常のscikit-learnモデル
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)

                # 評価指標計算
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                # クロスバリデーション（時系列分割）
                tscv = TimeSeriesSplit(n_splits=5)
                if name == "LightGBM_Native":
                    # LightGBMネイティブAPIの場合は手動でクロスバリデーション
                    cv_scores = []
                    for train_idx, val_idx in tscv.split(X_train_scaled):
                        X_cv_train, X_cv_val = (
                            X_train_scaled[train_idx],
                            X_train_scaled[val_idx],
                        )
                        y_cv_train, y_cv_val = (
                            y_train.iloc[train_idx],
                            y_train.iloc[val_idx],
                        )

                        cv_train_data = lgb.Dataset(X_cv_train, label=y_cv_train)
                        cv_val_data = lgb.Dataset(
                            X_cv_val, label=y_cv_val, reference=cv_train_data
                        )

                        cv_model = lgb.train(
                            params,
                            cv_train_data,
                            valid_sets=[cv_val_data],
                            num_boost_round=50,  # CVでは短く
                            callbacks=[lgb.log_evaluation(0)],
                        )

                        cv_pred = cv_model.predict(
                            X_cv_val, num_iteration=cv_model.best_iteration
                        )
                        cv_score = r2_score(y_cv_val, cv_pred)
                        cv_scores.append(cv_score)

                    cv_scores = np.array(cv_scores)
                else:
                    cv_scores = cross_val_score(
                        model, X_train_scaled, y_train, cv=tscv, scoring="r2"
                    )

                training_time = time.time() - start_time

                results[name] = {
                    "mse": mse,
                    "mae": mae,
                    "r2": r2,
                    "cv_r2_mean": cv_scores.mean(),
                    "cv_r2_std": cv_scores.std(),
                    "training_time": training_time,
                }

        else:  # classification
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import (
                accuracy_score,
                precision_score,
                recall_score,
                f1_score,
            )

            models = {
                "LogisticRegression": LogisticRegression(random_state=42),
                "RandomForest": RandomForestClassifier(
                    n_estimators=100, random_state=42
                ),
            }

            # LightGBMネイティブAPIを使用（既存実装に合わせる）
            if LIGHTGBM_AVAILABLE:
                models["LightGBM_Native"] = "lightgbm_native"  # 特別なマーカー

            for name, model in models.items():
                start_time = time.time()

                # LightGBMネイティブAPIの特別処理
                if name == "LightGBM_Native" and model == "lightgbm_native":
                    # LightGBMデータセットを作成
                    train_data = lgb.Dataset(X_train_scaled, label=y_train)
                    valid_data = lgb.Dataset(
                        X_test_scaled, label=y_test, reference=train_data
                    )

                    # LightGBMパラメータ（既存実装に合わせる）
                    params = {
                        "objective": "binary",
                        "metric": "binary_logloss",
                        "boosting_type": "gbdt",
                        "num_leaves": 31,
                        "learning_rate": 0.1,
                        "feature_fraction": 0.9,
                        "bagging_fraction": 0.8,
                        "bagging_freq": 5,
                        "verbose": -1,
                        "random_state": 42,
                    }

                    # モデル学習
                    lgb_model = lgb.train(
                        params,
                        train_data,
                        valid_sets=[train_data, valid_data],
                        valid_names=["train", "valid"],
                        num_boost_round=100,
                        callbacks=[
                            lgb.early_stopping(stopping_rounds=20),
                            lgb.log_evaluation(0),  # ログを抑制
                        ],
                    )

                    # 予測
                    y_pred_proba = lgb_model.predict(
                        X_test_scaled, num_iteration=lgb_model.best_iteration
                    )
                    y_pred = (y_pred_proba > 0.5).astype(int)
                else:
                    # 通常のscikit-learnモデル
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)

                # 評価指標計算
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average="weighted")
                recall = recall_score(y_test, y_pred, average="weighted")
                f1 = f1_score(y_test, y_pred, average="weighted")

                # クロスバリデーション
                tscv = TimeSeriesSplit(n_splits=5)
                if name == "LightGBM_Native":
                    # LightGBMネイティブAPIの場合は手動でクロスバリデーション
                    cv_scores = []
                    for train_idx, val_idx in tscv.split(X_train_scaled):
                        X_cv_train, X_cv_val = (
                            X_train_scaled[train_idx],
                            X_train_scaled[val_idx],
                        )
                        y_cv_train, y_cv_val = (
                            y_train.iloc[train_idx],
                            y_train.iloc[val_idx],
                        )

                        cv_train_data = lgb.Dataset(X_cv_train, label=y_cv_train)
                        cv_val_data = lgb.Dataset(
                            X_cv_val, label=y_cv_val, reference=cv_train_data
                        )

                        cv_model = lgb.train(
                            params,
                            cv_train_data,
                            valid_sets=[cv_val_data],
                            num_boost_round=50,  # CVでは短く
                            callbacks=[lgb.log_evaluation(0)],
                        )

                        cv_pred_proba = cv_model.predict(
                            X_cv_val, num_iteration=cv_model.best_iteration
                        )
                        cv_pred = (cv_pred_proba > 0.5).astype(int)
                        cv_score = accuracy_score(y_cv_val, cv_pred)
                        cv_scores.append(cv_score)

                    cv_scores = np.array(cv_scores)
                else:
                    cv_scores = cross_val_score(
                        model, X_train_scaled, y_train, cv=tscv, scoring="accuracy"
                    )

                training_time = time.time() - start_time

                results[name] = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "cv_accuracy_mean": cv_scores.mean(),
                    "cv_accuracy_std": cv_scores.std(),
                    "training_time": training_time,
                }

        return results

    def test_comprehensive_model_accuracy(self):
        """包括的なモデル精度測定テスト"""
        print("\n" + "=" * 80)
        print("🚀 AutoML特徴量を使用したモデル精度測定テスト")
        print("=" * 80)

        # リアルな金融データを生成
        print("📊 リアルな金融時系列データを生成中...")
        financial_data = self._generate_realistic_financial_data(1000)
        print(f"生成されたデータ: {financial_data.shape}")

        # 予測ターゲットを作成
        targets = self._create_prediction_targets(financial_data)
        print(f"予測ターゲット: {list(targets.keys())}")

        # AutoML特徴量を生成
        print("\n🔧 AutoML特徴量を生成中...")
        start_time = time.time()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # 全AutoMLライブラリを使用して特徴量生成
            enhanced_features = self.service.calculate_enhanced_features(
                ohlcv_data=financial_data,
                target=targets[
                    "next_return"
                ],  # リターン予測をメインターゲットとして使用
                lookback_periods={"short": 10, "medium": 20, "long": 50},
            )

        feature_generation_time = time.time() - start_time

        print(f"✅ 特徴量生成完了:")
        print(f"   - 総特徴量数: {len(enhanced_features.columns)}個")
        print(f"   - データポイント数: {len(enhanced_features)}行")
        print(f"   - 生成時間: {feature_generation_time:.2f}秒")

        # 特徴量の内訳を表示
        feature_types = {
            "Manual": len(
                [
                    col
                    for col in enhanced_features.columns
                    if not any(prefix in col for prefix in ["TSF_", "FT_", "AF_"])
                ]
            ),
            "TSFresh": len([col for col in enhanced_features.columns if "TSF_" in col]),
            "Featuretools": len(
                [col for col in enhanced_features.columns if "FT_" in col]
            ),
            "AutoFeat": len([col for col in enhanced_features.columns if "AF_" in col]),
        }

        print(f"   - 特徴量内訳: {feature_types}")

        # 各予測タスクでモデル性能を評価
        all_results = {}

        for target_name, target_series in targets.items():
            print(f"\n📈 {target_name} 予測タスクの評価中...")

            # インデックスを統一（タイムゾーン問題を回避）
            X_reset = enhanced_features.reset_index(drop=True)
            y_reset = target_series.reset_index(drop=True)

            # データの長さを合わせる
            min_length = min(len(X_reset), len(y_reset))
            X = X_reset.iloc[:min_length]
            y = y_reset.iloc[:min_length]

            # 無限値・NaN値を除去
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            mask = np.isfinite(X[numeric_cols]).all(axis=1) & np.isfinite(y)
            X_clean = X[mask]
            y_clean = y[mask]

            print(f"   - クリーンなデータ: {len(X_clean)}行")

            if len(X_clean) < 100:
                print(f"   ⚠️  データが不足しているためスキップ: {len(X_clean)}行")
                continue

            # タスクタイプを決定
            task_type = "classification" if target_name == "direction" else "regression"

            # モデル性能を評価
            results = self._evaluate_model_performance(X_clean, y_clean, task_type)
            all_results[target_name] = results

            # 結果を表示
            print(f"   📊 {target_name} 予測結果:")
            for model_name, metrics in results.items():
                if task_type == "regression":
                    print(f"      {model_name}:")
                    print(f"        - R² Score: {metrics['r2']:.4f}")
                    print(f"        - MSE: {metrics['mse']:.6f}")
                    print(f"        - MAE: {metrics['mae']:.6f}")
                    print(
                        f"        - CV R² (平均±標準偏差): {metrics['cv_r2_mean']:.4f}±{metrics['cv_r2_std']:.4f}"
                    )
                    print(f"        - 訓練時間: {metrics['training_time']:.2f}秒")
                else:
                    print(f"      {model_name}:")
                    print(f"        - Accuracy: {metrics['accuracy']:.4f}")
                    print(f"        - F1 Score: {metrics['f1']:.4f}")
                    print(
                        f"        - CV Accuracy (平均±標準偏差): {metrics['cv_accuracy_mean']:.4f}±{metrics['cv_accuracy_std']:.4f}"
                    )
                    print(f"        - 訓練時間: {metrics['training_time']:.2f}秒")

        # 総合結果のサマリー
        print(f"\n" + "=" * 80)
        print("📋 総合結果サマリー")
        print("=" * 80)

        print(f"🔧 特徴量生成:")
        print(f"   - 総特徴量数: {len(enhanced_features.columns)}個")
        print(f"   - 生成時間: {feature_generation_time:.2f}秒")
        print(f"   - 特徴量内訳: {feature_types}")

        print(f"\n🎯 最高性能モデル:")
        for target_name, results in all_results.items():
            if target_name in ["next_return", "return_5d"]:  # 回帰タスク
                best_model = max(results.items(), key=lambda x: x[1]["r2"])
                print(
                    f"   {target_name}: {best_model[0]} (R² = {best_model[1]['r2']:.4f})"
                )
            elif target_name == "direction":  # 分類タスク
                best_model = max(results.items(), key=lambda x: x[1]["accuracy"])
                print(
                    f"   {target_name}: {best_model[0]} (Accuracy = {best_model[1]['accuracy']:.4f})"
                )

        # テスト成功の確認
        assert (
            len(enhanced_features.columns) > 100
        ), f"特徴量数が不足: {len(enhanced_features.columns)}個"
        assert len(all_results) > 0, "モデル評価結果が空です"

        print(f"\n✅ モデル精度測定テスト完了!")
        print("=" * 80)

        return all_results
