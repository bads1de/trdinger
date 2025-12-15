import pytest
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score


class TestModelCapacity:
    """モデルの学習能力（明確なパターンを学習できるか）を検証するテスト"""

    @pytest.fixture
    def synthetic_pattern_data(self):
        """明確なパターンを持つ合成データを生成"""
        # 1000サンプルのデータ
        n_samples = 1000
        np.random.seed(42)

        # 特徴量1: ランダム
        feature1 = np.random.rand(n_samples)

        # 特徴量2: パターンを持つ (0.8以上ならターゲット1になりやすい)
        feature2 = np.random.rand(n_samples)

        # ターゲット生成
        # 基本は0
        y = np.zeros(n_samples)

        # ルール: feature2 > 0.7 なら y=1 (確率90%)
        mask = feature2 > 0.7
        y[mask] = np.random.choice([0, 1], size=mask.sum(), p=[0.1, 0.9])

        # DataFrame作成
        X = pd.DataFrame(
            {
                "feature1": feature1,
                "feature2": feature2,
                "noise": np.random.rand(n_samples),  # ノイズ特徴量
            }
        )

        return X, y

    def test_lightgbm_can_learn_simple_pattern(self, synthetic_pattern_data):
        """LightGBMが単純なパターンを学習できるか検証"""
        X, y = synthetic_pattern_data

        # 学習用とテスト用に分割
        train_size = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # LightGBMモデル設定 (現在の設定に近いものを使用)
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "n_estimators": 100,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "random_state": 42,
            "n_jobs": -1,
            "importance_type": "gain",
            # 正則化などは一旦デフォルト
        }

        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)

        # 予測
        y_pred = model.predict(X_test)

        # 評価
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)

        print(f"\nAccuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")

        # 特徴量重要度
        importances = model.feature_importances_
        print(f"Feature Importances: {importances}")

        # アサーション
        # 1. 精度が高いこと (ランダムならターゲット比率程度になるはず)
        # ターゲット比率は feature2 > 0.7 (30%) * 0.9 = 27% 程度
        assert acc > 0.8, "モデルが単純なパターンを学習できていません (Accuracy)"

        # 2. Precisionが高いこと
        assert prec > 0.8, "モデルのPrecisionが低すぎます"

        # 3. feature2が最も重要であること
        # feature1, feature2, noise の順
        assert (
            importances[1] > importances[0]
        ), "重要な特徴量(feature2)が正しく評価されていません"
        assert (
            importances[1] > importances[2]
        ), "重要な特徴量(feature2)がノイズより低く評価されています"

    def test_lightgbm_with_class_weight(self, synthetic_pattern_data):
        """class_weight='balanced' の影響を確認"""
        X, y = synthetic_pattern_data

        # 不均衡データを模倣 (y=1を減らす)
        # feature2 > 0.9 のみ y=1 にする (10%程度)
        y_imbalanced = np.zeros(len(X))
        mask = X["feature2"] > 0.9
        y_imbalanced[mask] = 1

        train_size = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y_imbalanced[:train_size], y_imbalanced[train_size:]

        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "n_estimators": 100,
            "learning_rate": 0.05,
            "class_weight": "balanced",  # ここがポイント
            "random_state": 42,
        }

        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        prec = precision_score(y_test, y_pred)
        from sklearn.metrics import recall_score

        rec = recall_score(y_test, y_pred)

        print(f"\nBalanced - Precision: {prec:.4f}, Recall: {rec:.4f}")

        # Balancedの場合、Recallは高くなるがPrecisionは下がる傾向がある
        # しかし、明確なパターンがあれば両方高くなるはず
        assert rec > 0.9, "Balanced設定でRecallが低すぎます"
        assert (
            prec > 0.8
        ), "Balanced設定でも明確なパターンならPrecisionは維持されるべきです"




