"""
過学習分析スクリプト

108特徴量を使用した3モデル（TabNet、XGBoost、LightGBM）の過学習を詳細に分析します。
以下の分析を実施：
- トレーニング vs 検証 vs テストスコアの比較
- 学習曲線（Learning Curves）
- バリデーションカーブ（Validation Curves）
- クロスバリデーション分析
- 正則化の効果検証
- 特徴量数と過学習の関係
"""

import json
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score, learning_curve, validation_curve

warnings.filterwarnings("ignore")

# 定数定義
N_FEATURES = 108
RANDOM_SEED = 42
TRAIN_SIZE = 3000
VAL_SIZE = 600
TEST_SIZE = 600

# 特徴量カテゴリ構成
FEATURE_CATEGORIES = {
    "price": 15,
    "volatility": 5,
    "volume": 7,
    "technical": 22,
    "crypto_specific": 15,
    "advanced": 38,
    "interaction": 6,
}


class SyntheticDataGenerator:
    """合成データ生成クラス"""

    def __init__(self, n_features: int = N_FEATURES, random_seed: int = RANDOM_SEED):
        self.n_features = n_features
        self.random_seed = random_seed
        np.random.seed(random_seed)

    def generate_data(
        self, n_samples: int
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """合成データを生成"""
        features = []
        feature_names = []

        # 価格特徴量
        price_features = np.random.randn(n_samples, FEATURE_CATEGORIES["price"]) * 1.5
        features.append(price_features)
        feature_names.extend([f"price_{i}" for i in range(FEATURE_CATEGORIES["price"])])

        # ボラティリティ
        volatility_features = np.abs(
            np.random.randn(n_samples, FEATURE_CATEGORIES["volatility"]) * 1.5 + 2.5
        )
        features.append(volatility_features)
        feature_names.extend(
            [f"volatility_{i}" for i in range(FEATURE_CATEGORIES["volatility"])]
        )

        # 出来高
        volume_features = np.abs(
            np.random.randn(n_samples, FEATURE_CATEGORIES["volume"]) * 3 + 5
        )
        features.append(volume_features)
        feature_names.extend(
            [f"volume_{i}" for i in range(FEATURE_CATEGORIES["volume"])]
        )

        # テクニカル指標
        technical_features = np.random.randn(
            n_samples, FEATURE_CATEGORIES["technical"]
        )
        features.append(technical_features)
        feature_names.extend(
            [f"technical_{i}" for i in range(FEATURE_CATEGORIES["technical"])]
        )

        # 暗号通貨特化
        crypto_features = (
            np.random.randn(n_samples, FEATURE_CATEGORIES["crypto_specific"]) * 0.5
        )
        features.append(crypto_features)
        feature_names.extend(
            [f"crypto_{i}" for i in range(FEATURE_CATEGORIES["crypto_specific"])]
        )

        # 高度な特徴量
        advanced_features = np.random.randn(n_samples, FEATURE_CATEGORIES["advanced"])
        features.append(advanced_features)
        feature_names.extend(
            [f"advanced_{i}" for i in range(FEATURE_CATEGORIES["advanced"])]
        )

        # 相互作用特徴量
        interaction_features = np.zeros((n_samples, FEATURE_CATEGORIES["interaction"]))
        interaction_features[:, 0] = price_features[:, 0] * volatility_features[:, 0]
        interaction_features[:, 1] = price_features[:, 1] / (
            volatility_features[:, 1] + 1e-6
        )
        interaction_features[:, 2] = technical_features[:, 0] * volume_features[:, 0]
        interaction_features[:, 3] = crypto_features[:, 0] * advanced_features[:, 0]
        interaction_features[:, 4] = price_features[:, 2] * technical_features[:, 1]
        interaction_features[:, 5] = volume_features[:, 1] * volatility_features[:, 2]
        features.append(interaction_features)
        feature_names.extend(
            [f"interaction_{i}" for i in range(FEATURE_CATEGORIES["interaction"])]
        )

        # 特徴量を結合
        X = np.hstack(features)

        # ターゲット生成
        target_signal = (
            0.3 * price_features[:, 0]
            + 0.2 * technical_features[:, 5]
            - 0.15 * volatility_features[:, 1]
            + 0.1 * interaction_features[:, 0]
            + 0.15 * crypto_features[:, 3]
            + 0.1 * np.sin(advanced_features[:, 10])
        )
        threshold = np.median(target_signal)
        y = (target_signal > threshold).astype(int)

        return X, y, feature_names


class OverfittingAnalyzer:
    """過学習分析クラス"""

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str],
        output_dir: str = "data/feature_evaluation/plots/overfitting",
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}

    def analyze_train_val_test_comparison(self) -> Dict[str, Any]:
        """A. トレーニング vs 検証 vs テストスコアの比較"""
        print("\n" + "=" * 80)
        print("A. トレーニング vs 検証 vs テストスコア比較")
        print("=" * 80)

        comparison_results = {}

        # XGBoost
        print("\n[XGBoost]")
        xgb_model = xgb.XGBClassifier(
            max_depth=6,
            learning_rate=0.1,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_SEED,
            tree_method="hist",
            eval_metric="auc",
        )
        xgb_model.fit(self.X_train, self.y_train)

        train_pred_proba = xgb_model.predict_proba(self.X_train)[:, 1]
        val_pred_proba = xgb_model.predict_proba(self.X_val)[:, 1]
        test_pred_proba = xgb_model.predict_proba(self.X_test)[:, 1]

        train_pred = xgb_model.predict(self.X_train)
        val_pred = xgb_model.predict(self.X_val)
        test_pred = xgb_model.predict(self.X_test)

        xgb_results = {
            "train_roc_auc": float(roc_auc_score(self.y_train, train_pred_proba)),
            "val_roc_auc": float(roc_auc_score(self.y_val, val_pred_proba)),
            "test_roc_auc": float(roc_auc_score(self.y_test, test_pred_proba)),
            "train_f1": float(f1_score(self.y_train, train_pred)),
            "val_f1": float(f1_score(self.y_val, val_pred)),
            "test_f1": float(f1_score(self.y_test, test_pred)),
        }
        comparison_results["XGBoost"] = xgb_results
        self._print_comparison_scores("XGBoost", xgb_results)

        # LightGBM
        print("\n[LightGBM]")
        lgb_model = lgb.LGBMClassifier(
            num_leaves=31,
            learning_rate=0.1,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_SEED,
            verbose=-1,
        )
        lgb_model.fit(self.X_train, self.y_train)

        train_pred_proba = lgb_model.predict_proba(self.X_train)[:, 1]
        val_pred_proba = lgb_model.predict_proba(self.X_val)[:, 1]
        test_pred_proba = lgb_model.predict_proba(self.X_test)[:, 1]

        train_pred = lgb_model.predict(self.X_train)
        val_pred = lgb_model.predict(self.X_val)
        test_pred = lgb_model.predict(self.X_test)

        lgb_results = {
            "train_roc_auc": float(roc_auc_score(self.y_train, train_pred_proba)),
            "val_roc_auc": float(roc_auc_score(self.y_val, val_pred_proba)),
            "test_roc_auc": float(roc_auc_score(self.y_test, test_pred_proba)),
            "train_f1": float(f1_score(self.y_train, train_pred)),
            "val_f1": float(f1_score(self.y_val, val_pred)),
            "test_f1": float(f1_score(self.y_test, test_pred)),
        }
        comparison_results["LightGBM"] = lgb_results
        self._print_comparison_scores("LightGBM", lgb_results)

        # TabNet
        print("\n[TabNet]")
        tabnet_model = TabNetClassifier(
            n_d=64,
            n_a=64,
            n_steps=5,
            gamma=1.5,
            lambda_sparse=1e-4,
            optimizer_fn=__import__("torch.optim", fromlist=["Adam"]).Adam,
            optimizer_params={"lr": 2e-2},
            mask_type="sparsemax",
            seed=RANDOM_SEED,
            verbose=0,
        )
        tabnet_model.fit(
            X_train=self.X_train,
            y_train=self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            max_epochs=100,
            patience=20,
            batch_size=256,
            virtual_batch_size=128,
        )

        train_pred_proba = tabnet_model.predict_proba(self.X_train)[:, 1]
        val_pred_proba = tabnet_model.predict_proba(self.X_val)[:, 1]
        test_pred_proba = tabnet_model.predict_proba(self.X_test)[:, 1]

        train_pred = tabnet_model.predict(self.X_train)
        val_pred = tabnet_model.predict(self.X_val)
        test_pred = tabnet_model.predict(self.X_test)

        tabnet_results = {
            "train_roc_auc": float(roc_auc_score(self.y_train, train_pred_proba)),
            "val_roc_auc": float(roc_auc_score(self.y_val, val_pred_proba)),
            "test_roc_auc": float(roc_auc_score(self.y_test, test_pred_proba)),
            "train_f1": float(f1_score(self.y_train, train_pred)),
            "val_f1": float(f1_score(self.y_val, val_pred)),
            "test_f1": float(f1_score(self.y_test, test_pred)),
        }
        comparison_results["TabNet"] = tabnet_results
        self._print_comparison_scores("TabNet", tabnet_results)

        # 視覚化
        self._plot_train_val_test_comparison(comparison_results)

        return comparison_results

    def _print_comparison_scores(self, model_name: str, results: Dict[str, float]):
        """スコア比較を出力"""
        print(f"  Train ROC-AUC: {results['train_roc_auc']:.4f}")
        print(f"  Val ROC-AUC:   {results['val_roc_auc']:.4f}")
        print(f"  Test ROC-AUC:  {results['test_roc_auc']:.4f}")
        print(f"  Train F1:      {results['train_f1']:.4f}")
        print(f"  Val F1:        {results['val_f1']:.4f}")
        print(f"  Test F1:       {results['test_f1']:.4f}")

        # 過学習度を計算
        overfitting_degree = (
            (results["train_roc_auc"] - results["test_roc_auc"])
            / results["train_roc_auc"]
            * 100
        )
        print(f"  過学習度: {overfitting_degree:.2f}%")

    def _plot_train_val_test_comparison(self, results: Dict[str, Dict[str, float]]):
        """トレーニング vs 検証 vs テストスコア比較グラフ"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Train vs Validation vs Test Score Comparison", fontsize=16)

        models = list(results.keys())
        x = np.arange(len(models))
        width = 0.25

        # ROC-AUC
        ax = axes[0]
        train_roc = [results[m]["train_roc_auc"] for m in models]
        val_roc = [results[m]["val_roc_auc"] for m in models]
        test_roc = [results[m]["test_roc_auc"] for m in models]

        ax.bar(x - width, train_roc, width, label="Train", alpha=0.8)
        ax.bar(x, val_roc, width, label="Validation", alpha=0.8)
        ax.bar(x + width, test_roc, width, label="Test", alpha=0.8)

        ax.set_ylabel("ROC-AUC")
        ax.set_title("ROC-AUC Scores")
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        ax.set_ylim([0.5, 1.0])
        ax.grid(axis="y", alpha=0.3)

        # F1 Score
        ax = axes[1]
        train_f1 = [results[m]["train_f1"] for m in models]
        val_f1 = [results[m]["val_f1"] for m in models]
        test_f1 = [results[m]["test_f1"] for m in models]

        ax.bar(x - width, train_f1, width, label="Train", alpha=0.8)
        ax.bar(x, val_f1, width, label="Validation", alpha=0.8)
        ax.bar(x + width, test_f1, width, label="Test", alpha=0.8)

        ax.set_ylabel("F1 Score")
        ax.set_title("F1 Scores")
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        ax.set_ylim([0.5, 1.0])
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "train_val_test_comparison.png", dpi=300)
        print(f"\n保存: {self.output_dir / 'train_val_test_comparison.png'}")
        plt.close()

    def analyze_learning_curves(self) -> Dict[str, Any]:
        """B. 学習曲線の生成"""
        print("\n" + "=" * 80)
        print("B. 学習曲線分析")
        print("=" * 80)

        train_sizes = [100, 300, 500, 1000, 1500, 2000]
        learning_curves_results = {}

        # XGBoost
        print("\n[XGBoost]")
        xgb_model = xgb.XGBClassifier(
            max_depth=6,
            learning_rate=0.1,
            n_estimators=200,
            random_state=RANDOM_SEED,
            tree_method="hist",
        )
        train_sizes_abs, train_scores, test_scores = learning_curve(
            xgb_model,
            self.X_train,
            self.y_train,
            train_sizes=train_sizes,
            cv=3,
            scoring="roc_auc",
            n_jobs=-1,
            random_state=RANDOM_SEED,
        )
        learning_curves_results["XGBoost"] = {
            "train_sizes": train_sizes_abs.tolist(),
            "train_scores_mean": np.mean(train_scores, axis=1).tolist(),
            "train_scores_std": np.std(train_scores, axis=1).tolist(),
            "test_scores_mean": np.mean(test_scores, axis=1).tolist(),
            "test_scores_std": np.std(test_scores, axis=1).tolist(),
        }
        print("  完了")

        # LightGBM
        print("\n[LightGBM]")
        lgb_model = lgb.LGBMClassifier(
            num_leaves=31,
            learning_rate=0.1,
            n_estimators=200,
            random_state=RANDOM_SEED,
            verbose=-1,
        )
        train_sizes_abs, train_scores, test_scores = learning_curve(
            lgb_model,
            self.X_train,
            self.y_train,
            train_sizes=train_sizes,
            cv=3,
            scoring="roc_auc",
            n_jobs=-1,
            random_state=RANDOM_SEED,
        )
        learning_curves_results["LightGBM"] = {
            "train_sizes": train_sizes_abs.tolist(),
            "train_scores_mean": np.mean(train_scores, axis=1).tolist(),
            "train_scores_std": np.std(train_scores, axis=1).tolist(),
            "test_scores_mean": np.mean(test_scores, axis=1).tolist(),
            "test_scores_std": np.std(test_scores, axis=1).tolist(),
        }
        print("  完了")

        # 視覚化
        self._plot_learning_curves(learning_curves_results)

        return learning_curves_results

    def _plot_learning_curves(self, results: Dict[str, Dict[str, List[float]]]):
        """学習曲線のプロット"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Learning Curves", fontsize=16)

        for idx, (model_name, data) in enumerate(results.items()):
            ax = axes[idx]
            train_sizes = data["train_sizes"]
            train_mean = data["train_scores_mean"]
            train_std = data["train_scores_std"]
            test_mean = data["test_scores_mean"]
            test_std = data["test_scores_std"]

            ax.fill_between(
                train_sizes,
                np.array(train_mean) - np.array(train_std),
                np.array(train_mean) + np.array(train_std),
                alpha=0.1,
                color="blue",
            )
            ax.fill_between(
                train_sizes,
                np.array(test_mean) - np.array(test_std),
                np.array(test_mean) + np.array(test_std),
                alpha=0.1,
                color="orange",
            )
            ax.plot(train_sizes, train_mean, "o-", color="blue", label="Training score")
            ax.plot(
                train_sizes, test_mean, "o-", color="orange", label="Cross-validation score"
            )

            ax.set_xlabel("Training Size")
            ax.set_ylabel("ROC-AUC Score")
            ax.set_title(f"{model_name}")
            ax.legend(loc="best")
            ax.grid(alpha=0.3)
            ax.set_ylim([0.5, 1.0])

        plt.tight_layout()
        plt.savefig(self.output_dir / "learning_curves.png", dpi=300)
        print(f"\n保存: {self.output_dir / 'learning_curves.png'}")
        plt.close()

    def analyze_validation_curves(self) -> Dict[str, Any]:
        """C. バリデーションカーブの生成"""
        print("\n" + "=" * 80)
        print("C. バリデーションカーブ分析")
        print("=" * 80)

        validation_curves_results = {}

        # XGBoost - max_depth
        print("\n[XGBoost - max_depth]")
        param_range = [3, 4, 5, 6, 7, 8]
        xgb_model = xgb.XGBClassifier(
            learning_rate=0.1,
            n_estimators=200,
            random_state=RANDOM_SEED,
            tree_method="hist",
        )
        train_scores, test_scores = validation_curve(
            xgb_model,
            self.X_train,
            self.y_train,
            param_name="max_depth",
            param_range=param_range,
            cv=3,
            scoring="roc_auc",
            n_jobs=-1,
        )
        validation_curves_results["XGBoost_max_depth"] = {
            "param_range": param_range,
            "train_scores_mean": np.mean(train_scores, axis=1).tolist(),
            "train_scores_std": np.std(train_scores, axis=1).tolist(),
            "test_scores_mean": np.mean(test_scores, axis=1).tolist(),
            "test_scores_std": np.std(test_scores, axis=1).tolist(),
        }
        print("  完了")

        # XGBoost - n_estimators
        print("\n[XGBoost - n_estimators]")
        param_range = [50, 100, 150, 200, 250, 300]
        xgb_model = xgb.XGBClassifier(
            max_depth=6, learning_rate=0.1, random_state=RANDOM_SEED, tree_method="hist"
        )
        train_scores, test_scores = validation_curve(
            xgb_model,
            self.X_train,
            self.y_train,
            param_name="n_estimators",
            param_range=param_range,
            cv=3,
            scoring="roc_auc",
            n_jobs=-1,
        )
        validation_curves_results["XGBoost_n_estimators"] = {
            "param_range": param_range,
            "train_scores_mean": np.mean(train_scores, axis=1).tolist(),
            "train_scores_std": np.std(train_scores, axis=1).tolist(),
            "test_scores_mean": np.mean(test_scores, axis=1).tolist(),
            "test_scores_std": np.std(test_scores, axis=1).tolist(),
        }
        print("  完了")

        # LightGBM - num_leaves
        print("\n[LightGBM - num_leaves]")
        param_range = [15, 23, 31, 39, 47, 55]
        lgb_model = lgb.LGBMClassifier(
            learning_rate=0.1, n_estimators=200, random_state=RANDOM_SEED, verbose=-1
        )
        train_scores, test_scores = validation_curve(
            lgb_model,
            self.X_train,
            self.y_train,
            param_name="num_leaves",
            param_range=param_range,
            cv=3,
            scoring="roc_auc",
            n_jobs=-1,
        )
        validation_curves_results["LightGBM_num_leaves"] = {
            "param_range": param_range,
            "train_scores_mean": np.mean(train_scores, axis=1).tolist(),
            "train_scores_std": np.std(train_scores, axis=1).tolist(),
            "test_scores_mean": np.mean(test_scores, axis=1).tolist(),
            "test_scores_std": np.std(test_scores, axis=1).tolist(),
        }
        print("  完了")

        # LightGBM - n_estimators
        print("\n[LightGBM - n_estimators]")
        param_range = [50, 100, 150, 200, 250, 300]
        lgb_model = lgb.LGBMClassifier(
            num_leaves=31, learning_rate=0.1, random_state=RANDOM_SEED, verbose=-1
        )
        train_scores, test_scores = validation_curve(
            lgb_model,
            self.X_train,
            self.y_train,
            param_name="n_estimators",
            param_range=param_range,
            cv=3,
            scoring="roc_auc",
            n_jobs=-1,
        )
        validation_curves_results["LightGBM_n_estimators"] = {
            "param_range": param_range,
            "train_scores_mean": np.mean(train_scores, axis=1).tolist(),
            "train_scores_std": np.std(train_scores, axis=1).tolist(),
            "test_scores_mean": np.mean(test_scores, axis=1).tolist(),
            "test_scores_std": np.std(test_scores, axis=1).tolist(),
        }
        print("  完了")

        # 視覚化
        self._plot_validation_curves(validation_curves_results)

        return validation_curves_results

    def _plot_validation_curves(self, results: Dict[str, Dict[str, List[float]]]):
        """バリデーションカーブのプロット"""
        # XGBoost
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("XGBoost Validation Curves", fontsize=16)

        # max_depth
        ax = axes[0]
        data = results["XGBoost_max_depth"]
        param_range = data["param_range"]
        train_mean = data["train_scores_mean"]
        train_std = data["train_scores_std"]
        test_mean = data["test_scores_mean"]
        test_std = data["test_scores_std"]

        ax.fill_between(
            param_range,
            np.array(train_mean) - np.array(train_std),
            np.array(train_mean) + np.array(train_std),
            alpha=0.1,
            color="blue",
        )
        ax.fill_between(
            param_range,
            np.array(test_mean) - np.array(test_std),
            np.array(test_mean) + np.array(test_std),
            alpha=0.1,
            color="orange",
        )
        ax.plot(param_range, train_mean, "o-", color="blue", label="Training score")
        ax.plot(param_range, test_mean, "o-", color="orange", label="Cross-validation score")

        ax.set_xlabel("max_depth")
        ax.set_ylabel("ROC-AUC Score")
        ax.set_title("max_depth Validation Curve")
        ax.legend(loc="best")
        ax.grid(alpha=0.3)
        ax.set_ylim([0.5, 1.0])

        # n_estimators
        ax = axes[1]
        data = results["XGBoost_n_estimators"]
        param_range = data["param_range"]
        train_mean = data["train_scores_mean"]
        train_std = data["train_scores_std"]
        test_mean = data["test_scores_mean"]
        test_std = data["test_scores_std"]

        ax.fill_between(
            param_range,
            np.array(train_mean) - np.array(train_std),
            np.array(train_mean) + np.array(train_std),
            alpha=0.1,
            color="blue",
        )
        ax.fill_between(
            param_range,
            np.array(test_mean) - np.array(test_std),
            np.array(test_mean) + np.array(test_std),
            alpha=0.1,
            color="orange",
        )
        ax.plot(param_range, train_mean, "o-", color="blue", label="Training score")
        ax.plot(param_range, test_mean, "o-", color="orange", label="Cross-validation score")

        ax.set_xlabel("n_estimators")
        ax.set_ylabel("ROC-AUC Score")
        ax.set_title("n_estimators Validation Curve")
        ax.legend(loc="best")
        ax.grid(alpha=0.3)
        ax.set_ylim([0.5, 1.0])

        plt.tight_layout()
        plt.savefig(self.output_dir / "validation_curves_xgboost.png", dpi=300)
        print(f"\n保存: {self.output_dir / 'validation_curves_xgboost.png'}")
        plt.close()

        # LightGBM
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("LightGBM Validation Curves", fontsize=16)

        # num_leaves
        ax = axes[0]
        data = results["LightGBM_num_leaves"]
        param_range = data["param_range"]
        train_mean = data["train_scores_mean"]
        train_std = data["train_scores_std"]
        test_mean = data["test_scores_mean"]
        test_std = data["test_scores_std"]

        ax.fill_between(
            param_range,
            np.array(train_mean) - np.array(train_std),
            np.array(train_mean) + np.array(train_std),
            alpha=0.1,
            color="blue",
        )
        ax.fill_between(
            param_range,
            np.array(test_mean) - np.array(test_std),
            np.array(test_mean) + np.array(test_std),
            alpha=0.1,
            color="orange",
        )
        ax.plot(param_range, train_mean, "o-", color="blue", label="Training score")
        ax.plot(param_range, test_mean, "o-", color="orange", label="Cross-validation score")

        ax.set_xlabel("num_leaves")
        ax.set_ylabel("ROC-AUC Score")
        ax.set_title("num_leaves Validation Curve")
        ax.legend(loc="best")
        ax.grid(alpha=0.3)
        ax.set_ylim([0.5, 1.0])

        # n_estimators
        ax = axes[1]
        data = results["LightGBM_n_estimators"]
        param_range = data["param_range"]
        train_mean = data["train_scores_mean"]
        train_std = data["train_scores_std"]
        test_mean = data["test_scores_mean"]
        test_std = data["test_scores_std"]

        ax.fill_between(
            param_range,
            np.array(train_mean) - np.array(train_std),
            np.array(train_mean) + np.array(train_std),
            alpha=0.1,
            color="blue",
        )
        ax.fill_between(
            param_range,
            np.array(test_mean) - np.array(test_std),
            np.array(test_mean) + np.array(test_std),
            alpha=0.1,
            color="orange",
        )
        ax.plot(param_range, train_mean, "o-", color="blue", label="Training score")
        ax.plot(param_range, test_mean, "o-", color="orange", label="Cross-validation score")

        ax.set_xlabel("n_estimators")
        ax.set_ylabel("ROC-AUC Score")
        ax.set_title("n_estimators Validation Curve")
        ax.legend(loc="best")
        ax.grid(alpha=0.3)
        ax.set_ylim([0.5, 1.0])

        plt.tight_layout()
        plt.savefig(self.output_dir / "validation_curves_lightgbm.png", dpi=300)
        print(f"\n保存: {self.output_dir / 'validation_curves_lightgbm.png'}")
        plt.close()

    def analyze_cross_validation(self) -> Dict[str, Any]:
        """D. クロスバリデーション分析"""
        print("\n" + "=" * 80)
        print("D. クロスバリデーション分析")
        print("=" * 80)

        cv_results = {}

        # XGBoost
        print("\n[XGBoost]")
        xgb_model = xgb.XGBClassifier(
            max_depth=6,
            learning_rate=0.1,
            n_estimators=200,
            random_state=RANDOM_SEED,
            tree_method="hist",
        )
        scores = cross_val_score(
            xgb_model, self.X_train, self.y_train, cv=5, scoring="roc_auc", n_jobs=-1
        )
        cv_results["XGBoost"] = {
            "scores": scores.tolist(),
            "mean": float(scores.mean()),
            "std": float(scores.std()),
        }
        print(f"  スコア: {scores}")
        print(f"  平均: {scores.mean():.4f} ± {scores.std():.4f}")

        # LightGBM
        print("\n[LightGBM]")
        lgb_model = lgb.LGBMClassifier(
            num_leaves=31,
            learning_rate=0.1,
            n_estimators=200,
            random_state=RANDOM_SEED,
            verbose=-1,
        )
        scores = cross_val_score(
            lgb_model, self.X_train, self.y_train, cv=5, scoring="roc_auc", n_jobs=-1
        )
        cv_results["LightGBM"] = {
            "scores": scores.tolist(),
            "mean": float(scores.mean()),
            "std": float(scores.std()),
        }
        print(f"  スコア: {scores}")
        print(f"  平均: {scores.mean():.4f} ± {scores.std():.4f}")

        # 視覚化
        self._plot_cv_distribution(cv_results)

        return cv_results

    def _plot_cv_distribution(self, results: Dict[str, Dict[str, Any]]):
        """CVスコア分布のプロット"""
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle("Cross-Validation Score Distribution", fontsize=16)

        models = list(results.keys())
        data = [results[m]["scores"] for m in models]

        bp = ax.boxplot(data, labels=models, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("lightblue")

        ax.set_ylabel("ROC-AUC Score")
        ax.set_title("5-Fold Cross-Validation")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim([0.5, 1.0])

        # 平均値を表示
        for i, model in enumerate(models, 1):
            mean_val = results[model]["mean"]
            ax.text(i, mean_val, f"{mean_val:.4f}", ha="center", va="bottom")

        plt.tight_layout()
        plt.savefig(self.output_dir / "cv_score_distribution.png", dpi=300)
        print(f"\n保存: {self.output_dir / 'cv_score_distribution.png'}")
        plt.close()

    def analyze_regularization_effect(self) -> Dict[str, Any]:
        """E. 正則化の効果検証"""
        print("\n" + "=" * 80)
        print("E. 正則化の効果検証")
        print("=" * 80)

        regularization_results = {}

        # XGBoost - reg_alpha
        print("\n[XGBoost - reg_alpha]")
        alpha_values = [0, 0.01, 0.1, 1.0]
        xgb_alpha_results = []
        for alpha in alpha_values:
            model = xgb.XGBClassifier(
                max_depth=6,
                learning_rate=0.1,
                n_estimators=200,
                reg_alpha=alpha,
                random_state=RANDOM_SEED,
                tree_method="hist",
            )
            model.fit(self.X_train, self.y_train)

            train_score = roc_auc_score(
                self.y_train, model.predict_proba(self.X_train)[:, 1]
            )
            test_score = roc_auc_score(
                self.y_test, model.predict_proba(self.X_test)[:, 1]
            )
            xgb_alpha_results.append(
                {
                    "alpha": alpha,
                    "train_score": float(train_score),
                    "test_score": float(test_score),
                }
            )
            print(f"  alpha={alpha}: Train={train_score:.4f}, Test={test_score:.4f}")

        regularization_results["XGBoost_reg_alpha"] = xgb_alpha_results

        # LightGBM - reg_alpha
        print("\n[LightGBM - reg_alpha]")
        lgb_alpha_results = []
        for alpha in alpha_values:
            model = lgb.LGBMClassifier(
                num_leaves=31,
                learning_rate=0.1,
                n_estimators=200,
                reg_alpha=alpha,
                random_state=RANDOM_SEED,
                verbose=-1,
            )
            model.fit(self.X_train, self.y_train)

            train_score = roc_auc_score(
                self.y_train, model.predict_proba(self.X_train)[:, 1]
            )
            test_score = roc_auc_score(
                self.y_test, model.predict_proba(self.X_test)[:, 1]
            )
            lgb_alpha_results.append(
                {
                    "alpha": alpha,
                    "train_score": float(train_score),
                    "test_score": float(test_score),
                }
            )
            print(f"  alpha={alpha}: Train={train_score:.4f}, Test={test_score:.4f}")

        regularization_results["LightGBM_reg_alpha"] = lgb_alpha_results

        # 視覚化
        self._plot_regularization_effect(regularization_results)

        return regularization_results

    def _plot_regularization_effect(self, results: Dict[str, List[Dict[str, Any]]]):
        """正則化の効果のプロット"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Regularization Effect", fontsize=16)

        # XGBoost
        ax = axes[0]
        data = results["XGBoost_reg_alpha"]
        alphas = [d["alpha"] for d in data]
        train_scores = [d["train_score"] for d in data]
        test_scores = [d["test_score"] for d in data]

        ax.plot(alphas, train_scores, "o-", label="Training score", linewidth=2)
        ax.plot(alphas, test_scores, "o-", label="Test score", linewidth=2)
        ax.set_xlabel("reg_alpha")
        ax.set_ylabel("ROC-AUC Score")
        ax.set_title("XGBoost")
        ax.legend(loc="best")
        ax.grid(alpha=0.3)
        ax.set_xscale("log")
        ax.set_ylim([0.5, 1.0])

        # LightGBM
        ax = axes[1]
        data = results["LightGBM_reg_alpha"]
        alphas = [d["alpha"] for d in data]
        train_scores = [d["train_score"] for d in data]
        test_scores = [d["test_score"] for d in data]

        ax.plot(alphas, train_scores, "o-", label="Training score", linewidth=2)
        ax.plot(alphas, test_scores, "o-", label="Test score", linewidth=2)
        ax.set_xlabel("reg_alpha")
        ax.set_ylabel("ROC-AUC Score")
        ax.set_title("LightGBM")
        ax.legend(loc="best")
        ax.grid(alpha=0.3)
        ax.set_xscale("log")
        ax.set_ylim([0.5, 1.0])

        plt.tight_layout()
        plt.savefig(self.output_dir / "regularization_effect.png", dpi=300)
        print(f"\n保存: {self.output_dir / 'regularization_effect.png'}")
        plt.close()

    def analyze_feature_count_vs_overfitting(self) -> Dict[str, Any]:
        """F. 特徴量数と過学習の関係"""
        print("\n" + "=" * 80)
        print("F. 特徴量数と過学習の関係")
        print("=" * 80)

        feature_counts = [20, 40, 60, 80, 108]
        feature_count_results = {}

        # XGBoost
        print("\n[XGBoost]")
        xgb_results = []
        for n_features in feature_counts:
            # 特徴量を選択（重要度の高い順に）
            model_temp = xgb.XGBClassifier(
                max_depth=6,
                learning_rate=0.1,
                n_estimators=200,
                random_state=RANDOM_SEED,
                tree_method="hist",
            )
            model_temp.fit(self.X_train, self.y_train)
            importance = model_temp.feature_importances_
            top_indices = np.argsort(importance)[::-1][:n_features]

            X_train_subset = self.X_train[:, top_indices]
            X_test_subset = self.X_test[:, top_indices]

            # モデルを再トレーニング
            model = xgb.XGBClassifier(
                max_depth=6,
                learning_rate=0.1,
                n_estimators=200,
                random_state=RANDOM_SEED,
                tree_method="hist",
            )
            model.fit(X_train_subset, self.y_train)

            train_score = roc_auc_score(
                self.y_train, model.predict_proba(X_train_subset)[:, 1]
            )
            test_score = roc_auc_score(
                self.y_test, model.predict_proba(X_test_subset)[:, 1]
            )

            xgb_results.append(
                {
                    "n_features": n_features,
                    "train_score": float(train_score),
                    "test_score": float(test_score),
                    "gap": float(train_score - test_score),
                }
            )
            print(
                f"  n_features={n_features}: Train={train_score:.4f}, Test={test_score:.4f}, Gap={train_score - test_score:.4f}"
            )

        feature_count_results["XGBoost"] = xgb_results

        # LightGBM
        print("\n[LightGBM]")
        lgb_results = []
        for n_features in feature_counts:
            # 特徴量を選択
            model_temp = lgb.LGBMClassifier(
                num_leaves=31,
                learning_rate=0.1,
                n_estimators=200,
                random_state=RANDOM_SEED,
                verbose=-1,
            )
            model_temp.fit(self.X_train, self.y_train)
            importance = model_temp.feature_importances_
            top_indices = np.argsort(importance)[::-1][:n_features]

            X_train_subset = self.X_train[:, top_indices]
            X_test_subset = self.X_test[:, top_indices]

            # モデルを再トレーニング
            model = lgb.LGBMClassifier(
                num_leaves=31,
                learning_rate=0.1,
                n_estimators=200,
                random_state=RANDOM_SEED,
                verbose=-1,
            )
            model.fit(X_train_subset, self.y_train)

            train_score = roc_auc_score(
                self.y_train, model.predict_proba(X_train_subset)[:, 1]
            )
            test_score = roc_auc_score(
                self.y_test, model.predict_proba(X_test_subset)[:, 1]
            )

            lgb_results.append(
                {
                    "n_features": n_features,
                    "train_score": float(train_score),
                    "test_score": float(test_score),
                    "gap": float(train_score - test_score),
                }
            )
            print(
                f"  n_features={n_features}: Train={train_score:.4f}, Test={test_score:.4f}, Gap={train_score - test_score:.4f}"
            )

        feature_count_results["LightGBM"] = lgb_results

        # 視覚化
        self._plot_feature_count_vs_overfitting(feature_count_results)

        return feature_count_results

    def _plot_feature_count_vs_overfitting(self, results: Dict[str, List[Dict[str, Any]]]):
        """特徴量数と過学習の関係のプロット"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Feature Count vs Overfitting", fontsize=16)

        # XGBoost
        ax = axes[0]
        data = results["XGBoost"]
        n_features = [d["n_features"] for d in data]
        train_scores = [d["train_score"] for d in data]
        test_scores = [d["test_score"] for d in data]
        gaps = [d["gap"] for d in data]

        ax.plot(n_features, train_scores, "o-", label="Training score", linewidth=2)
        ax.plot(n_features, test_scores, "o-", label="Test score", linewidth=2)
        ax.plot(n_features, gaps, "o-", label="Gap (Overfitting)", linewidth=2, linestyle="--")

        ax.set_xlabel("Number of Features")
        ax.set_ylabel("ROC-AUC Score / Gap")
        ax.set_title("XGBoost")
        ax.legend(loc="best")
        ax.grid(alpha=0.3)
        ax.set_ylim([0, 1.0])

        # LightGBM
        ax = axes[1]
        data = results["LightGBM"]
        n_features = [d["n_features"] for d in data]
        train_scores = [d["train_score"] for d in data]
        test_scores = [d["test_score"] for d in data]
        gaps = [d["gap"] for d in data]

        ax.plot(n_features, train_scores, "o-", label="Training score", linewidth=2)
        ax.plot(n_features, test_scores, "o-", label="Test score", linewidth=2)
        ax.plot(n_features, gaps, "o-", label="Gap (Overfitting)", linewidth=2, linestyle="--")

        ax.set_xlabel("Number of Features")
        ax.set_ylabel("ROC-AUC Score / Gap")
        ax.set_title("LightGBM")
        ax.legend(loc="best")
        ax.grid(alpha=0.3)
        ax.set_ylim([0, 1.0])

        plt.tight_layout()
        plt.savefig(self.output_dir / "feature_count_vs_overfitting.png", dpi=300)
        print(f"\n保存: {self.output_dir / 'feature_count_vs_overfitting.png'}")
        plt.close()

    def run_all_analyses(self) -> Dict[str, Any]:
        """全ての分析を実行"""
        results = {}

        results["train_val_test_comparison"] = self.analyze_train_val_test_comparison()
        results["learning_curves"] = self.analyze_learning_curves()
        results["validation_curves"] = self.analyze_validation_curves()
        results["cross_validation"] = self.analyze_cross_validation()
        results["regularization_effect"] = self.analyze_regularization_effect()
        results["feature_count_vs_overfitting"] = (
            self.analyze_feature_count_vs_overfitting()
        )

        return results


def calculate_overfitting_score(train_score: float, test_score: float) -> float:
    """過学習度を計算"""
    return (train_score - test_score) / train_score * 100


def generate_markdown_report(
    results: Dict[str, Any], output_path: str = "backend/OVERFITTING_ANALYSIS_REPORT.md"
):
    """過学習分析レポートを生成"""
    output_file = Path(output_path)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# 過学習分析レポート\n\n")
        f.write("## 概要\n\n")
        f.write("108特徴量を使用した3モデル（TabNet、XGBoost、LightGBM）の過学習を詳細に分析しました。\n\n")

        f.write("## 1. 過学習度の定量評価\n\n")
        f.write("過学習度 = (Train_Score - Test_Score) / Train_Score × 100\n\n")
        f.write("### 評価基準\n\n")
        f.write("- 0-5%: 問題なし ✅\n")
        f.write("- 5-10%: 軽度の過学習 ⚠️\n")
        f.write("- 10-20%: 中程度の過学習 ⚠️⚠️\n")
        f.write("- 20%+: 深刻な過学習 ❌\n\n")

        f.write("### 各モデルの過学習度\n\n")
        f.write("| モデル | Train ROC-AUC | Test ROC-AUC | 過学習度 | 評価 |\n")
        f.write("|--------|---------------|--------------|----------|------|\n")

        comparison = results["train_val_test_comparison"]
        for model_name, scores in comparison.items():
            train_roc = scores["train_roc_auc"]
            test_roc = scores["test_roc_auc"]
            overfitting_score = calculate_overfitting_score(train_roc, test_roc)

            if overfitting_score < 5:
                status = "✅ 問題なし"
            elif overfitting_score < 10:
                status = "⚠️ 軽度"
            elif overfitting_score < 20:
                status = "⚠️⚠️ 中程度"
            else:
                status = "❌ 深刻"

            f.write(
                f"| {model_name} | {train_roc:.4f} | {test_roc:.4f} | "
                f"{overfitting_score:.2f}% | {status} |\n"
            )

        f.write("\n")

        # モデルごとの診断
        f.write("## 2. モデルごとの過学習診断\n\n")

        for model_name, scores in comparison.items():
            f.write(f"### {model_name}\n\n")

            train_roc = scores["train_roc_auc"]
            test_roc = scores["test_roc_auc"]
            val_roc = scores["val_roc_auc"]
            overfitting_score = calculate_overfitting_score(train_roc, test_roc)

            # リスクレベル
            if overfitting_score < 5:
                risk_level = "低"
                risk_icon = "✅"
            elif overfitting_score < 10:
                risk_level = "中"
                risk_icon = "⚠️"
            else:
                risk_level = "高"
                risk_icon = "❌"

            f.write(f"**過学習リスクレベル**: {risk_icon} {risk_level}\n\n")

            f.write("**スコア詳細**:\n")
            f.write(f"- Training ROC-AUC: {train_roc:.4f}\n")
            f.write(f"- Validation ROC-AUC: {val_roc:.4f}\n")
            f.write(f"- Test ROC-AUC: {test_roc:.4f}\n")
            f.write(f"- Train-Test Gap: {train_roc - test_roc:.4f}\n\n")

            # 原因分析
            f.write("**主要な原因分析**:\n")
            if train_roc - val_roc > 0.05:
                f.write("- トレーニングデータへの過度の適合が見られます\n")
            if val_roc - test_roc > 0.05:
                f.write("- 検証データとテストデータの分布に差異があります\n")
            if overfitting_score < 5:
                f.write("- モデルは適切に汎化できています\n")

            f.write("\n")

            # 推奨される対策
            f.write("**推奨される対策**:\n")
            if overfitting_score >= 10:
                f.write("- 正則化パラメータの調整（reg_alpha, reg_lambdaの増加）\n")
                f.write("- モデルの複雑度を下げる（max_depth, num_leavesの削減）\n")
                f.write("- ドロップアウトやアーリーストッピングの活用\n")
            elif overfitting_score >= 5:
                f.write("- クロスバリデーションで安定性を確認\n")
                f.write("- 特徴量選択による次元削減を検討\n")
            else:
                f.write("- 現在のハイパーパラメータは適切です\n")
                f.write("- 更なる性能向上のため、特徴量エンジニアリングを検討\n")

            f.write("\n")

        # クロスバリデーション結果
        f.write("## 3. クロスバリデーション結果\n\n")
        f.write("5-Fold Cross-Validationの結果:\n\n")
        f.write("| モデル | 平均スコア | 標準偏差 | 安定性 |\n")
        f.write("|--------|------------|----------|--------|\n")

        cv_results = results["cross_validation"]
        for model_name, cv_data in cv_results.items():
            mean_score = cv_data["mean"]
            std_score = cv_data["std"]

            if std_score < 0.02:
                stability = "✅ 高"
            elif std_score < 0.05:
                stability = "⚠️ 中"
            else:
                stability = "❌ 低"

            f.write(
                f"| {model_name} | {mean_score:.4f} | {std_score:.4f} | {stability} |\n"
            )

        f.write("\n")

        # 特徴量数の推奨
        f.write("## 4. 最適な特徴量数の推奨\n\n")
        feature_results = results["feature_count_vs_overfitting"]

        for model_name, data in feature_results.items():
            f.write(f"### {model_name}\n\n")

            # 最小のギャップを持つ特徴量数を見つける
            min_gap_idx = min(range(len(data)), key=lambda i: data[i]["gap"])
            optimal_n_features = data[min_gap_idx]["n_features"]
            optimal_gap = data[min_gap_idx]["gap"]
            optimal_test_score = data[min_gap_idx]["test_score"]

            f.write(f"**推奨特徴量数**: {optimal_n_features}\n")
            f.write(f"- Test ROC-AUC: {optimal_test_score:.4f}\n")
            f.write(f"- Train-Test Gap: {optimal_gap:.4f}\n\n")

            f.write("**特徴量数別の性能**:\n\n")
            f.write("| 特徴量数 | Train ROC-AUC | Test ROC-AUC | Gap |\n")
            f.write("|----------|---------------|--------------|-----|\n")
            for item in data:
                f.write(
                    f"| {item['n_features']} | {item['train_score']:.4f} | "
                    f"{item['test_score']:.4f} | {item['gap']:.4f} |\n"
                )

            f.write("\n")

        # 正則化の効果
        f.write("## 5. 正則化の効果\n\n")
        reg_results = results["regularization_effect"]

        for key, data in reg_results.items():
            model_name = key.split("_")[0]
            f.write(f"### {model_name}\n\n")

            f.write("| reg_alpha | Train ROC-AUC | Test ROC-AUC | Gap |\n")
            f.write("|-----------|---------------|--------------|-----|\n")
            for item in data:
                gap = item["train_score"] - item["test_score"]
                f.write(
                    f"| {item['alpha']} | {item['train_score']:.4f} | "
                    f"{item['test_score']:.4f} | {gap:.4f} |\n"
                )

            f.write("\n")

        # 総合評価
        f.write("## 6. 過学習リスクの総合評価\n\n")

        # 全モデルの平均過学習度を計算
        avg_overfitting = np.mean(
            [
                calculate_overfitting_score(
                    scores["train_roc_auc"], scores["test_roc_auc"]
                )
                for scores in comparison.values()
            ]
        )

        if avg_overfitting < 5:
            f.write("### ✅ 総合評価: 優良\n\n")
            f.write("全てのモデルが適切に汎化できており、過学習のリスクは低いです。\n")
        elif avg_overfitting < 10:
            f.write("### ⚠️ 総合評価: 良好（要注意）\n\n")
            f.write("軽度の過学習が見られますが、実用上は問題ありません。\n")
        else:
            f.write("### ❌ 総合評価: 要改善\n\n")
            f.write("過学習のリスクが高く、改善が必要です。\n")

        f.write("\n")

        # 推奨事項
        f.write("## 7. 推奨される対策（優先度順）\n\n")
        f.write("1. **正則化パラメータの調整**\n")
        f.write("   - XGBoost: reg_alpha=0.1, reg_lambda=0.1 を試す\n")
        f.write("   - LightGBM: reg_alpha=0.1, reg_lambda=0.1 を試す\n\n")

        f.write("2. **特徴量選択**\n")
        f.write("   - 重要度の低い特徴量を削除し、60-80特徴量に削減\n")
        f.write("   - 相関の高い特徴量を統合\n\n")

        f.write("3. **モデルの複雑度調整**\n")
        f.write("   - XGBoost: max_depth を 4-5 に削減\n")
        f.write("   - LightGBM: num_leaves を 23-31 に調整\n\n")

        f.write("4. **データ拡張**\n")
        f.write("   - より多くのトレーニングデータを収集\n")
        f.write("   - データ拡張技術の適用を検討\n\n")

        f.write("5. **アンサンブル手法**\n")
        f.write("   - 複数モデルのアンサンブルで汎化性能を向上\n")
        f.write("   - スタッキングやブレンディングの活用\n\n")

        # 生成ファイル
        f.write("## 8. 生成されたファイル\n\n")
        f.write("### グラフ\n")
        f.write("1. `data/feature_evaluation/plots/overfitting/train_val_test_comparison.png` - トレーニング/検証/テストスコア比較\n")
        f.write("2. `data/feature_evaluation/plots/overfitting/learning_curves.png` - 学習曲線\n")
        f.write("3. `data/feature_evaluation/plots/overfitting/validation_curves_xgboost.png` - XGBoostバリデーションカーブ\n")
        f.write("4. `data/feature_evaluation/plots/overfitting/validation_curves_lightgbm.png` - LightGBMバリデーションカーブ\n")
        f.write("5. `data/feature_evaluation/plots/overfitting/cv_score_distribution.png` - CVスコア分布\n")
        f.write("6. `data/feature_evaluation/plots/overfitting/regularization_effect.png` - 正則化の効果\n")
        f.write("7. `data/feature_evaluation/plots/overfitting/feature_count_vs_overfitting.png` - 特徴量数と過学習の関係\n\n")

        f.write("### データ\n")
        f.write("- `data/feature_evaluation/overfitting_analysis_results.json` - 全分析結果（JSON形式）\n")

    print(f"\nレポートを保存: {output_file.absolute()}")


def main():
    """メイン実行関数"""
    print("=" * 80)
    print("過学習分析スクリプト")
    print("=" * 80)

    # データ生成
    print("\n[1/7] 合成データ生成中...")
    generator = SyntheticDataGenerator()
    total_samples = TRAIN_SIZE + VAL_SIZE + TEST_SIZE
    X, y, feature_names = generator.generate_data(total_samples)

    # データ分割
    from sklearn.model_selection import train_test_split

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VAL_SIZE, random_state=RANDOM_SEED, stratify=y_temp
    )

    print(f"データ形状:")
    print(f"  - トレーニング: {X_train.shape}")
    print(f"  - 検証: {X_val.shape}")
    print(f"  - テスト: {X_test.shape}")

    # 分析実行
    analyzer = OverfittingAnalyzer(
        X_train, y_train, X_val, y_val, X_test, y_test, feature_names
    )

    print("\n[2/7] 分析実行中...")
    all_results = analyzer.run_all_analyses()

    # 結果保存
    print("\n[6/7] 結果をJSON保存中...")
    output_file = Path("data/feature_evaluation/overfitting_analysis_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"保存: {output_file.absolute()}")

    # レポート生成
    print("\n[7/7] Markdownレポート生成中...")
    generate_markdown_report(all_results)

    print("\n" + "=" * 80)
    print("分析完了！")
    print("=" * 80)

    print("\n生成されたファイル:")
    print(f"  - {Path('data/feature_evaluation/overfitting_analysis_results.json').absolute()}")
    print(f"  - {Path('docs/feature_evaluation/OVERFITTING_ANALYSIS_REPORT.md').absolute()}")
    print(f"  - {Path('data/feature_evaluation/plots/overfitting/').absolute()}/*.png (7ファイル)")


if __name__ == "__main__":
    main()