"""
108特徴量を使用した3モデル（TabNet、XGBoost、LightGBM）の評価スクリプト

合成データを生成し、各モデルをトレーニング・評価して性能を比較します。
"""

import json
import os
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# 定数定義
N_FEATURES = 108
RANDOM_SEED = 42
TRAIN_SIZE = 3000
VAL_SIZE = 600
TEST_SIZE = 600

# 特徴量カテゴリ構成
FEATURE_CATEGORIES = {
    "price": 15,  # 価格特徴量
    "volatility": 5,  # ボラティリティ
    "volume": 7,  # 出来高
    "technical": 22,  # テクニカル指標
    "crypto_specific": 15,  # 暗号通貨特化
    "advanced": 38,  # 高度な特徴量
    "interaction": 6,  # 相互作用
}


class SyntheticDataGenerator:
    """合成データ生成クラス"""

    def __init__(self, n_features: int = N_FEATURES, random_seed: int = RANDOM_SEED):
        """
        初期化

        Args:
            n_features: 特徴量数
            random_seed: 乱数シード
        """
        self.n_features = n_features
        self.random_seed = random_seed
        np.random.seed(random_seed)

    def generate_data(
        self, n_samples: int
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        合成データを生成

        Args:
            n_samples: サンプル数

        Returns:
            特徴量、ターゲット、特徴量名のタプル
        """
        features = []
        feature_names = []
        idx = 0

        # 価格特徴量（正規分布、範囲: -3～3）
        price_features = np.random.randn(n_samples, FEATURE_CATEGORIES["price"]) * 1.5
        features.append(price_features)
        feature_names.extend([f"price_{i}" for i in range(FEATURE_CATEGORIES["price"])])
        idx += FEATURE_CATEGORIES["price"]

        # ボラティリティ（正規分布、範囲: 0～5）
        volatility_features = np.abs(
            np.random.randn(n_samples, FEATURE_CATEGORIES["volatility"]) * 1.5 + 2.5
        )
        features.append(volatility_features)
        feature_names.extend(
            [f"volatility_{i}" for i in range(FEATURE_CATEGORIES["volatility"])]
        )
        idx += FEATURE_CATEGORIES["volatility"]

        # 出来高（正規分布、範囲: 0～10）
        volume_features = np.abs(
            np.random.randn(n_samples, FEATURE_CATEGORIES["volume"]) * 3 + 5
        )
        features.append(volume_features)
        feature_names.extend(
            [f"volume_{i}" for i in range(FEATURE_CATEGORIES["volume"])]
        )
        idx += FEATURE_CATEGORIES["volume"]

        # テクニカル指標（正規分布、範囲: -2～2）
        technical_features = np.random.randn(
            n_samples, FEATURE_CATEGORIES["technical"]
        )
        features.append(technical_features)
        feature_names.extend(
            [f"technical_{i}" for i in range(FEATURE_CATEGORIES["technical"])]
        )
        idx += FEATURE_CATEGORIES["technical"]

        # 暗号通貨特化（正規分布、範囲: -1～1）
        crypto_features = np.random.randn(
            n_samples, FEATURE_CATEGORIES["crypto_specific"]
        ) * 0.5
        features.append(crypto_features)
        feature_names.extend(
            [f"crypto_{i}" for i in range(FEATURE_CATEGORIES["crypto_specific"])]
        )
        idx += FEATURE_CATEGORIES["crypto_specific"]

        # 高度な特徴量（正規分布、範囲: -2～2）
        advanced_features = np.random.randn(n_samples, FEATURE_CATEGORIES["advanced"])
        features.append(advanced_features)
        feature_names.extend(
            [f"advanced_{i}" for i in range(FEATURE_CATEGORIES["advanced"])]
        )
        idx += FEATURE_CATEGORIES["advanced"]

        # 相互作用特徴量（特徴量同士の積や比率）
        interaction_features = np.zeros((n_samples, FEATURE_CATEGORIES["interaction"]))
        interaction_features[:, 0] = price_features[:, 0] * volatility_features[:, 0]
        interaction_features[:, 1] = price_features[:, 1] / (
            volatility_features[:, 1] + 1e-6
        )
        interaction_features[:, 2] = technical_features[:, 0] * volume_features[:, 0]
        interaction_features[:, 3] = (
            crypto_features[:, 0] * advanced_features[:, 0]
        )
        interaction_features[:, 4] = price_features[:, 2] * technical_features[:, 1]
        interaction_features[:, 5] = volume_features[:, 1] * volatility_features[:, 2]
        features.append(interaction_features)
        feature_names.extend(
            [f"interaction_{i}" for i in range(FEATURE_CATEGORIES["interaction"])]
        )

        # 特徴量を結合
        X = np.hstack(features)

        # 非線形な関係を持つターゲットを生成
        # 複数の特徴量の組み合わせで決定
        target_signal = (
            0.3 * price_features[:, 0]
            + 0.2 * technical_features[:, 5]
            - 0.15 * volatility_features[:, 1]
            + 0.1 * interaction_features[:, 0]
            + 0.15 * crypto_features[:, 3]
            + 0.1 * np.sin(advanced_features[:, 10])
        )

        # クラスバランスを約50:50にするため、閾値を中央値に設定
        threshold = np.median(target_signal)
        y = (target_signal > threshold).astype(int)

        return X, y, feature_names


class ModelTrainer:
    """モデルトレーニングクラス"""

    def __init__(self, random_seed: int = RANDOM_SEED):
        """
        初期化

        Args:
            random_seed: 乱数シード
        """
        self.random_seed = random_seed

    def train_tabnet(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Tuple[TabNetClassifier, Dict[str, Any]]:
        """
        TabNetモデルをトレーニング

        Args:
            X_train: トレーニングデータ特徴量
            y_train: トレーニングデータターゲット
            X_val: 検証データ特徴量
            y_val: 検証データターゲット

        Returns:
            トレーニング済みモデルとトレーニング情報
        """
        print("\n=== TabNet トレーニング開始 ===")
        start_time = time.time()

        model = TabNetClassifier(
            n_d=64,
            n_a=64,
            n_steps=5,
            gamma=1.5,
            lambda_sparse=1e-4,
            optimizer_fn=__import__("torch.optim", fromlist=["Adam"]).Adam,
            optimizer_params={"lr": 2e-2},
            scheduler_params={
                "step_size": 10,
                "gamma": 0.9,
            },
            scheduler_fn=__import__(
                "torch.optim.lr_scheduler", fromlist=["StepLR"]
            ).StepLR,
            mask_type="sparsemax",
            seed=self.random_seed,
            verbose=0,
        )

        model.fit(
            X_train=X_train,
            y_train=y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=["auc"],
            max_epochs=100,
            patience=20,
            batch_size=256,
            virtual_batch_size=128,
        )

        training_time = time.time() - start_time
        print(f"トレーニング時間: {training_time:.2f}秒")

        training_info = {
            "training_time": training_time,
            "best_epoch": model.best_epoch,
        }

        return model, training_info

    def train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Tuple[xgb.XGBClassifier, Dict[str, Any]]:
        """
        XGBoostモデルをトレーニング

        Args:
            X_train: トレーニングデータ特徴量
            y_train: トレーニングデータターゲット
            X_val: 検証データ特徴量
            y_val: 検証データターゲット

        Returns:
            トレーニング済みモデルとトレーニング情報
        """
        print("\n=== XGBoost トレーニング開始 ===")
        start_time = time.time()

        model = xgb.XGBClassifier(
            max_depth=6,
            learning_rate=0.1,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_seed,
            tree_method="hist",
            eval_metric="auc",
            early_stopping_rounds=20,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        training_time = time.time() - start_time
        print(f"トレーニング時間: {training_time:.2f}秒")

        training_info = {
            "training_time": training_time,
            "best_iteration": getattr(model, "best_iteration", model.n_estimators),
        }

        return model, training_info

    def train_lightgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Tuple[lgb.LGBMClassifier, Dict[str, Any]]:
        """
        LightGBMモデルをトレーニング

        Args:
            X_train: トレーニングデータ特徴量
            y_train: トレーニングデータターゲット
            X_val: 検証データ特徴量
            y_val: 検証データターゲット

        Returns:
            トレーニング済みモデルとトレーニング情報
        """
        print("\n=== LightGBM トレーニング開始 ===")
        start_time = time.time()

        model = lgb.LGBMClassifier(
            num_leaves=31,
            learning_rate=0.1,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_seed,
            verbose=-1,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
        )

        training_time = time.time() - start_time
        print(f"トレーニング時間: {training_time:.2f}秒")

        training_info = {
            "training_time": training_time,
            "best_iteration": getattr(model, "best_iteration_", model.n_estimators),
        }

        return model, training_info


class ModelEvaluator:
    """モデル評価クラス"""

    def evaluate_model(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str,
        feature_names: List[str],
    ) -> Dict[str, Any]:
        """
        モデルを評価

        Args:
            model: 評価するモデル
            X_test: テストデータ特徴量
            y_test: テストデータターゲット
            model_name: モデル名
            feature_names: 特徴量名リスト

        Returns:
            評価結果の辞書
        """
        print(f"\n=== {model_name} 評価開始 ===")

        # 推論時間測定
        start_time = time.time()
        if model_name == "TabNet":
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)
        else:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)
        inference_time = time.time() - start_time

        # メトリクス計算
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        # PR-AUC計算
        precision_curve, recall_curve, _ = precision_recall_curve(
            y_test, y_pred_proba
        )
        pr_auc = auc(recall_curve, precision_curve)

        # 特徴量重要度取得
        if model_name == "TabNet":
            # TabNetの場合
            importance = model.feature_importances_
        elif model_name == "XGBoost":
            # XGBoostの場合
            importance = model.feature_importances_
        else:  # LightGBM
            # LightGBMの場合
            importance = model.feature_importances_

        # 上位20特徴量
        top_indices = np.argsort(importance)[::-1][:20]
        top_features = {
            feature_names[i]: float(importance[i]) for i in top_indices
        }

        results = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "inference_time": float(inference_time),
            "inference_time_per_sample": float(inference_time / len(X_test)),
            "top_20_features": top_features,
            "predictions": {
                "y_pred": y_pred.tolist() if hasattr(y_pred, "tolist") else y_pred,
                "y_pred_proba": y_pred_proba.tolist(),
            },
        }

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"PR-AUC: {pr_auc:.4f}")
        print(f"推論時間: {inference_time:.4f}秒")

        return results


class Visualizer:
    """視覚化クラス"""

    def __init__(self, output_dir: str = "data/feature_evaluation/plots/evaluation"):
        """
        初期化

        Args:
            output_dir: 出力ディレクトリ
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_metrics_comparison(self, results: Dict[str, Dict[str, Any]]) -> None:
        """
        メトリクス比較棒グラフを作成

        Args:
            results: 各モデルの評価結果
        """
        metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc"]
        models = list(results.keys())

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("Model Performance Comparison", fontsize=16)

        for idx, metric in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]
            values = [results[model][metric] for model in models]
            bars = ax.bar(models, values)
            ax.set_ylabel(metric.upper().replace("_", " "))
            ax.set_ylim([0, 1])
            ax.grid(axis="y", alpha=0.3)

            # 値をバーの上に表示
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                )

        plt.tight_layout()
        plt.savefig(self.output_dir / "metrics_comparison.png", dpi=300)
        print(f"メトリクス比較グラフを保存: {self.output_dir / 'metrics_comparison.png'}")
        plt.close()

    def plot_roc_curves(
        self,
        y_test: np.ndarray,
        results: Dict[str, Dict[str, Any]],
    ) -> None:
        """
        ROC曲線を作成

        Args:
            y_test: テストデータターゲット
            results: 各モデルの評価結果
        """
        plt.figure(figsize=(10, 8))

        for model_name, result in results.items():
            y_pred_proba = np.array(result["predictions"]["y_pred_proba"])
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = result["roc_auc"]
            plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.3f})", linewidth=2)

        plt.plot([0, 1], [0, 1], "k--", label="Random (AUC = 0.500)")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.title("ROC Curves Comparison", fontsize=14)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "roc_curves.png", dpi=300)
        print(f"ROC曲線を保存: {self.output_dir / 'roc_curves.png'}")
        plt.close()

    def plot_pr_curves(
        self,
        y_test: np.ndarray,
        results: Dict[str, Dict[str, Any]],
    ) -> None:
        """
        PR曲線を作成

        Args:
            y_test: テストデータターゲット
            results: 各モデルの評価結果
        """
        plt.figure(figsize=(10, 8))

        for model_name, result in results.items():
            y_pred_proba = np.array(result["predictions"]["y_pred_proba"])
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            pr_auc = result["pr_auc"]
            plt.plot(
                recall, precision, label=f"{model_name} (AUC = {pr_auc:.3f})", linewidth=2
            )

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall", fontsize=12)
        plt.ylabel("Precision", fontsize=12)
        plt.title("Precision-Recall Curves Comparison", fontsize=14)
        plt.legend(loc="best", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "pr_curves.png", dpi=300)
        print(f"PR曲線を保存: {self.output_dir / 'pr_curves.png'}")
        plt.close()

    def plot_feature_importance(
        self, results: Dict[str, Dict[str, Any]], top_n: int = 20
    ) -> None:
        """
        特徴量重要度グラフを作成

        Args:
            results: 各モデルの評価結果
            top_n: 表示する上位特徴量数
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"Top {top_n} Feature Importance by Model", fontsize=16)

        for idx, (model_name, result) in enumerate(results.items()):
            ax = axes[idx]
            top_features = result["top_20_features"]
            features = list(top_features.keys())
            importances = list(top_features.values())

            ax.barh(range(len(features)), importances)
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features, fontsize=8)
            ax.set_xlabel("Importance", fontsize=10)
            ax.set_title(model_name, fontsize=12)
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "feature_importance.png", dpi=300)
        print(f"特徴量重要度グラフを保存: {self.output_dir / 'feature_importance.png'}")
        plt.close()


def save_results(
    results: Dict[str, Any], output_path: str = "data/feature_evaluation/model_evaluation_results.json"
) -> None:
    """
    結果をJSONファイルに保存

    Args:
        results: 評価結果
        output_path: 出力ファイルパス
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n結果をJSONファイルに保存: {output_file.absolute()}")


def create_markdown_report(
    results: Dict[str, Any], output_path: str = "docs/feature_evaluation/MODEL_EVALUATION_REPORT.md"
) -> None:
    """
    評価レポートをMarkdownファイルに作成

    Args:
        results: 評価結果
        output_path: 出力ファイルパス
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# モデル評価レポート\n\n")
        f.write("## 概要\n\n")
        f.write(
            f"- 特徴量数: {N_FEATURES}\n"
            f"- トレーニングサンプル数: {TRAIN_SIZE}\n"
            f"- 検証サンプル数: {VAL_SIZE}\n"
            f"- テストサンプル数: {TEST_SIZE}\n"
            f"- 乱数シード: {RANDOM_SEED}\n\n"
        )

        f.write("## 特徴量カテゴリ構成\n\n")
        f.write("| カテゴリ | 特徴量数 |\n")
        f.write("|---------|----------|\n")
        for category, count in FEATURE_CATEGORIES.items():
            f.write(f"| {category} | {count} |\n")
        f.write(f"| **合計** | **{N_FEATURES}** |\n\n")

        f.write("## モデル性能比較\n\n")
        f.write("### メトリクス比較表\n\n")
        f.write(
            "| モデル | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |\n"
        )
        f.write("|--------|----------|-----------|--------|----------|---------|--------|\n")
        for model_name, result in results.items():
            if model_name != "metadata":
                f.write(
                    f"| {model_name} | "
                    f"{result['accuracy']:.4f} | "
                    f"{result['precision']:.4f} | "
                    f"{result['recall']:.4f} | "
                    f"{result['f1_score']:.4f} | "
                    f"{result['roc_auc']:.4f} | "
                    f"{result['pr_auc']:.4f} |\n"
                )
        f.write("\n")

        f.write("### トレーニング時間とメモリ効率\n\n")
        f.write("| モデル | トレーニング時間(秒) | 推論時間(秒) | サンプルあたり推論時間(ms) |\n")
        f.write("|--------|---------------------|--------------|---------------------------|\n")
        for model_name, result in results.items():
            if model_name != "metadata":
                f.write(
                    f"| {model_name} | "
                    f"{result['training_time']:.2f} | "
                    f"{result['inference_time']:.4f} | "
                    f"{result['inference_time_per_sample']*1000:.2f} |\n"
                )
        f.write("\n")

        f.write("## 主要な発見\n\n")
        f.write("### 各モデルの上位5特徴量\n\n")
        for model_name, result in results.items():
            if model_name != "metadata":
                f.write(f"#### {model_name}\n\n")
                top_5 = list(result["top_20_features"].items())[:5]
                for i, (feature, importance) in enumerate(top_5, 1):
                    f.write(f"{i}. {feature}: {importance:.6f}\n")
                f.write("\n")

        f.write("## 108特徴量の適切性評価\n\n")

        # 最高性能のモデルを特定
        best_model = max(
            [m for m in results.keys() if m != "metadata"],
            key=lambda m: results[m]["roc_auc"],
        )
        best_auc = results[best_model]["roc_auc"]

        f.write(f"### 総合評価\n\n")
        f.write(f"- **最高性能モデル**: {best_model}\n")
        f.write(f"- **最高ROC-AUC**: {best_auc:.4f}\n\n")

        if best_auc > 0.85:
            f.write(
                "✅ **評価**: 108特徴量は優れた予測性能を示しています。"
                "特徴量セットは適切に設計されており、モデルは効果的に学習できています。\n\n"
            )
        elif best_auc > 0.75:
            f.write(
                "⚠️ **評価**: 108特徴量は良好な予測性能を示していますが、"
                "改善の余地があります。特徴量エンジニアリングの最適化を検討してください。\n\n"
            )
        else:
            f.write(
                "❌ **評価**: 108特徴量の予測性能は期待以下です。"
                "特徴量の見直しや追加のデータ収集を検討する必要があります。\n\n"
            )

        f.write("### 推奨事項\n\n")
        f.write(
            "1. 特徴量重要度の低い特徴量を削除し、モデルを単純化することを検討\n"
        )
        f.write("2. 上位の重要な特徴量に焦点を当てた特徴量エンジニアリングを実施\n")
        f.write("3. 相互作用特徴量をさらに追加して非線形な関係を捉える\n")
        f.write("4. ドメイン知識を活用した特徴量選択とフィルタリング\n\n")

        f.write("## 生成ファイル\n\n")
        f.write("- `data/feature_evaluation/model_evaluation_results.json`: 詳細な評価結果データ\n")
        f.write("- `data/feature_evaluation/plots/evaluation/metrics_comparison.png`: メトリクス比較グラフ\n")
        f.write("- `data/feature_evaluation/plots/evaluation/roc_curves.png`: ROC曲線比較\n")
        f.write("- `data/feature_evaluation/plots/evaluation/pr_curves.png`: PR曲線比較\n")
        f.write("- `data/feature_evaluation/plots/evaluation/feature_importance.png`: 特徴量重要度グラフ\n")

    print(f"Markdownレポートを保存: {output_file.absolute()}")


def main() -> None:
    """メイン実行関数"""
    print("=" * 80)
    print("108特徴量を使用した3モデル評価スクリプト")
    print("=" * 80)

    # データ生成
    print("\n[1/6] 合成データ生成中...")
    generator = SyntheticDataGenerator()
    total_samples = TRAIN_SIZE + VAL_SIZE + TEST_SIZE
    X, y, feature_names = generator.generate_data(total_samples)

    # データ分割
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=VAL_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_temp,
    )

    print(f"データ形状:")
    print(f"  - トレーニング: {X_train.shape}")
    print(f"  - 検証: {X_val.shape}")
    print(f"  - テスト: {X_test.shape}")
    print(f"  - クラス分布（テスト）: {np.bincount(y_test)}")

    # モデルトレーニング
    trainer = ModelTrainer()
    evaluator = ModelEvaluator()

    results = {}

    # TabNet
    print("\n[2/6] TabNetトレーニング...")
    tabnet_model, tabnet_info = trainer.train_tabnet(X_train, y_train, X_val, y_val)
    print("\n[3/6] TabNet評価...")
    tabnet_results = evaluator.evaluate_model(
        tabnet_model, X_test, y_test, "TabNet", feature_names
    )
    tabnet_results.update(tabnet_info)
    results["TabNet"] = tabnet_results

    # XGBoost
    print("\n[4/6] XGBoostトレーニングと評価...")
    xgb_model, xgb_info = trainer.train_xgboost(X_train, y_train, X_val, y_val)
    xgb_results = evaluator.evaluate_model(
        xgb_model, X_test, y_test, "XGBoost", feature_names
    )
    xgb_results.update(xgb_info)
    results["XGBoost"] = xgb_results

    # LightGBM
    print("\n[5/6] LightGBMトレーニングと評価...")
    lgb_model, lgb_info = trainer.train_lightgbm(X_train, y_train, X_val, y_val)
    lgb_results = evaluator.evaluate_model(
        lgb_model, X_test, y_test, "LightGBM", feature_names
    )
    lgb_results.update(lgb_info)
    results["LightGBM"] = lgb_results

    # メタデータ追加
    results["metadata"] = {
        "n_features": N_FEATURES,
        "train_size": TRAIN_SIZE,
        "val_size": VAL_SIZE,
        "test_size": TEST_SIZE,
        "random_seed": RANDOM_SEED,
        "feature_categories": FEATURE_CATEGORIES,
    }

    # 結果保存
    print("\n[6/6] 結果保存と視覚化...")
    save_results(results)
    create_markdown_report(results)

    # 視覚化
    visualizer = Visualizer()
    visualizer.plot_metrics_comparison(
        {k: v for k, v in results.items() if k != "metadata"}
    )
    visualizer.plot_roc_curves(
        y_test, {k: v for k, v in results.items() if k != "metadata"}
    )
    visualizer.plot_pr_curves(
        y_test, {k: v for k, v in results.items() if k != "metadata"}
    )
    visualizer.plot_feature_importance(
        {k: v for k, v in results.items() if k != "metadata"}
    )

    print("\n" + "=" * 80)
    print("評価完了！")
    print("=" * 80)
    print("\n生成されたファイル:")
    print(f"  - {Path('data/feature_evaluation/model_evaluation_results.json').absolute()}")
    print(f"  - {Path('docs/feature_evaluation/MODEL_EVALUATION_REPORT.md').absolute()}")
    print(f"  - {Path('data/feature_evaluation/plots/evaluation/metrics_comparison.png').absolute()}")
    print(f"  - {Path('data/feature_evaluation/plots/evaluation/roc_curves.png').absolute()}")
    print(f"  - {Path('data/feature_evaluation/plots/evaluation/pr_curves.png').absolute()}")
    print(f"  - {Path('data/feature_evaluation/plots/evaluation/feature_importance.png').absolute()}")


if __name__ == "__main__":
    main()