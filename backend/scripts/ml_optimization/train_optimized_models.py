"""
Optunaを使用したLightGBMとXGBoostのハイパーパラメータ最適化スクリプト

このスクリプトは以下を実行します：
1. データベースからOHLCVデータを取得
2. FeatureEngineeringServiceを使用して特徴量を生成
3. ラベル生成（3クラス分類: UP/RANGE/DOWN）
4. Optunaを使用してLightGBMとXGBoostのハイパーパラメータを最適化
5. 最適化されたモデルを保存
6. 最適化プロセスの詳細レポートを出力
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import lightgbm as lgb
import optuna
import pandas as pd
import xgboost as xgb
from optuna.visualization import (
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services.ml.feature_engineering.feature_engineering_service import (  # type: ignore  # noqa: E501
    FeatureEngineeringService,
)
from database.connection import SessionLocal  # type: ignore
from database.repositories.ohlcv_repository import OHLCVRepository  # type: ignore
from scripts.feature_evaluation.common_feature_evaluator import (  # type: ignore
    CommonFeatureEvaluator,
)

# ロギング設定（UTF-8エンコーディングを強制）
logging.basicConfig(
    level=logging.DEBUG,  # DEBUGレベルに変更して診断ログを出力
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("ml_optimization.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ログハンドラのエンコーディングをUTF-8に設定
for handler in logging.root.handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.stream.reconfigure(encoding="utf-8")


class OptunaModelOptimizer:
    """Optunaを使用したモデル最適化クラス"""

    def __init__(
        self,
        symbol: str = "BTC/USDT:USDT",
        timeframe: str = "1h",
        n_trials: int = 50,
        n_jobs: int = 4,
        min_samples: int = 2000,
    ):
        """
        初期化

        Args:
            symbol: 取引ペア
            timeframe: 時間軸
            n_trials: Optuna最適化の試行回数
            n_jobs: 並列実行数
            min_samples: 最小データサンプル数
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.min_samples = min_samples

        self.db = SessionLocal()
        self.ohlcv_repo = OHLCVRepository(self.db)
        self.feature_service = FeatureEngineeringService()
        self.evaluator = CommonFeatureEvaluator()

        # モデル保存ディレクトリ（変更なし）
        self.models_dir = Path(__file__).parent.parent.parent / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # 結果出力ディレクトリ（新規追加）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = (
            Path(__file__).parent.parent.parent
            / "results"
            / "optimization"
            / f"optimization_{timestamp}"
        )
        self.results_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"OptunaModelOptimizer初期化完了: {symbol} {timeframe}")
        logger.info(f"モデル保存先: {self.models_dir}")
        logger.info(f"結果保存先: {self.results_dir}")

    def prepare_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        データ準備: OHLCV取得、特徴量生成、ラベル生成

        Returns:
            Tuple[pd.DataFrame, pd.Series]: 特徴量とラベル
        """
        logger.info("データ準備を開始")

        # OHLCVデータ取得
        data = self.evaluator.fetch_data(
            symbol=self.symbol, timeframe=self.timeframe, limit=10000
        )

        if data.ohlcv.empty:
            raise ValueError("OHLCVデータが空です")

        logger.info(f"OHLCVデータ取得完了: {len(data.ohlcv)}行")

        # 特徴量生成
        logger.info("特徴量生成を開始")
        features_df = self.feature_service.calculate_advanced_features(
            ohlcv_data=data.ohlcv,
            funding_rate_data=data.fr,
            open_interest_data=data.oi,
        )

        logger.info(f"特徴量生成完了: {len(features_df.columns)}個の特徴量")

        # ラベル生成（3クラス分類）
        logger.info("ラベル生成を開始")
        labels = self.evaluator.create_labels_from_config(
            ohlcv_df=data.ohlcv, price_column="close"
        )

        # ラベルの分布を確認
        label_counts = labels.value_counts()
        logger.info(f"ラベル分布:\n{label_counts}")

        # データのアライメント（共通インデックスを使用）
        common_index = features_df.index.intersection(labels.index)
        features_df = features_df.loc[common_index]
        labels = labels.loc[common_index]

        # NaN除去
        valid_idx = ~labels.isna()
        features_df = features_df.loc[valid_idx]
        labels = labels.loc[valid_idx]

        # OHLCV列を除外（closeは保持）
        feature_cols = [
            col
            for col in features_df.columns
            if col not in ["open", "high", "low", "volume"]
        ]
        X = features_df[feature_cols].copy()

        # NaN値の最終確認と処理
        if X.isna().any().any():
            logger.warning("特徴量にNaN値が含まれています。中央値で補完します。")
            X = X.fillna(X.median())

        # ラベルを数値に変換
        label_mapping = {"DOWN": 0, "RANGE": 1, "UP": 2}
        y = labels.map(label_mapping)

        logger.info(f"データ準備完了: X={X.shape}, y={y.shape}")

        # サンプル数チェック
        if len(X) < self.min_samples:
            raise ValueError(
                f"データサンプル数が不足しています: {len(X)} < {self.min_samples}"
            )

        return X, y

    def optimize_lightgbm(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_train: pd.Series,  # type: ignore
        y_val: pd.Series,  # type: ignore
    ) -> Tuple[Dict[str, Any], optuna.Study]:
        """
        LightGBMのハイパーパラメータ最適化

        Args:
            X_train: 訓練データ特徴量
            X_val: 検証データ特徴量
            y_train: 訓練データラベル
            y_val: 検証データラベル

        Returns:
            Tuple[Dict[str, Any], optuna.Study]: 最適なパラメータとStudyオブジェクト
        """
        logger.info("LightGBMハイパーパラメータ最適化を開始")

        def objective(trial: optuna.Trial) -> float:  # type: ignore
            """Optuna目的関数"""
            params = {
                "objective": "multiclass",
                "num_class": 3,
                "metric": "multi_logloss",
                "verbosity": -1,
                "boosting_type": "gbdt",
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "num_leaves": trial.suggest_int("num_leaves", 20, 100),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "random_state": 42,
            }

            # モデル学習
            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
            )

            # 予測と評価
            y_pred = model.predict(X_val)
            f1 = f1_score(y_val, y_pred, average="macro")

            return f1

        # Optuna最適化実行
        study = optuna.create_study(direction="maximize", study_name="lightgbm")
        study.optimize(objective, n_trials=self.n_trials, n_jobs=self.n_jobs)

        logger.info(f"LightGBM最適化完了: ベストF1スコア={study.best_value:.4f}")
        logger.info(f"ベストパラメータ:\n{json.dumps(study.best_params, indent=2)}")

        return study.best_params, study

    def optimize_xgboost(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_train: pd.Series,  # type: ignore
        y_val: pd.Series,  # type: ignore
    ) -> Tuple[Dict[str, Any], optuna.Study]:
        """
        XGBoostのハイパーパラメータ最適化

        Args:
            X_train: 訓練データ特徴量
            X_val: 検証データ特徴量
            y_train: 訓練データラベル
            y_val: 検証データラベル

        Returns:
            Tuple[Dict[str, Any], optuna.Study]: 最適なパラメータとStudyオブジェクト
        """
        logger.info("XGBoostハイパーパラメータ最適化を開始")

        def objective(trial: optuna.Trial) -> float:  # type: ignore
            """Optuna目的関数"""
            params = {
                "objective": "multi:softprob",
                "num_class": 3,
                "eval_metric": "mlogloss",
                "verbosity": 0,
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "gamma": trial.suggest_float("gamma", 0, 5),
                "random_state": 42,
            }

            # モデル学習
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            # 予測と評価
            y_pred = model.predict(X_val)
            f1 = f1_score(y_val, y_pred, average="macro")

            return f1

        # Optuna最適化実行
        study = optuna.create_study(direction="maximize", study_name="xgboost")
        study.optimize(objective, n_trials=self.n_trials, n_jobs=self.n_jobs)

        logger.info(f"XGBoost最適化完了: ベストF1スコア={study.best_value:.4f}")
        logger.info(f"ベストパラメータ:\n{json.dumps(study.best_params, indent=2)}")

        return study.best_params, study

    def train_final_model(
        self,
        model_type: str,
        best_params: Dict[str, Any],
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        最適パラメータで最終モデルを学習

        Args:
            model_type: モデルタイプ（'lightgbm' or 'xgboost'）
            best_params: 最適なハイパーパラメータ
            X_train: 訓練データ特徴量
            X_test: テストデータ特徴量
            y_train: 訓練データラベル
            y_test: テストデータラベル

        Returns:
            Tuple[Any, Dict[str, Any]]: 学習済みモデルと評価指標
        """
        logger.info(f"{model_type.upper()}最終モデルを学習中...")

        if model_type == "lightgbm":
            params = {
                "objective": "multiclass",
                "num_class": 3,
                "metric": "multi_logloss",
                "verbosity": -1,
                "boosting_type": "gbdt",
                "random_state": 42,
                **best_params,
            }
            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_test, y_test)],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
            )

        elif model_type == "xgboost":
            params = {
                "objective": "multi:softprob",
                "num_class": 3,
                "eval_metric": "mlogloss",
                "verbosity": 0,
                "random_state": 42,
                **best_params,
            }
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_test, y_test)],
                verbose=False,
            )

        else:
            raise ValueError(f"サポートされていないモデルタイプ: {model_type}")

        # モデル評価
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        # 評価指標を計算
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average="macro")),
            "recall": float(recall_score(y_test, y_pred, average="macro")),
            "f1_score": float(f1_score(y_test, y_pred, average="macro")),
        }

        # ROC-AUC（マルチクラス: ovr方式）
        try:
            metrics["roc_auc"] = float(
                roc_auc_score(y_test, y_pred_proba, multi_class="ovr", average="macro")
            )
        except Exception as e:
            logger.warning(f"ROC-AUC計算エラー: {e}")
            metrics["roc_auc"] = 0.0

        # 混同行列
        cm = confusion_matrix(y_test, y_pred)
        metrics["confusion_matrix"] = cm.tolist()

        # 分類レポート
        class_names = ["DOWN", "RANGE", "UP"]
        report = classification_report(
            y_test, y_pred, target_names=class_names, output_dict=True
        )
        metrics["classification_report"] = report

        logger.info(f"{model_type.upper()}最終モデル学習完了")
        logger.info(f"評価指標:\n{json.dumps(metrics, indent=2, default=str)}")

        return model, metrics

    def save_model(
        self,
        model: Any,
        model_type: str,
        best_params: Dict[str, Any],
        metrics: Dict[str, Any],
    ) -> None:
        """
        モデルとパラメータを保存

        Args:
            model: 学習済みモデル
            model_type: モデルタイプ
            best_params: 最適なハイパーパラメータ
            metrics: 評価指標
        """
        # モデル保存
        model_path = self.models_dir / f"{model_type}_optimized.joblib"
        joblib.dump(model, model_path)
        logger.info(f"モデルを保存: {model_path}")

        # パラメータ保存（結果ディレクトリに変更）
        params_path = self.results_dir / f"{model_type}_best_params.json"
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(best_params, f, indent=2)
        logger.info(f"パラメータを保存: {params_path}")

        # 評価指標保存（結果ディレクトリに変更）
        metrics_path = self.results_dir / f"{model_type}_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, default=str)
        logger.info(f"評価指標を保存: {metrics_path}")

    def visualize_optimization(
        self,
        study: optuna.Study,
        model_type: str,  # type: ignore
    ) -> None:
        """
        最適化プロセスを可視化

        Args:
            study: Optuna Studyオブジェクト
            model_type: モデルタイプ
        """
        try:
            import matplotlib

            matplotlib.use("Agg")  # GUIなし環境対応

            # 最適化履歴（結果ディレクトリに変更）
            fig1 = plot_optimization_history(study)
            fig1.write_image(
                str(self.results_dir / f"{model_type}_optimization_history.png")
            )
            logger.info(f"{model_type}最適化履歴を保存")

            # パラメータ重要度（結果ディレクトリに変更）
            fig2 = plot_param_importances(study)
            fig2.write_image(
                str(self.results_dir / f"{model_type}_param_importances.png")
            )
            logger.info(f"{model_type}パラメータ重要度を保存")

            # パラレルコーディネート（結果ディレクトリに変更）
            fig3 = plot_parallel_coordinate(study)
            fig3.write_image(
                str(self.results_dir / f"{model_type}_parallel_coordinate.png")
            )
            logger.info(f"{model_type}パラレルコーディネートを保存")

        except ImportError:
            logger.warning("plotlyまたはkaleido未インストール。可視化をスキップ")
        except Exception as e:
            logger.warning(f"可視化エラー: {e}")

    def run(self) -> None:
        """最適化プロセス全体を実行"""
        start_time = time.time()

        try:
            # データ準備
            X, y = self.prepare_data()

            # データ分割（訓練:検証:テスト = 60:20:20）
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
            )

            logger.info("データ分割完了:")
            logger.info(f"  訓練: {X_train.shape}")
            logger.info(f"  検証: {X_val.shape}")
            logger.info(f"  テスト: {X_test.shape}")

            # LightGBM最適化
            logger.info("=" * 80)
            logger.info("LightGBM最適化開始")
            logger.info("=" * 80)
            lgb_best_params, lgb_study = self.optimize_lightgbm(
                X_train, X_val, y_train, y_val
            )

            # LightGBM最終モデル学習
            lgb_model, lgb_metrics = self.train_final_model(
                "lightgbm", lgb_best_params, X_train, X_test, y_train, y_test
            )

            # LightGBM保存
            self.save_model(lgb_model, "lightgbm", lgb_best_params, lgb_metrics)

            # LightGBM可視化
            self.visualize_optimization(lgb_study, "lightgbm")

            # XGBoost最適化
            logger.info("=" * 80)
            logger.info("XGBoost最適化開始")
            logger.info("=" * 80)
            xgb_best_params, xgb_study = self.optimize_xgboost(
                X_train, X_val, y_train, y_val
            )

            # XGBoost最終モデル学習
            xgb_model, xgb_metrics = self.train_final_model(
                "xgboost", xgb_best_params, X_train, X_test, y_train, y_test
            )

            # XGBoost保存
            self.save_model(xgb_model, "xgboost", xgb_best_params, xgb_metrics)

            # XGBoost可視化
            self.visualize_optimization(xgb_study, "xgboost")

            # 最終レポート
            elapsed_time = time.time() - start_time
            logger.info("=" * 80)
            logger.info("最適化完了サマリー")
            logger.info("=" * 80)
            logger.info(f"総実行時間: {elapsed_time:.2f}秒")
            logger.info(f"LightGBM F1スコア: {lgb_metrics['f1_score']:.4f}")
            logger.info(f"XGBoost F1スコア: {xgb_metrics['f1_score']:.4f}")

            # より良いモデルを推奨
            recommended_model = (
                "LightGBM"
                if lgb_metrics["f1_score"] > xgb_metrics["f1_score"]
                else "XGBoost"
            )
            logger.info(f"推奨モデル: {recommended_model}")

            # マークダウンレポート生成
            self._generate_report(
                lgb_best_params,
                lgb_metrics,
                xgb_best_params,
                xgb_metrics,
                elapsed_time,
                recommended_model,
            )

        except Exception as e:
            logger.error(f"最適化プロセスでエラーが発生: {e}", exc_info=True)
            raise

        finally:
            # リソースクリーンアップ
            self.db.close()
            self.evaluator.close()

    def _generate_report(
        self,
        lgb_params: Dict[str, Any],
        lgb_metrics: Dict[str, Any],
        xgb_params: Dict[str, Any],
        xgb_metrics: Dict[str, Any],
        elapsed_time: float,
        recommended_model: str,
    ) -> None:
        """
        最適化レポートをマークダウン形式で生成

        Args:
            lgb_params: LightGBM最適パラメータ
            lgb_metrics: LightGBM評価指標
            xgb_params: XGBoost最適パラメータ
            xgb_metrics: XGBoost評価指標
            elapsed_time: 実行時間
            recommended_model: 推奨モデル
        """
        report_path = self.results_dir / "optimization_report.md"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# ML最適化レポート\n\n")
            f.write(f"**生成日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # 実行設定
            f.write("## 実行設定\n\n")
            f.write(f"- **シンボル**: {self.symbol}\n")
            f.write(f"- **時間軸**: {self.timeframe}\n")
            f.write(f"- **試行回数**: {self.n_trials}\n")
            f.write(f"- **並列実行数**: {self.n_jobs}\n")
            f.write(f"- **総実行時間**: {elapsed_time:.2f}秒\n\n")

            # 推奨モデル
            f.write("## 推奨モデル\n\n")
            f.write(f"**{recommended_model}**\n\n")

            # LightGBM結果
            f.write("## LightGBM結果\n\n")
            f.write("### 評価指標\n\n")
            f.write(f"- **Accuracy**: {lgb_metrics['accuracy']:.4f}\n")
            f.write(f"- **Precision**: {lgb_metrics['precision']:.4f}\n")
            f.write(f"- **Recall**: {lgb_metrics['recall']:.4f}\n")
            f.write(f"- **F1 Score**: {lgb_metrics['f1_score']:.4f}\n")
            if "roc_auc" in lgb_metrics:
                f.write(f"- **ROC-AUC**: {lgb_metrics['roc_auc']:.4f}\n")
            f.write("\n")

            f.write("### 最適パラメータ\n\n")
            f.write("```json\n")
            f.write(json.dumps(lgb_params, indent=2))
            f.write("\n```\n\n")

            # XGBoost結果
            f.write("## XGBoost結果\n\n")
            f.write("### 評価指標\n\n")
            f.write(f"- **Accuracy**: {xgb_metrics['accuracy']:.4f}\n")
            f.write(f"- **Precision**: {xgb_metrics['precision']:.4f}\n")
            f.write(f"- **Recall**: {xgb_metrics['recall']:.4f}\n")
            f.write(f"- **F1 Score**: {xgb_metrics['f1_score']:.4f}\n")
            if "roc_auc" in xgb_metrics:
                f.write(f"- **ROC-AUC**: {xgb_metrics['roc_auc']:.4f}\n")
            f.write("\n")

            f.write("### 最適パラメータ\n\n")
            f.write("```json\n")
            f.write(json.dumps(xgb_params, indent=2))
            f.write("\n```\n\n")

            # 出力ファイル
            f.write("## 出力ファイル\n\n")
            f.write("### モデルファイル\n")
            f.write(f"- `{self.models_dir}/lightgbm_optimized.joblib`\n")
            f.write(f"- `{self.models_dir}/xgboost_optimized.joblib`\n\n")

            f.write("### 結果ファイル\n")
            f.write(f"- `{self.results_dir}/lightgbm_best_params.json`\n")
            f.write(f"- `{self.results_dir}/lightgbm_metrics.json`\n")
            f.write(f"- `{self.results_dir}/xgboost_best_params.json`\n")
            f.write(f"- `{self.results_dir}/xgboost_metrics.json`\n")
            f.write(f"- `{self.results_dir}/optimization_report.md`\n\n")

            f.write("### 可視化ファイル（オプション）\n")
            f.write(f"- `{self.results_dir}/lightgbm_optimization_history.png`\n")
            f.write(f"- `{self.results_dir}/lightgbm_param_importances.png`\n")
            f.write(f"- `{self.results_dir}/lightgbm_parallel_coordinate.png`\n")
            f.write(f"- `{self.results_dir}/xgboost_optimization_history.png`\n")
            f.write(f"- `{self.results_dir}/xgboost_param_importances.png`\n")
            f.write(f"- `{self.results_dir}/xgboost_parallel_coordinate.png`\n")

        logger.info(f"最適化レポートを保存: {report_path}")


def main() -> None:
    """メイン関数"""
    parser = argparse.ArgumentParser(description="OptunaでLightGBMとXGBoostを最適化")
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTC/USDT:USDT",
        help="取引ペア（デフォルト: BTC/USDT:USDT）",
    )
    parser.add_argument(
        "--timeframe", type=str, default="1h", help="時間軸（デフォルト: 1h）"
    )
    parser.add_argument(
        "--n-trials", type=int, default=50, help="Optuna試行回数（デフォルト: 50）"
    )
    parser.add_argument(
        "--n-jobs", type=int, default=4, help="並列実行数（デフォルト: 4）"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=2000,
        help="最小データサンプル数（デフォルト: 2000）",
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("Optunaハイパーパラメータ最適化スクリプト")
    logger.info("=" * 80)
    logger.info("設定:")
    logger.info(f"  シンボル: {args.symbol}")
    logger.info(f"  時間軸: {args.timeframe}")
    logger.info(f"  試行回数: {args.n_trials}")
    logger.info(f"  並列実行数: {args.n_jobs}")
    logger.info(f"  最小サンプル数: {args.min_samples}")

    # 最適化実行
    optimizer = OptunaModelOptimizer(
        symbol=args.symbol,
        timeframe=args.timeframe,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        min_samples=args.min_samples,
    )
    optimizer.run()


if __name__ == "__main__":
    main()
