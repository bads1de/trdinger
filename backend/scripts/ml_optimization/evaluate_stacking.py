"""
スタッキングアンサンブル専用評価スクリプト（2値分類: TREND vs RANGE）

このスクリプトは以下を実行します：
1. データベースからOHLCVデータを取得
2. FeatureEngineeringServiceを使用して特徴量を生成
3. ラベル生成（2値分類: TREND vs RANGE）
   - UP/DOWN → TREND (1)
   - RANGE → RANGE (0)
4. スタッキングアンサンブルモデルの学習と評価
5. ベースモデル（LightGBM, XGBoost）と比較
6. 詳細な評価レポートを出力

スタッキングの構成:
- ベースモデル: LightGBM, XGBoost
- メタモデル: LogisticRegression
- CV: StratifiedKFold (5-fold)
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import catboost as cb
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold

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
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("stacking_evaluation.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ログハンドラのエンコーディングをUTF-8に設定
for handler in logging.root.handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.stream.reconfigure(encoding="utf-8")


class StackingEvaluator:
    """スタッキングアンサンブル専用評価クラス"""

    def __init__(
        self,
        symbol: str = "BTC/USDT:USDT",
        timeframe: str = "1h",
        min_samples: int = 2000,
        cv_folds: int = 5,
        use_optuna_params: bool = False,
        use_class_weight: bool = True,
    ):
        """
        初期化

        Args:
            symbol: 取引ペア
            timeframe: 時間軸
            min_samples: 最小データサンプル数
            cv_folds: クロスバリデーション分割数
            use_optuna_params: Optunaで最適化されたパラメータを使用するか
            use_class_weight: クラスウェイトを使用するか（デフォルト: True）
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.min_samples = min_samples
        self.cv_folds = cv_folds
        self.use_optuna_params = use_optuna_params
        self.use_class_weight = use_class_weight

        self.db = SessionLocal()
        self.ohlcv_repo = OHLCVRepository(self.db)
        self.feature_service = FeatureEngineeringService()
        self.evaluator = CommonFeatureEvaluator()

        # モデル保存ディレクトリ
        self.models_dir = Path(__file__).parent.parent.parent / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # 結果出力ディレクトリ
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = (
            Path(__file__).parent.parent.parent
            / "results"
            / "stacking_evaluation"
            / f"stacking_eval_{timestamp}"
        )
        self.results_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"StackingEvaluator初期化完了: {symbol} {timeframe}")
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

        # ラベル生成（3クラス分類から2値分類へ変換: TREND vs RANGE）
        logger.info("ラベル生成を開始（2値分類: TREND vs RANGE）")
        labels = self.evaluator.create_labels_from_config(
            ohlcv_df=data.ohlcv, price_column="close"
        )

        # ラベルの分布を確認（変換前）
        label_counts = labels.value_counts()
        logger.info(f"元のラベル分布（3クラス）:\n{label_counts}")

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

        # ラベルを数値に変換（2値分類: TREND vs RANGE）
        # UP/DOWN → TREND(1)
        # RANGE → RANGE(0)
        label_mapping = {"DOWN": 1, "RANGE": 0, "UP": 1}
        y = labels.map(label_mapping)

        logger.info(f"2値分類ラベル分布（TREND=1, RANGE=0）:\n{y.value_counts()}")
        logger.info(f"データ準備完了: X={X.shape}, y={y.shape}")

        # サンプル数チェック
        if len(X) < self.min_samples:
            raise ValueError(
                f"データサンプル数が不足しています: {len(X)} < {self.min_samples}"
            )

        return X, y

    def get_base_model_params(self, model_type: str) -> Dict[str, Any]:
        """
        ベースモデルのパラメータを取得

        Args:
            model_type: モデルタイプ（'lightgbm' or 'xgboost'）

        Returns:
            Dict[str, Any]: モデルパラメータ
        """
        if self.use_optuna_params:
            # Optunaで最適化されたパラメータを読み込む
            params_path = self.models_dir / f"{model_type}_best_params.json"
            if params_path.exists():
                with open(params_path, "r", encoding="utf-8") as f:
                    params = json.load(f)
                logger.info(f"{model_type}の最適化パラメータを読み込みました")
                return params
            else:
                logger.warning(
                    f"{model_type}の最適化パラメータが見つかりません。デフォルトパラメータを使用します。"
                )

        # デフォルトパラメータ
        if model_type == "lightgbm":
            params = {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 10,
                "num_leaves": 31,
                "min_child_samples": 20,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "verbose": -1,
            }
            # クラスウェイトを適用
            if self.use_class_weight:
                params["class_weight"] = "balanced"
                logger.info(f"{model_type}: class_weight='balanced' を適用")
            return params
        elif model_type == "xgboost":
            params = {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 6,
                "min_child_weight": 1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "verbosity": 0,
            }
            # XGBoostの場合はscale_pos_weightで対応
            # （学習時に動的に計算する必要があるためここでは設定しない）
            if self.use_class_weight:
                logger.info(f"{model_type}: scale_pos_weight を学習時に設定します")
            return params
        elif model_type == "catboost":
            params = {
                "iterations": 100,
                "learning_rate": 0.1,
                "depth": 6,
                "l2_leaf_reg": 3.0,
                "random_seed": 42,
                "verbose": 0,
                "allow_writing_files": False,  # 一時ファイル作成を無効化
            }
            # CatBoostのクラスウェイト設定
            if self.use_class_weight:
                # CatBoostはauto_class_weightsパラメータを使用
                params["auto_class_weights"] = "Balanced"
                logger.info(f"{model_type}: auto_class_weights='Balanced' を適用")
            return params
        else:
            raise ValueError(f"サポートされていないモデルタイプ: {model_type}")

    def create_stacking_classifier(self) -> StackingClassifier:
        """
        スタッキングアンサンブルモデルを作成

        Returns:
            StackingClassifier: スタッキングモデル
        """
        logger.info("スタッキングアンサンブルモデルを作成中...")

        # ベースモデルのパラメータを取得
        lgb_params = self.get_base_model_params("lightgbm")
        xgb_params = self.get_base_model_params("xgboost")
        cb_params = self.get_base_model_params("catboost")

        # ベースモデルを作成
        estimators = [
            ("lightgbm", lgb.LGBMClassifier(**lgb_params)),
            ("xgboost", xgb.XGBClassifier(**xgb_params)),
            ("catboost", cb.CatBoostClassifier(**cb_params)),
        ]

        # メタモデル（LogisticRegression）
        final_estimator = LogisticRegression(
            max_iter=1000,
            random_state=42,
            solver="saga",
            penalty="l2",
            C=1.0,
        )

        # クロスバリデーション設定
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

        # スタッキングアンサンブル
        stacking_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=cv,
            stack_method="predict_proba",
            n_jobs=-1,
            passthrough=False,  # 元特徴量をメタモデルに渡さない
            verbose=0,
        )

        logger.info("スタッキングアンサンブルモデル作成完了")
        logger.info(f"  ベースモデル: {[name for name, _ in estimators]}")
        logger.info(
            f"  メタモデル: LogisticRegression (solver=saga, C=1.0, penalty=l2)"
        )
        logger.info(f"  CV分割数: {self.cv_folds}")

        return stacking_clf

    def evaluate_model(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str,
    ) -> Dict[str, Any]:
        """
        モデルを評価

        Args:
            model: 学習済みモデル
            X_test: テストデータ特徴量
            y_test: テストデータラベル
            model_name: モデル名

        Returns:
            Dict[str, Any]: 評価指標
        """
        logger.info(f"{model_name}を評価中...")

        # 予測
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        # 評価指標を計算（2値分類用）
        metrics = {
            "model_name": model_name,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),  # binary
            "recall": float(recall_score(y_test, y_pred)),  # binary
            "f1_score": float(f1_score(y_test, y_pred)),  # binary
        }

        # 混同行列
        cm = confusion_matrix(y_test, y_pred)
        metrics["confusion_matrix"] = cm.tolist()

        # 分類レポート（2値分類用）
        class_names = ["RANGE", "TREND"]
        report = classification_report(
            y_test, y_pred, target_names=class_names, output_dict=True
        )
        metrics["classification_report"] = report

        logger.info(f"{model_name}評価完了:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  F1 Score: {metrics['f1_score']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")

        return metrics

    def train_and_evaluate_single_model(
        self,
        model_type: str,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        単一モデルを学習・評価（比較用）

        Args:
            model_type: モデルタイプ（'lightgbm' or 'xgboost'）
            X_train: 訓練データ特徴量
            X_test: テストデータ特徴量
            y_train: 訓練データラベル
            y_test: テストデータラベル

        Returns:
            Tuple[Any, Dict[str, Any]]: 学習済みモデルと評価指標
        """
        logger.info(f"{model_type.upper()}単一モデルを学習中...")

        params = self.get_base_model_params(model_type)

        if model_type == "lightgbm":
            model = lgb.LGBMClassifier(**params)
        elif model_type == "xgboost":
            # XGBoostの場合、scale_pos_weightを動的に計算
            if self.use_class_weight:
                # クラス数をカウント
                n_negative = (y_train == 0).sum()  # RANGE
                n_positive = (y_train == 1).sum()  # TREND
                scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1.0
                params["scale_pos_weight"] = scale_pos_weight
                logger.info(f"XGBoost: scale_pos_weight={scale_pos_weight:.4f} を設定")
            model = xgb.XGBClassifier(**params)
        elif model_type == "catboost":
            # CatBoostは既にauto_class_weightsが設定済み
            model = cb.CatBoostClassifier(**params)
        else:
            raise ValueError(f"サポートされていないモデルタイプ: {model_type}")

        # 学習
        model.fit(X_train, y_train)

        # 評価
        metrics = self.evaluate_model(model, X_test, y_test, model_type.upper())

        return model, metrics

    def run(self) -> None:
        """評価プロセス全体を実行"""
        start_time = time.time()

        try:
            # データ準備
            X, y = self.prepare_data()

            # 時系列分割（先読みバイアス防止）
            logger.info("時系列データ分割を開始（先読みバイアス防止）")
            total_samples = len(X)
            test_size = int(total_samples * 0.2)

            X_train = X.iloc[:-test_size]
            X_test = X.iloc[-test_size:]

            y_train = y.iloc[:-test_size]
            y_test = y.iloc[-test_size:]

            logger.info("データ分割完了（時系列順）:")
            logger.info(f"  訓練: {X_train.shape}")
            logger.info(f"  テスト: {X_test.shape}")

            # スタッキングアンサンブル学習
            logger.info("=" * 80)
            logger.info("スタッキングアンサンブル学習開始")
            logger.info("=" * 80)

            stacking_clf = self.create_stacking_classifier()
            stacking_clf.fit(X_train, y_train)

            logger.info("スタッキングアンサンブル学習完了")

            # スタッキングアンサンブル評価
            stacking_metrics = self.evaluate_model(
                stacking_clf, X_test, y_test, "Stacking Ensemble"
            )

            # LightGBM単体評価（比較用）
            logger.info("=" * 80)
            logger.info("LightGBM単体モデル学習・評価")
            logger.info("=" * 80)

            lgb_model, lgb_metrics = self.train_and_evaluate_single_model(
                "lightgbm", X_train, X_test, y_train, y_test
            )

            # XGBoost単体評価（比較用）
            logger.info("=" * 80)
            logger.info("XGBoost単体モデル学習・評価")
            logger.info("=" * 80)

            xgb_model, xgb_metrics = self.train_and_evaluate_single_model(
                "xgboost", X_train, X_test, y_train, y_test
            )

            # CatBoost単体評価（比較用）
            logger.info("=" * 80)
            logger.info("CatBoost単体モデル学習・評価")
            logger.info("=" * 80)

            cb_model, cb_metrics = self.train_and_evaluate_single_model(
                "catboost", X_train, X_test, y_train, y_test
            )

            # モデル保存
            logger.info("モデルを保存中...")
            stacking_path = self.models_dir / "stacking_ensemble.joblib"
            joblib.dump(stacking_clf, stacking_path)
            logger.info(f"スタッキングモデルを保存: {stacking_path}")

            # 結果を保存
            results = {
                "stacking": stacking_metrics,
                "lightgbm": lgb_metrics,
                "xgboost": xgb_metrics,
                "catboost": cb_metrics,
            }

            results_path = self.results_dir / "evaluation_results.json"
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"評価結果を保存: {results_path}")

            # 最終レポート
            elapsed_time = time.time() - start_time
            logger.info("=" * 80)
            logger.info("評価完了サマリー")
            logger.info("=" * 80)
            logger.info(f"総実行時間: {elapsed_time:.2f}秒")
            logger.info("")
            logger.info("F1 Score 比較（2値分類: TREND vs RANGE）:")
            logger.info(
                f"  Stacking Ensemble: {stacking_metrics['f1_score']:.4f} ⭐"
            )
            logger.info(f"  LightGBM:          {lgb_metrics['f1_score']:.4f}")
            logger.info(f"  XGBoost:           {xgb_metrics['f1_score']:.4f}")
            logger.info(f"  CatBoost:          {cb_metrics['f1_score']:.4f}")
            logger.info("")

            # スタッキングの優位性を計算
            improvement_vs_lgb = (
                stacking_metrics["f1_score"] - lgb_metrics["f1_score"]
            ) * 100
            improvement_vs_xgb = (
                stacking_metrics["f1_score"] - xgb_metrics["f1_score"]
            ) * 100
            improvement_vs_cb = (
                stacking_metrics["f1_score"] - cb_metrics["f1_score"]
            ) * 100

            logger.info("スタッキングの改善率:")
            logger.info(f"  vs LightGBM: {improvement_vs_lgb:+.2f}%")
            logger.info(f"  vs XGBoost:  {improvement_vs_xgb:+.2f}%")
            logger.info(f"  vs CatBoost: {improvement_vs_cb:+.2f}%")

            # マークダウンレポート生成
            self._generate_report(
                stacking_metrics,
                lgb_metrics,
                xgb_metrics,
                cb_metrics,
                elapsed_time,
                improvement_vs_lgb,
                improvement_vs_xgb,
                improvement_vs_cb,
            )

        except Exception as e:
            logger.error(f"評価プロセスでエラーが発生: {e}", exc_info=True)
            raise

        finally:
            # リソースクリーンアップ
            self.db.close()
            self.evaluator.close()

    def _generate_report(
        self,
        stacking_metrics: Dict[str, Any],
        lgb_metrics: Dict[str, Any],
        xgb_metrics: Dict[str, Any],
        cb_metrics: Dict[str, Any],
        elapsed_time: float,
        improvement_vs_lgb: float,
        improvement_vs_xgb: float,
        improvement_vs_cb: float,
    ) -> None:
        """
        評価レポートをマークダウン形式で生成

        Args:
            stacking_metrics: スタッキング評価指標
            lgb_metrics: LightGBM評価指標
            xgb_metrics: XGBoost評価指標
            cb_metrics: CatBoost評価指標
            elapsed_time: 実行時間
            improvement_vs_lgb: LightGBMに対する改善率
            improvement_vs_xgb: XGBoostに対する改善率
            improvement_vs_cb: CatBoostに対する改善率
        """
        report_path = self.results_dir / "stacking_evaluation_report.md"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# スタッキングアンサンブル評価レポート（2値分類: TREND vs RANGE）\n\n")
            f.write(f"**生成日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # 実行設定
            f.write("## 実行設定\n\n")
            f.write(f"- **シンボル**: {self.symbol}\n")
            f.write(f"- **時間軸**: {self.timeframe}\n")
            f.write(f"- **CV分割数**: {self.cv_folds}\n")
            f.write(
                f"- **Optunaパラメータ使用**: {'はい' if self.use_optuna_params else 'いいえ'}\n"
            )
            f.write(f"- **総実行時間**: {elapsed_time:.2f}秒\n\n")

            # スタッキング構成
            f.write("## スタッキング構成\n\n")
            f.write("- **ベースモデル**: LightGBM, XGBoost, CatBoost\n")
            f.write("- **メタモデル**: LogisticRegression (solver=saga, C=1.0)\n")
            f.write("- **スタッキング方法**: predict_proba\n")
            f.write("- **Passthrough**: False（元特徴量は使用しない）\n\n")

            # 結果比較
            f.write("## 評価結果比較\n\n")
            f.write("### F1 Score（2値分類）\n\n")
            f.write("| モデル | F1 Score | 改善率 |\n")
            f.write("|--------|----------|--------|\n")
            f.write(
                f"| **Stacking Ensemble** | **{stacking_metrics['f1_score']:.4f}** | - |\n"
            )
            f.write(
                f"| LightGBM | {lgb_metrics['f1_score']:.4f} | {improvement_vs_lgb:+.2f}% |\n"
            )
            f.write(
                f"| XGBoost | {xgb_metrics['f1_score']:.4f} | {improvement_vs_xgb:+.2f}% |\n"
            )
            f.write(
                f"| CatBoost | {cb_metrics['f1_score']:.4f} | {improvement_vs_cb:+.2f}% |\n\n"
            )

            f.write("### その他の指標\n\n")
            f.write("| モデル | Accuracy | Precision | Recall |\n")
            f.write("|--------|----------|-----------|--------|\n")
            f.write(
                f"| **Stacking** | {stacking_metrics['accuracy']:.4f} | {stacking_metrics['precision']:.4f} | {stacking_metrics['recall']:.4f} |\n"
            )
            f.write(
                f"| LightGBM | {lgb_metrics['accuracy']:.4f} | {lgb_metrics['precision']:.4f} | {lgb_metrics['recall']:.4f} |\n"
            )
            f.write(
                f"| XGBoost | {xgb_metrics['accuracy']:.4f} | {xgb_metrics['precision']:.4f} | {xgb_metrics['recall']:.4f} |\n"
            )
            f.write(
                f"| CatBoost | {cb_metrics['accuracy']:.4f} | {cb_metrics['precision']:.4f} | {cb_metrics['recall']:.4f} |\n\n"
            )

            # 詳細な分類レポート
            f.write("## スタッキングアンサンブル詳細レポート\n\n")
            f.write("### クラス別評価（2値分類）\n\n")
            f.write("| クラス | Precision | Recall | F1-Score | Support |\n")
            f.write("|--------|-----------|--------|----------|----------|\n")

            report = stacking_metrics["classification_report"]
            for class_name in ["RANGE", "TREND"]:
                class_data = report[class_name]
                f.write(
                    f"| {class_name} | {class_data['precision']:.4f} | {class_data['recall']:.4f} | {class_data['f1-score']:.4f} | {int(class_data['support'])} |\n"
                )

            f.write("\n### 混同行列（2値分類）\n\n")
            f.write("```\n")
            cm = np.array(stacking_metrics["confusion_matrix"])
            f.write("           Predicted\n")
            f.write("           RANGE  TREND\n")
            f.write("Actual RANGE  {:4d}   {:4d}\n".format(*cm[0]))
            f.write("       TREND  {:4d}   {:4d}\n".format(*cm[1]))
            f.write("```\n\n")

            # 結論
            f.write("## 結論\n\n")

            if improvement_vs_lgb > 0 and improvement_vs_xgb > 0:
                f.write(
                    "✅ **スタッキングアンサンブルは両方の単一モデルを上回りました！**\n\n"
                )
                f.write(
                    f"- LightGBMに対して {improvement_vs_lgb:.2f}% の改善\n"
                )
                f.write(f"- XGBoostに対して {improvement_vs_xgb:.2f}% の改善\n\n")
                f.write(
                    "スタッキングの利点が確認できました。複数モデルの強みを組み合わせることで、より堅牢な予測が可能になっています。\n"
                )
            elif improvement_vs_lgb > 0 or improvement_vs_xgb > 0:
                f.write("⚠️ **スタッキングアンサンブルは部分的に改善を示しました。**\n\n")
                f.write("一部のモデルを上回っていますが、全体的な改善は限定的です。\n")
                f.write("ベースモデルのパラメータ調整や特徴量エンジニアリングの改善を検討してください。\n")
            else:
                f.write("❌ **スタッキングアンサンブルは期待した改善を示しませんでした。**\n\n")
                f.write("考えられる原因:\n")
                f.write("- ベースモデルが十分に多様でない可能性\n")
                f.write("- メタモデルのパラメータ調整が必要\n")
                f.write("- データの質や量が不十分\n")

            # 出力ファイル
            f.write("\n## 出力ファイル\n\n")
            f.write("### モデルファイル\n")
            f.write(f"- `{self.models_dir}/stacking_ensemble.joblib`\n\n")

            f.write("### 結果ファイル\n")
            f.write(f"- `{self.results_dir}/evaluation_results.json`\n")
            f.write(f"- `{self.results_dir}/stacking_evaluation_report.md`\n")

        logger.info(f"評価レポートを保存: {report_path}")


def main() -> None:
    """メイン関数"""
    parser = argparse.ArgumentParser(description="スタッキングアンサンブル専用評価スクリプト")
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
        "--min-samples",
        type=int,
        default=2000,
        help="最小データサンプル数（デフォルト: 2000）",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="クロスバリデーション分割数（デフォルト: 5）",
    )
    parser.add_argument(
        "--use-optuna-params",
        action="store_true",
        help="Optunaで最適化されたパラメータを使用する",
    )
    parser.add_argument(
        "--use-class-weight",
        action="store_true",
        default=True,
        help="クラスウェイト（balanced）を使用する（デフォルト: True）",
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("スタッキングアンサンブル専用評価スクリプト")
    logger.info("=" * 80)
    logger.info("設定:")
    logger.info(f"  シンボル: {args.symbol}")
    logger.info(f"  時間軸: {args.timeframe}")
    logger.info(f"  最小サンプル数: {args.min_samples}")
    logger.info(f"  CV分割数: {args.cv_folds}")
    logger.info(f"  Optunaパラメータ使用: {args.use_optuna_params}")
    logger.info(f"  クラスウェイト使用: {args.use_class_weight}")

    # 評価実行
    evaluator = StackingEvaluator(
        symbol=args.symbol,
        timeframe=args.timeframe,
        min_samples=args.min_samples,
        cv_folds=args.cv_folds,
        use_optuna_params=args.use_optuna_params,
        use_class_weight=args.use_class_weight,
    )
    evaluator.run()


if __name__ == "__main__":
    main()
