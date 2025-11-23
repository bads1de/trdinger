"""
Optuna統合: ラベル生成パラメータ + モデルハイパーパラメータの同時最適化（レンジ vs トレンド 2値分類版）

Phase 1.5実装: プリセットの手動変更を不要にし、動的に最適解を発見します。

使用例:
    # 統合最適化を実行（50 trials）
    conda run -n trading python backend/scripts/ml_optimization/train_with_label_optimization.py --n-trials 50
    
    # 小規模テスト（10 trials）
    conda run -n trading python backend/scripts/ml_optimization/train_with_label_optimization.py --n-trials 10
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import lightgbm as lgb
import optuna
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Windowsでの文字化け対策
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

from app.services.ml.feature_engineering.feature_engineering_service import (  # type: ignore  # noqa: E501
    FeatureEngineeringService,
)
from app.services.ml.label_cache import LabelCache  # type: ignore
from database.connection import SessionLocal  # type: ignore
from database.repositories.ohlcv_repository import OHLCVRepository  # type: ignore
from scripts.feature_evaluation.common_feature_evaluator import (  # type: ignore
    CommonFeatureEvaluator,
)

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("integrated_optimization.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


class IntegratedOptimizer:
    """ラベル生成とモデルハイパーパラメータの統合最適化（Trend vs Range）"""

    def __init__(
        self,
        symbol: str = "BTC/USDT:USDT",
        timeframe: str = "1h",
        n_trials: int = 50,
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.n_trials = n_trials

        self.db = SessionLocal()
        self.ohlcv_repo = OHLCVRepository(self.db)
        self.feature_service = FeatureEngineeringService()
        self.evaluator = CommonFeatureEvaluator()

        # 結果保存ディレクトリ
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = (
            Path(__file__).parent.parent.parent
            / "results"
            / "integrated_optimization"
            / f"run_{timestamp}"
        )
        self.results_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"統合最適化初期化: {symbol} {timeframe}")
        logger.info(f"結果保存先: {self.results_dir}")

    def prepare_data(self):
        """データ準備"""
        logger.info("データ準備開始")

        # OHLCVデータ取得
        data = self.evaluator.fetch_data(
            symbol=self.symbol, timeframe=self.timeframe, limit=10000
        )

        if data.ohlcv.empty:
            raise ValueError("OHLCVデータが空です")

        logger.info(f"OHLCVデータ取得: {len(data.ohlcv)}行")

        # 特徴量生成
        features_df = self.feature_service.calculate_advanced_features(
            ohlcv_data=data.ohlcv,
            funding_rate_data=data.fr,
            open_interest_data=data.oi,
        )

        logger.info(f"特徴量生成完了: {len(features_df.columns)}個")

        # OHLCV列を除外
        feature_cols = [
            col
            for col in features_df.columns
            if col not in ["open", "high", "low", "volume"]
        ]
        X = features_df[feature_cols].copy()

        # NaN補完
        if X.isna().any().any():
            X = X.fillna(X.median())

        logger.info(f"特徴量準備完了: {X.shape}")

        return X, data.ohlcv

    def run_optimization(self):
        """統合最適化実行"""
        start_time = time.time()

        try:
            # データ準備
            X, ohlcv_df = self.prepare_data()

            # インスタンス変数として保存（後で使用）
            self.X = X
            self.ohlcv_df = ohlcv_df

            # LabelCacheを初期化
            label_cache = LabelCache(ohlcv_df)

            # グローバル変数として保存（objective関数から参照）
            self.label_cache = label_cache

            def objective(trial):
                """統合目的関数 - LightGBMとXGBoostの両方をサポート"""

                # ========== モデルタイプの選択 ==========
                model_type = trial.suggest_categorical("model_type", ["lightgbm", "xgboost"])

                # ========== ラベル生成パラメータ ==========
                horizon_n = trial.suggest_int("horizon_n", 4, 16, step=2)
                threshold_method = trial.suggest_categorical(
                    "threshold_method",
                    ["QUANTILE", "KBINS_DISCRETIZER", "DYNAMIC_VOLATILITY"],
                )

                # 閾値（方法によって意味が変わる）
                if threshold_method == "QUANTILE":
                    threshold = trial.suggest_float("quantile_threshold", 0.25, 0.40)
                elif threshold_method == "KBINS_DISCRETIZER":
                    threshold = trial.suggest_float("kbins_threshold", 0.001, 0.005)
                else:  # DYNAMIC_VOLATILITY
                    threshold = trial.suggest_float("volatility_threshold", 0.5, 2.0)

                # ========== ラベル取得（キャッシュ活用） ==========
                try:
                    labels = label_cache.get_labels(
                        horizon_n=horizon_n,
                        threshold_method=threshold_method,
                        threshold=threshold,
                        timeframe=self.timeframe,
                        price_column="close",
                    )
                except Exception as e:
                    logger.warning(f"ラベル生成エラー: {e}")
                    raise optuna.exceptions.TrialPruned()

                # データのアライメント
                common_index = X.index.intersection(labels.index)
                X_aligned = X.loc[common_index]
                labels_aligned = labels.loc[common_index]

                # NaN除去
                valid_idx = ~labels_aligned.isna()
                X_clean = X_aligned.loc[valid_idx]
                labels_clean = labels_aligned.loc[valid_idx]

                # ラベルを数値に変換 (TREND=1, RANGE=0)
                # UP/DOWN -> 1
                # RANGE -> 0
                label_mapping = {"DOWN": 1, "RANGE": 0, "UP": 1}
                y = labels_clean.map(label_mapping)

                # データ数チェック
                if len(y) < 100:
                    logger.warning(f"データ数不足: {len(y)}行")
                    raise optuna.exceptions.TrialPruned()

                # データ分割 (時系列順: Train 60%, Val 20%, Test 20%)
                # 先読みバイアス（リーク）を防ぐため、シャッフルせずに時系列順に分割
                n_total = len(X_clean)
                test_start_idx = int(n_total * 0.8)
                val_start_idx = int(n_total * 0.6)

                X_train = X_clean.iloc[:val_start_idx]
                y_train = y.iloc[:val_start_idx]
                
                X_val = X_clean.iloc[val_start_idx:test_start_idx]
                y_val = y.iloc[val_start_idx:test_start_idx]
                
                X_test = X_clean.iloc[test_start_idx:]
                y_test = y.iloc[test_start_idx:]

                # ========== クラス不均衡対策 (レンジ検出強化のため強制有効化) ==========
                use_class_weight = True
                # use_smote = trial.suggest_categorical("use_smote", [True, False])
                use_smote = False # Class Weightだけで十分な場合が多いので一旦無効化

                # SMOTEによるオーバーサンプリング (今回は無効化)
                if use_smote:
                    try:
                        from imblearn.over_sampling import SMOTE

                        smote = SMOTE(random_state=42)
                        X_train_resampled, y_train_resampled = smote.fit_resample(
                            X_train, y_train
                        )
                        X_train = X_train_resampled
                        y_train = y_train_resampled
                    except ImportError:
                        logger.warning("imblearn未インストール。SMOTEスキップ")
                    except Exception as e:
                        logger.warning(f"SMOTEエラー: {e}")

                # ========== モデル学習 (LightGBM or XGBoost) ==========
                try:
                    if model_type == "lightgbm":
                        # LightGBMパラメータ
                        params = {
                            "objective": "binary",
                            "metric": "binary_logloss",
                            "verbosity": -1,
                            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                            "learning_rate": trial.suggest_float(
                                "learning_rate", 0.001, 0.5, log=True
                            ),
                            "max_depth": trial.suggest_int("max_depth", 3, 15),
                            "num_leaves": trial.suggest_int("num_leaves", 20, 200),
                            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
                            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                            "random_state": 42,
                            "class_weight": "balanced" # 強制有効化
                        }

                        model = lgb.LGBMClassifier(**params)
                        model.fit(
                            X_train,
                            y_train,
                            eval_set=[(X_val, y_val)],
                            callbacks=[lgb.early_stopping(50, verbose=False)],
                        )

                    else:  # XGBoost
                        import xgboost as xgb

                        # XGBoostパラメータ
                        params = {
                            "objective": "binary:logistic",
                            "eval_metric": "logloss",
                            "verbosity": 0,
                            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                            "learning_rate": trial.suggest_float(
                                "learning_rate", 0.001, 0.5, log=True
                            ),
                            "max_depth": trial.suggest_int("max_depth", 3, 15),
                            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                            "gamma": trial.suggest_float("gamma", 0, 5),
                            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                            "random_state": 42,
                        }

                        # XGBoostはscale_pos_weightでクラスウェイト対応
                        neg_count = len(y_train) - sum(y_train)
                        pos_count = sum(y_train)
                        # ゼロ除算防止
                        if pos_count > 0:
                            params["scale_pos_weight"] = neg_count / pos_count
                        else:
                            params["scale_pos_weight"] = 1.0

                        model = xgb.XGBClassifier(**params)
                        model.fit(
                            X_train,
                            y_train,
                            eval_set=[(X_val, y_val)],
                            verbose=False,
                        )

                except Exception as e:
                    logger.warning(f"学習エラー ({model_type}): {e}")
                    raise optuna.exceptions.TrialPruned()

                # ========== Validation精度評価 (カスタムスコア: 期待値最大化) ==========
                y_val_pred = model.predict(X_val)
                
                from sklearn.metrics import classification_report
                val_report = classification_report(y_val, y_val_pred, output_dict=True, zero_division=0)
                
                # クラス 0: RANGE, 1: TREND
                trend_recall = val_report['1']['recall']
                trend_precision = val_report['1']['precision']
                range_recall = val_report['0']['recall']
                
                # カスタムスコア: トレンド検出数(Recall)と確度(Precision)を重視し、レンジ回避(Range Recall)は補助的に
                # これにより「高勝率だが低回数」ではなく「十分な回数でトータルプラス」を目指す
                custom_score = (trend_recall * 1.5) + trend_precision + (range_recall * 0.5)
                
                # ========== Test精度評価 (過学習チェック) ==========
                y_test_pred = model.predict(X_test)
                # test_f1 = f1_score(y_test, y_test_pred, average="macro") # もう使わないのでコメントアウト

                # クラス別精度を取得
                # from sklearn.metrics import classification_report # 上でimport済み

                report = classification_report(
                    y_test, y_test_pred, target_names=["RANGE", "TREND"], output_dict=True
                )

                range_f1 = report["RANGE"]["f1-score"]
                trend_f1 = report["TREND"]["f1-score"]

                logger.info(
                    f"Trial {trial.number} ({model_type}): Custom_Score={custom_score:.4f}, "
                    f"Trend_Rec={trend_recall:.4f}, Trend_Prec={trend_precision:.4f}, Range_Rec={range_recall:.4f}"
                )

                # Validation精度を返す（過学習を防ぐため）
                return custom_score

            # Optunaで最適化
            pruner = optuna.pruners.MedianPruner(n_startup_trials=10)
            study = optuna.create_study(
                direction="maximize",
                study_name="integrated",
                pruner=pruner,
            )

            logger.info("=" * 80)
            logger.info("統合最適化開始 (Trend vs Range)")
            logger.info("=" * 80)

            study.optimize(objective, n_trials=self.n_trials, n_jobs=1)

            # キャッシュ統計
            cache_stats = label_cache.get_stats()
            logger.info("=" * 80)
            logger.info("最適化完了")
            logger.info("=" * 80)
            logger.info(f"ベストF1スコア: {study.best_value:.4f}")
            logger.info(f"キャッシュヒット率: {cache_stats['hit_rate_pct']:.1f}%")

            # 結果保存
            self._save_results(study, cache_stats, time.time() - start_time)

        except Exception as e:
            logger.error(f"エラー発生: {e}", exc_info=True)
            raise
        finally:
            self.db.close()
            self.evaluator.close()

    def _save_results(self, study, cache_stats, elapsed_time):
        """結果をJSON + マークダウンで保存（拡張版）"""

        # ベストtrialからTest精度を再計算
        best_trial = study.best_trial
        best_params = best_trial.params
        best_params["use_class_weight"] = True # 強制的に追加

        # 最適パラメータで最終評価
        logger.info("=" * 80)
        logger.info("ベストパラメータで最終評価実行")
        logger.info("=" * 80)

        final_metrics = self._evaluate_best_model(best_params)

        # JSON保存
        results = {
            "best_value": study.best_value,
            "best_params": best_params,
            "n_trials": len(study.trials),
            "cache_stats": cache_stats,
            "elapsed_time": elapsed_time,
            "final_metrics": final_metrics,  # 追加
        }

        json_path = self.results_dir / "best_params.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"結果保存: {json_path}")

        # 拡張マークダウンレポート
        md_path = self.results_dir / "report.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# 統合最適化レポート (Phase 1.5 拡張版 - Trend vs Range)\n\n")
            f.write(f"**生成日時**: {datetime.now()}\n\n")

            # ベスト結果
            f.write("## ベスト結果\n\n")
            f.write(f"- **選択モデル**: {best_params.get('model_type', 'lightgbm').upper()}\n")
            f.write(f"- **Validation Custom Score**: {study.best_value:.4f}\n")
            f.write(f"- **Test F1スコア**: {final_metrics['test_f1']:.4f}\n")
            # f.write(f"- **過学習度**: {(study.best_value - final_metrics['test_f1']) * 100:.2f}%\n") # 比較不能なので削除
            f.write(f"- **実行時間**: {elapsed_time:.1f}秒\n")
            f.write(f"- **試行回数**: {len(study.trials)}\n\n")

            # クラス別性能
            f.write("## クラス別性能 (Test)\n\n")
            f.write("| クラス | Precision | Recall | F1-Score | Support |\n")
            f.write("|--------|-----------|--------|----------|----------|\n")
            for cls in ["RANGE", "TREND"]:
                metrics = final_metrics["class_report"][cls]
                f.write(
                    f"| {cls} | {metrics['precision']:.4f} | "
                    f"{metrics['recall']:.4f} | {metrics['f1-score']:.4f} | "
                    f"{int(metrics['support'])} |\n"
                )
            f.write("\n")

            # 最適ラベル生成パラメータ
            f.write("## 最適ラベル生成パラメータ\n\n")
            f.write(f"- **horizon_n**: {best_params['horizon_n']}時間先\n")
            f.write(f"- **threshold_method**: {best_params['threshold_method']}\n")
            threshold_key = [k for k in best_params if "threshold" in k and k != "threshold_method"][0]
            f.write(f"- **threshold**: {best_params[threshold_key]:.4f}\n")
            f.write(f"- **use_class_weight**: True (Fixed)\n")
            f.write(f"- **use_smote**: {best_params.get('use_smote', False)}\n\n")

            # 最適モデルハイパーパラメータ
            f.write("## 最適モデルハイパーパラメータ\n\n")
            f.write("```json\n")
            model_params = {
                k: v
                for k, v in best_params.items()
                if k
                not in [
                    "horizon_n",
                    "threshold_method",
                    "quantile_threshold",
                    "kbins_threshold",
                    "volatility_threshold",
                    "use_class_weight",
                    "use_smote",
                ]
            }
            f.write(json.dumps(model_params, indent=2, ensure_ascii=False))
            f.write("\n```\n\n")

            # キャッシュ統計
            f.write("## キャッシュ統計\n\n")
            f.write(f"- ヒット数: {cache_stats['hit_count']}\n")
            f.write(f"- ミス数: {cache_stats['miss_count']}\n")
            f.write(f"- ヒット率: {cache_stats['hit_rate_pct']:.1f}%\n\n")

            # 推奨事項
            f.write("## 推奨事項\n\n")
            overfit = (study.best_value - final_metrics["test_f1"]) * 100
            if overfit > 5:
                f.write("⚠️ **過学習の兆候**: ValidationとTestの差が5%以上あります。\n")
                f.write("- 正則化を強化してください\n")
                f.write("- データを増やすことを検討してください\n\n")
            else:
                f.write("✅ **過学習なし**: ValidationとTestの差が小さく、良好です。\n\n")

            if final_metrics["class_report"]["TREND"]["recall"] < 0.5:
                f.write("⚠️ **TRENDクラスのRecallが低い**: トレンドの検出が不十分です。\n")
                f.write("- SMOTEの有効化を検討してください\n")
                f.write("- class_weightの調整を検討してください\n\n")

        logger.info(f"レポート保存: {md_path}")

    def _evaluate_best_model(self, best_params):
        """ベストパラメータで最終評価"""
        # LabelCacheを再初期化
        from app.services.ml.label_cache import LabelCache

        label_cache = LabelCache(self.ohlcv_df)

        # ラベル生成パラメータ
        horizon_n = best_params["horizon_n"]
        threshold_method = best_params["threshold_method"]

        # 閾値パラメータを取得
        threshold_key = [
            k for k in best_params if "threshold" in k and k != "threshold_method"
        ][0]
        threshold = best_params[threshold_key]

        # ラベル取得
        labels = label_cache.get_labels(
            horizon_n=horizon_n,
            threshold_method=threshold_method,
            threshold=threshold,
            timeframe=self.timeframe,
            price_column="close",
        )

        # データのアライメント
        common_index = self.X.index.intersection(labels.index)
        X_aligned = self.X.loc[common_index]
        labels_aligned = labels.loc[common_index]

        # NaN除去
        valid_idx = ~labels_aligned.isna()
        X_clean = X_aligned.loc[valid_idx]
        labels_clean = labels_aligned.loc[valid_idx]

        # ラベルを数値に変換 (Trend=1, Range=0)
        label_mapping = {"DOWN": 1, "RANGE": 0, "UP": 1}
        y = labels_clean.map(label_mapping)

        # データ分割 (時系列順: Train 60%, Val 20%, Test 20%)
        n_total = len(X_clean)
        test_start_idx = int(n_total * 0.8)
        val_start_idx = int(n_total * 0.6)

        X_train = X_clean.iloc[:val_start_idx]
        y_train = y.iloc[:val_start_idx]
        
        X_val = X_clean.iloc[val_start_idx:test_start_idx]
        y_val = y.iloc[val_start_idx:test_start_idx]
        
        X_test = X_clean.iloc[test_start_idx:]
        y_test = y.iloc[test_start_idx:]

        # SMOTE適用
        if best_params.get("use_smote", False):
            try:
                from imblearn.over_sampling import SMOTE

                smote = SMOTE(random_state=42)
                X_train, y_train = smote.fit_resample(X_train, y_train)
            except Exception as e:
                logger.warning(f"SMOTE適用エラー: {e}")

        # モデル学習
        model_type = best_params.get("model_type", "lightgbm")

        if model_type == "lightgbm":
            params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "verbosity": -1,
                "n_estimators": best_params["n_estimators"],
                "learning_rate": best_params["learning_rate"],
                "max_depth": best_params["max_depth"],
                "num_leaves": best_params["num_leaves"],
                "min_child_samples": best_params["min_child_samples"],
                "subsample": best_params["subsample"],
                "colsample_bytree": best_params["colsample_bytree"],
                "random_state": 42,
            }

            if best_params.get("use_class_weight", True):
                params["class_weight"] = "balanced"

            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )

        else:  # XGBoost
            import xgboost as xgb

            params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "verbosity": 0,
                "n_estimators": best_params["n_estimators"],
                "learning_rate": best_params["learning_rate"],
                "max_depth": best_params["max_depth"],
                "min_child_weight": best_params["min_child_weight"],
                "subsample": best_params["subsample"],
                "colsample_bytree": best_params["colsample_bytree"],
                "gamma": best_params["gamma"],
                "reg_alpha": best_params["reg_alpha"],
                "reg_lambda": best_params["reg_lambda"],
                "random_state": 42,
            }

            if best_params.get("use_class_weight", True):
                neg_count = len(y_train) - sum(y_train)
                pos_count = sum(y_train)
                if pos_count > 0:
                    params["scale_pos_weight"] = neg_count / pos_count
                else:
                    params["scale_pos_weight"] = 1.0

            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

        # Test精度評価
        y_test_pred = model.predict(X_test)
        test_f1 = f1_score(y_test, y_test_pred, average="macro") # Macro F1

        # クラス別レポート
        from sklearn.metrics import classification_report

        report = classification_report(
            y_test,
            y_test_pred,
            target_names=["RANGE", "TREND"],
            output_dict=True,
        )

        return {"test_f1": test_f1, "class_report": report}


def main():
    parser = argparse.ArgumentParser(description="統合最適化: ラベル生成+モデル (Trend vs Range)")
    parser.add_argument("--symbol", default="BTC/USDT:USDT", help="シンボル")
    parser.add_argument("--timeframe", default="1h", help="時間軸")
    parser.add_argument("--n-trials", type=int, default=50, help="試行回数")

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("Phase 1.5: Optuna統合最適化 (Trend vs Range)")
    logger.info("=" * 80)
    logger.info(f"シンボル: {args.symbol}")
    logger.info(f"時間軸: {args.timeframe}")
    logger.info(f"試行回数: {args.n_trials}")

    optimizer = IntegratedOptimizer(
        symbol=args.symbol,
        timeframe=args.timeframe,
        n_trials=args.n_trials,
    )

    optimizer.run_optimization()


if __name__ == "__main__":
    main()