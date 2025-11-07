"""
ファンディングレート特徴量の評価スクリプト

実データを使用して、新しく実装したTier 1特徴量（15個）の有効性を検証します。

実行方法:
    cd backend
    python -m scripts.feature_evaluation.evaluate_funding_rate_features
    python -m scripts.feature_evaluation.evaluate_funding_rate_features --days 90
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services.ml.feature_engineering.funding_rate_features import (
    FundingRateFeatureCalculator,
)
from database.connection import SessionLocal
from database.repositories.funding_rate_repository import FundingRateRepository
from database.repositories.ohlcv_repository import OHLCVRepository

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class FundingRateFeatureEvaluator:
    """
    ファンディングレート特徴量評価クラス
    
    Tier 1特徴量（15個）の有効性を実データで検証します。
    """

    # Tier 1特徴量リスト（15個）
    TIER1_FEATURES = [
        # 基本金利指標（4個）
        "funding_rate_raw",
        "fr_lag_1p",
        "fr_lag_2p",
        "fr_lag_3p",
        # 時間サイクル（3個）
        "fr_hours_since_settlement",
        "fr_cycle_sin",
        "fr_cycle_cos",
        # モメンタム（3個）
        "fr_velocity",
        "fr_ema_3periods",
        "fr_ema_7periods",
        # レジーム（2個）
        "fr_regime_encoded",
        "regime_duration",
        # 価格相互作用（2個）
        "fr_price_corr_24h",
        "fr_volatility_adjusted",
    ]

    def __init__(self, symbol: str = "BTC/USDT:USDT", timeframe: str = "1h"):
        """
        初期化
        
        Args:
            symbol: 取引ペア
            timeframe: 時間足
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.db = SessionLocal()
        self.ohlcv_repo = OHLCVRepository(self.db)
        self.fr_repo = FundingRateRepository(self.db)
        self.calculator = FundingRateFeatureCalculator()
        self.evaluation_results = {}

    def __enter__(self):
        """コンテキストマネージャー: 入場"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャー: 退場"""
        self.db.close()

    def load_data(
        self, start_date: str, end_date: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        データベースからOHLCVとファンディングレートデータを読み込み
        
        Args:
            start_date: 開始日（YYYY-MM-DD）
            end_date: 終了日（YYYY-MM-DD）
        
        Returns:
            (ohlcv_df, funding_df)のタプル
        """
        logger.info(f"データ読み込み開始: {start_date} 〜 {end_date}")

        try:
            # 日付をdatetimeに変換
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)

            # OHLCVデータ取得
            ohlcv_df = self.ohlcv_repo.get_ohlcv_dataframe(
                symbol=self.symbol,
                timeframe=self.timeframe,
                start_time=start_dt,
                end_time=end_dt,
            )

            if ohlcv_df.empty:
                raise ValueError(f"OHLCVデータが見つかりません: {self.symbol}")

            # インデックスがtimestampの場合、カラムとしてもリセット
            if ohlcv_df.index.name == "timestamp":
                ohlcv_df = ohlcv_df.reset_index()

            logger.info(f"OHLCV: {len(ohlcv_df)}行取得")

            # ファンディングレートデータ取得
            fr_records = self.fr_repo.get_funding_rate_data(
                symbol=self.symbol,
                start_time=start_dt,
                end_time=end_dt,
            )

            if not fr_records:
                raise ValueError(
                    f"ファンディングレートデータが見つかりません: {self.symbol}"
                )

            # DataFrameに変換（funding_timestampをtimestampに名前変更）
            funding_df = pd.DataFrame(
                [
                    {
                        "timestamp": r.funding_timestamp,
                        "funding_rate": r.funding_rate,
                    }
                    for r in fr_records
                ]
            )

            logger.info(f"FR: {len(funding_df)}行取得")

            return ohlcv_df, funding_df

        except Exception as e:
            logger.error(f"データ読み込みエラー: {e}")
            raise

    def evaluate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        データ品質を評価
        
        Returns:
            {
                'missing_values': 欠損値の数と割合,
                'imputed_ratio': 補間された行の割合,
                'outliers': 異常値の数,
                'date_range': データ期間,
                'total_rows': 総行数
            }
        """
        logger.info("データ品質評価開始")

        quality_metrics = {
            "total_rows": len(df),
            "date_range": {
                "start": df["timestamp"].min().isoformat() if not df.empty and "timestamp" in df.columns else None,
                "end": df["timestamp"].max().isoformat() if not df.empty and "timestamp" in df.columns else None,
            },
            "missing_values": {},
            "imputed_ratio": 0.0,
            "outliers": {},
        }

        # 欠損値チェック
        for col in self.TIER1_FEATURES:
            if col in df.columns:
                missing_count = df[col].isna().sum()
                missing_ratio = missing_count / len(df) if len(df) > 0 else 0
                quality_metrics["missing_values"][col] = {
                    "count": int(missing_count),
                    "ratio": float(missing_ratio),
                }

        # 補間フラグがある場合、補間率を計算
        if "fr_imputed_flag" in df.columns:
            imputed_count = (df["fr_imputed_flag"] == 1).sum()
            quality_metrics["imputed_ratio"] = float(imputed_count / len(df))

        # 異常値検出（Z-score法、閾値=3）
        for col in self.TIER1_FEATURES:
            if col in df.columns and df[col].notna().sum() > 0:
                values = df[col].dropna()
                if len(values) > 0:
                    z_scores = np.abs(stats.zscore(values))
                    outlier_count = (z_scores > 3).sum()
                    quality_metrics["outliers"][col] = int(outlier_count)

        logger.info("データ品質評価完了")
        return quality_metrics

    def calculate_feature_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        各特徴量の基本統計量を計算
        
        Returns:
            統計量のDataFrame（mean, std, min, max, skew, kurt等）
        """
        logger.info("基本統計量計算開始")

        stats_list = []
        for col in self.TIER1_FEATURES:
            if col not in df.columns:
                continue

            values = df[col].dropna()
            if len(values) == 0:
                continue

            stats_list.append(
                {
                    "feature": col,
                    "count": len(values),
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "min": float(values.min()),
                    "25%": float(values.quantile(0.25)),
                    "50%": float(values.median()),
                    "75%": float(values.quantile(0.75)),
                    "max": float(values.max()),
                    "skewness": float(values.skew()),
                    "kurtosis": float(values.kurtosis()),
                }
            )

        stats_df = pd.DataFrame(stats_list)
        logger.info("基本統計量計算完了")
        return stats_df

    def analyze_feature_correlations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        特徴量間の相関行列を計算
        
        Returns:
            相関行列（15x15）
        """
        logger.info("特徴量間相関分析開始")

        available_features = [f for f in self.TIER1_FEATURES if f in df.columns]
        corr_matrix = df[available_features].corr()

        logger.info("特徴量間相関分析完了")
        return corr_matrix

    def calculate_target_correlations(
        self, df: pd.DataFrame, target_col: str = "returns_1h"
    ) -> pd.Series:
        """
        各特徴量とターゲット変数の相関を計算
        
        Args:
            target_col: ターゲット変数のカラム名
        
        Returns:
            各特徴量の相関係数（降順）
        """
        logger.info("ターゲット相関分析開始")

        # ターゲット変数を作成（1時間先のリターン）
        if "close" not in df.columns:
            raise ValueError("closeカラムが見つかりません")

        df[target_col] = df["close"].pct_change().shift(-1)

        available_features = [f for f in self.TIER1_FEATURES if f in df.columns]

        # 相関係数を計算
        correlations = df[available_features].corrwith(df[target_col])
        correlations = correlations.abs().sort_values(ascending=False)

        logger.info("ターゲット相関分析完了")
        return correlations

    def calculate_mutual_information(
        self, df: pd.DataFrame, target_col: str = "returns_1h"
    ) -> pd.Series:
        """
        相互情報量を計算
        
        Returns:
            各特徴量のMIスコア（降順）
        """
        logger.info("相互情報量計算開始")

        # ターゲット変数を作成
        if target_col not in df.columns:
            if "close" not in df.columns:
                raise ValueError("closeカラムが見つかりません")
            df[target_col] = df["close"].pct_change().shift(-1)

        available_features = [f for f in self.TIER1_FEATURES if f in df.columns]

        # NaN、inf、-infを除去
        valid_idx = ~(
            df[available_features].isna().any(axis=1)
            | df[target_col].isna()
            | np.isinf(df[available_features]).any(axis=1)
            | np.isinf(df[target_col])
        )
        X = df.loc[valid_idx, available_features]
        y = df.loc[valid_idx, target_col]

        if len(X) < 100:
            logger.warning("サンプル数不足、相互情報量計算をスキップ")
            return pd.Series()

        # 相互情報量を計算
        mi_scores = mutual_info_regression(X, y, random_state=42)
        mi_series = pd.Series(mi_scores, index=available_features).sort_values(
            ascending=False
        )

        logger.info("相互情報量計算完了")
        return mi_series

    def evaluate_prediction_contribution(
        self, df: pd.DataFrame, target_col: str = "returns_1h"
    ) -> Dict[str, Any]:
        """
        LightGBMを使用して特徴量の予測への寄与度を評価
        
        Returns:
            {
                'feature_importance': {特徴量名: 重要度スコア},
                'baseline_rmse': ベースライン（FR特徴量なし）のRMSE,
                'with_fr_rmse': FR特徴量ありのRMSE,
                'improvement': 改善率（%）
            }
        """
        logger.info("予測性能評価開始")

        # ターゲット変数を作成
        if target_col not in df.columns:
            if "close" not in df.columns:
                raise ValueError("closeカラムが見つかりません")
            df[target_col] = df["close"].pct_change().shift(-1)

        # 基本的なテクニカル指標を作成（ベースライン用）
        baseline_features = []
        if "close" in df.columns:
            df["returns"] = df["close"].pct_change()
            df["sma_20"] = df["close"].rolling(20).mean()
            df["volatility_20"] = df["close"].pct_change().rolling(20).std()
            baseline_features = ["returns", "sma_20", "volatility_20"]

        available_fr_features = [f for f in self.TIER1_FEATURES if f in df.columns]

        # NaN、inf、-infを除去
        all_features = baseline_features + available_fr_features
        valid_idx = ~(
            df[all_features].isna().any(axis=1)
            | df[target_col].isna()
            | np.isinf(df[all_features]).any(axis=1)
            | np.isinf(df[target_col])
        )
        df_clean = df.loc[valid_idx].copy()

        if len(df_clean) < 100:
            logger.warning("サンプル数不足、予測性能評価をスキップ")
            return {}

        # 時系列分割
        tscv = TimeSeriesSplit(n_splits=5)

        # ベースライン評価（FR特徴量なし）
        baseline_rmse_scores = []
        for train_idx, test_idx in tscv.split(df_clean):
            X_train = df_clean.iloc[train_idx][baseline_features]
            y_train = df_clean.iloc[train_idx][target_col]
            X_test = df_clean.iloc[test_idx][baseline_features]
            y_test = df_clean.iloc[test_idx][target_col]

            model = lgb.LGBMRegressor(
                objective="regression",
                n_estimators=100,
                learning_rate=0.05,
                random_state=42,
                verbosity=-1,
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            baseline_rmse_scores.append(rmse)

        baseline_rmse = np.mean(baseline_rmse_scores)

        # FR特徴量あり評価
        fr_rmse_scores = []
        feature_importance_sum = np.zeros(len(available_fr_features))

        for train_idx, test_idx in tscv.split(df_clean):
            X_train = df_clean.iloc[train_idx][all_features]
            y_train = df_clean.iloc[train_idx][target_col]
            X_test = df_clean.iloc[test_idx][all_features]
            y_test = df_clean.iloc[test_idx][target_col]

            model = lgb.LGBMRegressor(
                objective="regression",
                n_estimators=100,
                learning_rate=0.05,
                random_state=42,
                verbosity=-1,
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            fr_rmse_scores.append(rmse)

            # FR特徴量の重要度のみ集計
            importance = model.feature_importances_
            fr_importance = importance[len(baseline_features) :]
            feature_importance_sum += fr_importance

        fr_rmse = np.mean(fr_rmse_scores)

        # 重要度を正規化
        feature_importance_avg = feature_importance_sum / tscv.n_splits
        if feature_importance_avg.sum() > 0:
            feature_importance_avg = (
                feature_importance_avg / feature_importance_avg.sum()
            )

        feature_importance = dict(
            zip(available_fr_features, feature_importance_avg.tolist())
        )

        # 改善率を計算
        improvement = ((baseline_rmse - fr_rmse) / baseline_rmse) * 100

        result = {
            "baseline_rmse": float(baseline_rmse),
            "baseline_rmse_std": float(np.std(baseline_rmse_scores)),
            "with_fr_rmse": float(fr_rmse),
            "with_fr_rmse_std": float(np.std(fr_rmse_scores)),
            "improvement_pct": float(improvement),
            "feature_importance": {
                k: float(v) for k, v in feature_importance.items()
            },
        }

        logger.info(f"予測性能評価完了 - 改善率: {improvement:.2f}%")
        return result

    def plot_visualizations(
        self, df: pd.DataFrame, output_dir: str = "feature_evaluation_plots"
    ):
        """
        評価結果の可視化
        
        Args:
            df: 特徴量DataFrame
            output_dir: 出力ディレクトリ
        """
        logger.info("可視化開始")

        # 出力ディレクトリ作成
        plot_dir = Path(output_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)

        # スタイル設定
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (12, 8)

        available_features = [f for f in self.TIER1_FEATURES if f in df.columns]

        # 1. 特徴量の分布（ヒストグラム）
        n_features = len(available_features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
        axes = axes.flatten() if n_features > 1 else [axes]

        for idx, feature in enumerate(available_features):
            if idx < len(axes):
                # inf、-inf、NaNを除外
                values = df[feature].replace([np.inf, -np.inf], np.nan).dropna()
                if len(values) > 0:
                    values.hist(bins=50, ax=axes[idx], edgecolor="black")
                    axes[idx].set_title(feature, fontsize=10)
                    axes[idx].set_xlabel("Value")
                    axes[idx].set_ylabel("Frequency")
                else:
                    axes[idx].text(0.5, 0.5, 'No valid data', ha='center', va='center')
                    axes[idx].set_title(feature, fontsize=10)

        # 未使用のサブプロットを非表示
        for idx in range(n_features, len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()
        plt.savefig(plot_dir / "feature_distributions.png", dpi=300, bbox_inches="tight")
        plt.close()

        # 2. 相関ヒートマップ
        corr_matrix = self.analyze_feature_correlations(df)
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=0.5,
        )
        plt.title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(plot_dir / "correlation_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close()

        # 3. 特徴量重要度（予測性能評価から）
        if "prediction_contribution" in self.evaluation_results:
            importance = self.evaluation_results["prediction_contribution"].get(
                "feature_importance", {}
            )
            if importance:
                importance_sorted = dict(
                    sorted(importance.items(), key=lambda x: x[1], reverse=True)
                )
                plt.figure(figsize=(10, 8))
                plt.barh(list(importance_sorted.keys()), list(importance_sorted.values()))
                plt.xlabel("Importance Score")
                plt.ylabel("Feature")
                plt.title(
                    "Feature Importance (LightGBM)", fontsize=14, fontweight="bold"
                )
                plt.tight_layout()
                plt.savefig(
                    plot_dir / "feature_importance.png", dpi=300, bbox_inches="tight"
                )
                plt.close()

        # 4. ターゲット相関（散布図 - TOP 6特徴量）
        target_corr = self.calculate_target_correlations(df)
        top_features = target_corr.head(6).index.tolist()

        if "returns_1h" in df.columns and len(top_features) > 0:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()

            for idx, feature in enumerate(top_features[:6]):
                valid_idx = df[[feature, "returns_1h"]].notna().all(axis=1)
                x = df.loc[valid_idx, feature]
                y = df.loc[valid_idx, "returns_1h"]

                axes[idx].scatter(x, y, alpha=0.3, s=10)
                axes[idx].set_xlabel(feature)
                axes[idx].set_ylabel("Returns (1h)")
                axes[idx].set_title(
                    f"{feature}\n(corr={target_corr[feature]:.4f})", fontsize=10
                )

            plt.tight_layout()
            plt.savefig(
                plot_dir / "target_correlation_scatter.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

        logger.info(f"可視化完了: {plot_dir}")

    def generate_report(self, output_path: str = "funding_rate_evaluation_report.md"):
        """
        評価結果をMarkdownレポートとして出力
        
        Args:
            output_path: 出力ファイルパス
        """
        logger.info("レポート生成開始")

        report_lines = [
            "# ファンディングレート特徴量評価レポート\n",
            f"**生成日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            f"**評価対象**: {self.symbol} ({self.timeframe})\n",
            "\n---\n",
            "\n## 1. エグゼクティブサマリー\n",
        ]

        # データ品質サマリー
        if "data_quality" in self.evaluation_results:
            quality = self.evaluation_results["data_quality"]
            report_lines.extend(
                [
                    "\n### データ品質\n",
                    f"- **総行数**: {quality['total_rows']:,}行\n",
                    f"- **データ期間**: {quality['date_range']['start']} 〜 {quality['date_range']['end']}\n",
                    f"- **補間率**: {quality['imputed_ratio']*100:.2f}%\n",
                ]
            )

        # 予測性能サマリー
        if "prediction_contribution" in self.evaluation_results:
            pred = self.evaluation_results["prediction_contribution"]
            report_lines.extend(
                [
                    "\n### 予測性能への寄与\n",
                    f"- **ベースラインRMSE**: {pred['baseline_rmse']:.6f} (±{pred['baseline_rmse_std']:.6f})\n",
                    f"- **FR特徴量ありRMSE**: {pred['with_fr_rmse']:.6f} (±{pred['with_fr_rmse_std']:.6f})\n",
                    f"- **改善率**: {pred['improvement_pct']:.2f}%\n",
                ]
            )

        report_lines.append("\n---\n")

        # 2. データ品質詳細
        if "data_quality" in self.evaluation_results:
            quality = self.evaluation_results["data_quality"]
            report_lines.extend(["\n## 2. データ品質詳細\n", "\n### 欠損値\n"])

            missing_data = []
            for feature, info in quality["missing_values"].items():
                missing_data.append(
                    f"- **{feature}**: {info['count']}個 ({info['ratio']*100:.2f}%)\n"
                )
            report_lines.extend(sorted(missing_data))

            report_lines.append("\n### 異常値（Z-score > 3）\n")
            outlier_data = []
            for feature, count in quality["outliers"].items():
                outlier_data.append(f"- **{feature}**: {count}個\n")
            report_lines.extend(sorted(outlier_data))

        # 3. 基本統計量
        if "feature_statistics" in self.evaluation_results:
            report_lines.append("\n---\n\n## 3. 基本統計量\n")
            stats_df = self.evaluation_results["feature_statistics"]
            report_lines.append("\n```\n")
            report_lines.append(stats_df.to_string(index=False))
            report_lines.append("\n```\n")

        # 4. 相関分析
        if "target_correlations" in self.evaluation_results:
            report_lines.append("\n---\n\n## 4. ターゲット相関（TOP 10）\n")
            corr = self.evaluation_results["target_correlations"]
            for idx, (feature, value) in enumerate(corr.head(10).items(), 1):
                report_lines.append(f"{idx}. **{feature}**: {value:.6f}\n")

        # 5. 相互情報量
        if "mutual_information" in self.evaluation_results:
            report_lines.append("\n---\n\n## 5. 相互情報量（TOP 10）\n")
            mi = self.evaluation_results["mutual_information"]
            if not mi.empty:
                for idx, (feature, value) in enumerate(mi.head(10).items(), 1):
                    report_lines.append(f"{idx}. **{feature}**: {value:.6f}\n")

        # 6. 特徴量重要度
        if "prediction_contribution" in self.evaluation_results:
            pred = self.evaluation_results["prediction_contribution"]
            importance = pred.get("feature_importance", {})
            if importance:
                report_lines.append("\n---\n\n## 6. 特徴量重要度（LightGBM）\n")
                importance_sorted = sorted(
                    importance.items(), key=lambda x: x[1], reverse=True
                )
                for idx, (feature, value) in enumerate(importance_sorted, 1):
                    report_lines.append(f"{idx}. **{feature}**: {value:.6f}\n")

        # 7. 推奨事項
        report_lines.extend(
            [
                "\n---\n",
                "\n## 7. 推奨事項と次のステップ\n",
                "\n### 主要な発見\n",
            ]
        )

        # 予測性能の改善があるか
        if "prediction_contribution" in self.evaluation_results:
            pred = self.evaluation_results["prediction_contribution"]
            if pred["improvement_pct"] > 0:
                report_lines.append(
                    f"✅ ファンディングレート特徴量により予測性能が**{pred['improvement_pct']:.2f}%改善**しました。\n"
                )
            else:
                report_lines.append(
                    f"⚠️ ファンディングレート特徴量による明確な改善は見られませんでした（{pred['improvement_pct']:.2f}%）。\n"
                )

        # 最も重要な特徴量
        if "prediction_contribution" in self.evaluation_results:
            importance = self.evaluation_results["prediction_contribution"].get(
                "feature_importance", {}
            )
            if importance:
                top_feature = max(importance.items(), key=lambda x: x[1])
                report_lines.append(
                    f"✅ 最も重要な特徴量: **{top_feature[0]}** (重要度: {top_feature[1]:.4f})\n"
                )

        report_lines.extend(
            [
                "\n### 次のステップ\n",
                "1. 🔄 **Tier 2特徴量の実装**: より高度な派生特徴量の追加\n",
                "2. 🎯 **ハイパーパラメータ最適化**: モデルパラメータの調整\n",
                "3. 📊 **アンサンブルモデル**: XGBoost、TabNetとの組み合わせテスト\n",
                "4. ⏱️ **時間窓の最適化**: 24時間窓以外の検証\n",
                "5. 🔍 **レジーム分析の強化**: より詳細なレジーム分類の検討\n",
                "\n---\n",
                "\n*このレポートは自動生成されました*\n",
            ]
        )

        # ファイルに書き込み
        with open(output_path, "w", encoding="utf-8") as f:
            f.writelines(report_lines)

        logger.info(f"レポート生成完了: {output_path}")

    def evaluate_all(self, df: pd.DataFrame):
        """
        全評価を実行
        
        Args:
            df: 特徴量を含むDataFrame
        """
        logger.info("=== ファンディングレート特徴量評価開始 ===")

        # 1. データ品質評価
        self.evaluation_results["data_quality"] = self.evaluate_data_quality(df)

        # 2. 基本統計量
        self.evaluation_results["feature_statistics"] = (
            self.calculate_feature_statistics(df)
        )

        # 3. 相関分析
        self.evaluation_results["feature_correlations"] = (
            self.analyze_feature_correlations(df)
        )

        # 4. ターゲット相関
        self.evaluation_results["target_correlations"] = (
            self.calculate_target_correlations(df)
        )

        # 5. 相互情報量
        self.evaluation_results["mutual_information"] = (
            self.calculate_mutual_information(df)
        )

        # 6. 予測性能評価
        self.evaluation_results["prediction_contribution"] = (
            self.evaluate_prediction_contribution(df)
        )

        logger.info("=== 全評価完了 ===")


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(
        description="ファンディングレート特徴量評価スクリプト"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTC/USDT:USDT",
        help="取引ペア（デフォルト: BTC/USDT:USDT）",
    )
    parser.add_argument(
        "--days", type=int, default=90, help="評価期間（日数、デフォルト: 90）"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="feature_evaluation_results",
        help="出力ディレクトリ（デフォルト: feature_evaluation_results）",
    )

    args = parser.parse_args()

    try:
        # 期間設定
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)

        logger.info(f"評価期間: {start_date.date()} 〜 {end_date.date()}")

        # 評価器を初期化
        with FundingRateFeatureEvaluator(symbol=args.symbol) as evaluator:
            # データ読み込み
            ohlcv_df, funding_df = evaluator.load_data(
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
            )

            # 特徴量計算
            df = evaluator.calculator.calculate_features(ohlcv_df, funding_df)

            logger.info(f"特徴量計算完了: {len(df)}行, {len(df.columns)}カラム")

            # 評価実行
            evaluator.evaluate_all(df)

            # 出力ディレクトリ作成
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # レポート生成
            report_path = output_dir / "funding_rate_evaluation_report.md"
            evaluator.generate_report(str(report_path))

            # 可視化
            plot_dir = output_dir / "plots"
            evaluator.plot_visualizations(df, str(plot_dir))

            # 統計CSVを保存
            if "feature_statistics" in evaluator.evaluation_results:
                stats_df = evaluator.evaluation_results["feature_statistics"]
                stats_path = output_dir / "feature_statistics.csv"
                stats_df.to_csv(stats_path, index=False)
                logger.info(f"統計CSV保存: {stats_path}")

            # 結果JSONを保存
            result_json = {
                "evaluation_date": datetime.now().isoformat(),
                "symbol": args.symbol,
                "period_days": args.days,
                "results": {
                    k: v.to_dict() if isinstance(v, pd.DataFrame) else v
                    if not isinstance(v, pd.Series)
                    else v.to_dict()
                    for k, v in evaluator.evaluation_results.items()
                },
            }

            json_path = output_dir / "evaluation_results.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result_json, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"結果JSON保存: {json_path}")

            # サマリーを表示
            print("\n" + "=" * 80)
            print("ファンディングレート特徴量評価 - サマリー")
            print("=" * 80)

            if "data_quality" in evaluator.evaluation_results:
                quality = evaluator.evaluation_results["data_quality"]
                print(f"\nデータ品質スコア:")
                print(f"  総行数: {quality['total_rows']:,}行")
                print(f"  補間率: {quality['imputed_ratio']*100:.2f}%")

            if "prediction_contribution" in evaluator.evaluation_results:
                pred = evaluator.evaluation_results["prediction_contribution"]
                print(f"\n予測性能:")
                print(
                    f"  ベースラインRMSE: {pred['baseline_rmse']:.6f} (±{pred['baseline_rmse_std']:.6f})"
                )
                print(
                    f"  FR特徴量ありRMSE: {pred['with_fr_rmse']:.6f} (±{pred['with_fr_rmse_std']:.6f})"
                )
                print(f"  改善率: {pred['improvement_pct']:+.2f}%")

                importance = pred.get("feature_importance", {})
                if importance:
                    top5 = sorted(importance.items(), key=lambda x: x[1], reverse=True)[
                        :5
                    ]
                    print(f"\nTOP 5 重要特徴量:")
                    for idx, (feat, imp) in enumerate(top5, 1):
                        print(f"  {idx}. {feat}: {imp:.6f}")

            if "target_correlations" in evaluator.evaluation_results:
                corr = evaluator.evaluation_results["target_correlations"]
                top5 = corr.head(5)
                print(f"\nTOP 5 ターゲット相関:")
                for idx, (feat, corr_val) in enumerate(top5.items(), 1):
                    print(f"  {idx}. {feat}: {corr_val:.6f}")

            print("\n" + "=" * 80)
            print(f"\n✅ 評価完了！結果は {output_dir} に保存されました。")
            print("=" * 80 + "\n")

    except Exception as e:
        logger.error(f"実行エラー: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()