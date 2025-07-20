"""
特徴量ノイズ分析スクリプト（SQLite直接アクセス版）

このスクリプトは機械学習モデルの特徴量を分析し、
精度を落としているノイズ特徴量を特定・可視化します。

SQLiteデータベースに直接アクセスしてデータを取得します。
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import warnings
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# データベースファイルパス（絶対パス）
db_file_path = Path("C:/Users/buti3/trading/backend/trdinger.db")

# 既存のサービスをインポート
from app.core.services.ml.feature_engineering.feature_engineering_service import (
    FeatureEngineeringService,
)
from app.core.services.ml.lightgbm_trainer import LightGBMTrainer

# 警告を抑制
warnings.filterwarnings("ignore")

# ログ設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 日本語フォント設定（matplotlib）
plt.rcParams["font.family"] = "DejaVu Sans"
sns.set_style("whitegrid")
sns.set_palette("husl")


class FeatureNoiseAnalyzerSQLite:
    """
    特徴量ノイズ分析クラス（SQLite直接アクセス版）
    """

    def __init__(self, symbol: str = "BTC/USDT:USDT", timeframe: str = "1h"):
        """
        初期化

        Args:
            symbol: 取引ペア
            timeframe: 時間軸
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.feature_service = FeatureEngineeringService()
        self.trainer = LightGBMTrainer()

        # データベースファイルパス
        self.db_path = db_file_path

        # 結果保存用
        self.results = {}
        self.feature_importance_results = {}
        self.permutation_importance_results = {}
        self.noise_features = []

        # 出力ディレクトリ
        self.output_dir = Path("backend/scripts/ml_analysis/results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"特徴量ノイズ分析を初期化: {symbol} - {timeframe}")
        logger.info(f"データベースファイル: {self.db_path}")

    def load_data(self, limit: int = 5000) -> pd.DataFrame:
        """
        SQLiteから直接データを読み込み

        Args:
            limit: 取得するデータ数

        Returns:
            OHLCVデータのDataFrame
        """
        logger.info(f"データ読み込み開始: {self.symbol} - {self.timeframe}")

        if not self.db_path.exists():
            raise FileNotFoundError(
                f"データベースファイルが見つかりません: {self.db_path}"
            )

        try:
            # SQLite接続
            conn = sqlite3.connect(str(self.db_path))

            # データ取得クエリ
            query = """
            SELECT timestamp, open, high, low, close, volume
            FROM ohlcv_data 
            WHERE symbol = ? AND timeframe = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """

            # データ取得
            df = pd.read_sql_query(
                query, conn, params=(self.symbol, self.timeframe, limit)
            )

            conn.close()

            if df.empty:
                raise ValueError(
                    f"データが見つかりません: {self.symbol} - {self.timeframe}"
                )

            # データ型変換
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp").reset_index(drop=True)
            df.set_index("timestamp", inplace=True)

            # カラム名を大文字に変換（特徴量計算用）
            df.columns = ["Open", "High", "Low", "Close", "Volume"]

            logger.info(f"データ読み込み完了: {len(df)}件")
            return df

        except Exception as e:
            logger.error(f"データ読み込みエラー: {e}")
            raise

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        特徴量を計算（全ての特徴量タイプを含む）

        Args:
            df: OHLCVデータ

        Returns:
            特徴量付きDataFrame
        """
        logger.info("特徴量計算開始")

        try:
            # ダミーの外部データを作成（特徴量計算を完全にするため）
            dummy_funding_rate = self._create_dummy_funding_rate_data(df)
            dummy_open_interest = self._create_dummy_open_interest_data(df)
            dummy_external_market = self._create_dummy_external_market_data(df)
            dummy_fear_greed = self._create_dummy_fear_greed_data(df)

            # 特徴量計算（全ての外部データを提供）
            features_df = self.feature_service.calculate_advanced_features(
                df,
                funding_rate_data=dummy_funding_rate,
                open_interest_data=dummy_open_interest,
                external_market_data=dummy_external_market,
                fear_greed_data=dummy_fear_greed,
            )

            if features_df is None or features_df.empty:
                raise ValueError("特徴量計算に失敗しました")

            # 全ての列を保持（数値、カテゴリカル、バイナリ）
            # ただし、オブジェクト型は数値に変換を試みる
            for col in features_df.columns:
                if features_df[col].dtype == "object":
                    try:
                        # カテゴリカル変数を数値に変換
                        features_df[col] = pd.to_numeric(
                            features_df[col], errors="coerce"
                        )
                    except (ValueError, TypeError):
                        # 変換できない場合はラベルエンコーディング
                        features_df[col] = pd.Categorical(features_df[col]).codes

            # ブール型を数値に変換
            bool_columns = features_df.select_dtypes(include=["bool"]).columns
            features_df[bool_columns] = features_df[bool_columns].astype(int)

            # NaN値を処理
            features_df = features_df.fillna(method="ffill").fillna(0)

            # 無限値を処理
            features_df = features_df.replace([np.inf, -np.inf], 0)

            logger.info(f"特徴量計算完了: {len(features_df.columns)}個の特徴量")
            logger.info(f"特徴量タイプ: {features_df.dtypes.value_counts().to_dict()}")

            return features_df

        except Exception as e:
            logger.error(f"特徴量計算エラー: {e}")
            raise

    def _create_dummy_funding_rate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """ダミーのファンディングレートデータを作成"""
        dummy_fr = pd.DataFrame(index=df.index)
        dummy_fr["funding_rate"] = np.random.normal(
            0.0001, 0.0005, len(df)
        )  # 現実的な範囲
        return dummy_fr

    def _create_dummy_open_interest_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """ダミーの建玉残高データを作成"""
        dummy_oi = pd.DataFrame(index=df.index)
        # 価格に連動した建玉残高を生成
        base_oi = 1000000000  # 10億ドル相当
        dummy_oi["open_interest"] = base_oi * (1 + np.random.normal(0, 0.1, len(df)))
        return dummy_oi

    def _create_dummy_external_market_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """ダミーの外部市場データを作成"""
        dummy_ext = pd.DataFrame(index=df.index)
        # S&P500風のデータ
        dummy_ext["sp500"] = 4000 + np.cumsum(np.random.normal(0, 10, len(df)))
        # DXY風のデータ
        dummy_ext["dxy"] = 100 + np.cumsum(np.random.normal(0, 0.5, len(df)))
        # VIX風のデータ
        dummy_ext["vix"] = 20 + np.random.exponential(5, len(df))
        return dummy_ext

    def _create_dummy_fear_greed_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """ダミーのFear & Greedデータを作成"""
        dummy_fg = pd.DataFrame(index=df.index)
        # 0-100の範囲でFear & Greed Index
        dummy_fg["fear_greed_index"] = np.random.randint(0, 101, len(df))
        return dummy_fg

    def prepare_training_data(
        self, features_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        学習用データを準備

        Args:
            features_df: 特徴量DataFrame

        Returns:
            特徴量とラベルのタプル
        """
        logger.info("学習データ準備開始")

        try:
            # ラベル生成（価格変動に基づく3クラス分類）
            close_prices = features_df["Close"]
            returns = close_prices.pct_change(periods=1).shift(-1)  # 1期間後のリターン

            # 閾値設定（標準偏差ベース）
            std_return = returns.std()
            threshold_up = 0.5 * std_return
            threshold_down = -0.5 * std_return

            # ラベル作成
            labels = pd.Series(index=returns.index, dtype=int)
            labels[returns > threshold_up] = 2  # 上昇
            labels[returns < threshold_down] = 0  # 下落
            labels[(returns >= threshold_down) & (returns <= threshold_up)] = (
                1  # レンジ
            )

            # 最後の行は予測できないので除外
            features_clean = features_df.iloc[:-1].copy()
            labels_clean = labels.iloc[:-1].copy()

            # 無効なデータを除外
            valid_mask = ~(features_clean.isnull().any(axis=1) | labels_clean.isnull())
            features_clean = features_clean[valid_mask]
            labels_clean = labels_clean[valid_mask]

            # OHLCV列を除外（特徴量のみ残す）
            feature_columns = [
                col
                for col in features_clean.columns
                if col not in ["Open", "High", "Low", "Close", "Volume"]
            ]
            X = features_clean[feature_columns]

            logger.info(
                f"学習データ準備完了: {len(X)}サンプル, {len(feature_columns)}特徴量"
            )
            logger.info(
                f"ラベル分布: 下落={sum(labels_clean==0)}, レンジ={sum(labels_clean==1)}, 上昇={sum(labels_clean==2)}"
            )

            return X, labels_clean

        except Exception as e:
            logger.error(f"学習データ準備エラー: {e}")
            raise

    def train_baseline_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        ベースラインモデルを学習

        Args:
            X: 特徴量DataFrame
            y: ラベルSeries

        Returns:
            学習結果
        """
        logger.info("ベースラインモデル学習開始")

        try:
            # データ分割（時系列なので時間順に分割）
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

            # モデル学習
            result = self.trainer._train_model_impl(
                X_train,
                X_test,
                y_train,
                y_test,
                n_estimators=100,
                max_depth=10,
                learning_rate=0.1,
                random_state=42,
            )

            # 特徴量重要度を保存
            if (
                hasattr(self.trainer, "model")
                and self.trainer.model is not None
                and self.trainer.is_trained
            ):
                try:
                    feature_importance = self.trainer.get_feature_importance()
                    self.feature_importance_results = feature_importance
                    logger.info(f"特徴量重要度取得完了: {len(feature_importance)}個")
                except Exception as e:
                    logger.warning(f"特徴量重要度取得エラー: {e}")
                    self.feature_importance_results = {}

            self.results["baseline"] = result
            logger.info(
                f"ベースラインモデル学習完了: 精度={result.get('accuracy', 0):.4f}"
            )

            return result

        except Exception as e:
            logger.error(f"ベースラインモデル学習エラー: {e}")
            raise

    def calculate_permutation_importance(
        self, X: pd.DataFrame, y: pd.Series, n_repeats: int = 3
    ) -> Dict[str, float]:
        """
        Permutation Importanceを計算

        Args:
            X: 特徴量DataFrame
            y: ラベルSeries
            n_repeats: 繰り返し回数

        Returns:
            Permutation Importanceの辞書
        """
        logger.info("Permutation Importance計算開始")

        try:
            from sklearn.metrics import accuracy_score

            # データ分割
            split_idx = int(len(X) * 0.8)
            X_test = X.iloc[split_idx:]
            y_test = y.iloc[split_idx:]

            # ベースライン精度
            if not hasattr(self.trainer, "model") or self.trainer.model is None:
                raise ValueError("学習済みモデルがありません")

            # 予測（確率値をクラスに変換）
            y_pred_proba = self.trainer.model.predict(X_test)
            if y_pred_proba.ndim > 1:
                y_pred = np.argmax(y_pred_proba, axis=1)
            else:
                y_pred = y_pred_proba
            baseline_score = accuracy_score(y_test, y_pred)

            # 手動でPermutation Importance計算（LightGBM Booster対応）
            logger.info(f"ベースライン精度: {baseline_score:.4f}")

            # 各特徴量のPermutation Importance計算
            importance_scores = []

            for feature in X.columns:
                scores = []

                for _ in range(n_repeats):
                    # 特徴量をシャッフル
                    X_test_permuted = X_test.copy()
                    X_test_permuted[feature] = np.random.permutation(
                        X_test_permuted[feature].values
                    )

                    # 予測
                    y_pred_perm_proba = self.trainer.model.predict(X_test_permuted)
                    if y_pred_perm_proba.ndim > 1:
                        y_pred_perm = np.argmax(y_pred_perm_proba, axis=1)
                    else:
                        y_pred_perm = y_pred_perm_proba

                    # 精度計算
                    permuted_score = accuracy_score(y_test, y_pred_perm)
                    scores.append(
                        baseline_score - permuted_score
                    )  # 重要度 = ベースライン - シャッフル後

                importance_scores.append(scores)

            # 結果を辞書に変換
            importance_dict = {}
            for i, feature in enumerate(X.columns):
                scores = importance_scores[i]
                importance_dict[feature] = {
                    "importance_mean": np.mean(scores),
                    "importance_std": np.std(scores),
                }

            self.permutation_importance_results = importance_dict
            logger.info("Permutation Importance計算完了")

            return importance_dict

        except Exception as e:
            logger.error(f"Permutation Importance計算エラー: {e}")
            raise

    def identify_noise_features(
        self, X: pd.DataFrame, y: pd.Series = None
    ) -> List[str]:
        """
        ノイズ特徴量を総合的に特定

        Args:
            X: 特徴量DataFrame
            y: ラベルSeries

        Returns:
            ノイズ特徴量のリスト
        """
        logger.info("ノイズ特徴量特定開始")

        try:
            noise_features = set()

            # 1. 低重要度特徴量（Feature Importance下位30%）
            if self.feature_importance_results:
                sorted_importance = sorted(
                    self.feature_importance_results.items(), key=lambda x: x[1]
                )
                low_importance_count = max(1, len(sorted_importance) // 3)  # 下位30%
                low_importance_features = [
                    feature for feature, _ in sorted_importance[:low_importance_count]
                ]
                noise_features.update(low_importance_features)
                logger.info(f"低重要度特徴量: {len(low_importance_features)}個")

            # 2. 負のPermutation Importance特徴量
            if self.permutation_importance_results:
                negative_perm_features = [
                    feature
                    for feature, importance in self.permutation_importance_results.items()
                    if importance["importance_mean"] < 0
                ]
                noise_features.update(negative_perm_features)
                logger.info(
                    f"負のPermutation Importance特徴量: {len(negative_perm_features)}個"
                )

            # 3. 低分散特徴量
            variances = X.var()
            low_var_features = variances[variances < 0.01].index.tolist()
            noise_features.update(low_var_features)
            logger.info(f"低分散特徴量: {len(low_var_features)}個")

            self.noise_features = list(noise_features)
            logger.info(f"総ノイズ特徴量数: {len(self.noise_features)}")

            return self.noise_features

        except Exception as e:
            logger.error(f"ノイズ特徴量特定エラー: {e}")
            raise

    def save_results(self, X: pd.DataFrame, baseline_result: Dict[str, Any]) -> Path:
        """
        分析結果をファイルに保存する

        Args:
            X: 特徴量DataFrame
            baseline_result: ベースラインモデルの学習結果

        Returns:
            保存先ディレクトリのパス
        """
        logger.info("分析結果の保存開始")

        try:
            # タイムスタンプ付きのディレクトリを作成
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_dir = (
                self.output_dir
                / f"{self.symbol.replace('/', '_').replace(':', '')}_{self.timeframe}_{timestamp}"
            )
            result_dir.mkdir(parents=True, exist_ok=True)

            # 1. 分析サマリー
            summary_path = result_dir / "summary.txt"
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write("=== 特徴量ノイズ分析サマリー ===\n")
                f.write(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"シンボル: {self.symbol}\n")
                f.write(f"時間軸: {self.timeframe}\n")
                f.write(f"総特徴量数: {len(X.columns)}\n")
                f.write(f"ノイズ特徴量数: {len(self.noise_features)}\n")
                f.write(f"ベースライン精度: {baseline_result.get('accuracy', 0):.4f}\n")
                if len(X.columns) > 0:
                    noise_ratio = len(self.noise_features) / len(X.columns) * 100
                    f.write(f"ノイズ除去推奨: {noise_ratio:.1f}%の特徴量\n")

            # 2. 特徴量重要度
            if self.feature_importance_results:
                importance_df = pd.DataFrame.from_dict(
                    self.feature_importance_results,
                    orient="index",
                    columns=["importance"],
                )
                importance_df = importance_df.sort_values("importance", ascending=False)
                importance_df.to_csv(
                    result_dir / "feature_importance.csv", encoding="utf-8-sig"
                )

            # 3. Permutation Importance
            if self.permutation_importance_results:
                perm_importance_df = pd.DataFrame.from_dict(
                    self.permutation_importance_results, orient="index"
                )
                perm_importance_df = perm_importance_df.sort_values(
                    "importance_mean", ascending=False
                )
                perm_importance_df.to_csv(
                    result_dir / "permutation_importance.csv", encoding="utf-8-sig"
                )

            # 4. ノイズ特徴量リスト
            if self.noise_features:
                import json

                noise_path = result_dir / "noise_features.json"
                with open(noise_path, "w", encoding="utf-8") as f:
                    json.dump(self.noise_features, f, indent=4, ensure_ascii=False)

            logger.info(f"テキストベースの分析結果を {result_dir} に保存しました")
            return result_dir

        except Exception as e:
            logger.error(f"結果保存エラー: {e}")
            raise

    def plot_feature_importance(self, result_dir: Path, top_n: int = 30):
        """
        特徴量重要度を可視化して保存

        Args:
            result_dir: 保存先ディレクトリ
            top_n: 上位何件を表示するか
        """
        if not self.feature_importance_results:
            logger.warning("特徴量重要度のデータがないため、グラフを作成できません")
            return

        try:
            sorted_importance = sorted(
                self.feature_importance_results.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:top_n]
            df = pd.DataFrame(sorted_importance, columns=["Feature", "Importance"])

            plt.figure(figsize=(12, max(8, top_n * 0.3)))
            sns.barplot(x="Importance", y="Feature", data=df, palette="viridis")
            plt.title(f"Feature Importance (Top {top_n})", fontsize=16)
            plt.xlabel("Importance", fontsize=12)
            plt.ylabel("Feature", fontsize=12)
            plt.tight_layout()
            plt.savefig(result_dir / "feature_importance.png")
            plt.close()
            logger.info("特徴量重要度グラフを保存しました")

        except Exception as e:
            logger.error(f"特徴量重要度グラフ作成エラー: {e}")

    def plot_permutation_importance(self, result_dir: Path, top_n: int = 30):
        """
        Permutation Importanceを可視化して保存

        Args:
            result_dir: 保存先ディレクトリ
            top_n: 上位何件を表示するか
        """
        if not self.permutation_importance_results:
            logger.warning(
                "Permutation Importanceのデータがないため、グラフを作成できません"
            )
            return

        try:
            sorted_perm = sorted(
                self.permutation_importance_results.items(),
                key=lambda x: x[1]["importance_mean"],
                reverse=True,
            )[:top_n]

            features = [item[0] for item in sorted_perm]
            means = [item[1]["importance_mean"] for item in sorted_perm]
            stds = [item[1]["importance_std"] for item in sorted_perm]

            df = pd.DataFrame(
                {"Feature": features, "Importance_mean": means, "Importance_std": stds}
            )

            plt.figure(figsize=(12, max(8, top_n * 0.3)))
            plt.barh(
                df["Feature"],
                df["Importance_mean"],
                xerr=df["Importance_std"],
                align="center",
                capsize=5,
                color=sns.color_palette("plasma", len(df)),
            )
            plt.title(f"Permutation Importance (Top {top_n})", fontsize=16)
            plt.xlabel("Importance", fontsize=12)
            plt.ylabel("Feature", fontsize=12)
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(result_dir / "permutation_importance.png")
            plt.close()
            logger.info("Permutation Importanceグラフを保存しました")

        except Exception as e:
            logger.error(f"Permutation Importanceグラフ作成エラー: {e}")


def main():
    """
    メイン実行関数
    """
    try:
        logger.info("=== 特徴量ノイズ分析開始（SQLite版） ===")

        # 分析器を初期化
        analyzer = FeatureNoiseAnalyzerSQLite(symbol="BTC/USDT:USDT", timeframe="1h")

        # 1. データ読み込み
        logger.info("1. データ読み込み")
        df = analyzer.load_data(limit=1000)  # テスト用に少なめに設定

        # 2. 特徴量計算
        logger.info("2. 特徴量計算")
        features_df = analyzer.calculate_features(df)

        # 3. 学習データ準備
        logger.info("3. 学習データ準備")
        X, y = analyzer.prepare_training_data(features_df)

        # 4. ベースラインモデル学習
        logger.info("4. ベースラインモデル学習")
        baseline_result = analyzer.train_baseline_model(X, y)

        # 5. Permutation Importance計算
        logger.info("5. Permutation Importance計算")
        analyzer.calculate_permutation_importance(X, y)

        # 6. ノイズ特徴量特定
        logger.info("6. ノイズ特徴量特定")
        noise_features = analyzer.identify_noise_features(X, y)

        # 7. 結果の保存と可視化
        logger.info("7. 結果の保存と可視化")
        result_dir = analyzer.save_results(X, baseline_result)
        analyzer.plot_feature_importance(result_dir)
        analyzer.plot_permutation_importance(result_dir)

        # 8. 結果の表示
        logger.info("8. 結果表示")

        # 特徴量重要度の表示
        if analyzer.feature_importance_results:
            print("\n=== 特徴量重要度 Top 10 ===")
            sorted_importance = sorted(
                analyzer.feature_importance_results.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:10]
            for i, (feature, importance) in enumerate(sorted_importance, 1):
                print(f"{i:2d}. {feature}: {importance:.4f}")

        # Permutation Importanceの表示
        if analyzer.permutation_importance_results:
            print("\n=== Permutation Importance Top 10 ===")
            sorted_perm = sorted(
                analyzer.permutation_importance_results.items(),
                key=lambda x: x[1]["importance_mean"],
                reverse=True,
            )[:10]
            for i, (feature, importance) in enumerate(sorted_perm, 1):
                mean = importance["importance_mean"]
                std = importance["importance_std"]
                print(f"{i:2d}. {feature}: {mean:.4f} ± {std:.4f}")

        # ノイズ特徴量の表示
        if noise_features:
            print(f"\n=== ノイズ特徴量 ({len(noise_features)}個) ===")
            for i, feature in enumerate(noise_features[:10], 1):
                print(f"{i:2d}. {feature}")
            if len(noise_features) > 10:
                print(f"... 他 {len(noise_features) - 10}個")

        # 結果サマリー表示
        logger.info("=== 分析完了 ===")
        logger.info(f"総特徴量数: {len(X.columns)}")
        logger.info(f"ノイズ特徴量数: {len(noise_features)}")
        logger.info(f"ベースライン精度: {baseline_result.get('accuracy', 0):.4f}")
        logger.info(f"結果保存先: {result_dir}")

        print("\n" + "=" * 60)
        print("特徴量ノイズ分析が完了しました！")
        print(f"総特徴量数: {len(X.columns)}")
        print(f"ノイズ特徴量数: {len(noise_features)}")
        print(f"ベースライン精度: {baseline_result.get('accuracy', 0):.4f}")
        if len(X.columns) > 0:
            noise_ratio = len(noise_features) / len(X.columns) * 100
            print(f"ノイズ除去推奨: {noise_ratio:.1f}%の特徴量")
        print(f"結果保存先: {result_dir}")
        print("=" * 60)

    except Exception as e:
        logger.error(f"分析実行エラー: {e}")
        raise


if __name__ == "__main__":
    main()
