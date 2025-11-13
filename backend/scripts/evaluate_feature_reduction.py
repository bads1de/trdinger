"""
特徴量削減後の性能評価スクリプト

19個の特徴量を削除し、79個から60個に削減した後の性能を比較評価します。
削除前後で同じデータ、同じハイパーパラメータを使用して公平な比較を行います。

実行方法:
    cd backend
    python scripts/evaluate_feature_reduction.py

出力:
    - コンソール: 詳細な比較レポート
    - CSV: backend/feature_reduction_evaluation.csv
"""

import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# パスを追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config.unified_config import unified_config
from app.services.ml.feature_engineering.feature_engineering_service import (
    FeatureEngineeringService,
)
from app.services.ml.models.lightgbm import LightGBMModel
# ラベル生成は簡易的な実装を使用するため、インポート不要
from database.connection import SessionLocal
from database.repositories.ohlcv_repository import OHLCVRepository

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class FeatureReductionEvaluator:
    """特徴量削減の性能評価クラス"""

    def __init__(self, min_samples: int = 1500):
        """
        初期化

        Args:
            min_samples: 最小サンプル数
        """
        self.min_samples = min_samples
        self.feature_service = FeatureEngineeringService()
        
        # 削除された特徴量のリスト（19個）
        self.removed_features = [
            # 高相関による削除(5個)
            "macd",
            "Stochastic_K",
            "Near_Resistance",
            "MA_Long",
            "BB_Position",
            # 低重要度による削除(14個)
            "close_lag_24",
            "cumulative_returns_24",
            "Close_mean_20",
            "Local_Max",
            "Aroon_Up",
            "BB_Lower",
            "Resistance_Level",
            "BB_Middle",
            "stochastic_k",
            "rsi_14",
            "bb_lower_20",
            "bb_upper_20",
            "stochastic_d",
            "Local_Min",
        ]

    def load_data(
        self, symbol: str = "BTC/USDT:USDT", timeframe: str = "1h"
    ) -> pd.DataFrame:
        """
        データベースからデータを読み込み

        Args:
            symbol: 取引ペア
            timeframe: 時間軸

        Returns:
            OHLCVデータのDataFrame
        """
        logger.info(f"📊 データ読み込み開始: {symbol} {timeframe}")

        db = SessionLocal()
        try:
            repo = OHLCVRepository(db)
            df = repo.get_ohlcv_dataframe(
                symbol=symbol, timeframe=timeframe, limit=self.min_samples + 500
            )

            if df.empty:
                raise ValueError(f"データが見つかりません: {symbol} {timeframe}")

            if len(df) < self.min_samples:
                raise ValueError(
                    f"データ不足: {len(df)}件 (必要: {self.min_samples}件以上)"
                )

            logger.info(f"✅ データ読み込み完了: {len(df)}件")
            return df

        finally:
            db.close()

    def generate_features(
        self, ohlcv_data: pd.DataFrame, use_allowlist: bool = True
    ) -> pd.DataFrame:
        """
        特徴量を生成

        Args:
            ohlcv_data: OHLCVデータ
            use_allowlist: allowlistを使用するか（False=全特徴量）

        Returns:
            特徴量DataFrame
        """
        if use_allowlist:
            logger.info("🔧 特徴量生成開始（allowlist適用: 60個）")
        else:
            logger.info("🔧 特徴量生成開始（全特徴量: 79個）")

        # allowlistを一時的に無効化する場合
        original_allowlist = None
        if not use_allowlist:
            original_allowlist = unified_config.ml.feature_engineering.feature_allowlist
            unified_config.ml.feature_engineering.feature_allowlist = None

        try:
            features = self.feature_service.calculate_advanced_features(
                ohlcv_data=ohlcv_data
            )

            logger.info(f"✅ 特徴量生成完了: {len(features.columns)}個")
            return features

        finally:
            # allowlistを元に戻す
            if not use_allowlist and original_allowlist is not None:
                unified_config.ml.feature_engineering.feature_allowlist = (
                    original_allowlist
                )

    def generate_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        ラベルを生成（シンプルな価格変動ベース）

        Args:
            df: 特徴量DataFrame

        Returns:
            ラベルSeries (0: DOWN, 1: RANGE, 2: UP)
        """
        logger.info("🏷️ ラベル生成開始")

        # 次の期間の価格変動率を計算（4本先を見る）
        horizon = 4
        future_returns = df["close"].pct_change(horizon).shift(-horizon)

        # 閾値を設定（変動率の標準偏差の0.5倍）
        threshold = future_returns.std() * 0.5

        # 3クラスに分類
        labels = pd.Series(index=df.index, dtype=int)
        labels[future_returns > threshold] = 2  # UP
        labels[future_returns < -threshold] = 0  # DOWN
        labels[
            (future_returns >= -threshold) & (future_returns <= threshold)
        ] = 1  # RANGE

        # NaNを除去
        valid_mask = labels.notna() & future_returns.notna()
        labels = labels[valid_mask]

        logger.info(f"✅ ラベル生成完了: {len(labels)}サンプル")
        logger.info(f"クラス分布: {dict(labels.value_counts().sort_index())}")
        logger.info(f"閾値: ±{threshold:.4f} ({threshold*100:.2f}%)")

        return labels

    def prepare_data(
        self, features: pd.DataFrame, labels: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        データを準備（分割とクリーニング）

        Args:
            features: 特徴量DataFrame
            labels: ラベルSeries

        Returns:
            (X_train, X_val, y_train, y_val)
        """
        logger.info("⚙️ データ準備開始")

        # インデックスを揃える
        common_index = features.index.intersection(labels.index)
        features = features.loc[common_index]
        labels = labels.loc[common_index]

        # 基本カラムを除外
        exclude_cols = ["open", "high", "low", "close", "volume"]
        feature_cols = [col for col in features.columns if col not in exclude_cols]
        X = features[feature_cols].copy()

        # 無限値とNaNを処理
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())

        # データを分割（80:20）
        X_train, X_val, y_train, y_val = train_test_split(
            X, labels, test_size=0.2, random_state=42, stratify=labels
        )

        logger.info(f"✅ データ準備完了:")
        logger.info(f"  - 学習データ: {len(X_train):,}サンプル")
        logger.info(f"  - 検証データ: {len(X_val):,}サンプル")
        logger.info(f"  - 特徴量数: {len(feature_cols)}個")

        return X_train, X_val, y_train, y_val

    def train_and_evaluate(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
        label: str,
    ) -> Dict:
        """
        モデルを学習して評価

        Args:
            X_train: 学習用特徴量
            X_val: 検証用特徴量
            y_train: 学習用ラベル
            y_val: 検証用ラベル
            label: 評価ラベル（"削除前" or "削除後"）

        Returns:
            評価結果の辞書
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"🤖 {label}モデルの学習・評価")
        logger.info(f"{'='*60}")

        # モデル作成
        model = LightGBMModel(random_state=42, n_estimators=100, learning_rate=0.1)

        # 学習時間計測
        logger.info("学習を開始...")
        train_start = time.time()
        training_result = model._train_model_impl(X_train, X_val, y_train, y_val)
        train_time = time.time() - train_start

        # 予測時間計測
        logger.info("予測を実行...")
        predict_start = time.time()
        y_pred_proba = model.predict_proba(X_val)
        predict_time = time.time() - predict_start

        # 結果をまとめる
        result = {
            "label": label,
            "feature_count": len(X_train.columns),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "train_time": train_time,
            "predict_time": predict_time,
            **training_result,
        }

        # 結果を表示
        logger.info(f"\n📊 {label}の評価結果:")
        logger.info(f"  - 特徴量数: {result['feature_count']}個")
        logger.info(f"  - Accuracy: {result.get('accuracy', 0.0):.4f}")
        logger.info(f"  - Precision: {result.get('precision', 0.0):.4f}")
        logger.info(f"  - Recall: {result.get('recall', 0.0):.4f}")
        logger.info(f"  - F1-Score: {result.get('f1_score', 0.0):.4f}")
        logger.info(f"  - AUC-ROC: {result.get('roc_auc', 0.0):.4f}")
        logger.info(f"  - 学習時間: {train_time:.2f}秒")
        logger.info(f"  - 予測時間: {predict_time:.4f}秒")

        return result

    def compare_results(
        self, before_result: Dict, after_result: Dict
    ) -> pd.DataFrame:
        """
        結果を比較

        Args:
            before_result: 削除前の結果
            after_result: 削除後の結果

        Returns:
            比較結果のDataFrame
        """
        logger.info(f"\n{'='*60}")
        logger.info("📊 結果比較")
        logger.info(f"{'='*60}")

        # 比較する指標
        metrics = [
            "feature_count",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
            "train_time",
            "predict_time",
        ]

        comparison_data = []
        for metric in metrics:
            before_val = before_result.get(metric, 0.0)
            after_val = after_result.get(metric, 0.0)

            # 変化率を計算
            if before_val != 0:
                change_pct = ((after_val - before_val) / before_val) * 100
            else:
                change_pct = 0.0

            comparison_data.append(
                {
                    "metric": metric,
                    "before": before_val,
                    "after": after_val,
                    "change": after_val - before_val,
                    "change_pct": change_pct,
                }
            )

        comparison_df = pd.DataFrame(comparison_data)
        return comparison_df

    def print_report(
        self, comparison_df: pd.DataFrame, total_samples: int, class_distribution: Dict
    ) -> None:
        """
        詳細レポートを出力

        Args:
            comparison_df: 比較結果DataFrame
            total_samples: 総サンプル数
            class_distribution: クラス分布
        """
        print("\n" + "=" * 80)
        print("特徴量削減による性能評価レポート")
        print("=" * 80)

        print("\n【データセット】")
        print(f"- 総サンプル数: {total_samples:,}件")
        train_samples = int(
            comparison_df[comparison_df["metric"] == "feature_count"]["before"].iloc[0]
        )
        val_samples = total_samples - train_samples
        print(f"- 学習データ: {train_samples:,}件")
        print(f"- 検証データ: {val_samples:,}件")

        # クラス分布を表示
        class_str = ", ".join(
            [f"クラス{k}={v}" for k, v in sorted(class_distribution.items())]
        )
        print(f"- クラス分布: {class_str}")

        # 削除前の結果
        print("\n【削除前】特徴量数: 79個")
        for _, row in comparison_df.iterrows():
            if row["metric"] == "feature_count":
                continue
            if row["metric"] in ["train_time", "predict_time"]:
                print(f"- {row['metric']}: {row['before']:.2f}秒")
            else:
                print(f"- {row['metric']}: {row['before']:.4f}")

        # 削除後の結果
        print("\n【削除後】特徴量数: 60個")
        for _, row in comparison_df.iterrows():
            if row["metric"] == "feature_count":
                continue

            change_sign = "+" if row["change_pct"] >= 0 else ""
            if row["metric"] in ["train_time", "predict_time"]:
                print(
                    f"- {row['metric']}: {row['after']:.2f}秒 "
                    f"({change_sign}{row['change_pct']:.1f}%)"
                )
            else:
                print(
                    f"- {row['metric']}: {row['after']:.4f} "
                    f"({change_sign}{row['change_pct']:.2f}%)"
                )

        # 削除された特徴量
        print(f"\n【削除された特徴量】19個:")
        for i, feature in enumerate(self.removed_features, 1):
            print(f"  {i:2d}. {feature}")

        # 結論
        print("\n【結論】")

        # 性能変化を判定
        avg_performance_change = comparison_df[
            comparison_df["metric"].isin(
                ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
            )
        ]["change_pct"].mean()

        if abs(avg_performance_change) < 1.0:
            performance_status = "維持"
            recommendation = "削除を推奨"
        elif avg_performance_change > 0:
            performance_status = "改善"
            recommendation = "削除を強く推奨"
        else:
            performance_status = "低下"
            if abs(avg_performance_change) < 5.0:
                recommendation = "削除を推奨（性能低下は許容範囲内）"
            else:
                recommendation = "削除を非推奨"

        print(f"- 予測性能: {performance_status} (平均変化率: {avg_performance_change:+.2f}%)")

        # 学習速度改善
        train_time_change = comparison_df[comparison_df["metric"] == "train_time"][
            "change_pct"
        ].iloc[0]
        print(f"- 学習速度: {abs(train_time_change):.1f}%改善")

        # 推奨
        print(f"- 推奨: {recommendation}")

        print("\n詳細結果: backend/feature_reduction_evaluation.csv")
        print("=" * 80)

    def save_results(
        self, comparison_df: pd.DataFrame, output_path: str = "backend/feature_reduction_evaluation.csv"
    ) -> None:
        """
        結果をCSVファイルに保存

        Args:
            comparison_df: 比較結果DataFrame
            output_path: 出力ファイルパス
        """
        comparison_df.to_csv(output_path, index=False, encoding="utf-8")
        logger.info(f"✅ 結果をCSVに保存: {output_path}")

    def run(
        self, symbol: str = "BTC/USDT:USDT", timeframe: str = "1h"
    ) -> pd.DataFrame:
        """
        評価を実行

        Args:
            symbol: 取引ペア
            timeframe: 時間軸

        Returns:
            比較結果DataFrame
        """
        try:
            logger.info("\n🚀 特徴量削減の性能評価を開始します\n")

            # 1. データ読み込み
            ohlcv_data = self.load_data(symbol, timeframe)

            # 2. 削除前の特徴量生成（全特徴量）
            features_before = self.generate_features(ohlcv_data, use_allowlist=False)

            # 3. ラベル生成
            labels = self.generate_labels(features_before)

            # 4. 削除前のデータ準備と評価
            X_train_before, X_val_before, y_train, y_val = self.prepare_data(
                features_before, labels
            )
            before_result = self.train_and_evaluate(
                X_train_before, X_val_before, y_train, y_val, "削除前（79個）"
            )

            # 5. 削除後の特徴量生成（allowlist適用）
            features_after = self.generate_features(ohlcv_data, use_allowlist=True)

            # 6. 削除後のデータ準備と評価（同じラベルを使用）
            X_train_after, X_val_after, y_train_after, y_val_after = self.prepare_data(
                features_after, labels
            )
            after_result = self.train_and_evaluate(
                X_train_after, X_val_after, y_train_after, y_val_after, "削除後（60個）"
            )

            # 7. 結果比較
            comparison_df = self.compare_results(before_result, after_result)

            # 8. レポート出力
            total_samples = len(X_train_before) + len(X_val_before)
            class_distribution = dict(y_train.value_counts().sort_index())
            self.print_report(comparison_df, total_samples, class_distribution)

            # 9. CSV保存
            self.save_results(comparison_df)

            logger.info("\n✅ 評価が正常に完了しました")
            return comparison_df

        except Exception as e:
            logger.error(f"\n❌ 評価エラー: {e}", exc_info=True)
            raise


def main():
    """メイン関数"""
    logger.info("=" * 80)
    logger.info("特徴量削減性能評価スクリプト")
    logger.info("=" * 80)

    evaluator = FeatureReductionEvaluator(min_samples=1500)

    try:
        results = evaluator.run()
        print("\n✅ 分析が正常に完了しました。")
        print("詳細結果: backend/feature_reduction_evaluation.csv")

    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()