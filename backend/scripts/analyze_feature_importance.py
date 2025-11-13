"""
科学的根拠に基づいた特徴量重要度分析スクリプト

複数の手法を組み合わせて特徴量の重要度を評価し、
削除推奨特徴量を科学的根拠に基づいて特定します。

使用手法:
1. LightGBM Feature Importance (Gain-based)
2. Permutation Importance
3. 相関分析

実行方法:
    python backend/scripts/analyze_feature_importance.py

出力:
    - コンソール: 分析結果の詳細レポート
    - CSV: backend/feature_importance_analysis.csv
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

# パスを追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.ml.feature_engineering.feature_engineering_service import (
    FeatureEngineeringService,
)
from database.connection import SessionLocal
from database.repositories.ohlcv_repository import OHLCVRepository

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class FeatureImportanceAnalyzer:
    """特徴量重要度分析クラス"""

    def __init__(self, min_samples: int = 1000):
        """
        初期化

        Args:
            min_samples: 最小サンプル数
        """
        self.min_samples = min_samples
        self.feature_service = FeatureEngineeringService()
        self.results: Dict = {}

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
        logger.info(f"データ読み込み開始: {symbol} {timeframe}")

        db = SessionLocal()
        try:
            repo = OHLCVRepository(db)
            df = repo.get_ohlcv_dataframe(
                symbol=symbol, timeframe=timeframe, limit=self.min_samples + 500
            )

            if df.empty:
                raise ValueError(f"データが見つかりません: {symbol} {timeframe}")

            logger.info(f"データ読み込み完了: {len(df)}件")
            return df

        finally:
            db.close()

    def generate_features(self, ohlcv_data: pd.DataFrame) -> pd.DataFrame:
        """
        特徴量を生成

        Args:
            ohlcv_data: OHLCVデータ

        Returns:
            特徴量DataFrame
        """
        logger.info("特徴量生成開始")

        features = self.feature_service.calculate_advanced_features(
            ohlcv_data=ohlcv_data
        )

        logger.info(f"特徴量生成完了: {len(features.columns)}個")
        return features

    def generate_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        ラベルを生成（価格変動の3クラス分類）

        Args:
            df: 特徴量DataFrame

        Returns:
            ラベルSeries (0: DOWN, 1: RANGE, 2: UP)
        """
        logger.info("ラベル生成開始")

        # 次の期間の価格変動率を計算
        future_returns = df["close"].pct_change(5).shift(-5)

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

        logger.info(f"ラベル生成完了: {len(labels)}サンプル")
        logger.info(f"ラベル分布: {labels.value_counts().to_dict()}")

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
        logger.info("データ準備開始")

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

        logger.info(f"データ準備完了:")
        logger.info(f"  学習データ: {len(X_train)}サンプル")
        logger.info(f"  検証データ: {len(X_val)}サンプル")
        logger.info(f"  特徴量数: {len(feature_cols)}個")

        return X_train, X_val, y_train, y_val

    def analyze_lightgbm_importance(
        self, X_train: pd.DataFrame, X_val: pd.DataFrame, y_train: pd.Series
    ) -> Dict[str, float]:
        """
        LightGBM Feature Importanceを計算

        Args:
            X_train: 学習用特徴量
            X_val: 検証用特徴量
            y_train: 学習用ラベル

        Returns:
            特徴量重要度の辞書
        """
        logger.info("【手法1】LightGBM Feature Importance 計算開始")

        model = LGBMClassifier(
            n_estimators=100,
            random_state=42,
            verbose=-1,
        )

        model.fit(X_train, y_train)

        # Gain-based重要度を取得
        importance = model.feature_importances_
        feature_importance = dict(zip(X_train.columns, importance))

        # 正規化（0-1）
        max_importance = max(importance)
        if max_importance > 0:
            feature_importance = {
                k: v / max_importance for k, v in feature_importance.items()
            }

        logger.info("LightGBM重要度計算完了")
        return feature_importance

    def analyze_permutation_importance(
        self,
        model: LGBMClassifier,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Dict[str, float]:
        """
        Permutation Importanceを計算

        Args:
            model: 学習済みモデル
            X_val: 検証用特徴量
            y_val: 検証用ラベル

        Returns:
            特徴量重要度の辞書
        """
        logger.info("【手法2】Permutation Importance 計算開始")

        result = permutation_importance(
            model, X_val, y_val, n_repeats=10, random_state=42, n_jobs=-1
        )

        perm_importance = dict(zip(X_val.columns, result.importances_mean))

        # 正規化（0-1）
        max_importance = max(result.importances_mean)
        if max_importance > 0:
            perm_importance = {
                k: v / max_importance for k, v in perm_importance.items()
            }

        logger.info("Permutation重要度計算完了")
        return perm_importance

    def analyze_correlation(
        self, X: pd.DataFrame, threshold: float = 0.95
    ) -> Dict[str, Tuple[str, float]]:
        """
        相関分析を実行

        Args:
            X: 特徴量DataFrame
            threshold: 高相関の閾値

        Returns:
            高相関ペアの辞書 {feature: (correlated_feature, correlation)}
        """
        logger.info("【手法3】相関分析 開始")

        # 相関行列を計算
        corr_matrix = X.corr().abs()

        # 上三角行列から高相関ペアを抽出
        high_corr_pairs = {}
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > threshold:
                    feature1 = corr_matrix.columns[i]
                    feature2 = corr_matrix.columns[j]
                    corr_value = corr_matrix.iloc[i, j]

                    # まだ記録されていない特徴量のみ追加
                    if feature1 not in high_corr_pairs:
                        high_corr_pairs[feature1] = (feature2, corr_value)
                    if feature2 not in high_corr_pairs:
                        high_corr_pairs[feature2] = (feature1, corr_value)

        logger.info(f"高相関ペア検出: {len(high_corr_pairs)}個")
        return high_corr_pairs

    def integrate_results(
        self,
        lgbm_importance: Dict[str, float],
        perm_importance: Dict[str, float],
        high_corr_pairs: Dict[str, Tuple[str, float]],
    ) -> pd.DataFrame:
        """
        結果を統合

        Args:
            lgbm_importance: LightGBM重要度
            perm_importance: Permutation重要度
            high_corr_pairs: 高相関ペア

        Returns:
            統合結果のDataFrame
        """
        logger.info("結果統合開始")

        # 全特徴量のリストを取得
        all_features = set(lgbm_importance.keys()) | set(perm_importance.keys())

        results = []
        for feature in all_features:
            lgbm_score = lgbm_importance.get(feature, 0.0)
            perm_score = perm_importance.get(feature, 0.0)
            avg_score = (lgbm_score + perm_score) / 2

            # 相関情報
            corr_feature = ""
            corr_value = 0.0
            if feature in high_corr_pairs:
                corr_feature, corr_value = high_corr_pairs[feature]

            # 削除推奨の判定
            recommendation = "keep"
            reason = ""

            if avg_score < 0.1:
                recommendation = "remove"
                reason = "低重要度"
            elif corr_value > 0.95:
                # 相関ペアの片方を削除（重要度が低い方）
                if feature in high_corr_pairs:
                    corr_feat = high_corr_pairs[feature][0]
                    corr_feat_score = (
                        lgbm_importance.get(corr_feat, 0.0)
                        + perm_importance.get(corr_feat, 0.0)
                    ) / 2
                    if avg_score < corr_feat_score:
                        recommendation = "remove"
                        reason = f"{corr_feature}と高相関(r={corr_value:.3f})"

            results.append(
                {
                    "feature_name": feature,
                    "lgbm_importance": lgbm_score,
                    "perm_importance": perm_score,
                    "avg_importance": avg_score,
                    "corr_feature": corr_feature,
                    "corr_value": corr_value,
                    "recommendation": recommendation,
                    "reason": reason,
                }
            )

        df = pd.DataFrame(results)
        df = df.sort_values("avg_importance", ascending=False)

        logger.info("結果統合完了")
        return df

    def print_report(self, results_df: pd.DataFrame) -> None:
        """
        分析結果レポートを出力

        Args:
            results_df: 結果DataFrame
        """
        print("\n" + "=" * 80)
        print("特徴量重要度分析結果")
        print("=" * 80)

        print(f"\n総特徴量数: {len(results_df)}個")

        # 上位特徴量
        print("\n【統合結果】平均重要度スコア 上位20特徴量:")
        top_20 = results_df.head(20)
        for idx, row in top_20.iterrows():
            print(
                f"  {idx+1:2d}. {row['feature_name']:30s} "
                f"平均: {row['avg_importance']:.4f} "
                f"(LGB: {row['lgbm_importance']:.4f}, Perm: {row['perm_importance']:.4f})"
            )

        # 削除推奨特徴量
        remove_features = results_df[results_df["recommendation"] == "remove"]
        print(f"\n【削除推奨特徴量】 {len(remove_features)}個:")
        for idx, row in remove_features.iterrows():
            print(
                f"  - {row['feature_name']:30s} "
                f"スコア: {row['avg_importance']:.4f} "
                f"理由: {row['reason']}"
            )

        # 保持推奨特徴量
        keep_features = results_df[results_df["recommendation"] == "keep"]
        print(f"\n残存推奨特徴量: {len(keep_features)}個")

        print("\n" + "=" * 80)

    def save_results(self, results_df: pd.DataFrame, output_path: str) -> None:
        """
        結果をCSVファイルに保存

        Args:
            results_df: 結果DataFrame
            output_path: 出力ファイルパス
        """
        results_df.to_csv(output_path, index=False, encoding="utf-8")
        logger.info(f"結果をCSVに保存: {output_path}")

    def run(
        self, symbol: str = "BTC/USDT:USDT", timeframe: str = "1h"
    ) -> pd.DataFrame:
        """
        分析を実行

        Args:
            symbol: 取引ペア
            timeframe: 時間軸

        Returns:
            結果DataFrame
        """
        try:
            # 1. データ読み込み
            ohlcv_data = self.load_data(symbol, timeframe)

            # 2. 特徴量生成
            features = self.generate_features(ohlcv_data)

            # 3. ラベル生成
            labels = self.generate_labels(features)

            # 4. データ準備
            X_train, X_val, y_train, y_val = self.prepare_data(features, labels)

            # 5. LightGBM重要度
            lgbm_importance = self.analyze_lightgbm_importance(
                X_train, X_val, y_train
            )

            # 6. モデルを学習（Permutation用）
            model = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
            model.fit(X_train, y_train)

            # 7. Permutation重要度
            perm_importance = self.analyze_permutation_importance(model, X_val, y_val)

            # 8. 相関分析
            X_all = pd.concat([X_train, X_val])
            high_corr_pairs = self.analyze_correlation(X_all)

            # 9. 結果統合
            results_df = self.integrate_results(
                lgbm_importance, perm_importance, high_corr_pairs
            )

            # 10. レポート出力
            self.print_report(results_df)

            # 11. CSV保存
            output_path = "backend/feature_importance_analysis.csv"
            self.save_results(results_df, output_path)

            logger.info("分析完了")
            return results_df

        except Exception as e:
            logger.error(f"分析エラー: {e}")
            raise


def main():
    """メイン関数"""
    analyzer = FeatureImportanceAnalyzer(min_samples=1000)

    try:
        results = analyzer.run()
        print(f"\n分析が正常に完了しました。")
        print(f"詳細結果: backend/feature_importance_analysis.csv")

    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()