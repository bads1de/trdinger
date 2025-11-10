"""
特徴量寄与度分析スクリプト

DBから実際のデータを取得し、全特徴量の寄与度を分析して、
低寄与度の特徴量を特定します。

TimeSeriesSplitを使用した時系列クロスバリデーションにより、
時系列データの特性を考慮した評価を実施します（将来的な拡張用）。
現在はRandomForestRegressorで特徴量重要度を分析します。

実行方法:
    cd backend
    python -m scripts.feature_evaluation.analyze_feature_importance

設定:
    - ターゲット変数: forward return (1時間先の収益率)
    - RandomForestパラメータ: ml_configから読み込み
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config.unified_config import unified_config
from scripts.feature_evaluation.common_feature_evaluator import (
    CommonFeatureEvaluator,
    EvaluationData,
)

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 分析基準
IMPORTANCE_THRESHOLD = 0.001  # RF重要度の閾値
CORRELATION_THRESHOLD = 0.01  # 相関係数の閾値
VARIANCE_THRESHOLD = 0.0001  # 分散の閾値
MIN_SAMPLES = 1000  # 最小サンプル数


class FeatureImportanceAnalyzer:
    """特徴量寄与度分析クラス"""

    def __init__(self):
        """初期化"""
        self.common = CommonFeatureEvaluator()
        self.results = {}

    def __enter__(self):
        """コンテキストマネージャー: 入場"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャー: 退場"""
        self.common.close()

    def fetch_data(
        self, symbol: str = "BTC/USDT:USDT", limit: int = 2000
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        DBからデータを取得

        Args:
            symbol: 取引ペア
            limit: 取得件数

        Returns:
            (OHLCV, FR, OI)のタプル
        """
        logger.info(f"データ取得開始: {symbol}, limit={limit}")

        try:
            data = self.common.fetch_data(symbol=symbol, timeframe="1h", limit=limit)
            ohlcv_df = data.ohlcv

            if ohlcv_df.empty:
                logger.warning(f"OHLCVデータが見つかりません: {symbol}")
                return pd.DataFrame(), None, None

            logger.info(f"OHLCV: {len(ohlcv_df)}行取得")

            # 時間範囲を取得
            start_time = ohlcv_df.index.min()
            end_time = ohlcv_df.index.max()

            # ファンディングレートデータ取得
            try:
                fr_records = self.fr_repo.get_funding_rate_data(
                    symbol=symbol, start_time=start_time, end_time=end_time
                )
                if fr_records:
                    fr_df = self.fr_repo.to_dataframe(
                        records=fr_records,
                        column_mapping={
                            "funding_timestamp": "funding_timestamp",
                            "funding_rate": "funding_rate",
                        },
                        index_column="funding_timestamp",
                    )
                    logger.info(f"FR: {len(fr_df)}行取得")
                else:
                    fr_df = None
                    logger.warning("ファンディングレートデータなし")
            except Exception as e:
                logger.warning(f"FR取得エラー: {e}")
                fr_df = None

            # オープンインタレストデータ取得
            try:
                oi_records = self.oi_repo.get_open_interest_data(
                    symbol=symbol, start_time=start_time, end_time=end_time
                )
                if oi_records:
                    oi_df = pd.DataFrame(
                        [
                            {
                                "data_timestamp": r.data_timestamp,
                                "open_interest_value": r.open_interest_value,
                            }
                            for r in oi_records
                        ]
                    )
                    oi_df.set_index("data_timestamp", inplace=True)
                    logger.info(f"OI: {len(oi_df)}行取得")
                else:
                    oi_df = None
                    logger.warning("オープンインタレストデータなし")
            except Exception as e:
                logger.warning(f"OI取得エラー: {e}")
                oi_df = None

            return ohlcv_df, fr_df, oi_df

        except Exception as e:
            logger.error(f"データ取得エラー: {e}")
            raise

    def calculate_features(
        self,
        ohlcv_df: pd.DataFrame,
        fr_df: Optional[pd.DataFrame],
        oi_df: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """
        特徴量計算（削減後の全特徴量147個を分析）

        Args:
            ohlcv_df: OHLCVデータ
            fr_df: ファンディングレートデータ
            oi_df: オープンインタレストデータ

        Returns:
            特徴量DataFrame
        """
        logger.info("特徴量計算開始（削減後の全特徴量147個を分析）")

        try:
            data = EvaluationData(ohlcv=ohlcv_df, fr=fr_df, oi=oi_df)
            features_df = self.common.build_basic_features(
                data=data,
                skip_crypto_and_advanced=False,
            )

            # 元のOHLCVカラムを除外
            ohlcv_cols = ["open", "high", "low", "close", "volume"]
            feature_cols = [col for col in features_df.columns if col not in ohlcv_cols]

            logger.info(f"特徴量計算完了: {len(feature_cols)}個の特徴量")
            return features_df[feature_cols]

        except Exception as e:
            logger.error(f"特徴量計算エラー: {e}")
            raise

    def create_target_variables(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        ターゲット変数作成（複数の時間軸）

        Args:
            df: 特徴量が計算されたDataFrame（closeカラムを含む）

        Returns:
            ターゲット変数の辞書
        """
        targets = {}

        # dfにcloseカラムがあることを確認
        if "close" not in df.columns:
            logger.error("closeカラムが見つかりません")
            return targets

        # 1時間先の収益率
        targets["return_1h"] = df["close"].pct_change(1).shift(-1)

        # 4時間先の収益率
        targets["return_4h"] = df["close"].pct_change(4).shift(-4)

        # 24時間先の収益率
        targets["return_24h"] = df["close"].pct_change(24).shift(-24)

        logger.info(f"ターゲット変数作成完了: {len(targets)}個")
        return targets

    def analyze_rf_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Random Forest重要度分析

        ml_configのパラメータを使用してRandomForestモデルを学習します。
        時系列データを考慮し、過去データで学習します。

        Args:
            X: 特徴量
            y: ターゲット

        Returns:
            特徴量重要度の辞書
        """
        logger.info("Random Forest重要度分析開始（時系列データ考慮）")

        try:
            # NaN除去
            valid_idx = ~(X.isna().any(axis=1) | y.isna())
            X_clean = X[valid_idx]
            y_clean = y[valid_idx]

            if len(X_clean) < 100:
                logger.warning(f"サンプル数不足: {len(X_clean)}行")
                return {}

            # ml_configからパラメータを読み込み
            rf_config = unified_config.ml.training
            logger.info(
                f"RandomForest設定: n_estimators={rf_config.rf_n_estimators}, "
                f"max_depth={rf_config.rf_max_depth}, "
                f"random_state={rf_config.random_state}"
            )

            # モデル学習
            rf = RandomForestRegressor(
                n_estimators=rf_config.rf_n_estimators,
                max_depth=rf_config.rf_max_depth,
                random_state=rf_config.random_state,
                n_jobs=-1,
            )
            rf.fit(X_clean, y_clean)

            # 重要度取得
            importances = dict(zip(X.columns, rf.feature_importances_))

            logger.info("Random Forest重要度分析完了")
            return importances

        except Exception as e:
            logger.error(f"RF重要度分析エラー: {e}")
            return {}

    def analyze_permutation_importance(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Dict[str, float]:
        """
        Permutation Importance分析

        ml_configのrandom_stateを使用してPermutation Importanceを計算します。
        時系列データを考慮し、過去データで学習します。

        Args:
            X: 特徴量
            y: ターゲット

        Returns:
            特徴量重要度の辞書
        """
        logger.info("Permutation Importance分析開始（時系列データ考慮）")

        try:
            # NaN除去
            valid_idx = ~(X.isna().any(axis=1) | y.isna())
            X_clean = X[valid_idx]
            y_clean = y[valid_idx]

            if len(X_clean) < 100:
                logger.warning(f"サンプル数不足: {len(X_clean)}行")
                return {}

            # ml_configからrandom_stateを読み込み
            random_state = unified_config.ml.training.random_state
            logger.info(f"Permutation Importance設定: random_state={random_state}")

            # モデル学習（軽量版）
            rf = RandomForestRegressor(
                n_estimators=50, max_depth=8, random_state=random_state, n_jobs=-1
            )
            rf.fit(X_clean, y_clean)

            # Permutation Importance計算
            perm_importance = permutation_importance(
                rf, X_clean, y_clean, n_repeats=5, random_state=random_state, n_jobs=-1
            )

            importances = dict(zip(X.columns, perm_importance.importances_mean))

            logger.info("Permutation Importance分析完了")
            return importances

        except Exception as e:
            logger.error(f"Permutation Importance分析エラー: {e}")
            return {}

    def analyze_correlation(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        相関分析

        Args:
            X: 特徴量
            y: ターゲット

        Returns:
            相関係数の辞書
        """
        logger.info("相関分析開始")

        try:
            correlations = {}
            for col in X.columns:
                # NaN除去
                valid_idx = ~(X[col].isna() | y.isna())
                if valid_idx.sum() < 50:
                    correlations[col] = 0.0
                    continue

                corr = X.loc[valid_idx, col].corr(y[valid_idx])
                correlations[col] = abs(corr) if not np.isnan(corr) else 0.0

            logger.info("相関分析完了")
            return correlations

        except Exception as e:
            logger.error(f"相関分析エラー: {e}")
            return {}

    def analyze_variance(self, X: pd.DataFrame) -> Dict[str, float]:
        """
        分散分析

        Args:
            X: 特徴量

        Returns:
            分散の辞書
        """
        logger.info("分散分析開始")

        try:
            variances = {}
            for col in X.columns:
                var = X[col].var()
                variances[col] = var if not np.isnan(var) else 0.0

            logger.info("分散分析完了")
            return variances

        except Exception as e:
            logger.error(f"分散分析エラー: {e}")
            return {}

    def combine_scores(
        self,
        rf_importance: Dict[str, float],
        perm_importance: Dict[str, float],
        correlation: Dict[str, float],
        variance: Dict[str, float],
    ) -> Dict[str, Dict]:
        """
        各スコアを統合

        Args:
            rf_importance: RF重要度
            perm_importance: Permutation重要度
            correlation: 相関係数
            variance: 分散

        Returns:
            統合された分析結果
        """
        logger.info("スコア統合開始")

        all_features = set(rf_importance.keys())
        combined = {}

        for feature in all_features:
            rf_score = rf_importance.get(feature, 0.0)
            perm_score = perm_importance.get(feature, 0.0)
            corr_score = correlation.get(feature, 0.0)
            var_score = variance.get(feature, 0.0)

            # 正規化された複合スコア（重み付き平均）
            combined_score = (
                0.4 * rf_score + 0.3 * perm_score + 0.2 * corr_score + 0.1 * var_score
            )

            # 低寄与度判定
            is_low_importance = (
                rf_score < IMPORTANCE_THRESHOLD
                and perm_score < IMPORTANCE_THRESHOLD
                and corr_score < CORRELATION_THRESHOLD
                and var_score < VARIANCE_THRESHOLD
            )

            combined[feature] = {
                "rf_importance": float(rf_score),
                "permutation_importance": float(perm_score),
                "correlation": float(corr_score),
                "variance": float(var_score),
                "combined_score": float(combined_score),
                "is_low_importance": bool(is_low_importance),  # numpyブール型を変換
            }

        # ランク付け
        sorted_features = sorted(
            combined.items(), key=lambda x: x[1]["combined_score"], reverse=True
        )
        for rank, (feature, scores) in enumerate(sorted_features, 1):
            combined[feature]["rank"] = int(rank)  # numpy intを変換

        logger.info("スコア統合完了")
        return combined

    def save_results(
        self,
        analysis_result: Dict,
        json_path: str = "feature_importance_analysis.json",
        csv_path: str = "feature_importance_summary.csv",
    ):
        """
        結果を保存

        Args:
            analysis_result: 分析結果
            json_path: JSON保存パス
            csv_path: CSV保存パス
        """
        logger.info("結果保存開始")

        try:
            # JSON保存
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(analysis_result, f, indent=2, ensure_ascii=False)
            logger.info(f"JSON保存完了: {json_path}")

            # CSV保存
            feature_importance = analysis_result.get("feature_importance", {})
            df = pd.DataFrame.from_dict(feature_importance, orient="index")
            df.index.name = "feature_name"
            df = df.sort_values("rank")
            df.to_csv(csv_path)
            logger.info(f"CSV保存完了: {csv_path}")

        except Exception as e:
            logger.error(f"結果保存エラー: {e}")
            raise

    def run_analysis(self, symbol: str = "BTC/USDT:USDT", limit: int = 2000):
        """
        分析実行

        Args:
            symbol: 分析対象シンボル
            limit: データ取得件数
        """
        start_time = time.time()
        logger.info("=" * 80)
        logger.info("特徴量寄与度分析開始")
        logger.info("=" * 80)

        try:
            # 1. データ取得
            ohlcv_df, fr_df, oi_df = self.fetch_data(symbol, limit)

            if ohlcv_df.empty:
                logger.error("データが取得できませんでした")
                return

            if len(ohlcv_df) < MIN_SAMPLES:
                logger.error(
                    f"サンプル数不足: {len(ohlcv_df)}行 (最小: {MIN_SAMPLES}行)"
                )
                return

            # 2. 特徴量計算
            features_df = self.calculate_features(ohlcv_df, fr_df, oi_df)

            # closeカラムを一時的に追加
            if "close" not in features_df.columns:
                features_df["close"] = ohlcv_df["close"]

            # 3. ターゲット変数作成
            targets = self.create_target_variables(features_df)

            # 代表的なターゲット（1時間先）を使用
            target = targets["return_1h"]

            # closeカラムを除外
            feature_cols = [
                col
                for col in features_df.columns
                if col not in ["open", "high", "low", "close", "volume"]
            ]
            X = features_df[feature_cols]

            # データを結合してNaN除去
            combined_df = pd.concat([X, target.rename("return_1h")], axis=1).dropna()
            X = combined_df[feature_cols]
            y = combined_df["return_1h"]

            logger.info(f"分析対象サンプル数: {len(X)}行")
            logger.info(f"特徴量数: {len(X.columns)}個")

            # 4. 各分析手法を実行
            rf_importance = self.analyze_rf_importance(X, y)
            perm_importance = self.analyze_permutation_importance(X, y)
            correlation = self.analyze_correlation(X, y)
            variance = self.analyze_variance(X)

            # 5. スコア統合
            combined_scores = self.combine_scores(
                rf_importance, perm_importance, correlation, variance
            )

            # 6. 低寄与度特徴量リスト作成
            low_importance_features = [
                feature
                for feature, scores in combined_scores.items()
                if scores["is_low_importance"]
            ]

            # 7. 結果サマリー作成
            analysis_result = {
                "analysis_date": datetime.now().isoformat(),
                "data_samples": len(X),
                "symbols_analyzed": [symbol],
                "total_features": len(combined_scores),
                "low_importance_features_count": len(low_importance_features),
                "thresholds": {
                    "rf_importance": IMPORTANCE_THRESHOLD,
                    "permutation_importance": IMPORTANCE_THRESHOLD,
                    "correlation": CORRELATION_THRESHOLD,
                    "variance": VARIANCE_THRESHOLD,
                },
                "feature_importance": combined_scores,
                "low_importance_features": low_importance_features,
                "recommendation": f"{len(low_importance_features)}個の低寄与度特徴量の削除を推奨",
            }

            # 8. 結果保存
            self.save_results(analysis_result)

            # 9. コンソール出力
            self.print_summary(analysis_result)

            elapsed_time = time.time() - start_time
            logger.info(f"分析完了（処理時間: {elapsed_time:.2f}秒）")

        except Exception as e:
            logger.error(f"分析実行エラー: {e}")
            raise

    def print_summary(self, analysis_result: Dict):
        """
        結果サマリーをコンソール出力

        Args:
            analysis_result: 分析結果
        """
        print("\n" + "=" * 80)
        print("特徴量寄与度分析結果サマリー")
        print("=" * 80)

        print(f"\n分析日時: {analysis_result['analysis_date']}")
        print(f"分析サンプル数: {analysis_result['data_samples']:,}行")
        print(f"分析シンボル: {', '.join(analysis_result['symbols_analyzed'])}")
        print(f"総特徴量数: {analysis_result['total_features']}個")
        print(f"低寄与度特徴量数: {analysis_result['low_importance_features_count']}個")

        # 上位20個の特徴量
        print("\n" + "-" * 80)
        print("【上位20個の重要特徴量】")
        print("-" * 80)
        feature_importance = analysis_result["feature_importance"]
        sorted_features = sorted(
            feature_importance.items(), key=lambda x: x[1]["rank"]
        )[:20]

        print(
            f"{'順位':<6} {'特徴量名':<40} {'複合スコア':<12} {'RF重要度':<12} {'相関':<10}"
        )
        print("-" * 80)
        for feature, scores in sorted_features:
            print(
                f"{scores['rank']:<6} {feature:<40} {scores['combined_score']:<12.6f} "
                f"{scores['rf_importance']:<12.6f} {scores['correlation']:<10.6f}"
            )

        # 低寄与度特徴量
        print("\n" + "-" * 80)
        print("【低寄与度特徴量（削除推奨、上位20個のみ表示）】")
        print("-" * 80)
        low_features = analysis_result["low_importance_features"][:20]

        if low_features:
            for i, feature in enumerate(low_features, 1):
                scores = feature_importance[feature]
                print(
                    f"{i:3}. {feature:<40} (RF: {scores['rf_importance']:.6f}, "
                    f"相関: {scores['correlation']:.6f}, 分散: {scores['variance']:.6f})"
                )
        else:
            print("低寄与度特徴量は見つかりませんでした。")

        print("\n" + "=" * 80)
        print(analysis_result["recommendation"])
        print("=" * 80 + "\n")


def main():
    """メイン実行関数"""
    try:
        with FeatureImportanceAnalyzer() as analyzer:
            analyzer.run_analysis(symbol="BTC/USDT:USDT", limit=2000)

    except Exception as e:
        logger.error(f"実行エラー: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
