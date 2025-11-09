"""
XGBoost/LightGBM特徴量重要度分析スクリプト

XGBoostとLightGBMの両方で特徴量重要度を分析し、低重要度の特徴量を
自動検出してマークダウンレポートを生成します。

実行方法:
    cd backend
    python scripts/feature_evaluation/detect_low_importance_features.py \
        --symbol BTC/USDT \
        --timeframe 1h \
        --lookback-days 90 \
        --threshold 0.2 \
        --output-dir data/feature_evaluation
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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


class LowImportanceFeatureDetector:
    """
    XGBoost/LightGBM特徴量重要度分析クラス

    両モデルで特徴量重要度を計算し、低重要度の特徴量を検出します。
    """

    def __init__(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
        lookback_days: int = 90,
        threshold: float = 0.2,
        output_dir: str = "data/feature_evaluation",
    ):
        """
        初期化

        Args:
            symbol: 取引ペア
            timeframe: 時間足
            lookback_days: データ取得期間（日数）
            threshold: 低重要度判定の閾値（下位X%）
            output_dir: 出力ディレクトリ
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.lookback_days = lookback_days
        self.threshold = threshold
        self.output_dir = Path(output_dir)

        # 出力ディレクトリを作成
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 共通評価ユーティリティ
        self.common = CommonFeatureEvaluator()

        # 結果格納用
        self.xgb_importance: Dict[str, Dict[str, float]] = {}
        self.lgb_importance: Dict[str, Dict[str, float]] = {}
        self.low_importance_features: List[Dict] = []

    def __enter__(self) -> "LowImportanceFeatureDetector":
        """コンテキストマネージャー: 入場"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """コンテキストマネージャー: 退場"""
        self.common.close()

    def fetch_data(
        self,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        DBからデータを取得

        Returns:
            (OHLCV, FR, OI)のタプル
        """
        logger.info(f"データ取得開始: {self.symbol}, timeframe={self.timeframe}")

        try:
            end_time = datetime.now().astimezone()
            start_time = end_time - timedelta(days=self.lookback_days)
            data = self.common.fetch_data(
                symbol=self.symbol,
                timeframe=self.timeframe,
                limit=10**9,
            )
            idx = data.ohlcv.index
            if getattr(idx, "tz", None) is None:
                # tz-naive → naiveな範囲でフィルタ
                ohlcv_df = data.ohlcv[(idx >= start_time.replace(tzinfo=None)) & (idx <= end_time.replace(tzinfo=None))]
            else:
                # tz-aware → awareな範囲でフィルタ
                ohlcv_df = data.ohlcv[(idx >= start_time) & (idx <= end_time)]
            if ohlcv_df.empty:
                logger.warning(f"OHLCVデータが見つかりません: {self.symbol}")
                return pd.DataFrame(), None, None
            return ohlcv_df, data.fr, data.oi
        except Exception as e:
            logger.error(f"データ取得エラー: {e}")
            raise

    def prepare_features_and_labels(
        self,
        ohlcv_df: pd.DataFrame,
        fr_df: Optional[pd.DataFrame],
        oi_df: Optional[pd.DataFrame],
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        特徴量とラベルを準備

        Args:
            ohlcv_df: OHLCVデータ
            fr_df: ファンディングレートデータ
            oi_df: オープンインタレストデータ

        Returns:
            (特徴量DataFrame, ラベルSeries)
        """
        logger.info("特徴量計算開始")

        try:
            data = EvaluationData(ohlcv=ohlcv_df, fr=fr_df, oi=oi_df)
            # crypto_featuresはOI前提のため、このスクリプトでは無効化して基本+advancedの安定部分のみ使用
            features_df = self.common.build_basic_features(data=data, skip_crypto_and_advanced=True)
            X = self.common.drop_ohlcv_columns(features_df, keep_close=False)
            y = self.common.create_forward_return_target(ohlcv_df["close"], periods=1)

            # NaN除去
            valid_idx = ~(X.isna().any(axis=1) | y.isna())
            X = X[valid_idx]
            y = y[valid_idx]

            logger.info(f"特徴量: {len(X.columns)}個, サンプル数: {len(X)}行")

            return X, y

        except Exception as e:
            logger.error(f"特徴量準備エラー: {e}")
            raise

    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series) -> xgb.Booster:
        """
        XGBoostモデルをトレーニング

        Args:
            X_train: 訓練データ（特徴量）
            y_train: 訓練データ（ラベル）

        Returns:
            トレーニング済みXGBoostモデル
        """
        logger.info("XGBoostトレーニング開始")

        try:
            # DMatrix作成
            dtrain = xgb.DMatrix(X_train, label=y_train)

            # デフォルトパラメータ
            params = {
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "seed": 42,
            }

            # トレーニング
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=100,
                verbose_eval=False,
            )

            logger.info("XGBoostトレーニング完了")
            return model

        except Exception as e:
            logger.error(f"XGBoostトレーニングエラー: {e}")
            raise

    def train_lightgbm(self, X_train: pd.DataFrame, y_train: pd.Series) -> lgb.Booster:
        """
        LightGBMモデルをトレーニング

        Args:
            X_train: 訓練データ（特徴量）
            y_train: 訓練データ（ラベル）

        Returns:
            トレーニング済みLightGBMモデル
        """
        logger.info("LightGBMトレーニング開始")

        try:
            # Dataset作成
            train_data = lgb.Dataset(X_train, label=y_train)

            # デフォルトパラメータ
            params = {
                "objective": "regression",
                "metric": "rmse",
                "seed": 42,
                "verbose": -1,
            }

            # トレーニング
            model = lgb.train(
                params,
                train_data,
                num_boost_round=100,
            )

            logger.info("LightGBMトレーニング完了")
            return model

        except Exception as e:
            logger.error(f"LightGBMトレーニングエラー: {e}")
            raise

    def extract_feature_importance(
        self, xgb_model: xgb.Booster, lgb_model: lgb.Booster, feature_names: List[str]
    ) -> None:
        """
        両モデルから特徴量重要度を抽出

        Args:
            xgb_model: XGBoostモデル
            lgb_model: LightGBMモデル
            feature_names: 特徴量名リスト
        """
        logger.info("特徴量重要度抽出開始")

        try:
            # XGBoost重要度
            xgb_gain = xgb_model.get_score(importance_type="gain")
            xgb_weight = xgb_model.get_score(importance_type="weight")
            xgb_cover = xgb_model.get_score(importance_type="cover")

            # 正規化
            xgb_gain_sum = sum(xgb_gain.values())
            xgb_weight_sum = sum(xgb_weight.values())
            xgb_cover_sum = sum(xgb_cover.values())

            for feature in feature_names:
                self.xgb_importance[feature] = {
                    "gain": (
                        xgb_gain.get(feature, 0.0) / xgb_gain_sum
                        if xgb_gain_sum > 0
                        else 0.0
                    ),
                    "weight": (
                        xgb_weight.get(feature, 0.0) / xgb_weight_sum
                        if xgb_weight_sum > 0
                        else 0.0
                    ),
                    "cover": (
                        xgb_cover.get(feature, 0.0) / xgb_cover_sum
                        if xgb_cover_sum > 0
                        else 0.0
                    ),
                }

            # LightGBM重要度
            lgb_gain = dict(
                zip(
                    lgb_model.feature_name(),
                    lgb_model.feature_importance(importance_type="gain"),
                )
            )
            lgb_split = dict(
                zip(
                    lgb_model.feature_name(),
                    lgb_model.feature_importance(importance_type="split"),
                )
            )

            # 正規化
            lgb_gain_sum = sum(lgb_gain.values())
            lgb_split_sum = sum(lgb_split.values())

            for feature in feature_names:
                self.lgb_importance[feature] = {
                    "gain": (
                        lgb_gain.get(feature, 0.0) / lgb_gain_sum
                        if lgb_gain_sum > 0
                        else 0.0
                    ),
                    "weight": (
                        lgb_split.get(feature, 0.0) / lgb_split_sum
                        if lgb_split_sum > 0
                        else 0.0
                    ),
                }

            logger.info("特徴量重要度抽出完了")

        except Exception as e:
            logger.error(f"特徴量重要度抽出エラー: {e}")
            raise

    def detect_low_importance_features(self) -> None:
        """
        低重要度特徴量を検出
        """
        logger.info("低重要度特徴量検出開始")

        try:
            # Gain重要度の閾値を計算
            xgb_gains = [imp["gain"] for imp in self.xgb_importance.values()]
            lgb_gains = [imp["gain"] for imp in self.lgb_importance.values()]

            xgb_gain_threshold = np.percentile(xgb_gains, self.threshold * 100)
            lgb_gain_threshold = np.percentile(lgb_gains, self.threshold * 100)

            # 低重要度特徴量を検出
            for feature in self.xgb_importance.keys():
                xgb_gain = self.xgb_importance[feature]["gain"]
                lgb_gain = self.lgb_importance[feature]["gain"]
                xgb_weight = self.xgb_importance[feature]["weight"]
                lgb_weight = self.lgb_importance[feature]["weight"]

                reasons = []

                # 条件1: 両モデルでGain重要度が下位20%
                if xgb_gain < xgb_gain_threshold and lgb_gain < lgb_gain_threshold:
                    reasons.append(
                        f"両モデルでGain重要度が下位{self.threshold*100:.0f}%"
                    )

                # 条件2: 両モデルでWeight重要度が0（使用されていない）
                if xgb_weight == 0.0 and lgb_weight == 0.0:
                    reasons.append("両モデルで未使用（Weight=0）")

                if reasons:
                    self.low_importance_features.append(
                        {
                            "feature": feature,
                            "xgb_gain": xgb_gain,
                            "lgb_gain": lgb_gain,
                            "xgb_weight": xgb_weight,
                            "lgb_weight": lgb_weight,
                            "reasons": reasons,
                        }
                    )

            logger.info(f"低重要度特徴量: {len(self.low_importance_features)}個検出")

        except Exception as e:
            logger.error(f"低重要度特徴量検出エラー: {e}")
            raise

    def generate_markdown_report(self) -> str:
        """
        マークダウンレポートを生成

        Returns:
            レポート内容
        """
        logger.info("マークダウンレポート生成開始")

        try:
            report = []

            # ヘッダー
            report.append("# XGBoost/LightGBM特徴量重要度分析レポート")
            report.append("")
            report.append(
                f"**生成日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            report.append("")

            # サマリー
            report.append("## サマリー")
            report.append("")
            report.append(f"- **分析対象シンボル**: {self.symbol}")
            report.append(f"- **時間足**: {self.timeframe}")
            report.append(f"- **データ期間**: {self.lookback_days}日")
            report.append(f"- **総特徴量数**: {len(self.xgb_importance)}個")
            report.append(
                f"- **低重要度特徴量数**: {len(self.low_importance_features)}個"
            )
            report.append(f"- **低重要度判定閾値**: 下位{self.threshold*100:.0f}%")
            report.append("")

            # 低重要度特徴量リスト
            report.append("## 低重要度特徴量リスト")
            report.append("")
            if self.low_importance_features:
                report.append(
                    "| 順位 | 特徴量名 | XGBoost Gain | LightGBM Gain | "
                    "XGBoost Weight | LightGBM Weight | 検出理由 |"
                )
                report.append(
                    "|------|----------|--------------|---------------|"
                    "----------------|-----------------|----------|"
                )

                for idx, feat in enumerate(self.low_importance_features, 1):
                    reasons = "; ".join(feat["reasons"])
                    report.append(
                        f"| {idx} | {feat['feature']} | {feat['xgb_gain']:.6f} | "
                        f"{feat['lgb_gain']:.6f} | {feat['xgb_weight']:.6f} | "
                        f"{feat['lgb_weight']:.6f} | {reasons} |"
                    )
                report.append("")
            else:
                report.append("低重要度特徴量は検出されませんでした。")
                report.append("")

            # 詳細分析
            report.append("## 詳細分析")
            report.append("")

            # 特徴量重要度の分布
            report.append("### 特徴量重要度の分布")
            report.append("")
            xgb_gains = [imp["gain"] for imp in self.xgb_importance.values()]
            lgb_gains = [imp["gain"] for imp in self.lgb_importance.values()]
            report.append(
                f"- **XGBoost Gain重要度**: 平均={np.mean(xgb_gains):.6f}, "
                f"標準偏差={np.std(xgb_gains):.6f}"
            )
            report.append(
                f"- **LightGBM Gain重要度**: 平均={np.mean(lgb_gains):.6f}, "
                f"標準偏差={np.std(lgb_gains):.6f}"
            )
            report.append("")

            # モデル間の相関
            report.append("### モデル間の相関")
            report.append("")
            correlation = np.corrcoef(xgb_gains, lgb_gains)[0, 1]
            report.append(
                f"- **XGBoost vs LightGBM Gain重要度の相関係数**: {correlation:.4f}"
            )
            report.append("")

            # 推奨アクション
            report.append("## 推奨アクション")
            report.append("")
            if len(self.low_importance_features) > 0:
                report.append(
                    f"以下の{len(self.low_importance_features)}個の特徴量の削除を推奨します:"
                )
                report.append("")
                for feat in self.low_importance_features[:10]:  # 上位10個のみ表示
                    report.append(f"- `{feat['feature']}`")
                report.append("")
                report.append(
                    "削除推奨特徴量の完全なリストは`features_to_remove_auto.json`を参照してください。"
                )
            else:
                report.append("現時点で削除を推奨する特徴量はありません。")
            report.append("")

            return "\n".join(report)

        except Exception as e:
            logger.error(f"マークダウンレポート生成エラー: {e}")
            raise

    def save_results(self) -> None:
        """
        分析結果を保存
        """
        logger.info("分析結果保存開始")

        try:
            # マークダウンレポート
            report = self.generate_markdown_report()
            report_path = self.output_dir / "low_importance_features_report.md"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report)
            logger.info(f"マークダウンレポート保存: {report_path}")

            # 詳細データ（CSV）
            detailed_data = []
            for feature in self.xgb_importance.keys():
                detailed_data.append(
                    {
                        "feature": feature,
                        "xgb_gain": self.xgb_importance[feature]["gain"],
                        "xgb_weight": self.xgb_importance[feature]["weight"],
                        "xgb_cover": self.xgb_importance[feature]["cover"],
                        "lgb_gain": self.lgb_importance[feature]["gain"],
                        "lgb_weight": self.lgb_importance[feature]["weight"],
                        "is_low_importance": feature
                        in [f["feature"] for f in self.low_importance_features],
                    }
                )

            detailed_df = pd.DataFrame(detailed_data)
            detailed_df = detailed_df.sort_values("xgb_gain", ascending=False)
            csv_path = self.output_dir / "feature_importance_detailed.csv"
            detailed_df.to_csv(csv_path, index=False)
            logger.info(f"詳細データ保存: {csv_path}")

            # 削除推奨リスト（JSON）
            remove_list = [feat["feature"] for feat in self.low_importance_features]
            json_path = self.output_dir / "features_to_remove_auto.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(remove_list, f, indent=2, ensure_ascii=False)
            logger.info(f"削除推奨リスト保存: {json_path}")

        except Exception as e:
            logger.error(f"分析結果保存エラー: {e}")
            raise

    def run_analysis(self) -> None:
        """
        分析を実行
        """
        start_time = datetime.now()
        logger.info("=" * 80)
        logger.info("XGBoost/LightGBM特徴量重要度分析開始")
        logger.info("=" * 80)

        try:
            # 1. データ取得
            logger.info("ステップ1: データ取得")
            ohlcv_df, fr_df, oi_df = self.fetch_data()

            if ohlcv_df.empty:
                logger.error("データが取得できませんでした")
                return

            # 2. 特徴量とラベル準備
            logger.info("ステップ2: 特徴量とラベル準備")
            X, y = self.prepare_features_and_labels(ohlcv_df, fr_df, oi_df)

            if len(X) < 100:
                logger.error(f"サンプル数不足: {len(X)}行（最小100行必要）")
                return

            # 3. データ分割（訓練:検証:テスト = 70:15:15）
            logger.info("ステップ3: データ分割")
            train_size = int(len(X) * 0.7)
            X_train = X.iloc[:train_size]
            y_train = y.iloc[:train_size]

            # 4. モデルトレーニング
            logger.info("ステップ4: モデルトレーニング")
            with tqdm(total=2, desc="モデルトレーニング") as pbar:
                xgb_model = self.train_xgboost(X_train, y_train)
                pbar.update(1)

                lgb_model = self.train_lightgbm(X_train, y_train)
                pbar.update(1)

            # 5. 特徴量重要度抽出
            logger.info("ステップ5: 特徴量重要度抽出")
            self.extract_feature_importance(xgb_model, lgb_model, X.columns.tolist())

            # 6. 低重要度特徴量検出
            logger.info("ステップ6: 低重要度特徴量検出")
            self.detect_low_importance_features()

            # 7. 結果保存
            logger.info("ステップ7: 結果保存")
            self.save_results()

            # 完了
            elapsed_time = (datetime.now() - start_time).total_seconds()
            logger.info("=" * 80)
            logger.info(f"分析完了（処理時間: {elapsed_time:.2f}秒）")
            logger.info(f"総特徴量数: {len(self.xgb_importance)}個")
            logger.info(f"低重要度特徴量数: {len(self.low_importance_features)}個")
            logger.info(f"結果は {self.output_dir} に保存されました")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"分析実行エラー: {e}")
            import traceback

            traceback.print_exc()
            raise


def parse_args() -> argparse.Namespace:
    """
    コマンドライン引数をパース

    Returns:
        パース済み引数
    """
    parser = argparse.ArgumentParser(
        description="XGBoost/LightGBM特徴量重要度分析スクリプト"
    )

    parser.add_argument(
        "--symbol",
        type=str,
        default="BTC/USDT",
        help="取引ペア（デフォルト: BTC/USDT）",
    )

    parser.add_argument(
        "--timeframe",
        type=str,
        default="1h",
        help="時間足（デフォルト: 1h）",
    )

    parser.add_argument(
        "--lookback-days",
        type=int,
        default=90,
        help="データ取得期間（日数、デフォルト: 90）",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="低重要度判定の閾値（下位X%、デフォルト: 0.2）",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/feature_evaluation",
        help="出力ディレクトリ（デフォルト: data/feature_evaluation）",
    )

    return parser.parse_args()


def main() -> None:
    """メイン実行関数"""
    try:
        # 引数パース
        args = parse_args()

        # 分析実行
        with LowImportanceFeatureDetector(
            symbol=args.symbol,
            timeframe=args.timeframe,
            lookback_days=args.lookback_days,
            threshold=args.threshold,
            output_dir=args.output_dir,
        ) as detector:
            detector.run_analysis()

    except Exception as e:
        logger.error(f"実行エラー: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
