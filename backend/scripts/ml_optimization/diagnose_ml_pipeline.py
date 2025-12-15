"""
ML Pipeline 診断スクリプト

現在のMLパイプラインの問題点を特定するための包括的な診断を実行します。
"""

import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from app.services.ml.common.common_data import CommonData
from app.services.ml.label_generation.label_generation_service import (
    LabelGenerationService,
)
from app.services.ml.feature_engineering.feature_engineering_service import (
    FeatureEngineeringService,
    DEFAULT_FEATURE_ALLOWLIST,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MLPipelineDiagnostics:
    """MLパイプラインの診断クラス"""

    def __init__(self):
        self.label_service = LabelGenerationService()
        self.feature_service = FeatureEngineeringService()
        self.common_data = CommonData()

    def diagnose_all(self, symbol: str = "BTC/USDT:USDT", timeframe: str = "1h"):
        """全診断を実行"""
        logger.info("=" * 80)
        logger.info("ML Pipeline 包括診断開始")
        logger.info("=" * 80)

        # データ取得
        logger.info(f"\n1. データ取得: {symbol} {timeframe}")
        ohlcv_data, fr_data, oi_data = self._fetch_data(symbol, timeframe)

        # ラベル生成と診断
        logger.info("\n2. ラベル生成診断")
        labels_df = self._diagnose_labels(ohlcv_data, fr_data, oi_data)

        # 特徴量診断
        logger.info("\n3. 特徴量診断")
        features_df = self._diagnose_features(ohlcv_data, fr_data, oi_data)

        # 時系列診断
        logger.info("\n4. 時系列診断")
        self._diagnose_temporal_patterns(ohlcv_data, labels_df)

        # 推奨事項
        logger.info("\n5. 推奨事項")
        self._generate_recommendations(labels_df, features_df, ohlcv_data)

        logger.info("\n" + "=" * 80)
        logger.info("診断完了")
        logger.info("=" * 80)

    def _fetch_data(self, symbol: str, timeframe: str):
        """データを取得"""
        data = self.common_data.prepare_data(
            symbol=symbol, timeframe=timeframe, limit=50000
        )

        logger.info(f"  OHLCV: {len(data.ohlcv)} rows")
        logger.info(f"  期間: {data.ohlcv.index[0]} ～ {data.ohlcv.index[-1]}")
        logger.info(f"  FR: {len(data.fr) if data.fr is not None else 0} rows")
        logger.info(f"  OI: {len(data.oi) if data.oi is not None else 0} rows")

        return data.ohlcv, data.fr, data.oi

    def _diagnose_labels(self, ohlcv_data, fr_data, oi_data):
        """ラベル生成を診断"""

        # ラベル生成
        labels_result = self.label_service.generate_labels(ohlcv_data, fr_data, oi_data)

        labels_df = labels_result["labels"]
        t_events = labels_result["t_events"]

        logger.info(f"\n  === CUSUMイベント検出 ===")
        logger.info(f"  検出イベント数: {len(t_events)}")
        logger.info(f"  全データ行数: {len(ohlcv_data)}")
        logger.info(f"  イベント検出率: {len(t_events) / len(ohlcv_data) * 100:.2f}%")

        # ラベル分布
        logger.info(f"\n  === ラベル分布 ===")
        label_counts = labels_df["label"].value_counts()
        total_labels = len(labels_df.dropna())

        for label_val, count in label_counts.items():
            percentage = (count / total_labels) * 100
            logger.info(f"  {label_val}: {count} ({percentage:.2f}%)")

        # ラベル品質指標
        logger.info(f"\n  === ラベル品質指標 ===")

        # 実際のリターンを計算（次の12バーの価格変化）
        future_returns = ohlcv_data["close"].pct_change(12).shift(-12)

        # ラベル=UPの実際のリターン
        up_mask = labels_df["label"] == "UP"
        if up_mask.sum() > 0:
            up_returns = future_returns[up_mask].dropna()
            logger.info(f"  UP ラベルの実際の平均リターン: {up_returns.mean():.4f}")
            logger.info(
                f"  UP ラベルで実際に上昇した比率: {(up_returns > 0).sum() / len(up_returns) * 100:.2f}%"
            )

        # ラベル=DOWNの実際のリターン
        down_mask = labels_df["label"] == "DOWN"
        if down_mask.sum() > 0:
            down_returns = future_returns[down_mask].dropna()
            logger.info(f"  DOWN ラベルの実際の平均リターン: {down_returns.mean():.4f}")
            logger.info(
                f"  DOWN ラベルで実際に下降した比率: {(down_returns < 0).sum() / len(down_returns) * 100:.2f}%"
            )

        return labels_df

    def _diagnose_features(self, ohlcv_data, fr_data, oi_data):
        """特徴量を診断"""

        # 特徴量計算
        features_df = self.feature_service.calculate_advanced_features(
            ohlcv_data, fr_data, oi_data
        )

        logger.info(f"\n  === 特徴量基本情報 ===")
        logger.info(f"  生成特徴量数: {len(features_df.columns)}")
        logger.info(f"  Allowlist特徴量数: {len(DEFAULT_FEATURE_ALLOWLIST)}")

        # Allowlist内の特徴量の存在確認
        missing_features = [
            f for f in DEFAULT_FEATURE_ALLOWLIST if f not in features_df.columns
        ]
        if missing_features:
            logger.warning(
                f"  ⚠️ Allowlistに含まれるが生成されていない特徴量: {len(missing_features)}"
            )
            logger.warning(f"    {missing_features[:5]}")  # 最初の5個のみ表示

        # 欠損値診断
        logger.info(f"\n  === 欠損値診断 ===")
        null_counts = features_df[DEFAULT_FEATURE_ALLOWLIST].isnull().sum()
        high_null_features = null_counts[
            null_counts > len(features_df) * 0.1
        ]  # 10%以上欠損

        if len(high_null_features) > 0:
            logger.warning(f"  ⚠️ 欠損値が10%以上の特徴量: {len(high_null_features)}")
            for feat, null_count in high_null_features.items():
                logger.warning(
                    f"    {feat}: {null_count / len(features_df) * 100:.2f}%"
                )
        else:
            logger.info(f"  ✅ すべての特徴量の欠損値は10%未満")

        # 分散診断
        logger.info(f"\n  === 分散診断 ===")
        variances = features_df[DEFAULT_FEATURE_ALLOWLIST].var()
        zero_var_features = variances[variances < 1e-10]

        if len(zero_var_features) > 0:
            logger.warning(f"  ⚠️ 分散がほぼゼロの特徴量: {len(zero_var_features)}")
            logger.warning(f"    {list(zero_var_features.index[:5])}")
        else:
            logger.info(f"  ✅ すべての特徴量に十分な分散あり")

        return features_df

    def _diagnose_temporal_patterns(self, ohlcv_data, labels_df):
        """時系列パターンを診断"""

        logger.info(f"\n  === 年別ラベル分布 ===")

        # データを年ごとに分割
        ohlcv_data["year"] = ohlcv_data.index.year
        labels_with_year = labels_df.copy()
        labels_with_year["year"] = labels_df.index.year

        for year in sorted(labels_with_year["year"].unique()):
            year_labels = labels_with_year[labels_with_year["year"] == year]["label"]
            if len(year_labels) > 0:
                up_pct = (year_labels == "UP").sum() / len(year_labels) * 100
                down_pct = (year_labels == "DOWN").sum() / len(year_labels) * 100
                range_pct = (year_labels == "RANGE").sum() / len(year_labels) * 100

                logger.info(
                    f"  {year}: UP={up_pct:.1f}%, DOWN={down_pct:.1f}%, RANGE={range_pct:.1f}%  (n={len(year_labels)})"
                )

        # ボラティリティ診断
        logger.info(f"\n  === 年別ボラティリティ ===")
        for year in sorted(ohlcv_data["year"].unique()):
            year_data = ohlcv_data[ohlcv_data["year"] == year]
            returns = year_data["close"].pct_change()
            volatility = returns.std() * np.sqrt(365 * 24)  # 年率換算
            logger.info(f"  {year}: {volatility:.4f}")

    def _generate_recommendations(self, labels_df, features_df, ohlcv_data):
        """推奨事項を生成"""

        recommendations = []

        # 1. イベント検出率チェック
        event_rate = len(labels_df.dropna()) / len(ohlcv_data)
        if event_rate > 0.2:
            recommendations.append(
                f"⚠️ CUSUMイベント検出率が高すぎます ({event_rate*100:.1f}%)。"
                f"cusum_vol_multiplier を 2.0 ～ 3.0 に引き上げることを推奨。"
            )
        elif event_rate < 0.05:
            recommendations.append(
                f"⚠️ CUSUMイベント検出率が低すぎます ({event_rate*100:.1f}%)。"
                f"学習サンプルが不足している可能性があります。"
            )

        # 2. ラベル不均衡チェック
        label_counts = labels_df["label"].value_counts()
        total = label_counts.sum()

        for label_val, count in label_counts.items():
            pct = count / total
            if pct < 0.1:
                recommendations.append(
                    f"⚠️ '{label_val}' ラベルが極端に少ない ({pct*100:.1f}%)。"
                    f"クラス不均衡により学習が困難な可能性。"
                )

        # 3. データ期間チェック
        data_start = ohlcv_data.index[0]
        data_end = ohlcv_data.index[-1]
        data_span_years = (data_end - data_start).days / 365.25

        if data_span_years > 3:
            recommendations.append(
                f"💡 データ期間が長すぎます ({data_span_years:.1f}年)。"
                f"市場環境の変化を考慮し、直近1～2年に限定することを推奨。"
            )

        # 推奨事項を表示
        logger.info("\n  === 推奨改善策 ===")
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                logger.info(f"  {i}. {rec}")
        else:
            logger.info("  ✅ 明らかな問題は検出されませんでした。")

        logger.info("\n  === 次のステップ ===")
        logger.info("  1. CUSUMパラメータの調整（vol_multiplier）")
        logger.info("  2. データ期間の限定（直近1～2年）")
        logger.info("  3. Triple Barrierパラメータの調整（pt/sl比）")


if __name__ == "__main__":
    diagnostics = MLPipelineDiagnostics()
    diagnostics.diagnose_all()



