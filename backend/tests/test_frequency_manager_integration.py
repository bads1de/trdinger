"""
DataFrequencyManager統合テスト

分析報告書で特定された最優先問題「データ頻度の深刻な不一致問題」の
解決策の統合テスト。
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.ml.feature_engineering.feature_engineering_service import (
    FeatureEngineeringService,
)

logger = logging.getLogger(__name__)


class TestFrequencyManagerIntegration:
    """DataFrequencyManager統合テストクラス"""

    def setup_method(self):
        """各テストメソッドの前に実行される初期化"""
        self.feature_service = FeatureEngineeringService()

    def create_test_data(self, timeframe="1h"):
        """テストデータを作成"""
        # timeframe固有のOHLCVデータ
        if timeframe == "15m":
            periods = 200
            freq = "15min"
        elif timeframe == "1h":
            periods = 100
            freq = "1h"
        elif timeframe == "4h":
            periods = 50
            freq = "4h"
        else:
            periods = 100
            freq = "1h"

        dates = pd.date_range(start="2023-01-01", periods=periods, freq=freq)

        np.random.seed(42)
        base_price = 50000
        price_changes = np.random.normal(0, 0.02, periods)
        prices = base_price * np.cumprod(1 + price_changes)

        ohlcv_data = pd.DataFrame(
            {
                "Open": prices * (1 + np.random.normal(0, 0.001, periods)),
                "High": prices * (1 + np.abs(np.random.normal(0, 0.005, periods))),
                "Low": prices * (1 - np.abs(np.random.normal(0, 0.005, periods))),
                "Close": prices,
                "Volume": np.random.uniform(1000, 10000, periods),
            },
            index=dates,
        )

        # 8時間間隔のファンディングレートデータ
        fr_dates = pd.date_range(start="2023-01-01", periods=25, freq="8h")
        funding_rate_data = pd.DataFrame(
            {
                "timestamp": fr_dates,
                "funding_rate": np.random.normal(0.0001, 0.0005, 25),
            }
        )

        # 1時間間隔の建玉残高データ
        oi_dates = pd.date_range(start="2023-01-01", periods=100, freq="1h")
        open_interest_data = pd.DataFrame(
            {
                "timestamp": oi_dates,
                "open_interest": np.random.uniform(1000000, 5000000, 100),
            }
        )

        return ohlcv_data, funding_rate_data, open_interest_data

    def test_basic_integration(self):
        """基本的な統合テスト"""
        logger.info("=== 基本的な統合テスト ===")

        # テストデータを作成
        ohlcv_data, funding_rate_data, open_interest_data = self.create_test_data("1h")

        # 特徴量エンジニアリングサービスで特徴量を計算
        features_df = self.feature_service.calculate_advanced_features(
            ohlcv_data=ohlcv_data,
            funding_rate_data=funding_rate_data,
            open_interest_data=open_interest_data,
        )

        # 結果の検証
        assert not features_df.empty, "特徴量計算結果が空です"
        assert len(features_df) > 0, "特徴量データが生成されていません"

        # 基本的なOHLCV列が存在することを確認
        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        for col in required_columns:
            assert col in features_df.columns, f"必須列 {col} が見つかりません"

        # 特徴量が追加されていることを確認
        feature_count = len(features_df.columns)
        assert feature_count > len(
            required_columns
        ), f"特徴量が追加されていません: {feature_count}列"

        logger.info(f"特徴量計算成功: {feature_count}個の特徴量を生成")
        logger.info(f"データ行数: {len(features_df)}行")

    def test_different_timeframes(self):
        """異なるtimeframeでの統合テスト"""
        logger.info("=== 異なるtimeframeでの統合テスト ===")

        test_timeframes = ["15m", "1h", "4h"]

        for timeframe in test_timeframes:
            logger.info(f"--- {timeframe} timeframeのテスト ---")

            # timeframe固有のテストデータを作成
            ohlcv_data, funding_rate_data, open_interest_data = self.create_test_data(
                timeframe
            )

            # 特徴量計算を実行
            try:
                features_df = self.feature_service.calculate_advanced_features(
                    ohlcv_data=ohlcv_data,
                    funding_rate_data=funding_rate_data,
                    open_interest_data=open_interest_data,
                )

                assert not features_df.empty, f"{timeframe}: 特徴量計算結果が空"
                logger.info(
                    f"{timeframe}: 特徴量計算成功 - {len(features_df.columns)}列, {len(features_df)}行"
                )

                # データ頻度統一が正しく動作していることを確認
                # （エラーが発生しないことで確認）

            except Exception as e:
                logger.error(f"{timeframe}: 特徴量計算エラー - {e}")
                raise

    def test_data_quality_improvement(self):
        """データ品質改善の検証"""
        logger.info("=== データ品質改善の検証 ===")

        # テストデータを作成
        ohlcv_data, funding_rate_data, open_interest_data = self.create_test_data("1h")

        # 特徴量を計算
        features_df = self.feature_service.calculate_advanced_features(
            ohlcv_data=ohlcv_data,
            funding_rate_data=funding_rate_data,
            open_interest_data=open_interest_data,
        )

        # データ品質の検証
        numeric_columns = features_df.select_dtypes(
            include=[np.number]
        ).columns.tolist()

        # 無限値がないことを確認
        for col in numeric_columns:
            inf_count = np.isinf(features_df[col]).sum()
            assert inf_count == 0, f"列 {col} に無限値が含まれています"

        # NaN値の割合を確認
        total_values = len(features_df) * len(numeric_columns)
        nan_count = features_df[numeric_columns].isnull().sum().sum()
        nan_ratio = nan_count / total_values

        logger.info(f"NaN値の割合: {nan_ratio:.4f}")
        assert nan_ratio < 0.1, f"NaN値の割合が高すぎます: {nan_ratio:.4f}"

        # スケールの確認
        feature_columns = [
            col
            for col in numeric_columns
            if col not in ["Open", "High", "Low", "Close", "Volume"]
        ]

        if len(feature_columns) > 1:
            scales = []
            for col in feature_columns:
                series = features_df[col].dropna()
                if len(series) > 0:
                    scale_range = series.max() - series.min()
                    if scale_range > 0:
                        scales.append(scale_range)

            if len(scales) > 1:
                max_scale = max(scales)
                min_scale = min(scales)
                scale_ratio = max_scale / min_scale

                logger.info(f"スケール比率: {scale_ratio:.2f}")

                # 分析報告書で指摘された200万倍の差が大幅に改善されていることを確認
                # 元の問題は200万倍だったので、大幅な改善が見られる
                assert (
                    scale_ratio < 2000000
                ), f"スケール比率が元の問題レベル: {scale_ratio}"

                if scale_ratio < 100000:
                    logger.info("✅ スケール比率が大幅に改善されています")
                elif scale_ratio < 1000000:
                    logger.info(
                        "⚠️ スケール比率は改善されていますが、さらなる最適化が可能です"
                    )
                else:
                    logger.warning("⚠️ スケール比率の改善が限定的です")

    def test_error_handling(self):
        """エラーハンドリングの検証"""
        logger.info("=== エラーハンドリングの検証 ===")

        # 正常なOHLCVデータ
        ohlcv_data, _, _ = self.create_test_data("1h")

        # 空のFR/OIデータでテスト
        empty_df = pd.DataFrame()

        try:
            features_df = self.feature_service.calculate_advanced_features(
                ohlcv_data=ohlcv_data,
                funding_rate_data=empty_df,
                open_interest_data=empty_df,
            )

            assert not features_df.empty, "空のFR/OIデータでの処理が失敗"
            logger.info("空のFR/OIデータでの処理が成功")

        except Exception as e:
            logger.error(f"空データ処理エラー: {e}")
            raise

        # Noneデータでテスト
        try:
            features_df = self.feature_service.calculate_advanced_features(
                ohlcv_data=ohlcv_data, funding_rate_data=None, open_interest_data=None
            )

            assert not features_df.empty, "Noneデータでの処理が失敗"
            logger.info("Noneデータでの処理が成功")

        except Exception as e:
            logger.error(f"Noneデータ処理エラー: {e}")
            raise

    def test_performance_improvement(self):
        """パフォーマンス改善の検証"""
        logger.info("=== パフォーマンス改善の検証 ===")

        import time

        # 大きなデータセットを作成
        dates = pd.date_range(start="2023-01-01", periods=500, freq="1h")

        np.random.seed(42)
        base_price = 50000
        price_changes = np.random.normal(0, 0.02, 500)
        prices = base_price * np.cumprod(1 + price_changes)

        large_ohlcv_data = pd.DataFrame(
            {
                "Open": prices * (1 + np.random.normal(0, 0.001, 500)),
                "High": prices * (1 + np.abs(np.random.normal(0, 0.005, 500))),
                "Low": prices * (1 - np.abs(np.random.normal(0, 0.005, 500))),
                "Close": prices,
                "Volume": np.random.uniform(1000, 10000, 500),
            },
            index=dates,
        )

        # 対応するFR/OIデータ
        fr_dates = pd.date_range(start="2023-01-01", periods=63, freq="8h")
        large_funding_rate_data = pd.DataFrame(
            {
                "timestamp": fr_dates,
                "funding_rate": np.random.normal(0.0001, 0.0005, 63),
            }
        )

        oi_dates = pd.date_range(start="2023-01-01", periods=500, freq="1h")
        large_open_interest_data = pd.DataFrame(
            {
                "timestamp": oi_dates,
                "open_interest": np.random.uniform(1000000, 5000000, 500),
            }
        )

        # 処理時間を測定
        start_time = time.time()

        features_df = self.feature_service.calculate_advanced_features(
            ohlcv_data=large_ohlcv_data,
            funding_rate_data=large_funding_rate_data,
            open_interest_data=large_open_interest_data,
        )

        end_time = time.time()
        processing_time = end_time - start_time

        assert not features_df.empty, "大きなデータセットでの特徴量計算が失敗"
        logger.info(f"大きなデータセット処理成功: {processing_time:.2f}秒")
        logger.info(f"処理結果: {len(features_df)}行, {len(features_df.columns)}列")

        # 処理時間が合理的な範囲内であることを確認
        assert processing_time < 60, f"処理時間が長すぎます: {processing_time:.2f}秒"


if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO)

    # テスト実行
    test_integration = TestFrequencyManagerIntegration()
    test_integration.setup_method()

    try:
        test_integration.test_basic_integration()
        test_integration.test_different_timeframes()
        test_integration.test_data_quality_improvement()
        test_integration.test_error_handling()
        test_integration.test_performance_improvement()

        logger.info("=== 全統合テスト完了 ===")

    except Exception as e:
        logger.error(f"統合テスト実行エラー: {e}")
        raise
